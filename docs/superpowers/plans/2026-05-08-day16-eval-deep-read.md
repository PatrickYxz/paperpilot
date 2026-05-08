# Day 16: deep-read 召回精度 eval 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 PaperPilot 出第一份第三方数据 + 多 baseline 对比的定量 eval 报告:QASPER NLP 子集 50 paper × 3 题 = 150 case,跑 3 组 baseline(abstract-only / full-text dump / PaperPilot 全栈),出三段递进通过率 + 失败归因 Top-N。

**Architecture:** 全部代码隔离在新增子包 `paperpilot/eval/`(4 模块)+ 3 个 `scripts/day16_*.py`。复用 `main.run(prompt, on_event=tracer)` 现有 hook 落 trace,**红线零侵入**(main.py / loop.py / adapter.py / builtin_tools/* / skills/* zero diff)。Pipeline 切 prepare → run → summarize 三步,run 阶段每个 case 完成立即 append jsonl,中断可续跑(skip 已完成 case_id)。

**Tech Stack:** Python 3.12, anthropic SDK over DeepSeek (现有 `LLMClient`), pytest, 标准库 json/re/dataclass。无新依赖。AI2 QASPER 数据集本地 dump。

**Spec:** `docs/superpowers/specs/2026-05-08-day16-eval-deep-read-design.md`

---

## 文件结构

| 路径 | 动作 | 责任 |
|---|---|---|
| `paperpilot/eval/__init__.py` | 新建,空 | 子包 marker |
| `paperpilot/eval/jsonl_tracer.py` | 新建 | `make_jsonl_tracer(run_id, out_dir) -> Callable[[str, dict], None]` |
| `paperpilot/eval/scorer.py` | 新建 | `is_pass()` + `cluster_failures()` 6 桶 |
| `paperpilot/eval/qasper_loader.py` | 新建 | `EvalCase` dataclass + `load_qasper_cases(path) -> list[EvalCase]` |
| `paperpilot/eval/baselines.py` | 新建 | `run_abstract_only` / `run_full_text_dump` / `run_paperpilot` |
| `tests/eval/__init__.py` | 新建,空 | |
| `tests/eval/fixtures/qasper_mini.json` | 新建 | 3 paper 测试 fixture |
| `tests/eval/test_jsonl_tracer.py` | 新建 | 3 test |
| `tests/eval/test_scorer.py` | 新建 | 8 test |
| `tests/eval/test_qasper_loader.py` | 新建 | 7 test |
| `scripts/day16_prepare_eval.py` | 新建 | 加载 QASPER → arxiv mapping → dry-run download → `data/eval/qasper_subset.jsonl` |
| `scripts/day16_run_eval.py` | 新建 | 对每 case 跑 3 baseline,append `data/eval/results_<baseline>.jsonl` |
| `scripts/day16_summarize.py` | 新建 | 三 result jsonl → `data/eval/summary.md` |
| `data/eval/.gitkeep` | 新建,空 | 目录入仓(注:nested .gitkeep 需 `git add -f` 越过 .gitignore `data/` 规则)|
| `data/traces/.gitkeep` | 新建,空 | 同上 |

**预计 fast suite 总数变化:** 116 → 134(+18: tracer 3 + scorer 8 + qasper 7)

---

## Task 1: `jsonl_tracer` + tests

**Files:**
- Create: `paperpilot/eval/__init__.py`
- Create: `paperpilot/eval/jsonl_tracer.py`
- Create: `tests/eval/__init__.py`
- Create: `tests/eval/test_jsonl_tracer.py`

### Step 1.1: 创建空 __init__.py

- [ ] 新建 `paperpilot/eval/__init__.py`,内容仅一行注释:

```python
"""Day 16 eval pipeline (deep-read recall accuracy)."""
```

- [ ] 新建 `tests/eval/__init__.py`,内容为空字符串(让 pytest 把 `tests/eval/` 当 package):

```python
```

### Step 1.2: 写失败 test

- [ ] 新建 `tests/eval/test_jsonl_tracer.py`,完整内容:

```python
"""Tests for paperpilot.eval.jsonl_tracer."""
import json
from pathlib import Path

from paperpilot.eval.jsonl_tracer import make_jsonl_tracer


def test_tracer_writes_two_events(tmp_path: Path) -> None:
    tracer = make_jsonl_tracer("case-001", tmp_path)
    tracer("tool_call", {"name": "load_skill", "arguments": {"name": "deep-read-paper"}})
    tracer("tool_result", {"name": "load_skill", "content": "skill body"})

    out = tmp_path / "case-001.jsonl"
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    assert e1["kind"] == "tool_call"
    assert e1["payload"]["name"] == "load_skill"
    e2 = json.loads(lines[1])
    assert e2["kind"] == "tool_result"


def test_tracer_creates_nested_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    tracer = make_jsonl_tracer("x", nested)
    tracer("guardrail_stop", {"reason": "max_iter"})
    assert (nested / "x.jsonl").exists()


def test_tracer_handles_non_json_payload(tmp_path: Path) -> None:
    """Anthropic SDK 返回 content 含不可直接 JSON 的对象,fallback 到 str()."""
    class Block:
        text = "hello world"

    tracer = make_jsonl_tracer("y", tmp_path)
    tracer("tool_result", {"name": "x", "content": [Block()]})
    out = tmp_path / "y.jsonl"
    line = json.loads(out.read_text(encoding="utf-8").strip())
    assert "hello world" in str(line["payload"])
```

### Step 1.3: 跑 test 确认失败

Run: `pytest tests/eval/test_jsonl_tracer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.eval.jsonl_tracer'`

### Step 1.4: 写实现

- [ ] 新建 `paperpilot/eval/jsonl_tracer.py`,完整内容:

```python
"""Eval-only tracer that persists agent_loop on_event payloads to JSONL.

Pure new module — does NOT modify main.run / loop / adapter.
Caller passes the returned fn via main.run(prompt, on_event=tracer).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def make_jsonl_tracer(run_id: str, out_dir: Path) -> Callable[[str, dict], None]:
    """Return a tracer fn that appends each (kind, payload) as a JSONL line.

    Truncates the target file on first call so reruns produce clean traces.
    Uses default=str for objects that aren't natively JSON-serializable
    (e.g. anthropic SDK content blocks).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    out_path.write_text("", encoding="utf-8")

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        record = {"kind": kind, "payload": payload}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return tracer
```

### Step 1.5: 跑 test 确认通过

Run: `pytest tests/eval/test_jsonl_tracer.py -v`
Expected: 3 passed

### Step 1.6: 跑全套 fast suite 确认无回归

Run: `pytest tests -q --ignore=tests/mcp_servers/test_colbert_via_client.py --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py --ignore=tests/test_per_paper_index_slow.py`
Expected: `119 passed`(116 + 3)

### Step 1.7: Commit

```powershell
git add paperpilot/eval/__init__.py paperpilot/eval/jsonl_tracer.py tests/eval/__init__.py tests/eval/test_jsonl_tracer.py
git commit -m "Day 16 Task 1: add eval JSONL tracer"
```

---

## Task 2: `scorer` + tests(`is_pass` + 失败归因 6 桶)

**Files:**
- Create: `paperpilot/eval/scorer.py`
- Create: `tests/eval/test_scorer.py`

### Step 2.1: 写失败 test

- [ ] 新建 `tests/eval/test_scorer.py`,完整内容:

```python
"""Tests for paperpilot.eval.scorer."""
from paperpilot.eval.scorer import cluster_failures, is_pass


def test_is_pass_contains_simple() -> None:
    assert is_pass("the dataset is SQuAD 2.0", ["SQuAD 2.0"])


def test_is_pass_case_insensitive() -> None:
    assert is_pass("we use squad 2.0 for training", ["SQuAD 2.0"])


def test_is_pass_strips_punct() -> None:
    assert is_pass("Answer: SQuAD.", ["SQuAD"])


def test_is_pass_normalizes_whitespace() -> None:
    assert is_pass("uses\nSQuAD\t2.0", ["SQuAD 2.0"])


def test_is_pass_multi_oracle_any_match() -> None:
    assert is_pass("uses MNLI dataset", ["SNLI", "MNLI"])


def test_is_pass_no_match() -> None:
    assert not is_pass("uses CoLA", ["SQuAD", "MNLI"])


def test_is_pass_empty_oracle_filtered() -> None:
    assert not is_pass("anything", ["", "  "])


def test_cluster_failures_bucket_priority() -> None:
    """Multiple buckets could match — first by priority wins."""
    records = [
        # no_load_skill (no load_skill call at all)
        {"passed": False, "tool_calls": ["mcp__arxiv__search_papers"]},
        # no_download (load_skill but no download)
        {"passed": False, "tool_calls": ["load_skill"]},
        # no_colbert_search (load + download but no colbert.search)
        {"passed": False, "tool_calls": [
            "load_skill", "mcp__arxiv__download_paper",
            "mcp__colbert__build_index",
        ]},
        # colbert_searched_low (only 2 searches)
        {"passed": False, "tool_calls": [
            "load_skill", "mcp__arxiv__download_paper",
            "mcp__colbert__build_index",
            "mcp__colbert__search", "mcp__colbert__search",
        ]},
        # synthesis_miss (full flow + 3+ searches)
        {"passed": False, "tool_calls": [
            "load_skill", "mcp__arxiv__download_paper",
            "mcp__colbert__build_index",
            "mcp__colbert__search", "mcp__colbert__search", "mcp__colbert__search",
        ]},
        # iter_exhausted via error field
        {"passed": False, "tool_calls": ["load_skill"], "error": "GuardrailStop: max_iter"},
        # passed → ignored
        {"passed": True, "tool_calls": []},
    ]
    counts = cluster_failures(records)
    assert counts == {
        "no_load_skill": 1,
        "no_download": 1,
        "no_colbert_search": 1,
        "colbert_searched_low": 1,
        "synthesis_miss": 1,
        "iter_exhausted": 1,
    }
```

### Step 2.2: 跑 test 确认失败

Run: `pytest tests/eval/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.eval.scorer'`

### Step 2.3: 写实现

- [ ] 新建 `paperpilot/eval/scorer.py`,完整内容:

```python
"""Pass/fail scoring + failure clustering for Day 16 eval."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


def _normalize(s: str) -> str:
    """Lowercase, strip outer punct/whitespace, fold internal whitespace."""
    return re.sub(r"\s+", " ", s.strip(" .,;:'\"\n\t")).lower()


def is_pass(predicted: str, oracle_spans: list[str] | tuple[str, ...]) -> bool:
    """True iff any non-empty oracle span (normalized) is a substring of
    normalized predicted text."""
    pred = _normalize(predicted)
    return any(_normalize(span) in pred for span in oracle_spans if span and span.strip())


def _classify(record: dict) -> str:
    """Bucket a single failed record by its tool_calls trace.

    Priority (top to bottom, first match wins):
        iter_exhausted -> no_load_skill -> no_download
        -> no_colbert_search -> colbert_searched_low -> synthesis_miss
    """
    err = record.get("error") or ""
    if "max_iter" in err or "GuardrailStop" in err:
        return "iter_exhausted"
    calls: list[str] = record.get("tool_calls") or []
    if "load_skill" not in calls:
        return "no_load_skill"
    if "mcp__arxiv__download_paper" not in calls:
        return "no_download"
    if "mcp__colbert__search" not in calls:
        return "no_colbert_search"
    n_search = sum(1 for c in calls if c == "mcp__colbert__search")
    if n_search < 3:
        return "colbert_searched_low"
    return "synthesis_miss"


def cluster_failures(records: Iterable[dict]) -> dict[str, int]:
    """Count failure buckets across records. Skips passed records."""
    fails = (r for r in records if not r.get("passed"))
    return dict(Counter(_classify(r) for r in fails))
```

### Step 2.4: 跑 test 确认通过

Run: `pytest tests/eval/test_scorer.py -v`
Expected: 8 passed

### Step 2.5: 跑全套 fast suite

Run: `pytest tests -q --ignore=tests/mcp_servers/test_colbert_via_client.py --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py --ignore=tests/test_per_paper_index_slow.py`
Expected: `127 passed`(119 + 8)

### Step 2.6: Commit

```powershell
git add paperpilot/eval/scorer.py tests/eval/test_scorer.py
git commit -m "Day 16 Task 2: add eval scorer (is_pass + 6-bucket failure clustering)"
```

---

## Task 3: `qasper_loader` + fixture + tests

**Files:**
- Create: `tests/eval/fixtures/qasper_mini.json`
- Create: `paperpilot/eval/qasper_loader.py`
- Create: `tests/eval/test_qasper_loader.py`

### Step 3.1: 写 fixture

- [ ] 新建 `tests/eval/fixtures/qasper_mini.json`,完整内容(3 paper:A 合规、B QA 不够、C 非 arxiv):

```json
{
  "paper_aaa": {
    "title": "Paper A: Attention Models",
    "abstract": "An abstract about attention.",
    "full_text": [
      {"section_name": "Introduction", "paragraphs": ["Intro p1.", "Intro p2."]},
      {"section_name": "Method", "paragraphs": ["Method p1."]}
    ],
    "qas": [
      {"question": "Q1?", "answers": [{"answer": {"extractive_spans": ["span-1a"]}}]},
      {"question": "Q2?", "answers": [{"answer": {"extractive_spans": ["span-2a", "span-2b"]}}]},
      {"question": "Q3?", "answers": [{"answer": {"extractive_spans": ["span-3a"]}}]},
      {"question": "Q4?", "answers": [{"answer": {"extractive_spans": ["span-4a"]}}]}
    ],
    "paper_url": "https://arxiv.org/abs/2001.12345"
  },
  "paper_bbb": {
    "title": "Paper B (only 2 extractive QA)",
    "abstract": "B abstract.",
    "full_text": [{"section_name": "X", "paragraphs": ["b1"]}],
    "qas": [
      {"question": "Q1?", "answers": [{"answer": {"extractive_spans": ["b-span1"]}}]},
      {"question": "Q2?", "answers": [{"answer": {"extractive_spans": ["b-span2"]}}]}
    ],
    "paper_url": "https://arxiv.org/abs/2002.45678"
  },
  "paper_ccc": {
    "title": "Paper C (no arxiv URL)",
    "abstract": "C abstract.",
    "full_text": [],
    "qas": [
      {"question": "Q?", "answers": [{"answer": {"extractive_spans": ["c1"]}}]},
      {"question": "Q?", "answers": [{"answer": {"extractive_spans": ["c2"]}}]},
      {"question": "Q?", "answers": [{"answer": {"extractive_spans": ["c3"]}}]}
    ],
    "paper_url": "https://aclweb.org/foo"
  }
}
```

### Step 3.2: 写失败 test

- [ ] 新建 `tests/eval/test_qasper_loader.py`,完整内容:

```python
"""Tests for paperpilot.eval.qasper_loader."""
from pathlib import Path

from paperpilot.eval.qasper_loader import (
    EvalCase,
    extract_arxiv_id,
    load_qasper_cases,
)

FIXTURE = Path(__file__).parent / "fixtures" / "qasper_mini.json"


def test_extract_arxiv_id_abs() -> None:
    assert extract_arxiv_id("https://arxiv.org/abs/2001.12345") == "2001.12345"


def test_extract_arxiv_id_pdf() -> None:
    assert extract_arxiv_id("https://arxiv.org/pdf/2001.12345.pdf") == "2001.12345"


def test_extract_arxiv_id_non_arxiv_returns_none() -> None:
    assert extract_arxiv_id("https://aclweb.org/foo") is None
    assert extract_arxiv_id(None) is None
    assert extract_arxiv_id("") is None


def test_load_qasper_filters_papers_with_lt_3_extractive_or_no_arxiv() -> None:
    cases = load_qasper_cases(FIXTURE)
    arxiv_ids = {c.arxiv_id for c in cases}
    # paper_bbb dropped (only 2 QA); paper_ccc dropped (not arxiv)
    assert arxiv_ids == {"2001.12345"}


def test_load_qasper_takes_first_three_qa_in_order() -> None:
    cases = load_qasper_cases(FIXTURE)
    a_cases = [c for c in cases if c.arxiv_id == "2001.12345"]
    assert len(a_cases) == 3
    assert [c.question for c in a_cases] == ["Q1?", "Q2?", "Q3?"]
    # case_id format
    assert a_cases[0].case_id == "qasper-2001.12345-q0"
    assert a_cases[2].case_id == "qasper-2001.12345-q2"


def test_load_qasper_full_text_concatenated() -> None:
    cases = load_qasper_cases(FIXTURE)
    a = next(c for c in cases if c.arxiv_id == "2001.12345")
    assert "## Introduction" in a.full_text
    assert "Intro p1." in a.full_text
    assert "## Method" in a.full_text
    assert "Method p1." in a.full_text
    assert isinstance(a, EvalCase)


def test_load_qasper_multi_span_oracle_preserved() -> None:
    cases = load_qasper_cases(FIXTURE)
    a_cases = [c for c in cases if c.arxiv_id == "2001.12345"]
    q2 = next(c for c in a_cases if c.question == "Q2?")
    assert q2.oracle_spans == ("span-2a", "span-2b")
```

### Step 3.3: 跑 test 确认失败

Run: `pytest tests/eval/test_qasper_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.eval.qasper_loader'`

### Step 3.4: 写实现

- [ ] 新建 `paperpilot/eval/qasper_loader.py`,完整内容:

```python
"""Load QASPER paper dump → EvalCase list, filtering for arxiv + extractive QA."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    arxiv_id: str
    paper_title: str
    abstract: str
    full_text: str
    question: str
    oracle_spans: tuple[str, ...]


def extract_arxiv_id(url: str | None) -> str | None:
    if not url:
        return None
    m = ARXIV_RE.search(url)
    return m.group(1) if m else None


def _join_full_text(full_text: list[dict] | None) -> str:
    """Concatenate QASPER section list into a single markdown-ish string."""
    if not full_text:
        return ""
    parts: list[str] = []
    for section in full_text:
        if not isinstance(section, dict):
            continue
        name = section.get("section_name") or ""
        paragraphs = section.get("paragraphs") or []
        if name:
            parts.append(f"## {name}")
        parts.extend(p for p in paragraphs if isinstance(p, str) and p)
    return "\n\n".join(parts)


def _collect_extractive_qas(qas: list) -> list[tuple[str, tuple[str, ...]]]:
    """Return [(question, oracle_spans), ...] in original order, dedup spans."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for qa in qas or []:
        if not isinstance(qa, dict):
            continue
        question = qa.get("question") or ""
        spans: list[str] = []
        for ans in qa.get("answers") or []:
            if not isinstance(ans, dict):
                continue
            inner = ans.get("answer") if isinstance(ans.get("answer"), dict) else ans
            es = inner.get("extractive_spans") if isinstance(inner, dict) else None
            if es:
                spans.extend(s for s in es if isinstance(s, str) and s.strip())
        if question and spans:
            seen: set[str] = set()
            uniq: list[str] = []
            for s in spans:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            out.append((question, tuple(uniq)))
    return out


def load_qasper_cases(qasper_path: Path) -> list[EvalCase]:
    """Parse QASPER dump → list of EvalCase.

    Per spec §6: paper must have arxiv_url AND ≥ 3 extractive QA (else dropped).
    Each qualifying paper contributes exactly 3 cases (first 3 in original order).
    """
    raw = json.loads(qasper_path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for paper in raw.values():
        if not isinstance(paper, dict):
            continue
        arxiv_id = extract_arxiv_id(paper.get("paper_url"))
        if arxiv_id is None:
            continue
        qa_list = _collect_extractive_qas(paper.get("qas") or [])
        if len(qa_list) < 3:
            continue
        full_text = _join_full_text(paper.get("full_text"))
        for idx, (q, spans) in enumerate(qa_list[:3]):
            cases.append(EvalCase(
                case_id=f"qasper-{arxiv_id}-q{idx}",
                arxiv_id=arxiv_id,
                paper_title=paper.get("title") or "",
                abstract=paper.get("abstract") or "",
                full_text=full_text,
                question=q,
                oracle_spans=spans,
            ))
    return cases
```

### Step 3.5: 跑 test 确认通过

Run: `pytest tests/eval/test_qasper_loader.py -v`
Expected: 7 passed

### Step 3.6: 跑全套 fast suite

Run: `pytest tests -q --ignore=tests/mcp_servers/test_colbert_via_client.py --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py --ignore=tests/test_per_paper_index_slow.py`
Expected: `134 passed`(127 + 7)

### Step 3.7: Commit

```powershell
git add paperpilot/eval/qasper_loader.py tests/eval/test_qasper_loader.py tests/eval/fixtures/qasper_mini.json
git commit -m "Day 16 Task 3: add QASPER loader (arxiv + extractive filter)"
```

---

## Task 4: `baselines.py`(3 组跑法,无单测,Task 6 dry-run 集成验证)

**Files:**
- Create: `paperpilot/eval/baselines.py`

### Step 4.1: 写实现

- [ ] 新建 `paperpilot/eval/baselines.py`,完整内容:

```python
"""Three baselines for Day 16 deep-read eval.

Each fn returns a dict with keys:
    predicted: str
    elapsed_s: float
    trace_path: str | None      # only paperpilot baseline
    tool_calls: list[str] | None # only paperpilot baseline
    error: str | None
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from paperpilot.core import LLMClient
from paperpilot.eval.jsonl_tracer import make_jsonl_tracer
from paperpilot.eval.qasper_loader import EvalCase

_TRACE_DIR = Path("data/traces")
# ~60K tokens at 4 chars/token (cl100k_base avg). Char-based to avoid tiktoken dep.
_FULL_TEXT_CHAR_BUDGET = 240_000


def run_abstract_only(case: EvalCase, client: LLMClient | None = None) -> dict[str, Any]:
    client = client or LLMClient()
    prompt = (
        f"你是学术论文助手。下面是论文 \"{case.paper_title}\" 的 abstract:\n\n"
        f"{case.abstract}\n\n"
        f"请基于 abstract 简洁回答以下问题。"
        f"如果 abstract 不含答案,直接说\"abstract 中未提及\":\n\n"
        f"Q: {case.question}\nA:"
    )
    return _one_shot(prompt, client)


def run_full_text_dump(case: EvalCase, client: LLMClient | None = None) -> dict[str, Any]:
    client = client or LLMClient()
    text = case.full_text or case.abstract
    truncated = len(text) > _FULL_TEXT_CHAR_BUDGET
    if truncated:
        body = text[:_FULL_TEXT_CHAR_BUDGET]
        notice = (
            f"⚠ 全文过长,已截至 ~{_FULL_TEXT_CHAR_BUDGET // 4} tokens, "
            f"后段省略\n\n"
        )
    else:
        body = text
        notice = ""
    prompt = (
        f"{notice}你是学术论文助手。下面是论文 \"{case.paper_title}\" 的全文:\n\n"
        f"{body}\n\n"
        f"请基于全文回答以下问题:\n\nQ: {case.question}\nA:"
    )
    return _one_shot(prompt, client)


def run_paperpilot(case: EvalCase, max_iter: int = 12) -> dict[str, Any]:
    """Run main.run end-to-end with composite tracer.

    Captures tool_calls list AND writes per-case JSONL trace under data/traces/.
    """
    from paperpilot.main import run as agent_run  # local import to avoid load on unit tests

    tool_calls: list[str] = []
    file_tracer = make_jsonl_tracer(case.case_id, _TRACE_DIR)
    trace_path = _TRACE_DIR / f"{case.case_id}.jsonl"

    def composite_tracer(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            tool_calls.append(payload.get("name", ""))
        file_tracer(kind, payload)

    prompt = (
        f"请精读 arxiv:{case.arxiv_id}(标题《{case.paper_title}》),"
        f"回答下面的问题。使用 deep-read-paper skill 的工作流"
        f"(load_skill -> download_paper -> build_index -> "
        f"多次 colbert.search -> 综合)。\n\n"
        f"Q: {case.question}\nA:"
    )

    t0 = time.time()
    predicted = ""
    error: str | None = None
    try:
        messages = agent_run(prompt, max_iter=max_iter, on_event=composite_tracer)
        predicted = _extract_final_text(messages[-1].get("content"))
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    elapsed = round(time.time() - t0, 1)

    return {
        "predicted": predicted,
        "elapsed_s": elapsed,
        "trace_path": str(trace_path),
        "tool_calls": tool_calls,
        "error": error,
    }


def _one_shot(prompt: str, client: LLMClient) -> dict[str, Any]:
    t0 = time.time()
    predicted = ""
    error: str | None = None
    try:
        resp = client.call(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system="",
        )
        predicted = resp.text or ""
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    elapsed = round(time.time() - t0, 1)
    return {
        "predicted": predicted,
        "elapsed_s": elapsed,
        "trace_path": None,
        "tool_calls": None,
        "error": error,
    }


def _extract_final_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    parts: list[str] = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(p for p in parts if p)
```

### Step 4.2: import smoke

Run(确保模块可 import,无语法错):
```
python -c "from paperpilot.eval.baselines import run_abstract_only, run_full_text_dump, run_paperpilot; print('ok')"
```
Expected: `ok`

### Step 4.3: 跑全套 fast suite 确认无回归

Run: `pytest tests -q --ignore=tests/mcp_servers/test_colbert_via_client.py --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py --ignore=tests/test_per_paper_index_slow.py`
Expected: `134 passed`(无新单测,但确认 import 没破坏现有)

### Step 4.4: Commit

```powershell
git add paperpilot/eval/baselines.py
git commit -m "Day 16 Task 4: add 3 baselines (abstract / full_text / paperpilot)"
```

---

## Task 5: `prepare_eval.py` + 真实生成 `qasper_subset.jsonl`

**Files:**
- Create: `scripts/day16_prepare_eval.py`
- Create: `data/eval/.gitkeep`(空)
- Create: `data/traces/.gitkeep`(空)

### Step 5.1: 用户先手工下载 QASPER

> 这一步**不在脚本内**,执行者需要先确认下载完成。

- [ ] 在 PowerShell 跑(或手工浏览器下载):

```powershell
New-Item -ItemType Directory -Force -Path data/eval/qasper-source | Out-Null
Invoke-WebRequest -Uri "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz" -OutFile data/eval/qasper-source/qasper.tgz
tar -xzf data/eval/qasper-source/qasper.tgz -C data/eval/qasper-source/
ls data/eval/qasper-source/
```

Expected: 解压出含 `qasper-train-v0.3.json`(~30MB)。**若下载失败**(如网络问题),fallback 用 HuggingFace:
```
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('allenai/qasper'); ds.save_to_disk('data/eval/qasper-source/hf')"
```
然后改 `QASPER_PATH` 指向相应 json(执行者根据实际文件位置调整下面 Step 5.2 中的路径)。

### Step 5.2: 写脚本

- [ ] 新建 `scripts/day16_prepare_eval.py`,完整内容:

```python
"""Day 16 prepare: QASPER → 50 paper × 3 题 EvalCase jsonl.

Usage:
    python scripts/day16_prepare_eval.py

Reads QASPER from data/eval/qasper-source/qasper-train-v0.3.json
(downloaded manually per plan Step 5.1).
Writes data/eval/qasper_subset.jsonl with up to 50 papers × 3 cases each
(=150 cases). Falls back to 40 papers minimum (else FAIL).

Each candidate paper is dry-run downloaded via mcp__arxiv__download_paper
to filter out unreachable arxiv IDs.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.eval.qasper_loader import load_qasper_cases
from paperpilot.tools.mcp_client import MCPClient

QASPER_PATH = Path("data/eval/qasper-source/qasper-train-v0.3.json")
OUT_PATH = Path("data/eval/qasper_subset.jsonl")
TARGET_PAPERS = 50
MIN_PAPERS = 40
MANIFEST = Path(__file__).resolve().parent.parent / "paperpilot" / "mcp_servers.json"


def main() -> None:
    if not QASPER_PATH.exists():
        sys.exit(
            f"FAIL: {QASPER_PATH} not found. "
            "Manually download per plan Step 5.1 first."
        )

    print(f"Loading QASPER from {QASPER_PATH}...")
    all_cases = load_qasper_cases(QASPER_PATH)
    print(f"  {len(all_cases)} candidate cases ({len(all_cases) // 3} papers)")

    # Group by paper for ordered iteration
    papers_in_order: list[str] = []
    cases_by_paper: dict[str, list] = {}
    for c in all_cases:
        if c.arxiv_id not in cases_by_paper:
            papers_in_order.append(c.arxiv_id)
            cases_by_paper[c.arxiv_id] = []
        cases_by_paper[c.arxiv_id].append(c)

    print(f"\nDry-running mcp__arxiv__download_paper on each paper...")
    mcp = MCPClient(MANIFEST)
    mcp.start()
    accepted_papers: list[str] = []
    rejected: list[tuple[str, str]] = []
    try:
        for arxiv_id in papers_in_order:
            if len(accepted_papers) >= TARGET_PAPERS:
                break
            try:
                result = mcp.call_tool(
                    "mcp__arxiv__download_paper",
                    {"arxiv_id": arxiv_id},
                )
                # 期望返回含 paper_id + text(>500 chars)的 dict
                txt = result.get("text") if isinstance(result, dict) else ""
                if not isinstance(txt, str) or len(txt) < 500:
                    rejected.append((arxiv_id, "text too short or missing"))
                    continue
                accepted_papers.append(arxiv_id)
                print(f"  [{len(accepted_papers):>2}/{TARGET_PAPERS}] OK arxiv:{arxiv_id} ({len(txt)} chars)")
            except Exception as e:  # noqa: BLE001
                rejected.append((arxiv_id, f"{type(e).__name__}: {e}"))
                print(f"  -- skip arxiv:{arxiv_id}: {type(e).__name__}")
    finally:
        mcp.close()

    print(f"\nAccepted: {len(accepted_papers)}, Rejected: {len(rejected)}")

    if len(accepted_papers) < MIN_PAPERS:
        sys.exit(
            f"FAIL: only {len(accepted_papers)} papers accepted "
            f"(need >= {MIN_PAPERS}). Check arxiv quota / try later."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for arxiv_id in accepted_papers:
            for c in cases_by_paper[arxiv_id]:
                row = asdict(c)
                row["oracle_spans"] = list(row["oracle_spans"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(accepted_papers) * 3
    print(f"\nWrote {OUT_PATH} with {len(accepted_papers)} papers x 3 = {total} cases")


if __name__ == "__main__":
    main()
```

### Step 5.3: 跑 prepare(真实)

Run:
```
python scripts/day16_prepare_eval.py
```
Expected: 终端打 `OK arxiv:NNNN.NNNNN` 共 50 行(或 ≥ 40),最终 `Wrote data/eval/qasper_subset.jsonl with N papers x 3 = M cases`。耗时 5-15 min(每篇 5-15s arxiv download)。

### Step 5.4: 创建 .gitkeep

- [ ] 创建空文件:

```powershell
New-Item -ItemType File -Force -Path data/eval/.gitkeep | Out-Null
New-Item -ItemType File -Force -Path data/traces/.gitkeep | Out-Null
```

### Step 5.5: 验证 qasper_subset.jsonl 内容

Run:
```powershell
(Get-Content data/eval/qasper_subset.jsonl | Measure-Object -Line).Lines
Get-Content data/eval/qasper_subset.jsonl -TotalCount 1
```
Expected: 行数 ≥ 120(40 paper × 3),且第 1 行是合法 JSON 含 `case_id`/`arxiv_id`/`question`/`oracle_spans` 等字段。

### Step 5.6: Commit(脚本 + .gitkeep,**不**入 jsonl)

> qasper_subset.jsonl 不入仓(被 .gitignore 的 `data/` 规则 cover);.gitkeep 需 `git add -f` 越过规则。

```powershell
git add scripts/day16_prepare_eval.py
git add -f data/eval/.gitkeep data/traces/.gitkeep
git commit -m "Day 16 Task 5: add prepare_eval script + data/eval & data/traces dirs"
```

---

## Task 6: `run_eval.py` + dry-run 验证

**Files:**
- Create: `scripts/day16_run_eval.py`

### Step 6.1: 写脚本

- [ ] 新建 `scripts/day16_run_eval.py`,完整内容:

```python
"""Day 16 run: 跑 baseline 之一(或全部)over qasper_subset.jsonl,
每 case 完成立即 append 到 results_<baseline>.jsonl,可中断续跑。

Usage:
    # dry-run 2 case 验证 pipeline
    python scripts/day16_run_eval.py --baseline abstract_only --limit 2

    # 跑全部 abstract baseline
    python scripts/day16_run_eval.py --baseline abstract_only

    # 跑全部 3 baseline 顺序执行
    python scripts/day16_run_eval.py --baseline all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.eval.baselines import (
    run_abstract_only,
    run_full_text_dump,
    run_paperpilot,
)
from paperpilot.eval.qasper_loader import EvalCase
from paperpilot.eval.scorer import is_pass

SUBSET_PATH = Path("data/eval/qasper_subset.jsonl")
OUT_DIR = Path("data/eval")

BASELINE_FNS: dict[str, Callable[[EvalCase], dict]] = {
    "abstract_only": run_abstract_only,
    "full_text": run_full_text_dump,
    "paperpilot": run_paperpilot,
}


def _load_cases() -> list[EvalCase]:
    if not SUBSET_PATH.exists():
        sys.exit(f"FAIL: {SUBSET_PATH} not found. Run day16_prepare_eval.py first.")
    cases: list[EvalCase] = []
    for line in SUBSET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cases.append(EvalCase(
            case_id=d["case_id"],
            arxiv_id=d["arxiv_id"],
            paper_title=d["paper_title"],
            abstract=d["abstract"],
            full_text=d["full_text"],
            question=d["question"],
            oracle_spans=tuple(d["oracle_spans"]),
        ))
    return cases


def _existing_case_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line)["case_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def run_one_baseline(baseline: str, cases: list[EvalCase], limit: int | None) -> None:
    fn = BASELINE_FNS[baseline]
    out_path = OUT_DIR / f"results_{baseline}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_case_ids(out_path)
    print(f"\n=== Baseline: {baseline} ===")
    print(f"  output: {out_path}")
    print(f"  total cases: {len(cases)}, already done: {len(done)}")

    todo = [c for c in cases if c.case_id not in done]
    if limit is not None:
        todo = todo[:limit]

    for i, case in enumerate(todo, 1):
        print(f"\n  [{i}/{len(todo)}] {case.case_id} :: {case.question[:60]}...")
        ans = fn(case)
        passed = is_pass(ans["predicted"], list(case.oracle_spans))
        record = {
            "case_id": case.case_id,
            "baseline": baseline,
            "question": case.question,
            "oracle_spans": list(case.oracle_spans),
            "predicted": ans["predicted"],
            "passed": passed,
            "elapsed_s": ans["elapsed_s"],
            "trace_path": ans.get("trace_path"),
            "tool_calls": ans.get("tool_calls"),
            "error": ans.get("error"),
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        status = "PASS" if passed else ("ERR " if ans.get("error") else "FAIL")
        print(f"    -> {status} ({ans['elapsed_s']}s) "
              f"predicted: {ans['predicted'][:120]}...")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        choices=["abstract_only", "full_text", "paperpilot", "all"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = _load_cases()
    print(f"Loaded {len(cases)} cases from {SUBSET_PATH}")

    baselines = (
        list(BASELINE_FNS.keys()) if args.baseline == "all" else [args.baseline]
    )
    for b in baselines:
        run_one_baseline(b, cases, args.limit)
    print("\nDone.")


if __name__ == "__main__":
    main()
```

### Step 6.2: dry-run abstract baseline 跑 2 case

Run:
```
python scripts/day16_run_eval.py --baseline abstract_only --limit 2
```
Expected: 2 个 `[1/2]` / `[2/2]` 行,每行后跟 `PASS` / `FAIL` / `ERR` 状态,最终 `data/eval/results_abstract_only.jsonl` 有 2 行。耗时 < 30s。

### Step 6.3: dry-run paperpilot baseline 跑 1 case 验证 trace 落盘

Run:
```
python scripts/day16_run_eval.py --baseline paperpilot --limit 1
```
Expected: 1 个 `[1/1]` 行,~60-120s,完成后:
- `data/eval/results_paperpilot.jsonl` 有 1 行,含 `trace_path` 和非空 `tool_calls`
- `data/traces/qasper-NNNN.NNNNN-q0.jsonl` 文件存在,内容是多行 JSONL,含 `tool_call` / `tool_result` 事件

验证 trace 不空:
```powershell
Get-Content (Get-ChildItem data/traces/*.jsonl | Select-Object -First 1).FullName | Select-Object -First 3
```

### Step 6.4: 续跑测试(再跑 --limit 2 应 skip 已做)

Run:
```
python scripts/day16_run_eval.py --baseline abstract_only --limit 2
```
Expected: 终端打 `already done: 2`,且 todo 列表去除已完成后 limit 为 2 但实际跑 0 case(或继续做下一批)。`results_abstract_only.jsonl` 仍只有 2 行(不重复 append)。

### Step 6.5: 跑全套 fast suite 确认无回归

Run: `pytest tests -q --ignore=tests/mcp_servers/test_colbert_via_client.py --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py --ignore=tests/test_per_paper_index_slow.py`
Expected: `134 passed`

### Step 6.6: Commit

```powershell
git add scripts/day16_run_eval.py
git commit -m "Day 16 Task 6: add run_eval script (3 baseline + resumable)"
```

---

## Task 7: 跑全部 3 baseline → 3 个 results jsonl

**Files:**
- 产出(不入仓): `data/eval/results_{abstract_only,full_text,paperpilot}.jsonl`
- 产出(不入仓): `data/traces/qasper-*.jsonl`(~150 个)

### Step 7.1: 跑 abstract_only(快,~15 min)

Run:
```
python scripts/day16_run_eval.py --baseline abstract_only
```
Expected: 150(或更少,看 prepare 实际数)case 全跑完,终端最后一行 `Done.`,`results_abstract_only.jsonl` 行数 = case 数。中途若挂了重跑会自动续跑。

### Step 7.2: 跑 full_text(中,~50 min)

Run:
```
python scripts/day16_run_eval.py --baseline full_text
```
Expected: 同上,`results_full_text.jsonl` 行数 = case 数。耗时 30-60 min。

### Step 7.3: 跑 paperpilot(慢,~2.5h,可后台)

Run(可后台 / 单独终端跑):
```
python scripts/day16_run_eval.py --baseline paperpilot
```
Expected: 每 case 60-120s,总 2-3h。`results_paperpilot.jsonl` 行数 = case 数。`data/traces/` 含每 case 一个 jsonl。

中途异常 case 会写一行 `error` 字段非 null,不阻塞剩余。最终允许 ≤ 5% case error,否则视为脚本不可信(对策:重跑 error case;由于 resume 机制,需手工先 grep `"error":` 行删掉再续跑)。

### Step 7.4: 三组数字快速 sanity

Run:
```powershell
foreach ($b in 'abstract_only','full_text','paperpilot') {
    $f = "data/eval/results_$b.jsonl"
    $total = (Get-Content $f | Measure-Object -Line).Lines
    $pass = (Select-String -Path $f -Pattern '"passed": true' | Measure-Object).Count
    $err  = (Select-String -Path $f -Pattern '"error": "[^n]' | Measure-Object).Count
    Write-Host "$b : total=$total pass=$pass err=$err"
}
```
Expected: 三行打印,`paperpilot pass` > `full_text pass` > `abstract_only pass`(若不成立,记录,Day 17 处理)。

### Step 7.5: 不 commit 大产物

> `results_*.jsonl` 和 `data/traces/*.jsonl` 全部走 .gitignore `data/` 规则,**不入仓**,只在本地。Day 17 case study 时从 `data/traces/` 挑亮点 case 引用。

无 commit。

---

## Task 8: `summarize.py` + 出 `summary.md` + 红线 diff + 最终验证

**Files:**
- Create: `scripts/day16_summarize.py`
- 产出: `data/eval/summary.md`(可入仓:Day 17 README 会引用)

### Step 8.1: 写脚本

- [ ] 新建 `scripts/day16_summarize.py`,完整内容:

```python
"""Day 16 summarize: 三份 results_<baseline>.jsonl → summary.md。

Usage:
    python scripts/day16_summarize.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.eval.scorer import cluster_failures

EVAL_DIR = Path("data/eval")
OUT_PATH = EVAL_DIR / "summary.md"
BASELINES = ["abstract_only", "full_text", "paperpilot"]


def _load(baseline: str) -> list[dict]:
    p = EVAL_DIR / f"results_{baseline}.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stats(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {"total": 0, "pass": 0, "fail": 0, "err": 0, "rate": 0.0, "avg_s": 0.0}
    err = sum(1 for r in records if r.get("error"))
    passed = sum(1 for r in records if r.get("passed"))
    fail = total - passed - err
    avg_s = mean(r.get("elapsed_s", 0) for r in records) if records else 0.0
    return {
        "total": total,
        "pass": passed,
        "fail": fail,
        "err": err,
        "rate": passed / total if total else 0.0,
        "avg_s": avg_s,
    }


def _fmt_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def main() -> None:
    all_records = {b: _load(b) for b in BASELINES}
    stats = {b: _stats(rs) for b, rs in all_records.items()}

    if any(s["total"] == 0 for s in stats.values()):
        missing = [b for b, s in stats.items() if s["total"] == 0]
        sys.exit(f"FAIL: results_{missing[0]}.jsonl is empty. Run day16_run_eval.py first.")

    pp_stats = stats["paperpilot"]
    fail_buckets = cluster_failures(all_records["paperpilot"])
    fail_total = sum(fail_buckets.values())
    fail_rows = sorted(fail_buckets.items(), key=lambda kv: -kv[1])

    lines: list[str] = []
    lines.append("# Day 16 deep-read 召回精度 eval")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("**Dataset**: AI2 QASPER NLP subset (extractive QA)")
    lines.append(f"**Cases per baseline**: {pp_stats['total']}")
    lines.append("**LLM**: DeepSeek (via anthropic SDK), shared across all baselines")
    lines.append("")
    lines.append("## 三段对比")
    lines.append("")
    lines.append("| Baseline | Pass | Fail | Error | Pass Rate | Avg latency |")
    lines.append("|---|---|---|---|---|---|")
    for b in BASELINES:
        s = stats[b]
        lines.append(
            f"| {b} | {s['pass']} | {s['fail']} | {s['err']} | "
            f"**{_fmt_pct(s['rate'])}** | {s['avg_s']:.1f}s |"
        )
    lines.append("")

    lines.append("## PaperPilot 失败归因")
    lines.append("")
    lines.append("| 桶 | 计数 | 占比 |")
    lines.append("|---|---|---|")
    for bucket, count in fail_rows:
        pct = (count / fail_total * 100) if fail_total else 0
        lines.append(f"| {bucket} | {count} | {pct:.0f}% |")
    if not fail_rows:
        lines.append("| (无失败) | 0 | - |")
    lines.append("")

    lines.append("## 解读")
    lines.append("")
    a = stats["abstract_only"]["rate"]
    f = stats["full_text"]["rate"]
    p = stats["paperpilot"]["rate"]
    if f > a:
        lines.append(
            f"- abstract → full_text 提升 **{(f - a) * 100:+.1f}pts**:"
            f"细节召回需要正文,abstract 远不够"
        )
    else:
        lines.append(
            f"- abstract → full_text **未提升**({(f - a) * 100:+.1f}pts):"
            f"可能题目对 abstract 已经友好,或 full_text 截断过严"
        )
    if p > f:
        lines.append(
            f"- full_text → paperpilot 提升 **{(p - f) * 100:+.1f}pts**:"
            f"colbert 选段 + 多次召回 比 全文一次性塞 LLM 更优,验证 RAG 路线价值"
        )
    else:
        lines.append(
            f"- full_text → paperpilot **未提升**({(p - f) * 100:+.1f}pts):"
            f"故事破产,Day 17 加 BM25 baseline 或换难题型重跑"
        )

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")
    print("\n--- preview ---")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

### Step 8.2: 跑 summarize

Run:
```
python scripts/day16_summarize.py
```
Expected: 终端打 `Wrote data/eval/summary.md` + 完整 markdown preview。`data/eval/summary.md` 文件就位。

### Step 8.3: 人眼读 summary.md 验收

- [ ] 打开 `data/eval/summary.md`,确认:
  - 三段通过率表完整 3 行,数字非 0
  - 失败归因 Top 桶 ≥ 1 行(若 paperpilot 通过率 100% 则可空)
  - 解读段两句中文,数字方向自动对齐(p > f 给"提升",p < f 给"破产")

### Step 8.4: 红线 zero-diff 验证

Run:
```
git diff main -- paperpilot/main.py paperpilot/core/loop.py paperpilot/core/adapter.py paperpilot/core/guardrail.py paperpilot/core/__init__.py paperpilot/builtin_tools/ paperpilot/skills/ paperpilot/mcp_servers.json
```
Expected: **空输出**(零字符 diff)。若有任何 diff → 立即排查并 revert。

### Step 8.5: 跑全套 fast suite 最终确认

Run: `pytest tests -q --ignore=tests/mcp_servers/test_colbert_via_client.py --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py --ignore=tests/test_per_paper_index_slow.py`
Expected: `134 passed`(116 基线 + 18 新单测:3 tracer + 8 scorer + 7 qasper = 18)

### Step 8.6: Commit summarize 脚本 + summary.md(入仓)

> `data/eval/summary.md` 是 Day 17 README 要引用的产物,需入仓。`data/` 走 .gitignore,需 `git add -f`。其它 results jsonl 和 traces 不入仓。

```powershell
git add scripts/day16_summarize.py
git add -f data/eval/summary.md
git commit -m "Day 16 Task 8: add summarize script + final summary.md"
```

---

## DoD 验证(全部 task 完成后跑一遍)

- [ ] `pytest tests -q --ignore=tests/mcp_servers/test_colbert_via_client.py --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py --ignore=tests/test_per_paper_index_slow.py` → `133 passed`
- [ ] `git diff main -- paperpilot/main.py paperpilot/core/ paperpilot/builtin_tools/ paperpilot/skills/ paperpilot/mcp_servers.json` → 空(红线零 diff)
- [ ] `data/eval/qasper_subset.jsonl` 行数 ≥ 120(40 paper × 3)
- [ ] `data/eval/results_abstract_only.jsonl` / `results_full_text.jsonl` / `results_paperpilot.jsonl` 三份就位,行数 = subset 行数(允许 ≤ 5% error 行)
- [ ] `data/traces/` 含 ≥ subset 行数 个 jsonl 文件,每个 ≥ 5 行 trace 事件
- [ ] `data/eval/summary.md` 含三段通过率表 + 失败归因 + 解读段
- [ ] `paperpilot pass rate > full_text pass rate`(若不成立,在 summary.md 末尾 + Day 17 计划里记录)
- [ ] `git log --oneline | head -10` 含 Day 16 spec / Day 16 plan(下个 commit) / Day 16 Task 1-8 共 ≥ 9 条