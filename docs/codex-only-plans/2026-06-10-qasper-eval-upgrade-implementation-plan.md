# QASPER Eval Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first QASPER eval-upgrade slice: enriched answer/evidence metadata, an enriched subset writer, a dataset profile report, and a semantic-audit schema without changing the current strict scorer.

**Architecture:** Keep the existing Day 16 strict eval path intact. Add enriched data structures and offline scripts beside the existing loader/runner so historical `results_*.jsonl` and `passed` semantics remain unchanged.

**Tech Stack:** Python stdlib dataclasses, JSONL files, pytest, existing `paperpilot.eval` package, existing `scripts/day16_*` eval layout.

---

## Scope

This plan implements the first slice from:

```text
docs/superpowers/specs/2026-06-10-qasper-eval-upgrade-design.md
```

It does not modify:

```text
paperpilot/eval/scorer.py
paperpilot/eval/baselines.py
scripts/day16_run_eval.py
PaperPilot answer generation prompts
deep-read workflow
```

The core rule is:

```text
existing strict span score stays exactly as it is
```

## File Structure

### Modify: `paperpilot/eval/qasper_loader.py`

Responsibility:

- Keep existing `EvalCase` and `load_qasper_cases()` behavior unchanged.
- Add enriched dataclasses for QASPER answer metadata.
- Add `load_qasper_enriched_cases()` that preserves answer metadata and gold evidence.

New dataclasses:

```python
@dataclass(frozen=True)
class QasperAnswer:
    extractive_spans: tuple[str, ...]
    free_form_answer: str
    yes_no: bool | None
    unanswerable: bool
    evidence: tuple[str, ...]
    highlighted_evidence: tuple[str, ...]


@dataclass(frozen=True)
class EnrichedEvalCase:
    case_id: str
    arxiv_id: str
    paper_title: str
    abstract: str
    full_text: str
    question: str
    oracle_spans: tuple[str, ...]
    answers: tuple[QasperAnswer, ...]
```

### Modify: `tests/eval/fixtures/qasper_mini.json`

Responsibility:

- Add representative enriched answer fields to existing fixture cases.
- Keep enough existing fixture behavior so old loader tests still pass.

### Create: `tests/eval/test_qasper_enriched.py`

Responsibility:

- Test enriched metadata preservation.
- Test filtering remains aligned with current extractive subset policy.
- Test JSON serialization shape used by scripts.

### Create: `scripts/day19_prepare_enriched_eval.py`

Responsibility:

- Read raw QASPER source from `data/eval/qasper-source/qasper-train-v0.3.json`.
- Read existing `data/eval/qasper_subset.jsonl`.
- Write `data/eval/qasper_subset_enriched.jsonl` in the exact same case order.
- Avoid arXiv/MCP dry-run. The existing strict subset already defines the accepted cases.

### Create: `scripts/day19_profile_enriched_eval.py`

Responsibility:

- Read `data/eval/qasper_subset_enriched.jsonl`.
- Write `data/eval/qasper_subset_profile.md`.
- Summarize answer metadata availability and rough question types.

### Create: `paperpilot/eval/semantic_audit.py`

Responsibility:

- Define allowed semantic audit labels.
- Validate audit records.
- Provide deterministic serialization helpers.
- No LLM calls in this slice.

### Create: `tests/eval/test_semantic_audit.py`

Responsibility:

- Test allowed labels.
- Test record validation.
- Test legacy `passed` compatibility through `strict_pass`.

## Task 1: Enriched QASPER Loader

**Files:**

- Modify: `paperpilot/eval/qasper_loader.py`
- Modify: `tests/eval/fixtures/qasper_mini.json`
- Create: `tests/eval/test_qasper_enriched.py`
- Existing test still relevant: `tests/eval/test_qasper_loader.py`

- [ ] **Step 1: Add enriched fields to the fixture**

Edit `tests/eval/fixtures/qasper_mini.json` so Paper A contains answer metadata.
Use this exact shape for the first three Paper A QAs:

```json
{
  "question": "Q1?",
  "answers": [
    {
      "answer": {
        "extractive_spans": ["span-1a"],
        "free_form_answer": "Free answer 1.",
        "yes_no": null,
        "unanswerable": false,
        "evidence": ["Evidence paragraph 1."],
        "highlighted_evidence": ["Highlighted sentence 1."]
      }
    }
  ]
}
```

For Q2, use two extractive spans and two evidence items:

```json
{
  "question": "Q2?",
  "answers": [
    {
      "answer": {
        "extractive_spans": ["span-2a", "span-2b"],
        "free_form_answer": "Free answer 2.",
        "yes_no": null,
        "unanswerable": false,
        "evidence": ["Evidence paragraph 2a.", "Evidence paragraph 2b."],
        "highlighted_evidence": ["Highlighted sentence 2."]
      }
    }
  ]
}
```

For Q3, include a second annotator answer:

```json
{
  "question": "Q3?",
  "answers": [
    {
      "answer": {
        "extractive_spans": ["span-3a"],
        "free_form_answer": "Free answer 3a.",
        "yes_no": null,
        "unanswerable": false,
        "evidence": ["Evidence paragraph 3a."],
        "highlighted_evidence": ["Highlighted sentence 3a."]
      }
    },
    {
      "answer": {
        "extractive_spans": ["span-3b"],
        "free_form_answer": "Free answer 3b.",
        "yes_no": null,
        "unanswerable": false,
        "evidence": ["Evidence paragraph 3b."],
        "highlighted_evidence": ["Highlighted sentence 3b."]
      }
    }
  ]
}
```

- [ ] **Step 2: Write failing enriched loader tests**

Create `tests/eval/test_qasper_enriched.py` with:

```python
"""Tests for enriched QASPER eval loading."""
from dataclasses import asdict
from pathlib import Path

from paperpilot.eval.qasper_loader import (
    EnrichedEvalCase,
    QasperAnswer,
    load_qasper_enriched_cases,
)

FIXTURE = Path(__file__).parent / "fixtures" / "qasper_mini.json"


def test_load_qasper_enriched_preserves_answer_metadata() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)
    q1 = next(c for c in cases if c.case_id == "qasper-2001.12345-q0")

    assert isinstance(q1, EnrichedEvalCase)
    assert q1.oracle_spans == ("span-1a",)
    assert len(q1.answers) == 1

    answer = q1.answers[0]
    assert isinstance(answer, QasperAnswer)
    assert answer.extractive_spans == ("span-1a",)
    assert answer.free_form_answer == "Free answer 1."
    assert answer.yes_no is None
    assert answer.unanswerable is False
    assert answer.evidence == ("Evidence paragraph 1.",)
    assert answer.highlighted_evidence == ("Highlighted sentence 1.",)


def test_load_qasper_enriched_preserves_multiple_answers() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)
    q3 = next(c for c in cases if c.case_id == "qasper-2001.12345-q2")

    assert q3.oracle_spans == ("span-3a", "span-3b")
    assert len(q3.answers) == 2
    assert q3.answers[0].extractive_spans == ("span-3a",)
    assert q3.answers[1].extractive_spans == ("span-3b",)
    assert q3.answers[1].evidence == ("Evidence paragraph 3b.",)


def test_load_qasper_enriched_matches_existing_filter_policy() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)

    assert {c.arxiv_id for c in cases} == {"2001.12345"}
    assert [c.case_id for c in cases] == [
        "qasper-2001.12345-q0",
        "qasper-2001.12345-q1",
        "qasper-2001.12345-q2",
    ]


def test_enriched_case_is_json_serializable_via_asdict() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)
    row = asdict(cases[0])

    assert row["case_id"] == "qasper-2001.12345-q0"
    assert row["oracle_spans"] == ("span-1a",)
    assert row["answers"][0]["extractive_spans"] == ("span-1a",)
```

- [ ] **Step 3: Run enriched loader test and verify it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_qasper_enriched.py -q
```

Expected result:

```text
ImportError or AttributeError for load_qasper_enriched_cases / EnrichedEvalCase / QasperAnswer
```

- [ ] **Step 4: Implement enriched dataclasses and parsing helpers**

Modify `paperpilot/eval/qasper_loader.py`.

Add the new dataclasses below `EvalCase`:

```python
@dataclass(frozen=True)
class QasperAnswer:
    extractive_spans: tuple[str, ...]
    free_form_answer: str
    yes_no: bool | None
    unanswerable: bool
    evidence: tuple[str, ...]
    highlighted_evidence: tuple[str, ...]


@dataclass(frozen=True)
class EnrichedEvalCase:
    case_id: str
    arxiv_id: str
    paper_title: str
    abstract: str
    full_text: str
    question: str
    oracle_spans: tuple[str, ...]
    answers: tuple[QasperAnswer, ...]
```

Add helper functions after `_answer_spans()`:

```python
def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _parse_qasper_answer(answer_record: dict[str, Any]) -> QasperAnswer:
    inner = answer_record.get("answer")
    if not isinstance(inner, dict):
        inner = answer_record
    return QasperAnswer(
        extractive_spans=_string_list(inner.get("extractive_spans")),
        free_form_answer=str(inner.get("free_form_answer") or ""),
        yes_no=_optional_bool(inner.get("yes_no")),
        unanswerable=bool(inner.get("unanswerable", False)),
        evidence=_string_list(inner.get("evidence")),
        highlighted_evidence=_string_list(inner.get("highlighted_evidence")),
    )
```

Add this public loader after `load_qasper_cases()`:

```python
def load_qasper_enriched_cases(qasper_path: Path) -> list[EnrichedEvalCase]:
    """Parse QASPER JSON and preserve answer/evidence metadata.

    This uses the same extractive-only policy as load_qasper_cases(): a paper is
    included only if it has at least three QAs with extractive spans, and only
    the first three such QAs are returned.
    """
    raw = json.loads(qasper_path.read_text(encoding="utf-8"))
    cases: list[EnrichedEvalCase] = []

    for paper_id, paper in raw.items():
        if not isinstance(paper, dict):
            continue
        arxiv_id = (
            extract_arxiv_id(paper_id)
            or extract_arxiv_id(paper.get("paper_url"))
        )
        if arxiv_id is None:
            continue

        full_text = _join_full_text(paper.get("full_text"))
        qa_cases: list[tuple[str, tuple[str, ...], tuple[QasperAnswer, ...]]] = []
        for qa in paper.get("qas") or []:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question") or ""
            answers = tuple(
                _parse_qasper_answer(answer)
                for answer in qa.get("answers") or []
                if isinstance(answer, dict)
            )
            spans: list[str] = []
            for answer in answers:
                spans.extend(answer.extractive_spans)
            if not question or not spans:
                continue

            seen: set[str] = set()
            uniq: list[str] = []
            for span in spans:
                if span in seen:
                    continue
                seen.add(span)
                uniq.append(span)
            qa_cases.append((question, tuple(uniq), answers))

        if len(qa_cases) < 3:
            continue

        for idx, (question, spans, answers) in enumerate(qa_cases[:3]):
            cases.append(EnrichedEvalCase(
                case_id=f"qasper-{arxiv_id}-q{idx}",
                arxiv_id=arxiv_id,
                paper_title=paper.get("title") or "",
                abstract=paper.get("abstract") or "",
                full_text=full_text,
                question=question,
                oracle_spans=spans,
                answers=answers,
            ))
    return cases
```

- [ ] **Step 5: Run loader tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_qasper_loader.py tests\eval\test_qasper_enriched.py -q
```

Expected result:

```text
12 passed
```

- [ ] **Step 6: Commit Task 1**

Run:

```powershell
git add paperpilot/eval/qasper_loader.py tests/eval/fixtures/qasper_mini.json tests/eval/test_qasper_enriched.py
git commit -m "Eval: preserve QASPER answer metadata"
```

## Task 2: Enriched Subset Writer

**Files:**

- Create: `scripts/day19_prepare_enriched_eval.py`
- Create: `tests/eval/test_prepare_enriched_eval_script.py`
- Modify: `paperpilot/eval/qasper_loader.py` only if Task 1 serialization needs a helper.

- [ ] **Step 1: Write failing tests for selecting existing subset cases**

Create `tests/eval/test_prepare_enriched_eval_script.py`:

```python
"""Tests for the enriched QASPER subset writer."""
import json
from dataclasses import asdict
from pathlib import Path

from paperpilot.eval.qasper_loader import load_qasper_enriched_cases
from scripts.day19_prepare_enriched_eval import (
    _index_enriched_cases,
    _load_subset_case_ids,
    _serialize_enriched_case,
)

FIXTURE = Path(__file__).parent / "fixtures" / "qasper_mini.json"


def test_load_subset_case_ids_preserves_file_order(tmp_path: Path) -> None:
    subset = tmp_path / "subset.jsonl"
    subset.write_text(
        "\n".join([
            json.dumps({"case_id": "qasper-2001.12345-q2"}),
            json.dumps({"case_id": "qasper-2001.12345-q0"}),
        ]) + "\n",
        encoding="utf-8",
    )

    assert _load_subset_case_ids(subset) == [
        "qasper-2001.12345-q2",
        "qasper-2001.12345-q0",
    ]


def test_index_enriched_cases_by_case_id() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)
    indexed = _index_enriched_cases(cases)

    assert indexed["qasper-2001.12345-q0"].question == "Q1?"
    assert indexed["qasper-2001.12345-q2"].answers[1].free_form_answer == "Free answer 3b."


def test_serialize_enriched_case_converts_tuples_to_lists() -> None:
    case = load_qasper_enriched_cases(FIXTURE)[0]
    row = _serialize_enriched_case(case)

    assert row["case_id"] == "qasper-2001.12345-q0"
    assert row["oracle_spans"] == ["span-1a"]
    assert row["answers"][0]["extractive_spans"] == ["span-1a"]
    assert row["answers"][0]["evidence"] == ["Evidence paragraph 1."]
    assert asdict(case)["oracle_spans"] == ("span-1a",)
```

- [ ] **Step 2: Run script tests and verify failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_prepare_enriched_eval_script.py -q
```

Expected result:

```text
ModuleNotFoundError: No module named 'scripts.day19_prepare_enriched_eval'
```

- [ ] **Step 3: Implement enriched subset writer**

Create `scripts/day19_prepare_enriched_eval.py`:

```python
"""Prepare an enriched QASPER subset matching the existing Day 16 subset.

This script does not call MCP or redownload papers. It reads the accepted case
ids from data/eval/qasper_subset.jsonl and writes enriched rows in that order.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.eval.qasper_loader import EnrichedEvalCase, load_qasper_enriched_cases

QASPER_PATH = Path("data/eval/qasper-source/qasper-train-v0.3.json")
SUBSET_PATH = Path("data/eval/qasper_subset.jsonl")
OUT_PATH = Path("data/eval/qasper_subset_enriched.jsonl")


def _load_subset_case_ids(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"FAIL: {path} not found. Run scripts/day16_prepare_eval.py first.")
    case_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_ids.append(str(row["case_id"]))
    return case_ids


def _index_enriched_cases(cases: list[EnrichedEvalCase]) -> dict[str, EnrichedEvalCase]:
    return {case.case_id: case for case in cases}


def _serialize_enriched_case(case: EnrichedEvalCase) -> dict[str, Any]:
    row = asdict(case)
    row["oracle_spans"] = list(case.oracle_spans)
    for answer in row["answers"]:
        answer["extractive_spans"] = list(answer["extractive_spans"])
        answer["evidence"] = list(answer["evidence"])
        answer["highlighted_evidence"] = list(answer["highlighted_evidence"])
    return row


def main() -> None:
    if not QASPER_PATH.exists():
        sys.exit(
            f"FAIL: {QASPER_PATH} not found. "
            "Download QASPER per Day 16 plan first."
        )

    case_ids = _load_subset_case_ids(SUBSET_PATH)
    enriched_by_id = _index_enriched_cases(load_qasper_enriched_cases(QASPER_PATH))
    missing = [case_id for case_id in case_ids if case_id not in enriched_by_id]
    if missing:
        preview = ", ".join(missing[:5])
        sys.exit(f"FAIL: {len(missing)} subset case ids missing from enriched source: {preview}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for case_id in case_ids:
            row = _serialize_enriched_case(enriched_by_id[case_id])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_PATH} with {len(case_ids)} cases")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run script tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_prepare_enriched_eval_script.py -q
```

Expected result:

```text
3 passed
```

- [ ] **Step 5: Run the enriched writer on local data**

Run:

```powershell
.venv\Scripts\python.exe scripts\day19_prepare_enriched_eval.py
```

Expected result:

```text
Wrote data/eval/qasper_subset_enriched.jsonl with 150 cases
```

If local QASPER source data is missing, expected result is:

```text
FAIL: data/eval/qasper-source/qasper-train-v0.3.json not found. Download QASPER per Day 16 plan first.
```

Missing local source is not a code failure if tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add scripts/day19_prepare_enriched_eval.py tests/eval/test_prepare_enriched_eval_script.py
git commit -m "Eval: add enriched QASPER subset writer"
```

Do not add `data/eval/qasper_subset_enriched.jsonl` unless the user explicitly wants runtime eval artifacts committed.

## Task 3: Enriched Subset Profile

**Files:**

- Create: `scripts/day19_profile_enriched_eval.py`
- Create: `tests/eval/test_profile_enriched_eval_script.py`

- [ ] **Step 1: Write failing profile tests**

Create `tests/eval/test_profile_enriched_eval_script.py`:

```python
"""Tests for enriched QASPER subset profiling."""
import json
from pathlib import Path

from scripts.day19_profile_enriched_eval import (
    _answer_type_counts,
    _question_type,
    _render_profile,
)


def test_question_type_numeric() -> None:
    assert _question_type("How many comments were used?") == "numeric"
    assert _question_type("What percentage of examples are labeled?") == "numeric"


def test_question_type_boolean() -> None:
    assert _question_type("Does the paper report macro F1?") == "boolean"


def test_question_type_fact_or_list() -> None:
    assert _question_type("Which languages are used?") == "fact_or_list"
    assert _question_type("What labels are available?") == "fact_or_list"


def test_answer_type_counts() -> None:
    rows = [
        {
            "answers": [
                {
                    "extractive_spans": ["A"],
                    "free_form_answer": "",
                    "yes_no": None,
                    "unanswerable": False,
                    "evidence": ["E"],
                    "highlighted_evidence": [],
                }
            ]
        },
        {
            "answers": [
                {
                    "extractive_spans": [],
                    "free_form_answer": "Free",
                    "yes_no": True,
                    "unanswerable": False,
                    "evidence": [],
                    "highlighted_evidence": ["H"],
                }
            ]
        },
    ]

    counts = _answer_type_counts(rows)

    assert counts["extractive"] == 1
    assert counts["free_form"] == 1
    assert counts["yes_no"] == 1
    assert counts["unanswerable"] == 0
    assert counts["with_evidence"] == 1
    assert counts["with_highlighted_evidence"] == 1


def test_render_profile_contains_core_sections(tmp_path: Path) -> None:
    rows = [
        {
            "case_id": "case-1",
            "question": "Which labels are available?",
            "oracle_spans": ["positive", "negative"],
            "answers": [
                {
                    "extractive_spans": ["positive", "negative"],
                    "free_form_answer": "",
                    "yes_no": None,
                    "unanswerable": False,
                    "evidence": ["The labels are positive and negative."],
                    "highlighted_evidence": ["positive and negative"],
                }
            ],
        }
    ]

    text = _render_profile(rows)

    assert "# QASPER Enriched Subset Profile" in text
    assert "| cases | 1 |" in text
    assert "| fact_or_list | 1 |" in text
    assert "| with_evidence | 1 |" in text
```

- [ ] **Step 2: Run profile tests and verify failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_profile_enriched_eval_script.py -q
```

Expected result:

```text
ModuleNotFoundError: No module named 'scripts.day19_profile_enriched_eval'
```

- [ ] **Step 3: Implement profile script**

Create `scripts/day19_profile_enriched_eval.py`:

```python
"""Profile the enriched QASPER subset."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IN_PATH = Path("data/eval/qasper_subset_enriched.jsonl")
OUT_PATH = Path("data/eval/qasper_subset_profile.md")


def _question_type(question: str) -> str:
    q = question.strip().lower()
    if re.match(r"^(how many|how much|what percentage|what percent|how big)\b", q):
        return "numeric"
    if re.match(r"^(does|do|did|is|are|was|were|can|could)\b", q):
        return "boolean"
    if re.match(
        r"^(which|what are|what were|what is|what data|what dataset|"
        r"what baselines|what metrics|what methods|what languages|what labels)\b",
        q,
    ):
        return "fact_or_list"
    if re.match(r"^(how|why)\b", q):
        return "mechanism_or_explanation"
    return "other"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        sys.exit(f"FAIL: {path} not found. Run scripts/day19_prepare_enriched_eval.py first.")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _answer_type_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for answer in row.get("answers") or []:
            if answer.get("extractive_spans"):
                counts["extractive"] += 1
            if answer.get("free_form_answer"):
                counts["free_form"] += 1
            if answer.get("yes_no") is not None:
                counts["yes_no"] += 1
            if answer.get("unanswerable"):
                counts["unanswerable"] += 1
            if answer.get("evidence"):
                counts["with_evidence"] += 1
            if answer.get("highlighted_evidence"):
                counts["with_highlighted_evidence"] += 1
    return counts


def _render_profile(rows: list[dict[str, Any]]) -> str:
    qtypes = Counter(_question_type(str(row.get("question", ""))) for row in rows)
    answer_counts = _answer_type_counts(rows)
    answer_total = sum(len(row.get("answers") or []) for row in rows)
    span_counts = [len(row.get("oracle_spans") or []) for row in rows]
    evidence_counts = [
        sum(len(answer.get("evidence") or []) for answer in row.get("answers") or [])
        for row in rows
    ]

    lines: list[str] = []
    lines.append("# QASPER Enriched Subset Profile")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| cases | {len(rows)} |")
    lines.append(f"| answers | {answer_total} |")
    lines.append(f"| max oracle spans per case | {max(span_counts, default=0)} |")
    lines.append(f"| max evidence paragraphs per case | {max(evidence_counts, default=0)} |")
    lines.append("")
    lines.append("## Question Types")
    lines.append("")
    lines.append("| type | count |")
    lines.append("|---|---:|")
    for key, count in sorted(qtypes.items()):
        lines.append(f"| {key} | {count} |")
    lines.append("")
    lines.append("## Answer Metadata")
    lines.append("")
    lines.append("| field | answer count |")
    lines.append("|---|---:|")
    for key in [
        "extractive",
        "free_form",
        "yes_no",
        "unanswerable",
        "with_evidence",
        "with_highlighted_evidence",
    ]:
        lines.append(f"| {key} | {answer_counts[key]} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = _load_rows(IN_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(_render_profile(rows), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run profile tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_profile_enriched_eval_script.py -q
```

Expected result:

```text
5 passed
```

- [ ] **Step 5: Run profile script on local enriched subset**

Run:

```powershell
.venv\Scripts\python.exe scripts\day19_profile_enriched_eval.py
```

Expected success:

```text
Wrote data/eval/qasper_subset_profile.md
```

If the enriched subset has not been generated:

```text
FAIL: data/eval/qasper_subset_enriched.jsonl not found. Run scripts/day19_prepare_enriched_eval.py first.
```

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
git add scripts/day19_profile_enriched_eval.py tests/eval/test_profile_enriched_eval_script.py
git commit -m "Eval: add enriched QASPER subset profile"
```

Do not add `data/eval/qasper_subset_profile.md` unless the user explicitly wants generated eval artifacts committed.

## Task 4: Semantic Audit Schema

**Files:**

- Create: `paperpilot/eval/semantic_audit.py`
- Create: `tests/eval/test_semantic_audit.py`

- [ ] **Step 1: Write failing semantic audit tests**

Create `tests/eval/test_semantic_audit.py`:

```python
"""Tests for semantic audit schema helpers."""
import pytest

from paperpilot.eval.semantic_audit import (
    ALLOWED_SEMANTIC_LABELS,
    SemanticAuditRecord,
    semantic_audit_record_from_dict,
)


def test_allowed_semantic_labels_are_fixed() -> None:
    assert ALLOWED_SEMANTIC_LABELS == {
        "correct",
        "partial",
        "incorrect",
        "contradictory",
        "unverifiable",
        "judge_uncertain",
    }


def test_semantic_audit_record_to_dict() -> None:
    record = SemanticAuditRecord(
        case_id="qasper-1-q0",
        baseline="paperpilot",
        strict_pass=True,
        semantic_label="partial",
        confidence="medium",
        reason="The answer includes one required span but misses another.",
        used_gold_evidence=True,
        audit_model=None,
        audit_version="v1",
    )

    assert record.to_dict() == {
        "case_id": "qasper-1-q0",
        "baseline": "paperpilot",
        "strict_pass": True,
        "semantic_label": "partial",
        "confidence": "medium",
        "reason": "The answer includes one required span but misses another.",
        "used_gold_evidence": True,
        "audit_model": None,
        "audit_version": "v1",
    }


def test_semantic_audit_record_rejects_bad_label() -> None:
    with pytest.raises(ValueError, match="invalid semantic_label"):
        SemanticAuditRecord(
            case_id="qasper-1-q0",
            baseline="paperpilot",
            strict_pass=True,
            semantic_label="almost_right",
            confidence="medium",
            reason="Bad label.",
            used_gold_evidence=True,
            audit_model=None,
            audit_version="v1",
        )


def test_semantic_audit_record_from_dict_uses_legacy_passed() -> None:
    record = semantic_audit_record_from_dict({
        "case_id": "qasper-1-q0",
        "baseline": "paperpilot",
        "passed": False,
        "semantic_label": "incorrect",
        "confidence": "high",
        "reason": "The answer does not answer the question.",
        "used_gold_evidence": True,
        "audit_model": "manual",
        "audit_version": "v1",
    })

    assert record.strict_pass is False
    assert record.audit_model == "manual"
```

- [ ] **Step 2: Run semantic audit tests and verify failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_semantic_audit.py -q
```

Expected result:

```text
ModuleNotFoundError: No module named 'paperpilot.eval.semantic_audit'
```

- [ ] **Step 3: Implement semantic audit schema**

Create `paperpilot/eval/semantic_audit.py`:

```python
"""Semantic audit schema for QASPER eval results.

This module defines data structures only. It does not call an LLM judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SemanticLabel = Literal[
    "correct",
    "partial",
    "incorrect",
    "contradictory",
    "unverifiable",
    "judge_uncertain",
]

ALLOWED_SEMANTIC_LABELS: set[str] = {
    "correct",
    "partial",
    "incorrect",
    "contradictory",
    "unverifiable",
    "judge_uncertain",
}


@dataclass(frozen=True)
class SemanticAuditRecord:
    case_id: str
    baseline: str
    strict_pass: bool
    semantic_label: SemanticLabel
    confidence: str
    reason: str
    used_gold_evidence: bool
    audit_model: str | None
    audit_version: str

    def __post_init__(self) -> None:
        if self.semantic_label not in ALLOWED_SEMANTIC_LABELS:
            raise ValueError(f"invalid semantic_label: {self.semantic_label!r}")
        if not self.case_id.strip():
            raise ValueError("case_id is required")
        if not self.baseline.strip():
            raise ValueError("baseline is required")
        if not self.reason.strip():
            raise ValueError("reason is required")
        if not self.audit_version.strip():
            raise ValueError("audit_version is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "baseline": self.baseline,
            "strict_pass": self.strict_pass,
            "semantic_label": self.semantic_label,
            "confidence": self.confidence,
            "reason": self.reason,
            "used_gold_evidence": self.used_gold_evidence,
            "audit_model": self.audit_model,
            "audit_version": self.audit_version,
        }


def semantic_audit_record_from_dict(row: dict[str, Any]) -> SemanticAuditRecord:
    strict_pass = row.get("strict_pass", row.get("passed"))
    if not isinstance(strict_pass, bool):
        raise ValueError("strict_pass or passed must be a boolean")
    return SemanticAuditRecord(
        case_id=str(row["case_id"]),
        baseline=str(row["baseline"]),
        strict_pass=strict_pass,
        semantic_label=str(row["semantic_label"]),  # type: ignore[arg-type]
        confidence=str(row.get("confidence") or "unknown"),
        reason=str(row["reason"]),
        used_gold_evidence=bool(row.get("used_gold_evidence", False)),
        audit_model=(
            None if row.get("audit_model") is None else str(row.get("audit_model"))
        ),
        audit_version=str(row.get("audit_version") or "v1"),
    )
```

- [ ] **Step 4: Run semantic audit tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_semantic_audit.py -q
```

Expected result:

```text
4 passed
```

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add paperpilot/eval/semantic_audit.py tests/eval/test_semantic_audit.py
git commit -m "Eval: define semantic audit schema"
```

## Task 5: Integration Validation

**Files:**

- No required code changes.
- Optional generated runtime files under `data/eval/` stay uncommitted by default.

- [ ] **Step 1: Run focused eval tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval -q
```

Expected result:

```text
all tests in tests/eval pass
```

- [ ] **Step 2: Run existing relevant regression tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval tests\test_main_integration.py -q
```

Expected result:

```text
tests pass; no scorer behavior changes
```

- [ ] **Step 3: Verify current scorer remains unchanged**

Run:

```powershell
git diff HEAD~4 -- paperpilot/eval/scorer.py
```

Expected result:

```text
no diff, unless unrelated earlier commits touched scorer before this plan began
```

If the commit count differs because tasks were squashed, use:

```powershell
git diff 850ed14 -- paperpilot/eval/scorer.py
```

Expected result:

```text
no diff
```

- [ ] **Step 4: Check generated eval artifacts are not staged**

Run:

```powershell
git status --short
```

Expected result:

```text
source files and tests may be modified or clean;
data/eval/qasper_subset_enriched.jsonl and data/eval/qasper_subset_profile.md are not staged
```

- [ ] **Step 5: Final implementation summary**

Prepare a summary for the user:

```text
Implemented first QASPER eval-upgrade slice:
- enriched loader preserves answer/evidence metadata
- enriched subset writer maps existing 150-case subset to enriched rows
- profile script summarizes answer/evidence/question-type distribution
- semantic audit schema defines future judge output
- current strict scorer and passed field remain unchanged
```

## Validation Commands

Use escalated project venv Python if sandboxed Python cannot access the local
interpreter:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval -q
.venv\Scripts\python.exe -m pytest tests\eval tests\test_main_integration.py -q
```

If the broad test suite is run, keep the known SOCKS test issue in mind:

```powershell
.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/mcp_servers/test_ss_client.py
```

## Commit Strategy

Prefer one commit per task:

```text
Eval: preserve QASPER answer metadata
Eval: add enriched QASPER subset writer
Eval: add enriched QASPER subset profile
Eval: define semantic audit schema
```

Do not commit generated `data/eval/*.jsonl` or `data/eval/*.md` artifacts unless
the user explicitly requests eval artifacts in git.

## Self-Review Checklist

- The plan keeps `is_pass()` unchanged.
- The plan keeps existing `load_qasper_cases()` behavior unchanged.
- The enriched writer uses the existing subset case IDs and avoids another MCP
  dry-run.
- The profile script is offline and deterministic.
- The semantic audit schema has no LLM calls.
- Runtime `data/eval/` artifacts remain uncommitted by default.
