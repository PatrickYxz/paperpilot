# Day 15: `write-research-report` 主题综述 skill 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 1 个顶层 skill `write-research-report`,把 arxiv→colbert→(可选 graph)→subagent 并发精读→半固定骨架 markdown 综述全链路串起来。纯 prose 零代码,落盘交给 smoke 脚本。

**Architecture:** 1 个新 skill md(prose 引导工作流 + 输出骨架 + 质量准则)+ 1 个 smoke 脚本(真 LLM 端到端 + tracer 断言 + 落盘 + 报告内容断言)+ 1 行 main_integration test 回归断言。**红线全守恒**:`main.py / loop.py / adapter.py / builtin_tools/*` zero diff。

**Tech Stack:** Python 3.12,现有 anthropic SDK over DeepSeek,现有 4 MCP server(arxiv/colbert/graph/vlm),现有 4 内嵌 tool(research_todo/paper_deep_read/load_skill/compact_context),pytest。

**Spec:** `docs/superpowers/specs/2026-05-06-day15-write-research-report-design.md`

---

## 文件结构

| 路径 | 动作 | 责任 |
|---|---|---|
| `paperpilot/skills/write-research-report.md` | 新建 | 综述工作流 prose + 半固定输出骨架 + 质量准则 |
| `tests/test_main_integration.py` | 改 1 行 | 加 `assert "write-research-report" in prompt` |
| `data/reports/.gitkeep` | 新建,空 | 报告产物目录入仓 |
| `scripts/day15_smoke.py` | 新建 | 真 LLM 跑一篇 + tracer + 落盘 + 报告断言 |

---

## Task 1: 新增 skill prose + main_integration 回归断言

**Files:**
- Modify: `tests/test_main_integration.py:9-18` (加 1 行 assert)
- Create: `paperpilot/skills/write-research-report.md`
- Create: `data/reports/.gitkeep` (空)

### Step 1.1: 在 main_integration test 加失败断言

- [ ] 打开 `tests/test_main_integration.py`,找到 `test_build_system_prompt_includes_skills`,在 `assert "analyze-figures" in prompt` 后追加 1 行:

```python
def test_build_system_prompt_includes_skills():
    prompt = _build_system_prompt()
    assert "## 可用 skill" in prompt
    assert "deep-read-paper" in prompt
    assert "explore-citations" in prompt
    assert "find-classics" in prompt
    assert "compare-papers" in prompt
    assert "analyze-figures" in prompt
    assert "write-research-report" in prompt
    assert "PaperPilot" in prompt
    assert "build_index" in prompt
```

### Step 1.2: 跑 test 确认失败

Run: `pytest tests/test_main_integration.py::test_build_system_prompt_includes_skills -v`
Expected: FAIL,`AssertionError: assert 'write-research-report' in prompt`(skill 还不存在,prompt 里没这个名字)

### Step 1.3: 新建 skill md

- [ ] 新建 `paperpilot/skills/write-research-report.md`,完整内容如下:

````markdown
---
name: write-research-report
description: 给定一个研究方向,产出一篇结构化综述 markdown
when_to_use: 用户给一个主题/方向(不是单篇 paper / 不是对比指定几篇),希望"写一篇综述"或"了解 XX 领域近年进展"
---

# Write Research Report

## 适用场景
- "总结一下 long-context retrieval 2024 后的进展"
- "我想入门 X 方向,给我一份 reading list 加每篇要点"
- 不适用:用户给定具体 paper(用 deep-read-paper / compare-papers)

## 工作流(推荐,非强制)
1. **规划**:`research_todo` 写下 plan(搜索关键词 / 候选数 / 精读数)
2. **粗搜**:`mcp__arxiv__search(query=..., max_results=8)`,标题摘要先筛
3. **细排**:把候选喂 `mcp__colbert__build_index` → `mcp__colbert__search(query=主题问题, k=5)` 做语义重排
4. **扩展(可选)**:若主题有明确"奠基 paper",用 `mcp__graph__cited_by / cites` 1 跳找经典或最新工作
5. **精读**:选 3-5 篇代表作,`paper_deep_read(papers=[...], n_workers=3-5)` 并发精读
6. **下笔**:用半固定骨架组织 markdown 输出(见下)

## 输出骨架(半固定,可按主题适配)
1. **Title** — 综述标题
2. **TL;DR** — 3-5 句话核心结论
3. **Background** — 问题定义、为什么重要
4. **Key Methods / Approaches** — 主流路线分类(每条带 1-3 篇代表作 + arxiv_id)
5. **Recent Trends (2024-)** — 最新进展
6. **Open Problems** — 未解、争议、benchmark gap
7. **Reading List** — 推荐阅读顺序(5-8 篇,每篇一句话点评 + arxiv_id 链接)

## 质量准则
- 每篇引用必须带 arxiv_id(写成 `[Title (arxiv:2401.12345)]`),用户能点开
- 至少 3 篇 paper 来自 paper_deep_read 的精读结果(不能全靠 abstract)
- 报告 ≥ 800 字
- 主题方法学 vs 应用偏向不同,骨架可砍可加(纯方法论可砍"应用",偏应用可并掉"Open Problems")

## 注意
- 不要把所有候选都精读 — 3-5 篇够;其余靠 abstract + colbert chunk 补
- compose 阶段先写完整 markdown 再输出,不要边搜边写
````

### Step 1.4: 新建报告产物目录

- [ ] 新建空文件 `data/reports/.gitkeep`(让目录入仓):

```powershell
New-Item -ItemType Directory -Force -Path data/reports | Out-Null
New-Item -ItemType File -Force -Path data/reports/.gitkeep | Out-Null
```

### Step 1.5: 跑 main_integration test 确认 PASS

Run: `pytest tests/test_main_integration.py -q --no-header`
Expected: 5 passed, 1 deselected(slow 那条)。`test_build_system_prompt_includes_skills` 含新 assert 全过。

### Step 1.6: 跑全套 fast suite 确认无回归

Run: `pytest tests -q --ignore=tests/mcp_servers/test_colbert_via_client.py --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py --ignore=tests/test_per_paper_index_slow.py`
Expected: `116 passed`(Day 14 末基线)— 数字不变,因为只加了 1 行 assert,新 skill md 自动被发现。

### Step 1.7: Commit

```powershell
git add paperpilot/skills/write-research-report.md tests/test_main_integration.py data/reports/.gitkeep
git commit -m "Day 15 Task 1: add write-research-report skill prose"
```

---

## Task 2: Day 15 真 LLM smoke

**Files:**
- Create: `scripts/day15_smoke.py`

### Step 2.1: 新建 smoke 脚本

- [ ] 新建 `scripts/day15_smoke.py`,完整内容如下:

```python
"""Day 15 smoke: write-research-report skill 端到端综述生成。

Expected tracer path:
load_skill('write-research-report')
  -> research_todo
  -> mcp__arxiv__search
  -> mcp__colbert__build_index + mcp__colbert__search
  -> paper_deep_read (n_workers >= 3)
  -> compose markdown

Final assertions:
- report >= 800 chars
- >= 3 distinct arxiv_ids cited
- >= 4 of 7 section markers hit
- file written to data/reports/<slug>-<timestamp>.md
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

TOPIC = "long-context retrieval beyond 100k tokens"
SLUG = "long-context-retrieval"

ARXIV_ID_RE = re.compile(r"arxiv[:/ ]\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
SECTION_MARKERS = (
    "TL;DR",
    "Background",
    "Key Methods",
    "Recent Trends",
    "Open Problems",
    "Reading List",
    "Title",
)


def main() -> None:
    saw_load_skill_args: list[dict[str, Any]] = []
    tool_calls: list[str] = []
    deep_read_calls: list[dict[str, Any]] = []

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        if kind == "tool_call":
            name = payload["name"]
            args = payload.get("arguments", {})
            tool_calls.append(name)
            if name == "load_skill":
                saw_load_skill_args.append(args)
            if name == "paper_deep_read":
                deep_read_calls.append(args)
            print(f"  -> {name}({_preview(args)})")
        elif kind == "tool_result":
            name = payload["name"]
            content = payload.get("content", "")
            content_text = content if isinstance(content, str) else str(content)
            print(f"  <- {name}: {content_text[:160]}...")
        elif kind == "guardrail_stop":
            print(f"  !! guardrail: {payload.get('reason')}")

    prompt = (
        f"请为我写一篇综述,主题:{TOPIC}。"
        " 按 write-research-report skill 操作:先 load_skill,然后跟 prose 走"
        " (research_todo 规划 -> arxiv 搜索 -> colbert 重排 -> 选 3-5 篇 paper_deep_read 并发精读 -> 半固定骨架 markdown)。"
        " 最后输出完整 markdown 综述,不要分段输出。"
    )
    messages = run(prompt, max_iter=30, on_event=tracer)

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    final_text = _final_text(last)
    print(final_text[:500] + ("..." if len(final_text) > 500 else ""))

    # ---- tracer assertions ----
    assert any(args.get("name") == "write-research-report" for args in saw_load_skill_args), (
        f"FAIL: did not load write-research-report skill; load_skill args = {saw_load_skill_args}"
    )
    assert "research_todo" in tool_calls, (
        f"FAIL: research_todo never called; tool_calls = {tool_calls}"
    )
    assert "mcp__arxiv__search" in tool_calls, (
        f"FAIL: mcp__arxiv__search never called; tool_calls = {tool_calls}"
    )
    assert "mcp__colbert__build_index" in tool_calls, (
        f"FAIL: mcp__colbert__build_index never called; tool_calls = {tool_calls}"
    )
    assert "mcp__colbert__search" in tool_calls, (
        f"FAIL: mcp__colbert__search never called; tool_calls = {tool_calls}"
    )
    assert deep_read_calls, (
        f"FAIL: paper_deep_read never called; tool_calls = {tool_calls}"
    )
    first_dr = deep_read_calls[0]
    n_workers = first_dr.get("n_workers")
    assert isinstance(n_workers, int) and n_workers >= 3, (
        f"FAIL: paper_deep_read first call n_workers must be int >= 3, got {n_workers!r}"
    )

    # ---- final report assertions ----
    assert len(final_text) >= 800, (
        f"FAIL: report too short ({len(final_text)} chars < 800)"
    )
    arxiv_ids = set(ARXIV_ID_RE.findall(final_text))
    assert len(arxiv_ids) >= 3, (
        f"FAIL: only {len(arxiv_ids)} distinct arxiv_ids in report (need >= 3); "
        f"found = {arxiv_ids}"
    )
    sections_hit = [m for m in SECTION_MARKERS if m in final_text]
    assert len(sections_hit) >= 4, (
        f"FAIL: only {len(sections_hit)} of 7 section markers hit (need >= 4); "
        f"hit = {sections_hit}"
    )

    # ---- save ----
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(f"data/reports/{SLUG}-{ts}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(final_text, encoding="utf-8")

    print(
        f"\nDay 15 smoke PASSED "
        f"(chars={len(final_text)}, arxiv_ids={len(arxiv_ids)}, "
        f"sections={len(sections_hit)}, total_tool_calls={len(tool_calls)})"
    )
    print(f"Report saved: {out}")


def _final_text(content: Any) -> str:
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
    return "\n".join(part for part in parts if part)


def _preview(args: dict[str, Any]) -> dict[str, Any]:
    if "documents" in args:
        docs = args.get("documents") or []
        return {
            **args,
            "documents": [
                {"paper_id": d.get("paper_id"), "text_len": len(d.get("text", ""))}
                for d in docs if isinstance(d, dict)
            ],
        }
    if "papers" in args:
        papers = args.get("papers") or []
        return {
            **args,
            "papers": [
                p.get("paper_id") if isinstance(p, dict) else p
                for p in papers
            ],
        }
    return args


if __name__ == "__main__":
    main()
```

### Step 2.2: 跑 smoke

Run(确保 `.env` 里有 `DEEPSEEK_API_KEY`):
```
python scripts/day15_smoke.py
```
Expected: 终端依次打出 `load_skill(write-research-report)` → `research_todo` → `mcp__arxiv__search` → `mcp__colbert__build_index` → `mcp__colbert__search` → `paper_deep_read` → 最后 `Day 15 smoke PASSED`,并打出报告路径。整体耗时 2-3 分钟。

### Step 2.3: 人眼验报告

- [ ] 打开 `data/reports/long-context-retrieval-<timestamp>.md`,确认:
  - 像"一篇综述"(连续叙述,不是 bullet 拼盘)
  - section 结构清晰
  - 引用带 arxiv_id 能点开

### Step 2.4: 失败排查(若 smoke FAIL,按下表对症)

| 症状 | 排查 |
|---|---|
| 没 load_skill | 检查 Step 1.5 是否真 PASS;手测 `python -c "from paperpilot.main import _build_system_prompt; print('write-research-report' in _build_system_prompt())"` |
| 没 paper_deep_read 或 n_workers < 3 | prompt 里加重一句"必须用 paper_deep_read 并发精读 ≥ 3 篇,n_workers=3";若仍不行,在 skill prose 第 5 步加重"必须 n_workers=3-5,不要串行" |
| 报告 < 800 字 | max_iter 调到 40;prompt 加"完整 markdown ≥ 800 字" |
| arxiv_id 数量 < 3 | prompt 加"引用必须包含 arxiv_id,至少 3 篇不同" |

**绝不:** 不要为了 smoke 过加 if-else 分支 / loop nudge 文本。skill prose 调整 OK,核心 loop 不动。

### Step 2.5: Commit

```powershell
git add scripts/day15_smoke.py
git commit -m "Day 15 Task 2: add day15 smoke for write-research-report"
```

(如果 Step 2.4 改了 skill prose,把 skill md 一并 add commit。)

---

## DoD 验证(全部 task 完成后跑一遍)

- [ ] `pytest tests -q --ignore=tests/mcp_servers/test_colbert_via_client.py --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py --ignore=tests/test_per_paper_index_slow.py` → `116 passed`(Day 14 末基线不退)
- [ ] `python scripts/day15_smoke.py` → `Day 15 smoke PASSED` + 报告落盘
- [ ] 人眼读报告:像综述,不是 bullet 拼盘
- [ ] `git diff main -- paperpilot/main.py paperpilot/core/loop.py paperpilot/core/adapter.py paperpilot/builtin_tools/subagent.py paperpilot/builtin_tools/skill_loader.py paperpilot/builtin_tools/research_todo.py paperpilot/builtin_tools/compact.py` → 空(红线全守恒)
- [ ] `git log --oneline | head -3` → 含 Day 15 spec / Day 15 Task 1 / Day 15 Task 2 三条 commit