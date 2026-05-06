"""Day 15 smoke: write-research-report skill end-to-end report generation.

Expected tracer path:
load_skill('write-research-report')
  -> research_todo
  -> mcp__arxiv__search_papers
  -> mcp__arxiv__download_paper
  -> mcp__colbert__build_index + mcp__colbert__search
  -> paper_deep_read (paper_ids count >= 3)
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
import os
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

TOPIC = "long-context retrieval beyond 100k tokens"
SLUG = "long-context-retrieval"
PAPER_IDS = ("2402.13718", "2407.14482", "2310.03025")

ARXIV_ID_RE = re.compile(r"arxiv[:/ ]\s*(\d{4}\.\d{4,5})", re.IGNORECASE)
SECTION_MARKER_GROUPS = (
    ("TL;DR", "核心结论"),
    ("Background", "背景", "问题定义"),
    ("Key Methods", "Methods", "方法", "路线"),
    ("Recent Trends", "Trends", "趋势", "进展"),
    ("Open Problems", "问题", "局限", "挑战"),
    ("Reading List", "阅读", "推荐"),
    ("Title", "综述"),
)


def main() -> None:
    os.environ.setdefault("BUDGET_TOKENS", "300000")
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
        f" (research_todo 规划 -> arxiv 搜索 -> 固定只选择这 3 篇候选: {', '.join(PAPER_IDS)} -> "
        "download_paper 下载这 3 篇全文 -> colbert 重排 -> "
        "paper_deep_read 并发精读这 3 篇 -> "
        "半固定骨架 markdown)。"
        " 为控制上下文,不要下载或分析其他 paper。"
        " 完成 colbert.search 后立刻调用 paper_deep_read。"
        " paper_deep_read 返回后不要再调用任何工具,直接写报告正文。"
        " 不要调用 compact_context,不要调用 VLM。"
        " 最后直接输出完整 markdown 综述正文,长度至少 1200 字。"
        " 不要输出任务执行总结、状态表或'已完成'说明。"
    )
    messages = run(prompt, max_iter=30, on_event=tracer)

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    final_text = _final_text(last)
    print(final_text[:500] + ("..." if len(final_text) > 500 else ""))

    assert any(args.get("name") == "write-research-report" for args in saw_load_skill_args), (
        "FAIL: did not load write-research-report skill; "
        f"load_skill args = {saw_load_skill_args}"
    )
    assert "research_todo" in tool_calls, (
        f"FAIL: research_todo never called; tool_calls = {tool_calls}"
    )
    assert "mcp__arxiv__search_papers" in tool_calls, (
        f"FAIL: mcp__arxiv__search_papers never called; tool_calls = {tool_calls}"
    )
    assert "mcp__arxiv__download_paper" in tool_calls, (
        f"FAIL: mcp__arxiv__download_paper never called; tool_calls = {tool_calls}"
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
    paper_ids = first_dr.get("paper_ids")
    assert isinstance(paper_ids, list) and len(paper_ids) >= 3, (
        "FAIL: paper_deep_read first call paper_ids must contain >= 3 ids, "
        f"got {paper_ids!r}"
    )
    assert set(paper_ids) == set(PAPER_IDS), (
        f"FAIL: paper_deep_read should use fixed paper_ids {PAPER_IDS}, got {paper_ids!r}"
    )

    assert len(final_text) >= 800, (
        f"FAIL: report too short ({len(final_text)} chars < 800)"
    )
    arxiv_ids = set(ARXIV_ID_RE.findall(final_text))
    assert len(arxiv_ids) >= 3, (
        f"FAIL: only {len(arxiv_ids)} distinct arxiv_ids in report (need >= 3); "
        f"found = {arxiv_ids}"
    )
    sections_hit = [
        group[0]
        for group in SECTION_MARKER_GROUPS
        if any(marker in final_text for marker in group)
    ]
    assert len(sections_hit) >= 4, (
        f"FAIL: only {len(sections_hit)} of 7 section markers hit (need >= 4); "
        f"hit = {sections_hit}"
    )

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
    if "paper_ids" in args:
        return {**args, "paper_ids": args.get("paper_ids") or []}
    return args


if __name__ == "__main__":
    main()
