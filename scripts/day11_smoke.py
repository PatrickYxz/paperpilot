"""Day 11 smoke for conservative paper_deep_read.

Main loop -> paper_deep_read -> serial isolated subagents -> markdown summaries
-> final synthesis. The subagent execution is intentionally serial on Day 11
because the current ColBERT server has one global index.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

PAPER_IDS = ["1706.03762", "2010.11929", "2005.14165"]


def main() -> None:
    saw_main: set[str] = set()
    paper_deep_read_calls: list[dict[str, Any]] = []
    main_guardrails: list[str] = []
    subagent_search_previews: list[tuple[str, str]] = []

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        sub_pid = payload.get("subagent_paper_id")
        if kind == "tool_call":
            name = payload["name"]
            args = payload.get("arguments", {})
            if sub_pid:
                print(f"  [sub:{sub_pid}] -> {name}({_preview(args)})")
            else:
                saw_main.add(name)
                if name == "paper_deep_read":
                    paper_deep_read_calls.append(args)
                print(f"  -> {name}({_preview(args)})")
        elif kind == "tool_result":
            name = payload["name"]
            content = payload.get("content", "")
            content_text = content if isinstance(content, str) else str(content)
            tag = f"[sub:{sub_pid}] " if sub_pid else ""
            print(f"  {tag}<- {name}: {content_text[:160]}...")
            if sub_pid and name.endswith("__search"):
                subagent_search_previews.append((sub_pid, content_text))
        elif kind == "guardrail_stop":
            reason = payload["reason"]
            if sub_pid:
                print(f"  [sub:{sub_pid}] !! guardrail: {reason}")
            else:
                main_guardrails.append(reason)
                print(f"  !! guardrail: {reason}")
        elif kind == "tool_arg_repair":
            print(f"  ~ repaired {payload['name']}: {payload['repaired']}")

    prompt = (
        "You must use paper_deep_read exactly once to deep-read these 3 arXiv "
        f"papers: {PAPER_IDS[0]} / {PAPER_IDS[1]} / {PAPER_IDS[2]}. "
        "Compare how self-attention is designed or used across the papers. "
        "After the tool returns, synthesize the comparison and mention at "
        "least two paper IDs in the final answer."
    )
    messages = run(prompt, max_iter=10, on_event=tracer)

    print("\n=== FINAL ===")
    final_text = _extract_text(messages[-1].get("content"))
    print(final_text)

    deep_read_result = _find_tool_result(messages, "paper_deep_read")

    assert "paper_deep_read" in saw_main, (
        f"FAIL: paper_deep_read not called by main agent; saw {saw_main}"
    )
    assert len(paper_deep_read_calls) >= 1, "FAIL: no paper_deep_read call args"
    first_call_args = paper_deep_read_calls[0]
    assert first_call_args.get("paper_ids") == PAPER_IDS, (
        f"FAIL: first paper_deep_read paper_ids mismatch: {first_call_args}"
    )
    assert not main_guardrails, f"FAIL: main guardrail stopped: {main_guardrails}"
    assert deep_read_result, "FAIL: could not find full paper_deep_read tool result"

    for paper_id in PAPER_IDS:
        assert f"### {paper_id}" in deep_read_result, (
            f"FAIL: paper_deep_read result missing segment for {paper_id}"
        )

    ok_count = len(re.findall(r"### \S+ \(status: ok\)", deep_read_result))
    assert ok_count >= 2, (
        f"FAIL: only {ok_count} paper(s) reached status=ok; expected >= 2"
    )
    assert subagent_search_previews, (
        "FAIL: no subagent mcp__colbert__search result was observed"
    )
    for sub_pid, preview in subagent_search_previews:
        assert sub_pid in preview, (
            "FAIL: subagent search preview does not mention its own paper_id; "
            f"subagent={sub_pid}, preview={preview!r}"
        )

    assert len(final_text.strip()) > 200, (
        f"FAIL: final answer too short ({len(final_text)} chars)"
    )
    pid_mentions = sum(1 for paper_id in PAPER_IDS if paper_id in final_text)
    assert pid_mentions >= 2, (
        f"FAIL: final answer mentions only {pid_mentions} paper IDs"
    )

    print("\nDay 11 smoke PASSED")


def _find_tool_result(messages: list[dict], tool_name: str) -> str | None:
    tool_use_ids: set[str] = set()
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = getattr(block, "type", None)
            block_name = getattr(block, "name", None)
            block_id = getattr(block, "id", None)
            if block_type == "tool_use" and block_name == tool_name and block_id:
                tool_use_ids.add(str(block_id))

    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if (
                block.get("type") == "tool_result"
                and block.get("tool_use_id") in tool_use_ids
            ):
                return str(block.get("content", ""))
    return None


def _extract_text(content: Any) -> str:
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
    if "paper_ids" in args and "user_query" in args:
        query = str(args["user_query"])
        return {"paper_ids": args["paper_ids"], "user_query": query[:80]}
    if "documents" in args:
        docs = args.get("documents") or []
        preview_docs = []
        for doc in docs:
            if isinstance(doc, dict):
                preview_docs.append({
                    "paper_id": doc.get("paper_id"),
                    "text_len": len(doc.get("text", "")),
                })
        return {**args, "documents": preview_docs}
    return args


if __name__ == "__main__":
    main()
