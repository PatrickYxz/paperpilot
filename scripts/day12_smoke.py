"""Day 12 smoke: compare-papers skill + paper_deep_read parallelism.

Concurrency evidence uses subagent lifecycle events. It intentionally avoids
search call/result windows because the MCP stdio/server layer may serialize
tool calls even when worker loops overlap.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

PAPER_IDS = ["1706.03762", "2010.11929", "2005.14165"]


def main() -> None:
    saw_main: set[str] = set()
    saw_load_skill_args: list[dict[str, Any]] = []
    paper_deep_read_calls: list[dict[str, Any]] = []
    main_guardrails: list[str] = []
    sub_search_calls: list[tuple[str, str]] = []
    subagent_windows: dict[str, list[float | None]] = {}

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        sub_pid = payload.get("subagent_paper_id")
        if kind == "subagent_start" and sub_pid:
            subagent_windows[sub_pid] = [time.time(), None]
            print(f"  [sub:{sub_pid}] == start")
            return
        if kind == "subagent_done" and sub_pid:
            subagent_windows.setdefault(sub_pid, [time.time(), None])[1] = time.time()
            print(f"  [sub:{sub_pid}] == done: {payload.get('status')}")
            return
        if kind == "tool_call":
            name = payload["name"]
            args = payload.get("arguments", {})
            if sub_pid:
                print(f"  [sub:{sub_pid}] -> {name}({_preview(args)})")
                if name == "mcp__colbert__search":
                    sub_search_calls.append((sub_pid, args.get("paper_id", "")))
            else:
                saw_main.add(name)
                if name == "load_skill":
                    saw_load_skill_args.append(args)
                if name == "paper_deep_read":
                    paper_deep_read_calls.append(args)
                print(f"  -> {name}({_preview(args)})")
        elif kind == "tool_result":
            name = payload["name"]
            content = payload.get("content", "")
            content_text = content if isinstance(content, str) else str(content)
            tag = f"[sub:{sub_pid}] " if sub_pid else ""
            print(f"  {tag}<- {name}: {content_text[:160]}...")
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
        "I want a comparative deep read of 3 arXiv papers. Please use the "
        "compare-papers skill: load it first, then follow it. The papers are: "
        f"{PAPER_IDS[0]} (Attention Is All You Need), "
        f"{PAPER_IDS[1]} (ViT), and {PAPER_IDS[2]} (GPT-3). "
        "Compare how self-attention is designed or used across them. "
        "Mention at least two paper IDs in the final answer."
    )
    messages = run(prompt, max_iter=12, on_event=tracer)

    print("\n=== FINAL ===")
    final_text = _extract_text(messages[-1].get("content"))
    print(final_text)

    deep_read_result = _find_tool_result(messages, "paper_deep_read") or ""

    assert any(args.get("name") == "compare-papers" for args in saw_load_skill_args), (
        f"FAIL: did not load compare-papers skill; args = {saw_load_skill_args}"
    )
    assert "paper_deep_read" in saw_main, "FAIL: paper_deep_read not called"
    assert paper_deep_read_calls, "FAIL: no paper_deep_read call args"
    assert paper_deep_read_calls[0].get("paper_ids") == PAPER_IDS, (
        f"FAIL: paper_deep_read paper_ids mismatch: {paper_deep_read_calls[0]}"
    )

    ok_count = len(re.findall(r"### \S+ \(status: ok\)", deep_read_result))
    assert ok_count >= 2, f"FAIL: only {ok_count} subagent(s) ok"

    assert sub_search_calls, "FAIL: no subagent search calls observed"
    for sub_pid, arg_pid in sub_search_calls:
        assert sub_pid == arg_pid, (
            f"FAIL: subagent {sub_pid} called search with paper_id={arg_pid}"
        )

    assert _has_overlapping_windows(subagent_windows), (
        f"FAIL: no concurrent subagent lifecycle windows; windows = {subagent_windows}"
    )

    pid_mentions = sum(1 for paper_id in PAPER_IDS if paper_id in final_text)
    assert pid_mentions >= 2, f"FAIL: final answer mentions only {pid_mentions} ids"
    assert not main_guardrails, f"FAIL: main guardrail = {main_guardrails}"

    print("\nDay 12 smoke PASSED")


def _has_overlapping_windows(windows: dict[str, list[float | None]]) -> bool:
    flat: list[tuple[float, float, str]] = []
    for paper_id, window in windows.items():
        if len(window) != 2 or window[0] is None or window[1] is None:
            continue
        flat.append((float(window[0]), float(window[1]), paper_id))
    for index, (start_a, end_a, paper_a) in enumerate(flat):
        for start_b, end_b, paper_b in flat[index + 1:]:
            if paper_a != paper_b and start_a < end_b and start_b < end_a:
                return True
    return False


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
