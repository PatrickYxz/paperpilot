"""Day 10 smoke: load_skill(find-classics) -> research_todo + arxiv + graph."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

EXPECT_TOOLS = {
    "load_skill",
    "research_todo",
    "mcp__arxiv__search_papers",
    "mcp__graph__build_graph",
    "mcp__graph__get_common_citations",
}


def main() -> None:
    saw: set[str] = set()
    research_todo_calls = 0
    load_skill_targets: list[str] = []
    guardrail_reasons: list[str] = []

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        nonlocal research_todo_calls
        if kind == "tool_call":
            name = payload["name"]
            saw.add(name)
            if name == "research_todo":
                research_todo_calls += 1
            elif name == "load_skill":
                load_skill_targets.append(
                    payload.get("arguments", {}).get("name", "")
                )
            print(f"  -> {name}({_preview(name, payload.get('arguments', {}))})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            preview = (
                content[:160] if isinstance(content, str) else str(content)[:160]
            )
            print(f"  <- {payload['name']}: {preview}...")
        elif kind == "guardrail_stop":
            guardrail_reasons.append(payload["reason"])
            print(f"  !! guardrail: {payload['reason']}")
        elif kind == "tool_arg_repair":
            print(f"  ~ repaired {payload['name']}: {payload['repaired']}")

    messages = run(
        "我想入门 retrieval-augmented generation 领域, 帮我找出该领域被反复"
        "引用的几篇必读经典。",
        max_iter=16,
        on_event=tracer,
    )

    print("\n=== FINAL ===")
    final_text = _extract_text(messages[-1].get("content"))
    print(final_text)

    missing = EXPECT_TOOLS - saw
    assert not missing, f"FAIL: expected tools missing: {missing}; saw {saw}"
    assert not guardrail_reasons, f"FAIL: guardrail stopped: {guardrail_reasons}"
    assert final_text.strip(), "FAIL: final answer is empty"
    assert research_todo_calls >= 2, (
        f"FAIL: research_todo only called {research_todo_calls} times; "
        "expected >= 2 (initial plan + at least 1 progress update)"
    )
    assert "find-classics" in load_skill_targets, (
        f"FAIL: load_skill never called with name='find-classics'; "
        f"saw load_skill targets: {load_skill_targets}"
    )
    print("\nDay 10 smoke PASSED")


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


def _preview(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "research_todo":
        todos = args.get("todos", [])
        return {
            "todos_count": len(todos),
            "in_progress": [
                item.get("content", "")[:30]
                for item in todos
                if item.get("status") == "in_progress"
            ],
            "completed_count": sum(
                1 for item in todos if item.get("status") == "completed"
            ),
        }
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
