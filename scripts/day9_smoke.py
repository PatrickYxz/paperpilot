"""Day 9 smoke: main loop -> load_skill -> arxiv -> colbert -> answer.

The prompt intentionally does not say "call load_skill first". The smoke fails
if the model skips skill loading, because Day 9 is validating that mechanism.
"""
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
    "mcp__arxiv__download_paper",
    "mcp__colbert__build_index",
    "mcp__colbert__search",
}


def main() -> None:
    saw: set[str] = set()

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        if kind == "tool_call":
            saw.add(payload["name"])
            print(f"  -> {payload['name']}({_preview(payload.get('arguments', {}))})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            preview = content[:160] if isinstance(content, str) else str(content)[:160]
            print(f"  <- {payload['name']}: {preview}...")
        elif kind == "guardrail_stop":
            print(f"  !! guardrail: {payload['reason']}")
        elif kind == "tool_arg_repair":
            print(f"  ~ repaired {payload['name']}: {payload['repaired']}")

    messages = run(
        "帮我深读 arxiv 论文 1706.03762,重点讲 multi-head attention 是怎么定义的。"
        "回答必须基于论文具体段落,不要只看 abstract。",
        max_iter=10,
        on_event=tracer,
    )

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for block in last:
            if hasattr(block, "text"):
                print(block.text)
    else:
        print(last)

    missing = EXPECT_TOOLS - saw
    assert not missing, f"FAIL: expected tools missing: {missing}; saw {saw}"
    assert "load_skill" in saw, "FAIL: skill loading did not take effect"
    print("\nDay 9 smoke PASSED")


def _preview(args: dict[str, Any]) -> dict[str, Any]:
    if "documents" not in args:
        return args

    docs = args.get("documents") or []
    preview_docs = []
    for doc in docs:
        if isinstance(doc, dict):
            preview_docs.append({
                "paper_id": doc.get("paper_id"),
                "text_len": len(doc.get("text", "")),
            })
    return {**args, "documents": preview_docs}


if __name__ == "__main__":
    main()

