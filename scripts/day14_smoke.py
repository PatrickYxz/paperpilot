"""Day 14 smoke: analyze-figures skill + vlm understand_paper_page.

Expected tracer path:
load_skill('analyze-figures') -> colbert text location -> understand_paper_page.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

PAPER_ID = "2010.11929"


def main() -> None:
    saw_load_skill_args: list[dict[str, Any]] = []
    tool_calls: list[str] = []
    vlm_calls: list[dict[str, Any]] = []

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        if kind == "tool_call":
            name = payload["name"]
            args = payload.get("arguments", {})
            tool_calls.append(name)
            if name == "load_skill":
                saw_load_skill_args.append(args)
            if name == "mcp__vlm__understand_paper_page":
                vlm_calls.append(args)
            print(f"  -> {name}({_preview(args)})")
        elif kind == "tool_result":
            name = payload["name"]
            content = payload.get("content", "")
            content_text = content if isinstance(content, str) else str(content)
            print(f"  <- {name}: {content_text[:160]}...")
        elif kind == "guardrail_stop":
            print(f"  !! guardrail: {payload.get('reason')}")

    prompt = (
        f"我想了解 arxiv {PAPER_ID} (Vision Transformer) 的 Figure 1 画的是什么。"
        " 请按 analyze-figures skill 操作: 先 load_skill, 然后跟 prose 走"
        " (colbert 文本定位 -> understand_paper_page)。"
    )
    messages = run(prompt, max_iter=15, on_event=tracer)

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for block in last:
            if hasattr(block, "text"):
                print(block.text)
    else:
        print(last)
    final_text = _final_text(last)

    assert any(args.get("name") == "analyze-figures" for args in saw_load_skill_args), (
        "FAIL: did not load analyze-figures skill; "
        f"load_skill args = {saw_load_skill_args}"
    )
    assert any(c.startswith("mcp__colbert__") for c in tool_calls), (
        "FAIL: no colbert call (text-locate step skipped)"
    )
    assert vlm_calls, f"FAIL: vlm tool never called; tool_calls={tool_calls}"
    assert vlm_calls[0].get("arxiv_id") == PAPER_ID
    assert isinstance(vlm_calls[0].get("page_num"), int)
    assert vlm_calls[0]["page_num"] >= 1

    keywords = ("patch", "16x16", "linear projection", "embedding")
    assert any(kw in final_text.lower() for kw in keywords), (
        f"FAIL: final answer missing ViT keywords {keywords}; got: {final_text!r}"
    )

    print(
        f"\nDay 14 smoke PASSED "
        f"(vlm_calls={len(vlm_calls)}, total_tool_calls={len(tool_calls)})"
    )


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
    return args


if __name__ == "__main__":
    main()
