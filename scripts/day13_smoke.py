"""Day 13 smoke: compact_context self-directed trigger.

The prompt first creates a long enough research conversation, then switches to
an unrelated short question and asks the model to compact stale history before
answering. The assertions verify that compact_context ran and compressed a
non-trivial middle section.
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


PAPER_ID = "1706.03762"


def main() -> None:
    saw_compact_start = 0
    saw_compact_done = 0
    tool_calls: list[str] = []
    compact_result_text: str | None = None
    middle_count_seen: int | None = None

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        nonlocal saw_compact_start
        nonlocal saw_compact_done
        nonlocal compact_result_text
        nonlocal middle_count_seen

        if kind == "compact_start":
            saw_compact_start += 1
            middle_count_seen = int(payload.get("middle_count", 0))
            print(f"  ## compact_start: middle={middle_count_seen}")
        elif kind == "compact_done":
            saw_compact_done += 1
            print(f"  ## compact_done: kept={payload.get('kept_recent')}")
        elif kind == "tool_call":
            name = payload["name"]
            tool_calls.append(name)
            print(f"  -> {name}")
        elif kind == "tool_result":
            name = payload["name"]
            content = payload.get("content", "")
            content_text = content if isinstance(content, str) else str(content)
            print(f"  <- {name}: {content_text[:160]}...")
            if name == "compact_context":
                compact_result_text = content_text
        elif kind == "guardrail_stop":
            print(f"  !! guardrail: {payload.get('reason')}")

    prompt = (
        f"先深读 arxiv {PAPER_ID} (Attention Is All You Need): 调 "
        "load_skill('deep-read-paper'), 然后按 skill 走 download -> build_index -> "
        "至少 4 次 colbert.search(query 覆盖 method/experiments/ablation/"
        "limitations), 把结果整合成一段精读摘要。\n\n"
        "完成上面之后, 我换一个完全无关的问题: 用一句话告诉我 Python list 和 "
        "tuple 的区别。回答这个问题前, 先调 compact_context() 把前面深读的对话 "
        "历史压缩 (history 已经很长, 后面这个问题不需要前文 tool_result 的细节)。"
    )

    messages = run(prompt, max_iter=20, on_event=tracer)

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for block in last:
            if hasattr(block, "text"):
                print(block.text)
    else:
        print(last)

    assert saw_compact_start >= 1, (
        f"FAIL: compact_start not observed; tool_calls={tool_calls}"
    )
    assert saw_compact_done >= 1, "FAIL: compact_done not observed"
    assert "compact_context" in tool_calls, (
        f"FAIL: compact_context tool never called; tool_calls={tool_calls}"
    )
    assert compact_result_text is not None, "FAIL: no compact_context tool_result"
    match = re.search(r"compacted (\d+) messages into summary", compact_result_text)
    assert match, (
        "FAIL: compact tool_result missing compacted count; "
        f"got: {compact_result_text!r}"
    )
    compacted_n = int(match.group(1))
    assert compacted_n >= 4, (
        f"FAIL: compacted only {compacted_n} messages; expected >= 4"
    )
    assert middle_count_seen == compacted_n, (
        f"FAIL: middle_count={middle_count_seen} != compacted_n={compacted_n}"
    )

    print(
        f"\nDay 13 smoke PASSED "
        f"(compacted={compacted_n}, final_messages={len(messages)}, "
        f"compact_calls={saw_compact_start})"
    )


if __name__ == "__main__":
    main()
