"""Day 5 冒烟:Main Loop ←→ MCPClient ←→ stdio ←→ arxiv-mcp ←→ arXiv API
              ←→ LLM(DeepSeek) 全链路验证。

跑一次 ~$0.01(DeepSeek),需联网 + 有效 DEEPSEEK_API_KEY。
用法:python scripts/day5_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

EXPECT_TOOL = "mcp__arxiv__search_papers"


def main() -> None:
    saw_tool_call = False
    saw_tool_result = False

    def check(kind: str, payload: dict) -> None:
        nonlocal saw_tool_call, saw_tool_result
        if kind == "tool_call" and payload["name"] == EXPECT_TOOL:
            saw_tool_call = True
        if kind == "tool_result" and payload["name"] == EXPECT_TOOL:
            saw_tool_result = True
        print(f"[{kind}] {payload}")

    messages = run(
        "找 3 篇 LoRA 微调相关的近期论文,告诉我标题、作者和一句话摘要。",
        max_iter=5,
        on_event=check,
    )

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for b in last:
            if hasattr(b, "text"):
                print(b.text)

    assert saw_tool_call, f"FAIL: 期望模型调用 {EXPECT_TOOL},未观察到"
    assert saw_tool_result, f"FAIL: 期望 {EXPECT_TOOL} 返回 result,未观察到"
    print("\n✅ Day 5 smoke PASSED")


if __name__ == "__main__":
    main()
