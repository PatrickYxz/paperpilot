"""Day 4 冒烟测试:验证 agent loop + LLMClient + guardrail 跑通一轮 tool use。

用法:
    python scripts/day4_smoke.py
    python scripts/day4_smoke.py --query "算一下 2+3"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from paperpilot.core import Guardrail, LLMClient, Tool, agent_loop


def make_add_tool() -> Tool:
    return Tool(
        name="add",
        description="返回两个整数之和。用于需要精确计算加法时。",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "第一个加数"},
                "b": {"type": "integer", "description": "第二个加数"},
            },
            "required": ["a", "b"],
        },
        handler=lambda args: args["a"] + args["b"],
    )


def make_echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="原样返回传入的 text。仅用于 debug。",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda args: args["text"],
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="请用 add 工具计算 127 + 348,然后简短告诉我结果。",
    )
    parser.add_argument("--max-iter", type=int, default=5)
    args = parser.parse_args()

    tools = [make_add_tool(), make_echo_tool()]
    messages = [{"role": "user", "content": args.query}]

    def on_event(kind: str, payload: dict) -> None:
        print(f"[{kind}] {payload}")

    agent_loop(
        messages,
        system="你是一个测试 agent。有 tool 可用时优先调 tool,不要自己心算。",
        tools=tools,
        client=LLMClient(),
        guardrail=Guardrail(max_iterations=args.max_iter),
        on_event=on_event,
    )

    print("\n=== FINAL MESSAGE ===")
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if text:
                print(text)
    else:
        print(content)


if __name__ == "__main__":
    main()