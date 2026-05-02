"""PaperPilot 顶层入口。Day 5 起。

CLI:python -m paperpilot.main --query "查 3 篇 LoRA 相关近期论文"
库:from paperpilot.main import run; run(query, max_iter=8)
"""
from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path

from dotenv import load_dotenv

from paperpilot.core import Guardrail, LLMClient, agent_loop
from paperpilot.tools.mcp_client import MCPClient

MANIFEST_PATH = Path(__file__).parent / "mcp_servers.json"

SYSTEM_PROMPT = """你是 PaperPilot,一个学术论文研究助手。

工作原则:
- 有 tool 可用时优先调 tool;不要自己编造论文标题、作者或 arxiv id
- 一次只解决用户问的事,不主动扩展任务范围
- tool 报错时,根据错误信息决定:重试(换参数) / 换工具 / 告诉用户失败原因
- 调 tool 时必须按 schema 传完整必填参数;如果错误提示缺字段,下一轮必须补齐字段,不要重复同一个空参数
- 调 mcp__colbert__build_index 时,documents 必须是非空列表,每项包含 paper_id 和 text;通常直接使用 mcp__arxiv__download_paper 返回的对象组成 documents=[download_result]
""".strip()


def _default_logger(kind: str, payload: dict) -> None:
    if kind == "tool_call":
        print(f"  → {payload['name']}({payload['arguments']})")
    elif kind == "tool_result":
        print(f"  ← {payload['name']}: {payload['content'][:120]}...")
    elif kind == "guardrail_stop":
        print(f"  ⚠ guardrail: {payload['reason']}")


def run(query: str, *, max_iter: int = 8, on_event=None) -> list[dict]:
    """跑一次完整 agent 会话,返回最终 messages。"""
    load_dotenv()

    mcp = MCPClient(MANIFEST_PATH)
    mcp.start()
    atexit.register(mcp.close)

    # Day 5: tools 只来自 MCP。Day 8+ 加 L2 inline tools 时改这一行:
    #   tools = INLINE_TOOLS + mcp.list_tools()
    tools = mcp.list_tools()

    messages = [{"role": "user", "content": query}]
    return agent_loop(
        messages,
        system=SYSTEM_PROMPT,
        tools=tools,
        client=LLMClient(),
        guardrail=Guardrail(
            max_iterations=max_iter,
            budget_tokens=int(os.environ.get("BUDGET_TOKENS", 50_000)),
        ),
        on_event=on_event or _default_logger,
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="paperpilot")
    p.add_argument("--query", required=True)
    p.add_argument("--max-iter", type=int, default=8)
    args = p.parse_args()
    messages = run(args.query, max_iter=args.max_iter)
    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for b in last:
            if hasattr(b, "text"):
                print(b.text)
    else:
        print(last)


if __name__ == "__main__":
    main()
