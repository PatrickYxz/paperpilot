"""PaperPilot top-level entrypoint.

CLI:
    python -m paperpilot.main --query "..."

Library:
    from paperpilot.main import run
    run(query, max_iter=8)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from paperpilot.builtin_tools.research_todo import (
    RESEARCH_TODO_NUDGE,
    TodoStore,
    research_todo_tool,
)
from paperpilot.builtin_tools.subagent import (
    PAPER_DEEP_READ_NUDGE,
    paper_deep_read_tool,
)
from paperpilot.builtin_tools.skill_loader import (
    SkillRegistry,
    load_skill_tool,
    render_skill_section,
)
from paperpilot.core import Guardrail, LLMClient, agent_loop
from paperpilot.core.adapter import Tool
from paperpilot.tools.mcp_client import MCPClient

MANIFEST_PATH = Path(__file__).parent / "mcp_servers.json"
SKILLS_DIR = Path(__file__).parent / "skills"

SYSTEM_PROMPT_BASE = """你是 PaperPilot,一个学术论文研究助手。
工作原则:
- 有 tool 可用时优先调 tool;不要自己编造论文标题、作者或 arxiv id
- 一次只解决用户问的事,不主动扩展任务范围
- tool 报错时,根据错误信息决定:重试(换参数 / 换工具) / 告诉用户失败原因
- 调 tool 时必须按 schema 传完整必填参数;如果错误提示缺字段,下一轮必须补齐字段,不要重复同一个空参数
- 调 mcp__colbert__build_index 时,documents 必须是非空列表,每项包含 paper_id 和 text;通常直接使用 mcp__arxiv__download_paper 返回的对象组成 documents=[download_result]
""".strip()


def _build_system_prompt(registry: SkillRegistry | None = None) -> str:
    registry = registry or SkillRegistry(SKILLS_DIR)
    return (
        SYSTEM_PROMPT_BASE
        + render_skill_section(registry.list_metadata())
        + "\n\n"
        + RESEARCH_TODO_NUDGE
        + "\n\n"
        + PAPER_DEEP_READ_NUDGE
    )


def _build_tools(
    registry: SkillRegistry | None = None,
    todo_store: TodoStore | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[list[Tool], MCPClient]:
    """Return (tools, mcp_client); caller is responsible for close()."""
    registry = registry or SkillRegistry(SKILLS_DIR)
    todo_store = todo_store or TodoStore()
    emit = on_event or _default_logger

    mcp = MCPClient(MANIFEST_PATH)
    try:
        mcp.start()
        mcp_tools = mcp.list_tools()
        tools: list[Tool] = [
            load_skill_tool(registry),
            research_todo_tool(todo_store),
            paper_deep_read_tool(
                client_factory=lambda: LLMClient(),
                mcp_tools=mcp_tools,
                on_event=emit,
            ),
            *mcp_tools,
        ]
        return tools, mcp
    except Exception:
        mcp.close()
        raise


def _default_logger(kind: str, payload: dict) -> None:
    if kind == "tool_call":
        print(f"  -> {payload['name']}({payload['arguments']})")
    elif kind == "tool_result":
        print(f"  <- {payload['name']}: {payload['content'][:120]}...")
    elif kind == "guardrail_stop":
        print(f"  !! guardrail: {payload['reason']}")


def run(query: str, *, max_iter: int = 8, on_event=None) -> list[dict]:
    """Run one complete agent conversation and return final messages."""
    load_dotenv()

    emit = on_event or _default_logger
    registry = SkillRegistry(SKILLS_DIR)
    todo_store = TodoStore()
    tools, mcp = _build_tools(registry, todo_store, on_event=emit)
    try:
        messages = [{"role": "user", "content": query}]
        return agent_loop(
            messages,
            system=_build_system_prompt(registry),
            tools=tools,
            client=LLMClient(),
            guardrail=Guardrail(
                max_iterations=max_iter,
                budget_tokens=int(os.environ.get("BUDGET_TOKENS", 50_000)),
            ),
            on_event=emit,
        )
    finally:
        mcp.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="paperpilot")
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-iter", type=int, default=8)
    args = parser.parse_args()
    messages = run(args.query, max_iter=args.max_iter)
    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for block in last:
            if hasattr(block, "text"):
                print(block.text)
    else:
        print(last)


if __name__ == "__main__":
    main()
