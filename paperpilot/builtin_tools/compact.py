"""compact_context built-in tool.

LLM decides when to call this tool. The handler closes over the main agent's
messages list and mutates it in place, replacing older turns with a structured
summary while keeping the most recent turns intact.
"""
from __future__ import annotations

from typing import Callable

from paperpilot.core.adapter import LLMClient, Tool
from paperpilot.core.loop import EventCallback


COMPACT_CONTEXT_NUDGE = """
## Long conversation
When tool results pile up and earlier turns no longer matter, call
compact_context() with no arguments. It rewrites the history into a
structured summary and frees context budget. Do not call it on a short
conversation or when you are mid-step (e.g. just got a tool result and
are about to act on it).
""".strip()

SUMMARIZE_SYSTEM = "You are a conversation history summarizer."

SUMMARIZE_PROMPT = """
你的任务是把下面的对话历史压缩成结构化 summary, 供后续 LLM 理解上下文用。

输出格式:
## 已完成的关键步骤
<按时间顺序列出 tool calls 和重要结论, 5-10 条>

## 关键发现 / 中间结果
<事实性内容、找到的段落或数据, 简洁列点>

## 待办 / 下一步
<如果 history 里 LLM 表达过未完成的计划, 列出来>

要求:
- 只保留事实和决策, 删掉客套和重复
- 引用具体 paper_id / chunk 摘要 / 数字结果
- 总长度控制在 800 token 内

对话历史:
""".strip()


def compact_context_tool(
    *,
    messages_ref: list[dict],
    client_factory: Callable[[], LLMClient],
    on_event: EventCallback,
    keep_recent_turns: int = 3,
) -> Tool:
    """Build the compact_context tool."""
    k = keep_recent_turns

    def _handler(args: dict) -> str:
        if len(messages_ref) <= 1 + 2 * k:
            return "already compact, nothing to summarize"

        head = messages_ref[0]
        tail_start = _tail_start_preserving_tool_results(messages_ref, 2 * k)
        tail = messages_ref[tail_start:]
        middle = messages_ref[1:tail_start]
        if not middle:
            return "already compact, nothing to summarize"

        on_event("compact_start", {"middle_count": len(middle)})
        summary_text = _summarize(middle, client_factory())
        on_event("compact_done", {"kept_recent": len(tail)})

        messages_ref[:] = [
            head,
            {
                "role": "user",
                "content": (
                    f"<context_summary>\n{summary_text}\n</context_summary>"
                ),
            },
            *tail,
        ]
        return (
            f"compacted {len(middle)} messages into summary; "
            f"kept last {len(tail)} turns"
        )

    return Tool(
        name="compact_context",
        description=(
            "Compress earlier conversation into a structured summary, freeing "
            "context budget. Call when the conversation is long and recent "
            "tool results have made earlier turns redundant."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=_handler,
    )


def _summarize(middle: list[dict], client: LLMClient) -> str:
    rendered = _render_messages_for_summary(middle)
    prompt = f"{SUMMARIZE_PROMPT}\n{rendered}"
    response = client.call(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        system=SUMMARIZE_SYSTEM,
    )
    return response.text or "(empty summary)"


def _render_messages_for_summary(middle: list[dict]) -> str:
    """Best-effort plain-text render for the summarizer LLM."""
    lines: list[str] = []
    for message in middle:
        role = message.get("role", "?")
        content = message.get("content")
        if isinstance(content, str):
            lines.append(f"[{role}] {content}")
            continue
        if not isinstance(content, list):
            lines.append(f"[{role}] {content!r}")
            continue
        for block in content:
            if isinstance(block, dict):
                lines.append(_render_dict_block(role, block))
                continue
            block_type = getattr(block, "type", None)
            if block_type == "text":
                lines.append(f"[{role} text] {getattr(block, 'text', '')}")
            elif block_type == "tool_use":
                lines.append(
                    f"[{role} tool_use] "
                    f"{getattr(block, 'name', '?')}"
                    f"({getattr(block, 'input', {})!r})"
                )
            else:
                lines.append(f"[{role} {block_type}] {block!r}")
    return "\n".join(lines)


def _render_dict_block(role: str, block: dict) -> str:
    if block.get("type") == "tool_result":
        inner = block.get("content", "")
        inner_text = inner if isinstance(inner, str) else str(inner)
        return (
            f"[tool_result for {block.get('tool_use_id', '?')}] "
            f"{inner_text[:500]}"
        )
    return f"[{role} {block.get('type', '?')}] {block!r}"


def _tail_start_preserving_tool_results(
    messages: list[dict],
    desired_count: int,
) -> int:
    """Return a suffix start that does not orphan a leading tool_result."""
    start = max(1, len(messages) - desired_count)
    if start < len(messages) and _is_tool_result_message(messages[start]):
        start = max(1, start - 1)
    return start


def _is_tool_result_message(message: dict) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    return (
        isinstance(content, list)
        and bool(content)
        and all(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in content
        )
    )
