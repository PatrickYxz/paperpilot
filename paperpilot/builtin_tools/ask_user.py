"""ask_user built-in tool.

The agent can call this tool when it needs a missing user decision before it
can continue the current turn.
"""
from __future__ import annotations

from typing import Callable

from paperpilot.core.adapter import Tool
from paperpilot.core.loop import EventCallback


ASK_USER_NUDGE = """
## Ask user
When required information is missing or a risky action needs explicit
confirmation, call ask_user(question=..., reason=...). Use it sparingly: do
not ask the user when you can answer or proceed safely with the available
context.
""".strip()


def ask_user_tool(
    *,
    input_provider: Callable[[str], str],
    on_event: EventCallback,
) -> Tool:
    """Build an ask_user tool backed by an injectable input provider."""

    def _handler(args: dict) -> str:
        question = str(args.get("question", "")).strip()
        reason = str(args.get("reason", "")).strip()
        if not question:
            raise ValueError("question is required")

        on_event("ask_user_prompt", {"question": question, "reason": reason})
        answer = input_provider(question)
        on_event("ask_user_answer", {"question": question, "answer": answer})
        return answer

    return Tool(
        name="ask_user",
        description=(
            "Ask the user a concise blocking question when required "
            "information is missing or explicit confirmation is needed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A concise question to ask the user before continuing."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Why this answer is needed.",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        handler=_handler,
    )

