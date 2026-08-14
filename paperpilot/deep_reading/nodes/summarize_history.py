"""Conversation-history summarization node and history-window helpers."""
from __future__ import annotations

import json
from collections.abc import Mapping

from langchain.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from pydantic import ValidationError

from ..research_agent import ResearchContractError
from ..schemas import ConversationSummary
from ..state import DeepReadingState
from .binding import _validate_runtime_binding
from .context import DeepReadingContext


def needs_summary(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> bool:
    """Return whether estimated checkpoint context exceeds the configured budget."""
    return _estimated_tokens(state) > runtime.context.summary_token_threshold


def summarize_history(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Summarize long history and replace current State messages with recent turns."""
    if not needs_summary(state, runtime):
        return {}

    context = runtime.context
    _validate_runtime_binding(state, context)
    summary_model = context.model.with_structured_output(
        ConversationSummary,
        include_raw=True,
    )
    summary_input: list[AnyMessage] = [
        SystemMessage(
            content=(
                "Summarize the conversation for continued scholarly paper reading. "
                "Preserve only confirmed facts, paper findings, comparison context, "
                "and open questions."
            )
        )
    ]
    previous_summary = state.get("conversation_summary")
    if previous_summary is not None:
        summary_input.append(
            HumanMessage(
                content=(
                    "Previous structured summary:\n"
                    + json.dumps(previous_summary, ensure_ascii=False, sort_keys=True)
                )
            )
        )
    summary_input.extend(state.get("messages", []))

    summary_envelope = summary_model.invoke(summary_input)
    if not isinstance(summary_envelope, Mapping):
        raise ResearchContractError(
            "model returned an invalid conversation summary"
        )
    parsing_error = summary_envelope.get("parsing_error")
    if isinstance(parsing_error, BaseException):
        raise ResearchContractError(
            "model returned an invalid conversation summary"
        ) from parsing_error
    if parsing_error is not None or "parsed" not in summary_envelope:
        raise ResearchContractError(
            "model returned an invalid conversation summary"
        )
    try:
        summary = ConversationSummary.model_validate(summary_envelope["parsed"])
    except ValidationError as exc:
        raise ResearchContractError(
            "model returned an invalid conversation summary"
        ) from exc
    recent_messages = _recent_turns(
        state.get("messages", []), context.summary_recent_turns
    )
    return {
        "conversation_summary": summary.model_dump(mode="json"),
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages],
    }


def _estimated_tokens(state: DeepReadingState) -> int:
    total_chars = sum(
        _message_character_count(message) for message in state.get("messages", [])
    )
    summary = state.get("conversation_summary")
    if summary is not None:
        total_chars += len(
            json.dumps(
                summary,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return max(1, total_chars // 4)


def _message_character_count(message: AnyMessage) -> int:
    content = message.content
    if isinstance(content, str):
        return len(content)
    return len(json.dumps(content, ensure_ascii=False, separators=(",", ":")))


def _recent_turns(messages: list[AnyMessage], turn_count: int) -> list[AnyMessage]:
    turns_seen = 0
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].type == "human":
            turns_seen += 1
            if turns_seen == turn_count:
                return messages[index:]
    return list(messages)
