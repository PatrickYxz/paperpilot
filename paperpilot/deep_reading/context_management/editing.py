"""Local request-copy editing for eligible historical tool results."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

from .models import FutureRetention, ToolResultDisposition
from .ports import EditedRequest


@dataclass(frozen=True)
class EditingDisposition:
    message_index: int
    disposition: ToolResultDisposition
    entered_successful_model_call: bool
    downstream_consumed: bool
    recoverable: bool


class LocalContextEditingAdapter:
    def __init__(
        self,
        *,
        token_counter,
        usable_input_budget: int,
        trigger_ratio: float = 0.70,
        min_reclaim_tokens: int = 8_000,
        min_reclaim_ratio: float = 0.10,
        keep_recent_tool_results: int = 3,
        event_sink: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        if usable_input_budget < 1:
            raise ValueError("usable_input_budget must be positive")
        if not 0 < trigger_ratio < 1:
            raise ValueError("trigger_ratio must be between 0 and 1")
        if keep_recent_tool_results < 0:
            raise ValueError("keep_recent_tool_results must be non-negative")
        self._token_counter = token_counter
        self._usable_input_budget = usable_input_budget
        self._trigger_ratio = trigger_ratio
        self._min_reclaim_tokens = min_reclaim_tokens
        self._min_reclaim_ratio = min_reclaim_ratio
        self._keep_recent_tool_results = keep_recent_tool_results
        self._event_sink = event_sink or (lambda _kind, _payload: None)

    def edit(
        self,
        request: Any,
        dispositions: Sequence[EditingDisposition],
    ) -> EditedRequest:
        messages = list(request.messages)
        tool_schemas = getattr(request, "tools", ()) or ()
        current_tokens = getattr(request, "input_tokens", None)
        if not isinstance(current_tokens, int) or current_tokens < 0:
            current_tokens = self._token_counter.count_messages(
                messages,
                tool_schemas=tool_schemas,
            )
        eligible = [item for item in dispositions if self._eligible(item)]
        reclaimable = sum(
            item.disposition.artifact_ref.token_estimate
            for item in eligible
            if item.disposition.artifact_ref is not None
        )
        required_reclaim = max(
            self._min_reclaim_tokens,
            int(self._usable_input_budget * self._min_reclaim_ratio + 0.999999),
        )
        if (
            current_tokens < self._usable_input_budget * self._trigger_ratio
            or reclaimable < required_reclaim
        ):
            return EditedRequest(tuple(messages), current_tokens)

        eligible.sort(key=lambda item: item.message_index)
        keep_count = self._keep_recent_tool_results
        selected = eligible if keep_count == 0 else eligible[:-keep_count]
        if not selected:
            return EditedRequest(tuple(messages), current_tokens)
        event_payload = {
            "stage": "micro",
            "compressor_version": "context-editing-v1",
            "reason": "eligible_history_batch",
            "before_tokens": current_tokens,
            "after_tokens": current_tokens,
            "reclaimed_tokens": 0,
            "protected_item_count": sum(
                item.disposition.future_retention is FutureRetention.PROTECTED
                for item in dispositions
            ),
            "archive_ref_count": 0,
            "artifact_ref_count": sum(
                item.disposition.artifact_ref is not None for item in selected
            ),
            "validation_failure_type": None,
            "breaker_state": "CLOSED",
        }
        self._event_sink("micro_compaction_started", dict(event_payload))
        for item in selected:
            if not 0 <= item.message_index < len(messages):
                continue
            messages[item.message_index] = _replace_tool_message(
                messages[item.message_index],
                item.disposition,
            )
        edited_tokens = self._token_counter.count_messages(
            messages,
            tool_schemas=tool_schemas,
        )
        event_payload["after_tokens"] = edited_tokens
        event_payload["reclaimed_tokens"] = max(0, current_tokens - edited_tokens)
        self._event_sink("micro_compaction_completed", dict(event_payload))
        return EditedRequest(tuple(messages), edited_tokens)

    @staticmethod
    def _eligible(item: EditingDisposition) -> bool:
        disposition = item.disposition
        return (
            item.entered_successful_model_call
            and item.downstream_consumed
            and item.recoverable
            and disposition.future_retention is FutureRetention.CLEARABLE_AFTER_USE
            and disposition.artifact_ref is not None
        )


class ResearchContextMiddleware(AgentMiddleware[Any, Any, Any]):
    """Apply ingestion before a result is next shown and edit request copies."""

    def __init__(
        self,
        *,
        ingestor=None,
        editing_adapter: LocalContextEditingAdapter | None = None,
        conversation_id: str = "",
        task_id: str = "",
    ) -> None:
        super().__init__()
        self._ingestor = ingestor
        self._editing_adapter = editing_adapter
        self._conversation_id = conversation_id
        self._task_id = task_id
        self._dispositions: list[EditingDisposition] = []

    @property
    def dispositions(self) -> tuple[EditingDisposition, ...]:
        return tuple(self._dispositions)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        result = handler(request)
        if self._ingestor is None or not isinstance(result, ToolMessage):
            return result
        tool_name = str(request.tool_call.get("name", ""))
        if tool_name not in {
            "search_related_papers",
            "prepare_paper",
            "retrieve_paper_evidence",
        }:
            return result
        try:
            raw_result = getattr(result, "artifact", None)
            if raw_result is None:
                raw_result = (
                    json.loads(result.content)
                    if isinstance(result.content, str)
                    else result.content
                )
        except json.JSONDecodeError:
            raw_result = result.content
        call_id = result.tool_call_id or str(request.tool_call.get("id", ""))
        ingested = self._ingestor.ingest(
            __import__(
                "paperpilot.deep_reading.context_management.artifacts",
                fromlist=["ToolResultIngestRequest"],
            ).ToolResultIngestRequest(
                conversation_id=self._conversation_id,
                task_id=self._task_id,
                tool_call_id=call_id,
                tool_name=tool_name,
                raw_result=raw_result,
            )
        )
        self._dispositions.append(
            EditingDisposition(
                message_index=-1,
                disposition=ingested.disposition,
                entered_successful_model_call=False,
                downstream_consumed=False,
                recoverable=ingested.disposition.artifact_ref is not None,
            )
        )
        return ToolMessage(
            content=ingested.model_content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            id=result.id,
            status=result.status,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        edited_request = request
        if self._editing_adapter is not None:
            prepared = self._editing_adapter.edit(request, self._indexed_dispositions(request))
            edited_request = request.override(messages=list(prepared.messages))
        response = handler(edited_request)
        self._mark_successful_input(edited_request.messages)
        return response

    def _indexed_dispositions(self, request: ModelRequest[Any]) -> list[EditingDisposition]:
        index_by_call_id = {
            call_id: index
            for index, message in enumerate(request.messages)
            if (call_id := _tool_call_id(message)) is not None
        }
        updated: list[EditingDisposition] = []
        for item in self._dispositions:
            if item.message_index >= 0:
                updated.append(item)
                continue
            message_index = index_by_call_id.get(item.disposition.tool_call_id, -1)
            updated.append(replace(item, message_index=message_index))
        self._dispositions = updated
        return list(updated)

    def _mark_successful_input(self, messages: Sequence[Any]) -> None:
        consumed_call_ids = {
            call_id
            for message in messages
            if (call_id := _tool_call_id(message)) is not None
        }
        self._dispositions = [
            replace(
                item,
                entered_successful_model_call=True,
                downstream_consumed=True,
            )
            if item.disposition.tool_call_id in consumed_call_ids
            else item
            for item in self._dispositions
        ]


def _tool_call_id(message: Any) -> str | None:
    value = (
        message.get("tool_call_id")
        if isinstance(message, dict)
        else getattr(message, "tool_call_id", None)
    )
    return value if isinstance(value, str) and value else None


def _replace_tool_message(message: Any, disposition: ToolResultDisposition) -> Any:
    ref = disposition.artifact_ref
    if ref is None:
        return message
    content = (
        f'<artifact_ref id="{ref.artifact_id}" '
        f'sha256="{ref.sha256}" '
        'summary="本结果已完成消费，原文可按需读取" />'
    )
    if isinstance(message, dict):
        updated = dict(message)
        updated["content"] = content
        return updated
    if isinstance(message, ToolMessage):
        return ToolMessage(
            content=content,
            tool_call_id=message.tool_call_id,
            name=message.name,
            id=message.id,
            status=message.status,
        )
    try:
        return message.model_copy(update={"content": content})
    except AttributeError:
        return message
