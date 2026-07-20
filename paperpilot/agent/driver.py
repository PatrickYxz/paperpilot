"""Serializable, one-boundary driver for durable agent runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from paperpilot.agent.models import StepKind
from paperpilot.agent.policy import RunPolicy
from paperpilot.agent.tool_executor import ToolExecutor, ToolInvocationResult
from paperpilot.builtin_tools.compact import compact_messages
from paperpilot.core.adapter import LLMClient, Tool, ToolCall, ToolResult
from paperpilot.core.context_manager import ContextManager
from paperpilot.core.loop import (
    EventCallback,
    extract_downloaded_document,
    should_repair_build_index_args,
)
from paperpilot.message_codec import decode_messages, encode_messages


class DriverNonRetryableError(RuntimeError):
    """A deterministic driver failure that a new attempt cannot repair."""

    retryable = False
    failure_class = "capacity"


class BudgetExceededError(DriverNonRetryableError):
    """The run exhausted an LLM iteration or token budget."""


class ContextOverflowError(DriverNonRetryableError):
    """Compaction could not bring the conversation below the critical limit."""


@dataclass(frozen=True)
class DriverState:
    messages: list[dict[str, Any]]
    turn_count: int = 0
    tokens_used: int = 0
    downloaded_documents: list[dict[str, Any]] = field(default_factory=list)
    todo_items: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_calls: list[ToolCall] = field(default_factory=list)
    pending_tool_results: list[ToolResult] = field(default_factory=list)
    final_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": encode_messages(self.messages),
            "turn_count": self.turn_count,
            "tokens_used": self.tokens_used,
            "downloaded_documents": deepcopy(self.downloaded_documents),
            "todo_items": deepcopy(self.todo_items),
            "pending_tool_calls": [
                tool_call_dict(call) for call in self.pending_tool_calls
            ],
            "pending_tool_results": [
                tool_result_dict(result) for result in self.pending_tool_results
            ],
            "final_text": self.final_text,
        }

    def runtime_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("messages")
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DriverState":
        return cls(
            messages=decode_messages(payload.get("messages", [])),
            turn_count=int(payload.get("turn_count", 0)),
            tokens_used=int(payload.get("tokens_used", 0)),
            downloaded_documents=deepcopy(
                payload.get("downloaded_documents", [])
            ),
            todo_items=deepcopy(payload.get("todo_items", [])),
            pending_tool_calls=[
                tool_call_from_dict(call)
                for call in payload.get("pending_tool_calls", [])
            ],
            pending_tool_results=[
                tool_result_from_dict(result)
                for result in payload.get("pending_tool_results", [])
            ],
            final_text=payload.get("final_text"),
        )


@dataclass(frozen=True)
class DriverAction:
    kind: StepKind
    input_data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "input_data": deepcopy(self.input_data)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DriverAction":
        return cls(kind=payload["kind"], input_data=deepcopy(payload["input_data"]))


@dataclass(frozen=True)
class DriverStepResult:
    state: DriverState
    output_data: dict[str, Any]
    final_text: str | None = None
    tool_completion: ToolInvocationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": self.state.to_dict(),
            "output_data": deepcopy(self.output_data),
            "final_text": self.final_text,
            "tool_completion": None,
        }
        if self.tool_completion is not None:
            completion = self.tool_completion
            payload["tool_completion"] = {
                "execution_id": completion.execution_id,
                "tool_result": tool_result_dict(completion.tool_result),
                "result_preview": completion.result_preview,
                "duration_ms": completion.duration_ms,
                "classification": completion.classification,
                "sanitized_arguments": deepcopy(completion.sanitized_arguments),
            }
        return payload


class AgentLoopDriver:
    """Plan and execute exactly one checkpointable ReAct boundary."""

    def __init__(
        self,
        *,
        client: LLMClient,
        tools: list[Tool],
        system: str,
        context_manager: ContextManager,
        policy: RunPolicy,
        tool_executor: ToolExecutor,
        resources: Any | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.system = system
        self.context_manager = context_manager
        self.policy = policy
        self.tool_executor = tool_executor
        self.resources = resources
        self.emit = on_event or (lambda _kind, _payload: None)

    def plan_next(self, state: DriverState) -> DriverAction:
        context = self.context_manager.inspect(
            system=self.system,
            tools=self.tools,
            messages=state.messages,
        )
        if state.final_text is not None:
            return DriverAction(kind="finalize", input_data={})
        if state.pending_tool_calls:
            call = state.pending_tool_calls[0]
            kind: StepKind = "subagent" if call.name == "paper_deep_read" else "tool"
            return DriverAction(
                kind=kind,
                input_data={"tool_call": tool_call_dict(call)},
            )
        if context.needs_compact:
            return DriverAction(
                kind="compact",
                input_data={"estimated_tokens": context.estimated_tokens},
            )
        self._check_llm_budget(state)
        return DriverAction(
            kind="llm",
            input_data={"message_count": len(state.messages)},
        )

    def execute(
        self,
        action: DriverAction,
        state: DriverState,
        *,
        run_id: str,
        step_id: str,
        owner_id: str,
    ) -> DriverStepResult:
        if action.kind == "llm":
            return self._execute_llm(state)
        if action.kind in {"tool", "subagent"}:
            return self._execute_tool(
                state,
                run_id=run_id,
                step_id=step_id,
                owner_id=owner_id,
            )
        if action.kind == "compact":
            return self._execute_compact(state)
        if action.kind == "finalize":
            return DriverStepResult(
                state=state,
                output_data={"final_text": state.final_text},
                final_text=state.final_text,
            )
        raise ValueError(f"unknown driver action: {action.kind}")

    def _execute_llm(self, state: DriverState) -> DriverStepResult:
        self._check_llm_budget(state)
        messages = deepcopy(state.messages)
        response = self.client.call(messages, self.tools, system=self.system)
        self.client.append_assistant_turn(messages, response)
        tokens = _usage_total_tokens(response.usage)
        final_text = None if response.tool_calls else (response.text or "")
        next_state = DriverState(
            messages=messages,
            turn_count=state.turn_count + 1,
            tokens_used=state.tokens_used + tokens,
            downloaded_documents=deepcopy(state.downloaded_documents),
            todo_items=deepcopy(state.todo_items),
            pending_tool_calls=[
                tool_call_from_dict(tool_call_dict(call))
                for call in response.tool_calls
            ],
            pending_tool_results=[],
            final_text=final_text,
        )
        return DriverStepResult(
            state=next_state,
            output_data={
                "text": response.text,
                "tool_calls": [call.name for call in response.tool_calls],
                "usage": deepcopy(response.usage),
            },
            final_text=final_text,
        )

    def _execute_tool(
        self,
        state: DriverState,
        *,
        run_id: str,
        step_id: str,
        owner_id: str,
    ) -> DriverStepResult:
        if not state.pending_tool_calls:
            raise ValueError("tool action requires a pending tool call")
        call = tool_call_from_dict(tool_call_dict(state.pending_tool_calls[0]))
        if should_repair_build_index_args(call, state.downloaded_documents):
            call.arguments = {
                **call.arguments,
                "documents": deepcopy(state.downloaded_documents),
            }
        completion = self.tool_executor.execute(
            run_id=run_id,
            step_id=step_id,
            owner_id=owner_id,
            tool_call=call,
            policy=self.policy,
        )

        messages = deepcopy(state.messages)
        remaining_calls = [
            tool_call_from_dict(tool_call_dict(item))
            for item in state.pending_tool_calls[1:]
        ]
        pending_results = [
            tool_result_from_dict(tool_result_dict(item))
            for item in state.pending_tool_results
        ]
        pending_results.append(
            tool_result_from_dict(tool_result_dict(completion.tool_result))
        )
        if not remaining_calls:
            self.client.append_tool_results(messages, pending_results)
            pending_results = []

        documents = deepcopy(state.downloaded_documents)
        document = extract_downloaded_document(
            call.name,
            completion.tool_result.content,
        )
        if document is not None:
            documents = [document]

        todo_items = deepcopy(state.todo_items)
        if call.name == "research_todo" and self.resources is not None:
            todo_store = getattr(self.resources, "todo_store", None)
            if todo_store is not None:
                todo_items = deepcopy(todo_store.items())

        next_state = DriverState(
            messages=messages,
            turn_count=state.turn_count,
            tokens_used=state.tokens_used,
            downloaded_documents=documents,
            todo_items=todo_items,
            pending_tool_calls=remaining_calls,
            pending_tool_results=pending_results,
            final_text=state.final_text,
        )
        return DriverStepResult(
            state=next_state,
            output_data={
                "tool_name": call.name,
                "tool_use_id": call.id,
                "result_preview": completion.result_preview,
                "duration_ms": completion.duration_ms,
                "classification": completion.classification,
            },
            final_text=state.final_text,
            tool_completion=completion,
        )

    def _execute_compact(self, state: DriverState) -> DriverStepResult:
        messages = deepcopy(state.messages)
        compact_result = compact_messages(
            messages_ref=messages,
            client_factory=lambda: self.client,
            on_event=self.emit,
        )
        context = self.context_manager.inspect(
            system=self.system,
            tools=self.tools,
            messages=messages,
        )
        if context.over_critical_limit:
            raise ContextOverflowError(
                "context remains above the critical limit after compaction: "
                f"{context.estimated_tokens} >= {context.critical_limit}"
            )
        next_state = DriverState(
            messages=messages,
            turn_count=state.turn_count,
            tokens_used=state.tokens_used,
            downloaded_documents=deepcopy(state.downloaded_documents),
            todo_items=deepcopy(state.todo_items),
            pending_tool_calls=[
                tool_call_from_dict(tool_call_dict(call))
                for call in state.pending_tool_calls
            ],
            pending_tool_results=[
                tool_result_from_dict(tool_result_dict(result))
                for result in state.pending_tool_results
            ],
            final_text=state.final_text,
        )
        return DriverStepResult(
            state=next_state,
            output_data={
                "result": compact_result,
                "estimated_tokens": context.estimated_tokens,
            },
            final_text=state.final_text,
        )

    def _check_llm_budget(self, state: DriverState) -> None:
        if state.turn_count >= self.policy.max_iterations:
            raise BudgetExceededError(
                f"iteration budget exhausted: {state.turn_count} >= "
                f"{self.policy.max_iterations}"
            )
        if state.tokens_used >= self.policy.token_budget:
            raise BudgetExceededError(
                f"token budget exhausted: {state.tokens_used} >= "
                f"{self.policy.token_budget}"
            )


def tool_call_dict(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments": deepcopy(call.arguments),
    }


def tool_call_from_dict(payload: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=str(payload["id"]),
        name=str(payload["name"]),
        arguments=deepcopy(payload.get("arguments", {})),
    )


def tool_result_dict(result: ToolResult) -> dict[str, Any]:
    return {
        "id": result.id,
        "content": result.content,
        "is_error": result.is_error,
    }


def tool_result_from_dict(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        id=str(payload["id"]),
        content=str(payload.get("content", "")),
        is_error=bool(payload.get("is_error", False)),
    )


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if total is not None:
        return int(total)
    return int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
