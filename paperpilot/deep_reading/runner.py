"""Durable Task-to-LangGraph execution and recovery boundary."""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain.messages import HumanMessage
from pydantic import ValidationError

from paperpilot.papers import PaperCandidate, search_arxiv_candidates
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.task_store import (
    ConversationRecord,
    MessageRecord,
    ResearchTask,
    TaskStore,
)

from .graph import build_deep_reading_graph
from .nodes import DeepReadingContext
from .research_agent import (
    CheckpointBindingError,
    CheckpointIncompleteError,
    CheckpointMissingError,
    DeepReadingTaskError,
    FinalCheckpointError,
    GraphVersionUnsupportedError,
    ResearchContractError,
    SchemaVersionUnsupportedError,
    TaskBindingError,
    TaskStatusError,
)
from .schemas import AnswerDraft
from .state import GRAPH_VERSION, SCHEMA_VERSION, DeepReadingState

if TYPE_CHECKING:
    from langchain_deepseek import ChatDeepSeek


ModelFactory = Callable[[], Any]
PaperSearch = Callable[[str, int], list[PaperCandidate]]
_STRUCTURED_RESPONSE_ATTEMPTS = 2
_LOGGER = logging.getLogger("paperpilot.web.runtime")


@dataclass(frozen=True)
class DeepReadingCheckpoint:
    """Read-only application view of one LangGraph checkpoint."""

    checkpoint_id: str
    state: DeepReadingState
    is_complete: bool


def build_deep_reading_model(
    *,
    research_max_output_tokens: int = 4096,
) -> ChatDeepSeek:
    """Build the production model lazily, only when a new task executes."""
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=research_max_output_tokens,
        max_retries=0,
    )


class DeepReadingRunner:
    """Run one conversation Task and reconcile its durable business head."""

    def __init__(
        self,
        *,
        task_store: TaskStore,
        checkpointer: Any,
        mcp_runtime: MCPRuntime,
        model_factory: ModelFactory | None = None,
        paper_search: PaperSearch = search_arxiv_candidates,
        summary_token_threshold: int = 32_000,
        summary_recent_turns: int = 6,
        research_recursion_limit: int = 24,
        research_model_call_limit: int = 8,
        research_tool_call_limit: int = 12,
        research_max_output_tokens: int = 4096,
        research_model_retries: int = 1,
    ) -> None:
        runtime_bounds = {
            "summary_token_threshold": summary_token_threshold,
            "summary_recent_turns": summary_recent_turns,
            "research_recursion_limit": research_recursion_limit,
            "research_max_output_tokens": research_max_output_tokens,
        }
        for name, value in runtime_bounds.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "research_model_call_limit": research_model_call_limit,
            "research_tool_call_limit": research_tool_call_limit,
        }.items():
            if value < _STRUCTURED_RESPONSE_ATTEMPTS:
                raise ValueError(
                    f"{name} must cover both structured-response attempts"
                )
        if research_model_retries < 0:
            raise ValueError("research_model_retries must be nonnegative")
        self._task_store = task_store
        self._checkpointer = checkpointer
        self._mcp_runtime = mcp_runtime
        self._model_factory = model_factory
        self._paper_search = paper_search
        self._summary_token_threshold = summary_token_threshold
        self._summary_recent_turns = summary_recent_turns
        self._research_recursion_limit = research_recursion_limit
        self._research_model_call_limit = research_model_call_limit
        self._research_tool_call_limit = research_tool_call_limit
        self._research_max_output_tokens = research_max_output_tokens
        self._research_model_retries = research_model_retries

    def run(self, task_id: str) -> None:
        """Execute or recover one Task; only expected task errors become failed."""
        try:
            self._run(task_id)
        except DeepReadingTaskError as exc:
            _LOGGER.error(
                "Deep-reading terminal contract failure",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={
                    "event": "task.deep_reading_terminal",
                    "reason": exc.error_code,
                    "exception_type": type(exc).__name__,
                },
            )
            self._task_store.fail_conversation_task(
                task_id=task_id,
                message=exc.public_message,
                stage="deep_reading_terminal",
                payload={
                    "error_code": exc.error_code,
                    "error_type": type(exc).__name__,
                },
            )

    def read_checkpoint(
        self,
        conversation_id: str,
        checkpoint_id: str,
    ) -> DeepReadingCheckpoint | None:
        """Return a frozen checkpoint view without exposing the raw saver."""
        cleaned_conversation_id = conversation_id.strip()
        cleaned_checkpoint_id = checkpoint_id.strip()
        if not cleaned_conversation_id or not cleaned_checkpoint_id:
            return None
        graph = build_deep_reading_graph(self._checkpointer)
        snapshot = graph.get_state(
            {
                "configurable": {
                    "thread_id": cleaned_conversation_id,
                    "checkpoint_id": cleaned_checkpoint_id,
                }
            }
        )
        actual_id = _snapshot_checkpoint_id(snapshot)
        if snapshot.created_at is None or actual_id != cleaned_checkpoint_id:
            return None
        return DeepReadingCheckpoint(
            checkpoint_id=actual_id,
            state=cast(DeepReadingState, deepcopy(dict(snapshot.values))),
            is_complete=snapshot.next == (),
        )

    def _run(self, task_id: str) -> None:
        task, user_message, conversation, primary_paper_id = (
            self._load_business_binding(task_id)
        )
        if task.status in {"completed", "failed"}:
            return
        if task.status not in {"pending", "running"}:
            raise TaskStatusError(f"cannot run task in status: {task.status!r}")

        graph = build_deep_reading_graph(self._checkpointer)
        _validate_active_task_base(
            graph,
            task_store=self._task_store,
            task=task,
            user_message=user_message,
            conversation=conversation,
        )

        assistant = self._task_store.get_task_message(task.id, "assistant")
        if assistant is not None:
            latest = graph.get_state(
                {"configurable": {"thread_id": conversation.id}}
            )
            trusted = _trusted_durable_snapshot(
                graph,
                latest,
                task=task,
                user_message=user_message,
                assistant=assistant,
            )
            if trusted is not None:
                self._finalize(task, assistant, trusted)
                return

        model = (
            self._model_factory()
            if self._model_factory is not None
            else build_deep_reading_model(
                research_max_output_tokens=self._research_max_output_tokens
            )
        )
        with self._mcp_runtime.lease_tools() as tools:
            context = DeepReadingContext(
                user_id=conversation.user_id,
                conversation_id=conversation.id,
                task_id=task.id,
                current_user_message_id=user_message.id,
                base_checkpoint_id=task.base_checkpoint_id,
                task_store=self._task_store,
                model=model,
                mcp_tools=_tool_map(tools),
                paper_search=self._paper_search,
                event_sink=lambda event_type, payload: self._record_event(
                    task.id, event_type, payload
                ),
                summary_token_threshold=self._summary_token_threshold,
                summary_recent_turns=self._summary_recent_turns,
                research_recursion_limit=self._research_recursion_limit,
                research_model_call_limit=self._research_model_call_limit,
                research_tool_call_limit=self._research_tool_call_limit,
                research_max_output_tokens=self._research_max_output_tokens,
                research_model_retries=self._research_model_retries,
            )
            config: dict[str, dict[str, str]] = {
                "configurable": {"thread_id": conversation.id}
            }
            if task.base_checkpoint_id is not None:
                config["configurable"]["checkpoint_id"] = task.base_checkpoint_id
            graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=user_message.content,
                            id=user_message.id,
                        )
                    ],
                    "current_task_id": task.id,
                    "current_user_message_id": user_message.id,
                    "primary_paper_id": primary_paper_id,
                },
                config=config,
                context=context,
            )
            final_snapshot = graph.get_state(
                {"configurable": {"thread_id": conversation.id}}
            )

        assistant = self._task_store.get_task_message(task.id, "assistant")
        if assistant is None:
            raise FinalCheckpointError(
                "graph completed without a published assistant"
            )
        trusted = _trusted_durable_snapshot(
            graph,
            final_snapshot,
            task=task,
            user_message=user_message,
            assistant=assistant,
        )
        if trusted is None:
            raise FinalCheckpointError(
                "graph completed without a trusted final checkpoint"
            )
        self._finalize(task, assistant, trusted)

    def _load_business_binding(
        self,
        task_id: str,
    ) -> tuple[ResearchTask, MessageRecord, ConversationRecord, str]:
        task = self._task_store.get_task(task_id)
        if task is None:
            raise TaskBindingError(f"task not found: {task_id}")
        if task.conversation_id is None or task.user_id is None:
            raise TaskBindingError("task is not attached to an owned conversation")
        detail = self._task_store.get_conversation_detail(
            task.conversation_id,
            user_id=task.user_id,
        )
        if detail is None:
            raise TaskBindingError("task conversation is unavailable for its owner")
        if detail.conversation.id != task.conversation_id:
            raise TaskBindingError("task conversation binding is inconsistent")
        user_message = self._task_store.get_task_message(task.id, "user")
        if user_message is None:
            raise TaskBindingError("conversation task has no user message")
        if (
            user_message.conversation_id != task.conversation_id
            or user_message.task_id != task.id
            or user_message.role != "user"
            or user_message.status != "complete"
            or user_message.content != task.question
        ):
            raise TaskBindingError(
                "conversation task user-message binding is inconsistent"
            )
        return (
            task,
            user_message,
            detail.conversation,
            detail.primary_paper.id,
        )

    def _record_event(
        self,
        task_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        raw_stage = payload.get("stage")
        stage = raw_stage if isinstance(raw_stage, str) else None
        raw_name = payload.get("name")
        name = raw_name if isinstance(raw_name, str) else event_type
        self._task_store.add_event(
            task_id=task_id,
            type=event_type,
            stage=stage,
            message=name,
            payload=payload,
        )

    def _finalize(self, task: ResearchTask, assistant: MessageRecord, snapshot) -> None:
        values = snapshot.values
        try:
            draft = AnswerDraft.model_validate(values.get("answer_draft"))
        except ValidationError as exc:
            raise FinalCheckpointError(
                "final checkpoint has an invalid answer draft"
            ) from exc
        active_paper_ids = values.get("active_paper_ids")
        if not isinstance(active_paper_ids, list) or not all(
            isinstance(paper_id, str) and paper_id.strip()
            for paper_id in active_paper_ids
        ):
            raise FinalCheckpointError(
                "final checkpoint has invalid active paper IDs"
            )
        checkpoint_id = _snapshot_checkpoint_id(snapshot)
        if checkpoint_id is None:
            raise FinalCheckpointError("final checkpoint has no checkpoint ID")
        self._task_store.finalize_conversation_task(
            task_id=task.id,
            assistant_message_id=assistant.id,
            final_checkpoint_id=checkpoint_id,
            result_quality=draft.result_quality,
            active_paper_ids=list(active_paper_ids),
        )


def _tool_map(tools: list[Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for tool in tools:
        if tool.name in mapped:
            raise ResearchContractError(f"duplicate MCP tool name: {tool.name}")
        mapped[tool.name] = tool
    return mapped


def _snapshot_checkpoint_id(snapshot) -> str | None:
    configurable = snapshot.config.get("configurable", {})
    checkpoint_id = configurable.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        return None
    return checkpoint_id


def _is_trusted_recovery_snapshot(
    snapshot,
    *,
    task: ResearchTask,
    user_message: MessageRecord,
    assistant: MessageRecord,
) -> bool:
    if snapshot.created_at is None or snapshot.next != ():
        return False
    values = snapshot.values
    return (
        values.get("current_task_id") == task.id
        and values.get("current_user_message_id") == user_message.id
        and values.get("graph_version") == GRAPH_VERSION
        and values.get("schema_version") == SCHEMA_VERSION
        and values.get("published_message_id") == assistant.id
        and _snapshot_checkpoint_id(snapshot) is not None
    )


def _trusted_durable_snapshot(
    graph,
    snapshot,
    *,
    task: ResearchTask,
    user_message: MessageRecord,
    assistant: MessageRecord,
):
    """Reject pending-write projections whose checkpoint ID is still incomplete."""
    _validate_complete_recovery_snapshot(
        snapshot,
        task=task,
        user_message=user_message,
        assistant=assistant,
    )
    if not _is_trusted_recovery_snapshot(
        snapshot,
        task=task,
        user_message=user_message,
        assistant=assistant,
    ):
        return None
    durable = graph.get_state(snapshot.config)
    _validate_complete_recovery_snapshot(
        durable,
        task=task,
        user_message=user_message,
        assistant=assistant,
    )
    if not _is_trusted_recovery_snapshot(
        durable,
        task=task,
        user_message=user_message,
        assistant=assistant,
    ):
        return None
    if _snapshot_checkpoint_id(durable) != _snapshot_checkpoint_id(snapshot):
        return None
    return durable


def _validate_complete_recovery_snapshot(
    snapshot,
    *,
    task: ResearchTask,
    user_message: MessageRecord,
    assistant: MessageRecord,
) -> None:
    """Reject deterministic corruption while allowing incomplete crash recovery."""
    if snapshot.created_at is None or snapshot.next != ():
        return
    values = snapshot.values
    if values.get("graph_version") != GRAPH_VERSION:
        raise GraphVersionUnsupportedError(
            "complete recovery checkpoint graph version is unsupported"
        )
    if values.get("schema_version") != SCHEMA_VERSION:
        raise SchemaVersionUnsupportedError(
            "complete recovery checkpoint schema version is unsupported"
        )
    if (
        values.get("current_task_id") != task.id
        or values.get("current_user_message_id") != user_message.id
        or values.get("published_message_id") != assistant.id
        or _snapshot_checkpoint_id(snapshot) is None
    ):
        raise FinalCheckpointError(
            "complete recovery checkpoint does not match the published task"
        )


def _validate_active_task_base(
    graph,
    *,
    task_store: TaskStore,
    task: ResearchTask,
    user_message: MessageRecord,
    conversation: ConversationRecord,
) -> None:
    """Validate the business head and exact historical checkpoint before I/O."""
    head_message_id = conversation.head_message_id
    head_checkpoint_id = conversation.head_checkpoint_id
    if (head_message_id is None) != (head_checkpoint_id is None):
        raise CheckpointBindingError(
            "conversation message/checkpoint heads must be paired"
        )
    if task.base_checkpoint_id != head_checkpoint_id:
        raise CheckpointBindingError(
            "task base checkpoint does not match conversation head"
        )
    if user_message.parent_message_id != head_message_id:
        raise CheckpointBindingError(
            "task user-message parent does not match conversation head"
        )
    if head_checkpoint_id is None:
        return
    if not head_checkpoint_id.strip() or head_message_id is None:
        raise CheckpointBindingError(
            "conversation head identifiers must be non-blank"
        )

    head_message = task_store.get_message(
        conversation.id,
        head_message_id,
        user_id=conversation.user_id,
    )
    if head_message is None:
        raise CheckpointBindingError(
            "conversation head message is unavailable for its owner"
        )
    if head_message.role != "assistant" or head_message.status != "complete":
        raise CheckpointBindingError(
            "conversation head must be a complete assistant message"
        )

    base_snapshot = graph.get_state(
        {
            "configurable": {
                "thread_id": conversation.id,
                "checkpoint_id": head_checkpoint_id,
            }
        }
    )
    if base_snapshot.created_at is None:
        raise CheckpointMissingError("conversation base checkpoint does not exist")
    if _snapshot_checkpoint_id(base_snapshot) != head_checkpoint_id:
        raise CheckpointBindingError(
            "conversation base checkpoint ID does not match"
        )
    if base_snapshot.next != ():
        raise CheckpointIncompleteError(
            "conversation base checkpoint is incomplete"
        )
    values = base_snapshot.values
    if values.get("graph_version") != GRAPH_VERSION:
        raise GraphVersionUnsupportedError(
            "conversation base checkpoint graph version is unsupported"
        )
    if values.get("schema_version") != SCHEMA_VERSION:
        raise SchemaVersionUnsupportedError(
            "conversation base checkpoint schema version is unsupported"
        )
    if values.get("published_message_id") != head_message_id:
        raise CheckpointBindingError(
            "conversation base checkpoint does not match message head"
        )
