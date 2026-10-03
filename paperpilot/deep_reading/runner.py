"""Durable Task-to-LangGraph execution and recovery boundary."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, cast

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from paperpilot.papers import PaperCandidate, search_arxiv_candidates
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.task_store import (
    ConversationRecord,
    MessageRecord,
    ResearchTask,
    TaskStore,
)
from paperpilot.web.store.records import TurnArchiveSeedRecord
from paperpilot.web.config import ContextManagementConfig, WebRuntimeConfig

from .graph import build_deep_reading_graph
from .model_usage import DeepSeekUsageCallback
from .context_management.archives import (
    ResearchExecutionOutcome,
    ResearchTrace,
    TurnArchiveNarrative,
    TurnArchiveSeedBuilder,
    ValidatedContextDelta,
)
from .context_management.compaction import ContextCapacityExhaustedError
from .context_management.models import ArchiveSupersession, ArtifactRef, ProtectedText
from .context_management.models import TurnArchiveSeed
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
from .state import (
    GRAPH_VERSION,
    LEGACY_GRAPH_VERSION,
    LEGACY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    DeepReadingState,
)

if TYPE_CHECKING:
    from langchain_deepseek import ChatDeepSeek


ModelFactory = Callable[..., Any]
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
    model_name: str = "deepseek-v4-flash",
    research_max_output_tokens: int = 4096,
) -> ChatDeepSeek:
    """Build the production model lazily, only when a new task executes."""
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=model_name,
        temperature=0,
        max_tokens=research_max_output_tokens,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
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
        research_model_name: str = "deepseek-v4-flash",
        paper_search: PaperSearch = search_arxiv_candidates,
        summary_token_threshold: int = 32_000,
        summary_recent_turns: int = 6,
        research_recursion_limit: int = 33,
        research_model_call_limit: int = 12,
        research_tool_call_limit: int = 12,
        research_max_output_tokens: int = 4096,
        research_model_retries: int = 1,
        context_management: ContextManagementConfig | None = None,
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
        self._research_model_name = research_model_name
        self._paper_search = paper_search
        self._summary_token_threshold = summary_token_threshold
        self._summary_recent_turns = summary_recent_turns
        self._research_recursion_limit = research_recursion_limit
        self._research_model_call_limit = research_model_call_limit
        self._research_tool_call_limit = research_tool_call_limit
        self._research_max_output_tokens = research_max_output_tokens
        self._research_model_retries = research_model_retries
        self._context_management = context_management or ContextManagementConfig()

    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        """Claim and execute one Conversation-bound Task."""
        claimed = self._task_store.claim_task(
            task_id,
            allow_running=allow_running,
        )
        if claimed is None:
            existing = self._task_store.get_task(task_id)
            if existing is None:
                raise ValueError(f"task not found: {task_id}")
            if existing.status in {"completed", "failed"}:
                self._archive_task_terminal_best_effort(
                    task_id,
                    terminal_status=(
                        "success" if existing.status == "completed" else "failed"
                    ),
                )
            return False
        if claimed.conversation_id is None:
            raise TaskBindingError("task is not bound to a Conversation")
        try:
            self._run(task_id)
        except DeepReadingTaskError as exc:
            self._fail_terminal(task_id, exc)
        except ContextCapacityExhaustedError as exc:
            self._fail_terminal(task_id, exc)  # type: ignore[arg-type]
        return True

    def fail_retry_exhausted(
        self,
        task_id: str,
        *,
        backend: str,
        attempts: int,
        max_retries: int,
        exc: Exception,
    ) -> None:
        """Idempotently fail active Conversation work after retry exhaustion."""
        _LOGGER.error(
            "Conversation execution failed after retry limit",
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={
                "event": "task.execution_retry_exhausted",
                "executor": backend,
                "reason": "retry_limit",
                "exception_type": type(exc).__name__,
            },
        )
        try:
            self._task_store.fail_conversation_task(
                task_id=task_id,
                message="Conversation execution failed after retry limit.",
                stage="execution_retry_exhausted",
                payload={
                    "backend": backend,
                    "attempts": attempts,
                    "max_retries": max_retries,
                    "error_type": type(exc).__name__,
                },
            )
        except ValueError:
            task = self._task_store.get_task(task_id)
            if (
                task is not None
                and task.conversation_id is not None
                and task.status == "completed"
            ):
                return
            raise
        self._archive_task_terminal_best_effort(task_id, terminal_status="failed")

    def fail_non_retryable(
        self,
        task_id: str,
        *,
        backend: str,
        attempts: int,
        exc: Exception,
    ) -> None:
        """Idempotently fail active work after a deterministic provider error."""
        status_code = _exception_status_code(exc)
        _LOGGER.error(
            "Conversation execution failed with a non-retryable error",
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={
                "event": "task.execution_non_retryable",
                "executor": backend,
                "reason": "non_retryable_exception",
                "exception_type": type(exc).__name__,
                "status_code": status_code,
            },
        )
        payload: dict[str, object] = {
            "backend": backend,
            "attempts": attempts,
            "error_type": type(exc).__name__,
        }
        if status_code is not None:
            payload["status_code"] = status_code
        try:
            self._task_store.fail_conversation_task(
                task_id=task_id,
                message="Conversation execution failed with a non-retryable error.",
                stage="execution_non_retryable",
                payload=payload,
            )
        except ValueError:
            task = self._task_store.get_task(task_id)
            if (
                task is not None
                and task.conversation_id is not None
                and task.status == "completed"
            ):
                return
            raise
        self._archive_task_terminal_best_effort(task_id, terminal_status="failed")

    def _fail_terminal(self, task_id: str, exc: DeepReadingTaskError) -> None:
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
        self._archive_task_terminal_best_effort(task_id, terminal_status="failed")

    def _archive_task_terminal_best_effort(
        self,
        task_id: str,
        *,
        terminal_status: str,
    ) -> None:
        try:
            task, user_message, _conversation, _primary_paper_id = (
                self._load_business_binding(task_id)
            )
        except Exception as exc:
            _LOGGER.warning(
                "Failed to load terminal Task for TurnArchive",
                extra={
                    "event": "task.turn_archive_binding_failed",
                    "reason": "binding_unavailable",
                    "exception_type": type(exc).__name__,
                },
            )
            return
        snapshot = None
        if terminal_status == "success":
            if task.final_checkpoint_id is None:
                _LOGGER.warning(
                    "Cannot repair successful TurnArchive without final checkpoint",
                    extra={
                        "event": "task.turn_archive_checkpoint_failed",
                        "reason": "final_checkpoint_unavailable",
                    },
                )
                return
            snapshot = self.read_checkpoint(
                task.conversation_id or "",
                task.final_checkpoint_id,
            )
            if snapshot is None or not snapshot.is_complete:
                _LOGGER.warning(
                    "Cannot repair successful TurnArchive from incomplete checkpoint",
                    extra={
                        "event": "task.turn_archive_checkpoint_failed",
                        "reason": "final_checkpoint_unavailable",
                    },
                )
                return
        self._archive_terminal_best_effort(
            task=task,
            user_message=user_message,
            terminal_status=terminal_status,
            snapshot=snapshot,
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
                self._archive_terminal_best_effort(
                    task=task,
                    user_message=user_message,
                    terminal_status="success",
                    snapshot=trusted,
                )
                return

        model = self._build_model()
        usage_callback = DeepSeekUsageCallback()
        from paperpilot.web.context_runtime import build_context_management_runtime

        context_runtime = build_context_management_runtime(
            self._task_store,
            WebRuntimeConfig(
                research_model_name=self._research_model_name,
                summary_token_threshold=self._summary_token_threshold,
                summary_recent_turns=self._summary_recent_turns,
                research_recursion_limit=self._research_recursion_limit,
                research_model_call_limit=self._research_model_call_limit,
                research_tool_call_limit=self._research_tool_call_limit,
                research_max_output_tokens=self._research_max_output_tokens,
                research_model_retries=self._research_model_retries,
                context_management=self._context_management,
            ),
            model,
            conversation_id=conversation.id,
            task_id=task.id,
            event_sink=lambda event_type, payload: self._record_event(
                task.id,
                event_type,
                payload,
            ),
        )
        try:
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
                    context_management=self._context_management,
                    context_management_runtime=context_runtime,
                    summary_token_threshold=self._summary_token_threshold,
                    summary_recent_turns=self._summary_recent_turns,
                    research_recursion_limit=self._research_recursion_limit,
                    research_model_call_limit=self._research_model_call_limit,
                    research_tool_call_limit=self._research_tool_call_limit,
                    research_max_output_tokens=self._research_max_output_tokens,
                    research_model_retries=self._research_model_retries,
                )
                configurable = {"thread_id": conversation.id}
                if task.base_checkpoint_id is not None:
                    configurable["checkpoint_id"] = task.base_checkpoint_id
                config: RunnableConfig = {
                    "configurable": configurable,
                    "callbacks": [usage_callback],
                }
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
        finally:
            self._record_model_usage_events(task.id, usage_callback)

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
        self._archive_terminal_best_effort(
            task=task,
            user_message=user_message,
            terminal_status="success",
            snapshot=trusted,
            model=model,
        )

    def _build_model(self) -> Any:
        if self._model_factory is None:
            return build_deep_reading_model(
                model_name=self._research_model_name,
                research_max_output_tokens=self._research_max_output_tokens,
            )
        factory = self._model_factory
        try:
            parameters = inspect.signature(factory).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        accepts_argument = any(
            parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            for parameter in parameters
        )
        if accepts_argument:
            return factory(self._research_model_name)
        return factory()

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

    def _record_model_usage_events(
        self,
        task_id: str,
        callback: DeepSeekUsageCallback,
    ) -> None:
        try:
            summaries = callback.stage_summaries()
        except Exception:
            _LOGGER.warning(
                "Failed to aggregate model usage",
                exc_info=True,
                extra={"event": "task.model_usage_persist_failed"},
            )
            return

        for summary in summaries:
            try:
                self._record_event(
                    task_id,
                    "model_usage",
                    summary.to_event_payload(),
                )
            except Exception:
                _LOGGER.warning(
                    "Failed to persist model usage",
                    exc_info=True,
                    extra={
                        "event": "task.model_usage_persist_failed",
                        "stage": summary.stage,
                        "prompt_version": summary.prompt_version,
                        "model": summary.model,
                    },
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

    def _archive_terminal_best_effort(
        self,
        *,
        task: ResearchTask,
        user_message: MessageRecord,
        terminal_status: str,
        snapshot: object | None = None,
        model: Any | None = None,
    ) -> None:
        """Persist an immutable Seed after the business terminal transaction."""
        if not self._context_management.enabled:
            return
        try:
            values = _archive_values(snapshot)
            raw_result = values.get("research_result") if isinstance(values, dict) else None
            research_result = None
            if raw_result is not None:
                from .schemas import ResearchResult

                research_result = ResearchResult.model_validate(raw_result)
            raw_draft = values.get("answer_draft") if isinstance(values, dict) else None
            answer_draft = None
            if raw_draft is not None:
                answer_draft = AnswerDraft.model_validate(raw_draft)
            trace = _trace_from_state(values if isinstance(values, dict) else {})
            artifact_refs = self._artifact_refs_for_trace(task, trace)
            context_delta = _context_delta_from_state(
                values if isinstance(values, dict) else {}
            )
            archive_id = _archive_id(
                task.conversation_id or "",
                user_message.id,
                "turn-archive-v1",
            )
            seed = TurnArchiveSeedBuilder().build(
                archive_id=archive_id,
                conversation_id=task.conversation_id or user_message.conversation_id,
                task_id=task.id,
                user_message=user_message,
                terminal_status=terminal_status,  # type: ignore[arg-type]
                research_result=research_result,
                answer_draft=answer_draft,
                context_delta=context_delta,
                trace=trace,
                artifact_refs=artifact_refs,
                created_at=user_message.created_at,
            )
            record = self._task_store.seed_turn_archive(
                seed=_seed_record(seed),
            )
            self._record_event(
                task.id,
                "turn_archive_seeded",
                {
                    "stage": "archive_seed",
                    "archive_id": record.archive_id,
                    "terminal_status": terminal_status,
                },
            )
        except Exception as exc:
            _LOGGER.warning(
                "Failed to persist TurnArchive seed",
                extra={
                    "event": "task.turn_archive_seed_failed",
                    "reason": "seed_persistence_failed",
                    "exception_type": type(exc).__name__,
                },
            )
            try:
                self._record_event(
                    task.id,
                    "turn_archive_failed",
                    {
                        "stage": "archive_seed",
                        "reason": "seed_persistence_failed",
                        "validation_failure_type": type(exc).__name__,
                    },
                )
            except Exception:
                _LOGGER.warning(
                    "Failed to persist TurnArchive failure event",
                    extra={"event": "task.turn_archive_event_failed"},
                )
            return

        self._enrich_archive_narrative(
            task=task,
            archive_id=record.archive_id,
            seed=seed,
            model=model,
        )

    def _artifact_refs_for_trace(
        self,
        task: ResearchTask,
        trace: ResearchTrace,
    ) -> list[ArtifactRef]:
        conversation_id = task.conversation_id or ""
        result: list[ArtifactRef] = []
        for artifact_id in trace.artifact_ids:
            record = self._task_store.get_context_artifact(
                artifact_id,
                conversation_id=conversation_id,
            )
            if record is None or record.task_id != task.id:
                raise ValueError(
                    f"archive artifact is not authoritative: {artifact_id}"
                )
            result.append(
                ArtifactRef(
                    artifact_id=record.artifact_id,
                    sha256=record.sha256,
                    token_estimate=record.token_estimate,
                    preview=record.preview,
                )
            )
        return result

    def _enrich_archive_narrative(
        self,
        *,
        task: ResearchTask,
        archive_id: str,
        seed: object,
        model: Any | None,
    ) -> None:
        claimed = None
        try:
            claimed = self._task_store.claim_turn_archive_narrative(archive_id)
            if claimed is None:
                return
            if model is None:
                raise RuntimeError("archive narrative model is unavailable")
            callback = DeepSeekUsageCallback()
            narrative_model = model.with_structured_output(TurnArchiveNarrative)
            response = narrative_model.invoke(
                [
                    SystemMessage(
                        content=(
                            "Summarize only the supplied TurnArchive Seed. "
                            "Do not add or modify IDs, decisions, constraints, "
                            "verification, or supersession relations."
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            seed.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                ],
                config={
                    "callbacks": [callback],
                    "metadata": {
                        "paperpilot_stage": "archive_narrative",
                        "prompt_version": "turn-archive-narrative-v1",
                    },
                },
            )
            narrative = TurnArchiveNarrative.model_validate(response)
            self._task_store.complete_turn_archive_narrative(
                archive_id,
                narrative_summary=narrative.summary,
            )
            self._record_model_usage_events(task.id, callback)
            self._record_event(
                task.id,
                "turn_archive_enriched",
                {
                    "stage": "archive_narrative",
                    "archive_id": archive_id,
                },
            )
        except Exception as exc:
            if claimed is not None:
                try:
                    self._task_store.fail_turn_archive_narrative(archive_id)
                except Exception:
                    _LOGGER.warning(
                        "Failed to mark TurnArchive narrative failed",
                        extra={"event": "task.turn_archive_narrative_state_failed"},
                    )
            _LOGGER.warning(
                "TurnArchive narrative enrichment failed",
                extra={
                    "event": "task.turn_archive_narrative_failed",
                    "reason": "narrative_failed",
                    "exception_type": type(exc).__name__,
                },
            )
            try:
                self._record_event(
                    task.id,
                    "turn_archive_failed",
                    {
                        "stage": "archive_narrative",
                        "archive_id": archive_id,
                        "reason": "narrative_failed",
                        "validation_failure_type": type(exc).__name__,
                    },
                )
            except Exception:
                _LOGGER.warning(
                    "Failed to persist TurnArchive narrative failure event",
                    extra={"event": "task.turn_archive_event_failed"},
                )


def _seed_record(seed: TurnArchiveSeed) -> TurnArchiveSeedRecord:
    payload = seed.model_dump(mode="json")
    return TurnArchiveSeedRecord(
        archive_id=seed.archive_id,
        conversation_id=seed.conversation_id,
        task_id=seed.task_id,
        user_message_id=seed.user_message_id,
        terminal_status=seed.terminal_status,
        archive_version=seed.archive_version,
        seed_json=payload,
        supersedes_json=[item.model_dump(mode="json") for item in seed.supersedes],
        created_at=seed.created_at,
    )


def _archive_values(snapshot: object | None) -> dict[str, object]:
    if snapshot is None:
        return {}
    values = getattr(snapshot, "values", None)
    if isinstance(values, Mapping):
        return dict(values)
    state = getattr(snapshot, "state", None)
    if isinstance(state, Mapping):
        return dict(state)
    if isinstance(snapshot, Mapping):
        return dict(snapshot)
    return {}


def _exception_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _trace_from_state(values: Mapping[str, object]) -> ResearchTrace:
    raw = values.get("research_trace")
    if not isinstance(raw, Mapping):
        return ResearchTrace()

    def read_list(name: str) -> tuple[str, ...]:
        value = raw.get(name, [])
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, str) and item.strip())

    return ResearchTrace(
        todos=read_list("todos"),
        tool_outcomes=read_list("tool_outcomes"),
        artifact_ids=read_list("artifact_ids"),
        verification=read_list("verification"),
    )


def _context_delta_from_state(
    values: Mapping[str, object],
) -> ValidatedContextDelta | None:
    raw = values.get("research_context_delta")
    if not isinstance(raw, Mapping):
        return None
    try:
        return ValidatedContextDelta(
            constraints=tuple(
                ProtectedText.model_validate(item)
                for item in raw.get("constraints", [])
                if isinstance(item, Mapping)
            ),
            decisions=tuple(
                ProtectedText.model_validate(item)
                for item in raw.get("decisions", [])
                if isinstance(item, Mapping)
            ),
            supersedes=tuple(
                ArchiveSupersession.model_validate(item)
                for item in raw.get("supersedes", [])
                if isinstance(item, Mapping)
            ),
        )
    except ValidationError:
        return None


def _archive_id(conversation_id: str, user_message_id: str, version: str) -> str:
    identity = f"{conversation_id}\x00{user_message_id}\x00{version}"
    return "archive_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


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


def _supported_checkpoint_version(values: Mapping[str, object]) -> bool:
    graph_version = values.get("graph_version")
    schema_version = values.get("schema_version")
    if graph_version not in {GRAPH_VERSION, LEGACY_GRAPH_VERSION}:
        raise GraphVersionUnsupportedError(
            "checkpoint graph version is unsupported"
        )
    if schema_version not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise SchemaVersionUnsupportedError(
            "checkpoint schema version is unsupported"
        )
    return True


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
        and _supported_checkpoint_version(values)
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
    if not _supported_checkpoint_version(values):
        raise GraphVersionUnsupportedError(
            "complete recovery checkpoint graph version is unsupported"
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
    if not _supported_checkpoint_version(values):
        raise GraphVersionUnsupportedError(
            "conversation base checkpoint graph version is unsupported"
        )
    if values.get("published_message_id") != head_message_id:
        raise CheckpointBindingError(
            "conversation base checkpoint does not match message head"
        )
