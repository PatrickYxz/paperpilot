"""Authenticated Conversation HTTP routes."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from paperpilot.deep_reading.runner import DeepReadingRunner
from paperpilot.deep_reading.state import GRAPH_VERSION, SCHEMA_VERSION
from paperpilot.papers import PaperCandidate, normalize_arxiv_id
from paperpilot.web.observability import RUNTIME_LOGGER_NAME
from paperpilot.web.routes.auth import RequireUser
from paperpilot.web.routes.papers import PaperSearch
from paperpilot.web.schemas import (
    ConversationAlternativesResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationMessagesResponse,
    ConversationResponse,
    CreateConversationRequest,
    CreateMessageRequest,
    CreateMessageResponse,
    PaperReference,
    RollbackRequest,
    UpdateConversationRequest,
)
from paperpilot.web.task_executor import (
    TaskExecutorAtCapacityError,
    TaskExecutorLike,
    TaskExecutorShuttingDownError,
)
from paperpilot.web.task_store import (
    ConversationBusyError,
    ConversationDetail,
    ConversationRecord,
    MessageRecord,
    PaperRecord,
    ResearchTask,
    StaleConversationHeadError,
    TaskStore,
    WebUser,
)


_LOGGER = logging.getLogger(RUNTIME_LOGGER_NAME)


def build_conversation_router(
    *,
    store: TaskStore,
    executor: TaskExecutorLike,
    require_user: RequireUser,
    deep_reading_runner: DeepReadingRunner,
    paper_search: PaperSearch,
    overload_retry_after_seconds: int,
) -> APIRouter:
    router = APIRouter(prefix="/api/conversations")

    @router.post("", response_model=ConversationResponse, status_code=201)
    def create_conversation(
        payload: CreateConversationRequest,
        user: WebUser = Depends(require_user),
    ) -> dict:
        paper = _resolve_paper_reference(payload.paper, paper_search)
        try:
            conversation = store.create_conversation(
                user_id=user.id,
                paper=paper,
                title=payload.title,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _conversation_dict(conversation)

    @router.get("", response_model=ConversationListResponse)
    def list_conversations(
        include_archived: bool = Query(default=False),
        limit: int = Query(default=100, ge=1, le=100),
        user: WebUser = Depends(require_user),
    ) -> dict:
        records = store.list_conversations(
            user_id=user.id,
            include_archived=include_archived,
            limit=limit,
        )
        return {"items": [_conversation_dict(record) for record in records]}

    @router.get("/{conversation_id}", response_model=ConversationDetailResponse)
    def get_conversation(
        conversation_id: str,
        user: WebUser = Depends(require_user),
    ) -> dict:
        detail = _owned_detail(store, conversation_id, user.id)
        return _detail_dict(detail)

    @router.patch("/{conversation_id}", response_model=ConversationResponse)
    def update_conversation(
        conversation_id: str,
        payload: UpdateConversationRequest,
        user: WebUser = Depends(require_user),
    ) -> dict:
        if store.get_conversation_detail(conversation_id, user_id=user.id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        try:
            record = store.update_conversation(
                conversation_id,
                user_id=user.id,
                title=payload.title,
                archived=payload.archived,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return _conversation_dict(record)

    @router.get(
        "/{conversation_id}/messages",
        response_model=ConversationMessagesResponse,
    )
    def list_messages(
        conversation_id: str,
        user: WebUser = Depends(require_user),
    ) -> dict:
        _owned_detail(store, conversation_id, user.id)
        messages = store.list_active_messages(conversation_id, user_id=user.id)
        if messages is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        unstable = store.get_unstable_turn(conversation_id, user_id=user.id)
        return {
            "items": [_message_dict(message) for message in messages],
            "unstable_turn": (
                {
                    "user_message": _message_dict(unstable.user_message),
                    "task": _task_dict(unstable.task),
                }
                if unstable is not None
                else None
            ),
        }

    @router.get(
        "/{conversation_id}/messages/{message_id}/alternatives",
        response_model=ConversationAlternativesResponse,
    )
    def list_alternatives(
        conversation_id: str,
        message_id: str,
        user: WebUser = Depends(require_user),
    ) -> dict:
        _owned_detail(store, conversation_id, user.id)
        try:
            alternatives = store.list_message_alternatives(
                conversation_id,
                message_id,
                user_id=user.id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if alternatives is None:
            raise HTTPException(status_code=404, detail="message not found")
        return {
            "items": [
                {
                    "user_message": _message_dict(item.user_message),
                    "assistant_message": _message_dict(item.assistant_message),
                }
                for item in alternatives
            ]
        }

    @router.post(
        "/{conversation_id}/messages",
        response_model=CreateMessageResponse,
        status_code=202,
    )
    def create_message(
        conversation_id: str,
        payload: CreateMessageRequest,
        user: WebUser = Depends(require_user),
    ) -> dict:
        detail = _owned_detail(store, conversation_id, user.id)
        _validate_turn_admission(detail, payload.expected_head_message_id)
        try:
            reservation = executor.reserve()
        except TaskExecutorAtCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail="task executor is at capacity",
                headers={"Retry-After": str(overload_retry_after_seconds)},
            ) from exc
        except TaskExecutorShuttingDownError as exc:
            raise HTTPException(
                status_code=503,
                detail="task executor is shutting down",
            ) from exc

        try:
            turn = store.create_conversation_turn(
                user_id=user.id,
                conversation_id=conversation_id,
                content=payload.content,
                depth=payload.depth,
                expected_head_message_id=payload.expected_head_message_id,
            )
        except ConversationBusyError as exc:
            reservation.release()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except StaleConversationHeadError as exc:
            reservation.release()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            reservation.release()
            if store.get_conversation_detail(
                conversation_id,
                user_id=user.id,
            ) is None:
                raise HTTPException(
                    status_code=404,
                    detail="conversation not found",
                ) from exc
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception:
            reservation.release()
            raise

        response = {
            "user_message": _message_dict(turn.user_message),
            "task": _task_dict(turn.task),
            "stable_head_message_id": detail.conversation.head_message_id,
        }
        try:
            reservation.submit(turn.task.id)
        except Exception as exc:
            try:
                failed = store.fail_pending_task(turn.task.id)
                if failed is None:
                    current = store.get_task(turn.task.id, user_id=user.id)
                    if current is not None and current.status in {
                        "running",
                        "completed",
                        "failed",
                    }:
                        return response
                else:
                    store.add_event(
                        task_id=turn.task.id,
                        type="failed",
                        stage="queue",
                        message="Task queue submission failed.",
                        payload={
                            "error_type": type(exc).__name__,
                        },
                    )
            except Exception:
                _LOGGER.exception(
                    "Conversation task submission cleanup failed",
                    extra={
                        "event": "conversation.submission_cleanup_failed",
                        "user_id": user.id,
                        "task_id": turn.task.id,
                    },
                )
            raise HTTPException(
                status_code=503,
                detail="task queue unavailable",
            ) from exc
        return response

    @router.post("/{conversation_id}/rollback", response_model=ConversationResponse)
    def rollback_conversation(
        conversation_id: str,
        payload: RollbackRequest,
        user: WebUser = Depends(require_user),
    ) -> dict:
        detail = _owned_detail(store, conversation_id, user.id)
        _validate_rollback_admission(detail, payload.expected_head_message_id)
        target = store.get_message(
            conversation_id,
            payload.message_id,
            user_id=user.id,
        )
        if target is None:
            raise HTTPException(status_code=404, detail="message not found")
        if target.role != "assistant" or target.status != "complete":
            raise HTTPException(
                status_code=409,
                detail="target must be a complete assistant message",
            )
        if target.task_id is None:
            raise HTTPException(status_code=409, detail="target task is unavailable")
        task = store.get_task(target.task_id, user_id=user.id)
        if task is None or task.conversation_id != conversation_id:
            raise HTTPException(status_code=409, detail="target task is unavailable")
        if task.status != "completed" or not task.final_checkpoint_id:
            raise HTTPException(status_code=409, detail="checkpoint is unavailable")

        checkpoint = deep_reading_runner.read_checkpoint(
            conversation_id,
            task.final_checkpoint_id,
        )
        active_paper_ids = _validate_rollback_checkpoint(
            checkpoint,
            checkpoint_id=task.final_checkpoint_id,
            task_id=task.id,
            message_id=target.id,
            primary_paper_id=detail.conversation.primary_paper_id,
        )
        try:
            switched = store.switch_conversation_head(
                conversation_id,
                user_id=user.id,
                expected_head_message_id=payload.expected_head_message_id,
                target_message_id=target.id,
                target_checkpoint_id=task.final_checkpoint_id,
                active_paper_ids=active_paper_ids,
            )
        except (ConversationBusyError, StaleConversationHeadError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if switched is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return _conversation_dict(switched)

    return router


def _resolve_paper_reference(
    reference: PaperReference,
    paper_search: PaperSearch,
) -> PaperCandidate:
    normalized = normalize_arxiv_id(reference.external_id)
    if normalized is None:
        raise HTTPException(status_code=422, detail="invalid arXiv id")
    candidates = paper_search(normalized, 1)
    for candidate in candidates:
        if (
            candidate.source == reference.source
            and normalize_arxiv_id(candidate.external_id) == normalized
        ):
            return candidate
    raise HTTPException(status_code=404, detail="paper not found")


def _owned_detail(
    store: TaskStore,
    conversation_id: str,
    user_id: str,
) -> ConversationDetail:
    detail = store.get_conversation_detail(conversation_id, user_id=user_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return detail


def _validate_turn_admission(
    detail: ConversationDetail,
    expected_head_message_id: str | None,
) -> None:
    if detail.conversation.archived_at is not None:
        raise HTTPException(status_code=409, detail="conversation is archived")
    if detail.active_task is not None:
        raise HTTPException(
            status_code=409,
            detail="conversation already has an active task",
        )
    if detail.conversation.head_message_id != expected_head_message_id:
        raise HTTPException(status_code=409, detail="conversation head has changed")


def _validate_rollback_admission(
    detail: ConversationDetail,
    expected_head_message_id: str | None,
) -> None:
    _validate_turn_admission(detail, expected_head_message_id)


def _validate_rollback_checkpoint(
    checkpoint: object,
    *,
    checkpoint_id: str,
    task_id: str,
    message_id: str,
    primary_paper_id: str,
) -> list[str]:
    if checkpoint is None:
        raise HTTPException(status_code=409, detail="checkpoint is unavailable")
    if getattr(checkpoint, "checkpoint_id", None) != checkpoint_id:
        raise HTTPException(status_code=409, detail="checkpoint is unavailable")
    state = getattr(checkpoint, "state", None)
    if not isinstance(state, dict):
        raise HTTPException(status_code=409, detail="checkpoint schema is unsupported")
    if state.get("schema_version") != SCHEMA_VERSION:
        raise HTTPException(status_code=409, detail="checkpoint schema is unsupported")
    if state.get("graph_version") != GRAPH_VERSION:
        raise HTTPException(status_code=409, detail="checkpoint graph is unsupported")
    if getattr(checkpoint, "is_complete", None) is not True:
        raise HTTPException(status_code=409, detail="checkpoint is not complete")
    if state.get("published_message_id") != message_id:
        raise HTTPException(
            status_code=409,
            detail="checkpoint does not match target message",
        )
    if state.get("current_task_id") != task_id:
        raise HTTPException(
            status_code=409,
            detail="checkpoint does not match target task",
        )
    active_paper_ids = state.get("active_paper_ids")
    if (
        not isinstance(active_paper_ids, list)
        or not active_paper_ids
        or any(
            not isinstance(paper_id, str) or not paper_id
            for paper_id in active_paper_ids
        )
        or len(set(active_paper_ids)) != len(active_paper_ids)
        or primary_paper_id not in active_paper_ids
    ):
        raise HTTPException(
            status_code=409,
            detail="checkpoint active paper ids are invalid",
        )
    return list(active_paper_ids)


def _paper_dict(record: PaperRecord) -> dict:
    return {
        "source": record.source,
        "external_id": record.external_id,
        "title": record.title,
        "authors": list(record.authors),
        "abstract": record.abstract,
        "source_url": record.source_url,
    }


def _conversation_dict(record: ConversationRecord) -> dict:
    return {
        "id": record.id,
        "primary_paper_id": record.primary_paper_id,
        "title": record.title,
        "head_message_id": record.head_message_id,
        "head_checkpoint_id": record.head_checkpoint_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "archived_at": record.archived_at,
    }


def _task_dict(task: ResearchTask) -> dict:
    return task.to_dict()


def _message_dict(message: MessageRecord) -> dict:
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "task_id": message.task_id,
        "parent_message_id": message.parent_message_id,
        "role": message.role,
        "content": message.content,
        "status": message.status,
        "metadata": message.metadata,
        "created_at": message.created_at,
    }


def _detail_dict(detail: ConversationDetail) -> dict:
    return {
        "conversation": _conversation_dict(detail.conversation),
        "primary_paper": _paper_dict(detail.primary_paper),
        "active_papers": [_paper_dict(paper) for paper in detail.active_papers],
        "active_task": (
            _task_dict(detail.active_task) if detail.active_task is not None else None
        ),
    }
