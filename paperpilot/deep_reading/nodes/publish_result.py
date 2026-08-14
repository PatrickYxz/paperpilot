"""Idempotent publication node."""
from __future__ import annotations

import json

from langgraph.runtime import Runtime
from pydantic import ValidationError

from paperpilot.web.task_store import MessageRecord, ResearchTask, UsedPaperInput

from ..research_agent import FinalCheckpointError, ResearchContractError
from ..schemas import AnswerDraft, ResearchResult
from ..state import DeepReadingState
from .binding import _validate_runtime_binding
from .context import DeepReadingContext
from .validation import (
    _validate_answer_citations,
    _validated_answer_draft,
    _validated_research_result,
)


def publish_result(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Idempotently publish one validated answer and its actually used papers."""
    context = runtime.context
    task, user_message, _detail = _validate_runtime_binding(state, context)
    candidate_result = _validated_research_result(state.get("research_result"))
    candidate_draft = _validated_answer_draft(state.get("answer_draft"))
    _validate_answer_citations(candidate_draft, candidate_result)

    existing = context.task_store.get_task_message(task.id, "assistant")
    if existing is None:
        candidate_metadata = _publication_metadata(candidate_result, candidate_draft)
        established = context.task_store.publish_conversation_result(
            task_id=task.id,
            content=candidate_draft.content,
            metadata=candidate_metadata,
            used_papers=[],
        )
        result, draft = _read_authoritative_publication(
            established.message,
            task=task,
            user_message=user_message,
        )
    else:
        result, draft = _read_authoritative_publication(
            existing,
            task=task,
            user_message=user_message,
        )

    published = context.task_store.publish_conversation_result(
        task_id=task.id,
        content=draft.content,
        metadata=_publication_metadata(result, draft),
        used_papers=_used_paper_inputs(result),
    )
    result, draft = _read_authoritative_publication(
        published.message,
        task=task,
        user_message=user_message,
    )
    return {
        "research_result": result.model_dump(mode="json"),
        "answer_draft": draft.model_dump(mode="json"),
        "published_message_id": published.message.id,
        "active_paper_ids": list(published.active_paper_ids),
        "messages": [
            {
                "role": "assistant",
                "content": published.message.content,
                "id": published.message.id,
            }
        ],
    }


def _publication_metadata(
    result: ResearchResult,
    draft: AnswerDraft,
) -> dict[str, object]:
    return json.loads(
        json.dumps(
            {
                "citations": [
                    citation.model_dump(mode="json") for citation in draft.citations
                ],
                "result_quality": draft.result_quality,
                "limitations": list(result.limitations),
                "research_result": result.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
    )


def _read_authoritative_publication(
    message: MessageRecord,
    *,
    task: ResearchTask,
    user_message: MessageRecord,
) -> tuple[ResearchResult, AnswerDraft]:
    if message.conversation_id != task.conversation_id:
        raise FinalCheckpointError(
            "persisted assistant belongs to another conversation"
        )
    if message.task_id != task.id:
        raise FinalCheckpointError("persisted assistant belongs to another task")
    if message.parent_message_id != user_message.id:
        raise FinalCheckpointError("persisted assistant has an invalid parent")
    if message.role != "assistant" or message.status != "complete":
        raise FinalCheckpointError("persisted assistant is not complete")
    if not isinstance(message.metadata, dict):
        raise FinalCheckpointError("persisted assistant metadata is not an object")
    try:
        result = ResearchResult.model_validate(message.metadata["research_result"])
        draft = AnswerDraft.model_validate(
            {
                "content": message.content,
                "citations": message.metadata["citations"],
                "result_quality": message.metadata["result_quality"],
            }
        )
        limitations = message.metadata["limitations"]
    except (KeyError, TypeError, ValidationError) as exc:
        raise FinalCheckpointError("persisted assistant metadata is invalid") from exc
    if limitations != result.limitations:
        raise FinalCheckpointError(
            "persisted assistant metadata limitations do not match research result"
        )
    try:
        _validate_answer_citations(draft, result)
    except ResearchContractError as exc:
        raise FinalCheckpointError(
            "persisted assistant citations are invalid"
        ) from exc
    return result, draft


def _used_paper_inputs(result: ResearchResult) -> list[UsedPaperInput]:
    return [
        UsedPaperInput(paper=paper_use.paper, role=paper_use.role)
        for paper_use in result.used_papers
    ]
