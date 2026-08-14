"""Structured answer-writing node."""
from __future__ import annotations

import json
from collections.abc import Mapping

from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import ValidationError

from ..research_agent import ResearchContractError
from ..schemas import AnswerDraft
from ..state import DeepReadingState
from .binding import _validate_runtime_binding
from .context import DeepReadingContext
from .summarize_history import _recent_turns
from .validation import (
    _required_text,
    _validate_answer_citations,
    _validated_research_result,
)


def write_answer(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Create and validate a structured answer from checkpoint-safe inputs."""
    context = runtime.context
    _task, _user_message, detail = _validate_runtime_binding(state, context)
    result = _validated_research_result(state.get("research_result"))
    primary_id = _required_text(state.get("primary_paper_id"), "primary_paper_id")
    if detail.primary_paper.id != primary_id:
        raise ResearchContractError(
            "state primary_paper_id does not match the conversation catalog"
        )

    model_input: list[AnyMessage] = [
        SystemMessage(
            content=(
                "Write a concise scholarly answer grounded only in the trusted "
                "research result and paper metadata below. Cite evidence by its "
                "exact evidence ID. Mark the result partial when evidence is "
                "insufficient."
            )
        )
    ]
    summary = state.get("conversation_summary")
    if summary is not None:
        model_input.append(
            HumanMessage(
                content=(
                    "Conversation summary:\n"
                    + json.dumps(summary, ensure_ascii=False, sort_keys=True)
                )
            )
        )
    model_input.extend(
        _recent_turns(state.get("messages", []), context.summary_recent_turns)
    )
    model_input.append(
        HumanMessage(
            content=(
                "Trusted paper and research data:\n"
                + json.dumps(
                    {
                        "paper_metadata": {
                            "primary": _paper_metadata(detail.primary_paper),
                            "active": [
                                _paper_metadata(paper)
                                for paper in detail.active_papers
                            ],
                        },
                        "research_result": result.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        )
    )

    structured_model = context.model.with_structured_output(
        AnswerDraft,
        include_raw=True,
    )
    draft_envelope = structured_model.invoke(model_input)
    if not isinstance(draft_envelope, Mapping):
        raise ResearchContractError("model returned an invalid answer draft")
    parsing_error = draft_envelope.get("parsing_error")
    if isinstance(parsing_error, BaseException):
        raise ResearchContractError(
            "model returned an invalid answer draft"
        ) from parsing_error
    if parsing_error is not None or "parsed" not in draft_envelope:
        raise ResearchContractError("model returned an invalid answer draft")
    try:
        draft = AnswerDraft.model_validate(draft_envelope["parsed"])
    except ValidationError as exc:
        raise ResearchContractError("model returned an invalid answer draft") from exc
    _validate_answer_citations(draft, result)
    return {"answer_draft": draft.model_dump(mode="json")}


def _paper_metadata(paper: object) -> dict[str, object]:
    return {
        "source": getattr(paper, "source"),
        "external_id": getattr(paper, "external_id"),
        "title": getattr(paper, "title"),
        "authors": list(getattr(paper, "authors")),
        "abstract": getattr(paper, "abstract"),
        "source_url": getattr(paper, "source_url"),
    }
