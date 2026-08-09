"""Initial state-management nodes for the deep-reading graph."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from langchain.messages import AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from pydantic import ValidationError

from paperpilot.core.adapter import Tool
from paperpilot.papers import PaperCandidate, normalize_arxiv_id
from paperpilot.web.task_store import (
    ConversationDetail,
    MessageRecord,
    ResearchTask,
    TaskStore,
    UsedPaperInput,
)

from .research_agent import ResearchContractError, run_research_agent
from .schemas import AnswerDraft, ConversationSummary, ResearchResult
from .state import GRAPH_VERSION, SCHEMA_VERSION, DeepReadingState

_DOWNLOAD_TOOL = "mcp__arxiv__download_paper"
_BUILD_TOOL = "mcp__colbert__build_index"

PaperSearch = Callable[[str, int], list[PaperCandidate]]
EventSink = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class DeepReadingContext:
    """Trusted, per-run dependencies and identifiers excluded from checkpoints."""

    user_id: str
    conversation_id: str
    task_id: str
    current_user_message_id: str
    base_checkpoint_id: str | None
    task_store: TaskStore
    model: Any
    mcp_tools: Mapping[str, Tool]
    paper_search: PaperSearch
    event_sink: EventSink
    summary_token_threshold: int = 32_000
    summary_recent_turns: int = 6
    research_recursion_limit: int = 12

    def __post_init__(self) -> None:
        for name in (
            "summary_token_threshold",
            "summary_recent_turns",
            "research_recursion_limit",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


def initialize_turn(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Reset per-turn outputs while retaining checkpointed conversation context."""
    del state
    context = runtime.context
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_version": GRAPH_VERSION,
        "current_task_id": context.task_id,
        "current_user_message_id": context.current_user_message_id,
        "research_result": None,
        "answer_draft": None,
        "published_message_id": None,
        "error": None,
    }


def needs_summary(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> bool:
    """Return whether estimated checkpoint context exceeds the configured budget."""
    return _estimated_tokens(state) > runtime.context.summary_token_threshold


def route_after_initialize(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> Literal["summarize_history", "prepare_primary_paper"]:
    """Route through the optional summary node using this invocation's context."""
    if needs_summary(state, runtime):
        return "summarize_history"
    return "prepare_primary_paper"


def summarize_history(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Summarize long history and replace current State messages with recent turns."""
    if not needs_summary(state, runtime):
        return {}

    context = runtime.context
    _validate_runtime_binding(state, context)
    summary_model = context.model.with_structured_output(ConversationSummary)
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

    summary = ConversationSummary.model_validate(summary_model.invoke(summary_input))
    recent_messages = _recent_turns(
        state.get("messages", []), context.summary_recent_turns
    )
    return {
        "conversation_summary": summary.model_dump(mode="json"),
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *recent_messages],
    }


def research_evidence(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Run bounded research and checkpoint the complete JSON result."""
    context = runtime.context
    _validate_runtime_binding(state, context)
    result = run_research_agent(state, context)
    return {"research_result": result.model_dump(mode="json")}


def prepare_primary_paper(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Prepare the business-authoritative primary paper for bounded retrieval."""
    context = runtime.context
    _task, _user_message, detail = _validate_runtime_binding(state, context)
    primary = detail.primary_paper
    state_primary_id = _required_text(
        state.get("primary_paper_id"),
        "primary_paper_id",
    )
    if primary.id != state_primary_id:
        raise ResearchContractError(
            "state primary_paper_id does not match the conversation catalog"
        )
    if primary.source != "arxiv":
        raise ResearchContractError("conversation primary paper source must be arxiv")
    external_id = _canonical_arxiv_id(
        primary.external_id,
        "conversation primary paper external ID",
    )

    downloaded = _call_prepare_mcp_json(
        context,
        name=_DOWNLOAD_TOOL,
        arguments={"arxiv_id": external_id},
    )
    downloaded_id = _canonical_arxiv_id(
        downloaded.get("paper_id"),
        "download paper ID",
    )
    if downloaded_id != external_id:
        raise ResearchContractError(
            "download MCP paper ID does not match the requested paper ID"
        )
    text = downloaded.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ResearchContractError("download MCP payload is missing paper text")

    build_result = _call_prepare_mcp_json(
        context,
        name=_BUILD_TOOL,
        arguments={"documents": [{"paper_id": external_id, "text": text}]},
    )
    if external_id not in _indexed_paper_ids(build_result):
        raise ResearchContractError(
            "build MCP paper IDs do not include the requested paper ID"
        )

    active_paper_ids = [primary.id]
    for paper in detail.active_papers:
        paper_id = _required_text(getattr(paper, "id", None), "active paper ID")
        if paper_id not in active_paper_ids:
            active_paper_ids.append(paper_id)
    return {
        "primary_paper_id": primary.id,
        "active_paper_ids": active_paper_ids,
    }


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

    structured_model = context.model.with_structured_output(AnswerDraft)
    draft = AnswerDraft.model_validate(structured_model.invoke(model_input))
    _validate_answer_citations(draft, result)
    return {"answer_draft": draft.model_dump(mode="json")}


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


def _validate_runtime_binding(
    state: DeepReadingState,
    context: DeepReadingContext,
) -> tuple[ResearchTask, MessageRecord, ConversationDetail]:
    task_id = _required_text(context.task_id, "runtime task_id")
    task = context.task_store.get_task(task_id, user_id=context.user_id)
    if task is None:
        raise ResearchContractError("runtime task is unavailable for the owner")
    if task.id != task_id or task.user_id != context.user_id:
        raise ResearchContractError("runtime task does not match the requested owner")

    detail = context.task_store.get_conversation_detail(
        context.conversation_id,
        user_id=context.user_id,
    )
    if detail is None:
        raise ResearchContractError("conversation is unavailable for the owner")
    conversation = detail.conversation
    if conversation.id != context.conversation_id:
        raise ResearchContractError(
            "conversation detail does not match runtime context"
        )
    if conversation.user_id != context.user_id:
        raise ResearchContractError("conversation detail does not match runtime owner")
    if task.conversation_id != conversation.id:
        raise ResearchContractError("runtime task belongs to another conversation")
    if task.base_checkpoint_id != context.base_checkpoint_id:
        raise ResearchContractError(
            "runtime task base checkpoint does not match context"
        )
    state_task_id = _required_text(state.get("current_task_id"), "current_task_id")
    if state_task_id != task.id:
        raise ResearchContractError("state current_task_id does not match runtime task")

    message = context.task_store.get_task_message(task.id, "user")
    if message is None:
        raise ResearchContractError("runtime task user message is unavailable")
    if message.id != context.current_user_message_id:
        raise ResearchContractError("runtime task is bound to another user message")
    if message.role != "user" or message.status != "complete":
        raise ResearchContractError("runtime task user message is not complete")
    if message.conversation_id != context.conversation_id:
        raise ResearchContractError(
            "runtime user message belongs to another conversation"
        )
    if message.task_id != task.id:
        raise ResearchContractError("runtime user message belongs to another task")
    if task.question != message.content:
        raise ResearchContractError("runtime task question does not match user message")
    state_message_id = _required_text(
        state.get("current_user_message_id"),
        "current_user_message_id",
    )
    if state_message_id != message.id:
        raise ResearchContractError(
            "state current_user_message_id does not match runtime user message"
        )

    state_messages = state.get("messages")
    if not isinstance(state_messages, list):
        raise ResearchContractError("state messages must be a list")
    matching_messages = [
        item for item in state_messages if getattr(item, "id", None) == message.id
    ]
    if len(matching_messages) != 1:
        raise ResearchContractError(
            "state must contain exactly one current user message"
        )
    current_human = matching_messages[0]
    if getattr(current_human, "type", None) != "human":
        raise ResearchContractError("state current user message must be human")
    if current_human.content != message.content:
        raise ResearchContractError(
            "state current user message content does not match business message"
        )
    human_messages = [
        item for item in state_messages if getattr(item, "type", None) == "human"
    ]
    if not human_messages or human_messages[-1] is not current_human:
        raise ResearchContractError("state current user message must be the last human")
    return task, message, detail


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchContractError(f"{field_name} must be a non-blank string")
    return value.strip()


def _canonical_arxiv_id(value: object, field_name: str) -> str:
    raw = _required_text(value, field_name)
    normalized = normalize_arxiv_id(raw)
    if normalized is None:
        raise ResearchContractError(f"{field_name} must be a valid arXiv ID or URL")
    return normalized


def _call_prepare_mcp_json(
    context: DeepReadingContext,
    *,
    name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    tool = context.mcp_tools.get(name)
    if tool is None:
        raise ResearchContractError(f"required MCP tool is unavailable: {name}")
    if tool.name != name:
        raise ResearchContractError(f"MCP tool map entry has mismatched name: {name}")
    context.event_sink(
        "tool_call",
        {
            "stage": "prepare",
            "name": name,
            "arguments": _safe_prepare_event_arguments(arguments),
        },
    )
    raw = tool.handler(arguments)
    if not isinstance(raw, str):
        raise ResearchContractError(f"{name} did not return JSON text")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        context.event_sink(
            "tool_result",
            {"stage": "prepare", "name": name, "content": "invalid JSON payload"},
        )
        raise ResearchContractError(f"{name} did not return valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ResearchContractError(f"{name} JSON payload must be an object")
    context.event_sink(
        "tool_result",
        {"stage": "prepare", "name": name, "content": "validated JSON object"},
    )
    return decoded


def _safe_prepare_event_arguments(
    arguments: Mapping[str, object],
) -> dict[str, object]:
    documents = arguments.get("documents")
    if isinstance(documents, list):
        return {
            "documents": [
                {
                    "paper_id": document.get("paper_id"),
                    "text_chars": len(str(document.get("text", ""))),
                }
                for document in documents
                if isinstance(document, Mapping)
            ]
        }
    return dict(arguments)


def _indexed_paper_ids(payload: Mapping[str, object]) -> set[str]:
    paper_ids: set[str] = set()
    found_list = False
    for field in ("fresh_papers", "cached_papers", "existing_papers"):
        value = payload.get(field)
        if value is None:
            continue
        found_list = True
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ResearchContractError(
                f"build MCP field {field} must be a string list"
            )
        paper_ids.update(
            _canonical_arxiv_id(item, f"build MCP {field} paper ID")
            for item in value
        )
    if not found_list:
        raise ResearchContractError("build MCP payload is missing indexed paper IDs")
    return paper_ids


def _validated_research_result(value: object) -> ResearchResult:
    try:
        return ResearchResult.model_validate(value)
    except ValidationError as exc:
        raise ResearchContractError("state research_result is invalid") from exc


def _validated_answer_draft(value: object) -> AnswerDraft:
    try:
        return AnswerDraft.model_validate(value)
    except ValidationError as exc:
        raise ResearchContractError("state answer_draft is invalid") from exc


def _validate_answer_citations(
    draft: AnswerDraft,
    result: ResearchResult,
) -> None:
    evidence_ids = {item.id for item in result.evidence_items}
    for citation in draft.citations:
        if citation.evidence_id not in evidence_ids:
            raise ResearchContractError(
                f"answer citation references unknown evidence: {citation.evidence_id}"
            )


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
        raise ResearchContractError(
            "persisted assistant belongs to another conversation"
        )
    if message.task_id != task.id:
        raise ResearchContractError("persisted assistant belongs to another task")
    if message.parent_message_id != user_message.id:
        raise ResearchContractError("persisted assistant has an invalid parent")
    if message.role != "assistant" or message.status != "complete":
        raise ResearchContractError("persisted assistant is not complete")
    if not isinstance(message.metadata, dict):
        raise ResearchContractError("persisted assistant metadata is not an object")
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
        raise ResearchContractError("persisted assistant metadata is invalid") from exc
    if limitations != result.limitations:
        raise ResearchContractError(
            "persisted assistant metadata limitations do not match research result"
        )
    _validate_answer_citations(draft, result)
    return result, draft


def _used_paper_inputs(result: ResearchResult) -> list[UsedPaperInput]:
    return [
        UsedPaperInput(paper=paper_use.paper, role=paper_use.role)
        for paper_use in result.used_papers
    ]


def _paper_metadata(paper: object) -> dict[str, object]:
    return {
        "source": getattr(paper, "source"),
        "external_id": getattr(paper, "external_id"),
        "title": getattr(paper, "title"),
        "authors": list(getattr(paper, "authors")),
        "abstract": getattr(paper, "abstract"),
        "source_url": getattr(paper, "source_url"),
    }
