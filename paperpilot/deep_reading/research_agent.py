"""Bounded LangChain research agent with authoritative local ledgers."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.agents.structured_output import StructuredOutputError, ToolStrategy
from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ConfigDict, ValidationError

from paperpilot.papers import PaperCandidate, normalize_arxiv_id
from paperpilot.tools.mcp_client import MCPToolError
from paperpilot.tools.types import Tool

from .schemas import EvidenceItem, PaperUse, ResearchResult
from .state import DeepReadingState

if TYPE_CHECKING:
    from .nodes import DeepReadingContext


_DOWNLOAD_TOOL = "mcp__arxiv__download_paper"
_BUILD_TOOL = "mcp__colbert__build_index"
_RETRIEVAL_TOOL = "mcp__colbert__planned_retrieval"
_MAX_AGENT_PAPERS = 20
_MAX_EVENT_TEXT = 500
_STRUCTURED_RESPONSE_ATTEMPTS = 2


class DeepReadingTaskError(RuntimeError):
    """Expected terminal failure for one deep-reading task."""

    error_code = "research_contract_invalid"
    public_message = "The deep-reading task did not satisfy a required contract."


class ResearchContractError(DeepReadingTaskError):
    """Raised when a tool, model decision, or authoritative ledger disagrees."""

    error_code = "research_contract_invalid"
    public_message = "The research result did not satisfy the required contract."


class TaskBindingError(ResearchContractError):
    """Raised when Task, Conversation, or User Message ownership is inconsistent."""

    error_code = "task_binding_invalid"
    public_message = "The task is not bound to the requested conversation."


class TaskStatusError(DeepReadingTaskError):
    """Raised when a Task cannot execute from its persisted status."""

    error_code = "task_status_invalid"
    public_message = "The task is not in an executable state."


class CheckpointBindingError(DeepReadingTaskError):
    """Raised when business heads and checkpoint identity disagree."""

    error_code = "checkpoint_binding_invalid"
    public_message = "The conversation checkpoint binding is invalid."


class CheckpointMissingError(DeepReadingTaskError):
    """Raised when the required checkpoint is absent for this thread."""

    error_code = "checkpoint_missing"
    public_message = "The conversation checkpoint is unavailable."


class CheckpointIncompleteError(DeepReadingTaskError):
    """Raised when a required historical checkpoint is not complete."""

    error_code = "checkpoint_incomplete"
    public_message = "The conversation checkpoint is incomplete."


class GraphVersionUnsupportedError(DeepReadingTaskError):
    """Raised when checkpointed graph data uses an unsupported version."""

    error_code = "graph_version_unsupported"
    public_message = "The conversation checkpoint uses an unsupported graph version."


class SchemaVersionUnsupportedError(DeepReadingTaskError):
    """Raised when checkpointed state uses an unsupported schema version."""

    error_code = "schema_version_unsupported"
    public_message = "The conversation checkpoint uses an unsupported schema version."


class AgentBudgetExceededError(DeepReadingTaskError):
    """Raised when a bounded Research Agent cannot finish within its limits."""

    error_code = "agent_budget_exhausted"
    public_message = "The research agent exhausted its bounded execution budget."


class FinalCheckpointError(ResearchContractError):
    """Raised when published answer state cannot be safely finalized."""

    error_code = "final_checkpoint_invalid"
    public_message = "The final conversation checkpoint is invalid."


class AgentPaperUseDecision(BaseModel):
    """Model-selected paper role expressed only through authoritative IDs."""

    model_config = ConfigDict(extra="forbid")

    external_id: str
    role: Literal["comparison", "citation", "background", "follow_up"]
    evidence_ids: list[str]


class AgentResearchDecision(BaseModel):
    """Model decision that cannot provide paper or evidence content."""

    model_config = ConfigDict(extra="forbid")

    selected_evidence_ids: list[str]
    paper_uses: list[AgentPaperUseDecision]
    limitations: list[str]


CreateAgentFactory = Callable[..., Any]


def run_research_agent(
    state: DeepReadingState,
    context: DeepReadingContext,
    *,
    create_agent_factory: CreateAgentFactory = create_agent,
) -> ResearchResult:
    """Run one bounded research turn and materialize only authoritative evidence."""
    candidate_ledger, primary_external_id = _trusted_candidates(state, context)
    prepared_ledger: dict[str, PaperCandidate] = {}
    evidence_ledger: dict[str, EvidenceItem] = {}

    search_tool = _build_search_tool(context, candidate_ledger)
    prepare_tool = _build_prepare_tool(
        context,
        candidate_ledger,
        prepared_ledger,
    )
    retrieval_tool = _build_retrieval_tool(
        context,
        prepared_ledger,
        evidence_ledger,
    )
    per_attempt_model_limit = (
        context.research_model_call_limit // _STRUCTURED_RESPONSE_ATTEMPTS
    )
    per_attempt_tool_limit = (
        context.research_tool_call_limit // _STRUCTURED_RESPONSE_ATTEMPTS
    )
    agent = create_agent_factory(
        model=context.model,
        tools=[search_tool, prepare_tool, retrieval_tool],
        response_format=ToolStrategy(
            AgentResearchDecision,
            handle_errors=False,
        ),
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=per_attempt_model_limit,
                exit_behavior="error",
            ),
            ToolCallLimitMiddleware(
                run_limit=per_attempt_tool_limit,
                exit_behavior="error",
            ),
            ModelRetryMiddleware(
                max_retries=context.research_model_retries,
                on_failure="error",
            ),
        ],
    )
    messages = _research_messages(
        state,
        primary_external_id=primary_external_id,
        active_external_ids=list(candidate_ledger),
    )

    last_structured_error: Exception | None = None
    for _attempt in range(_STRUCTURED_RESPONSE_ATTEMPTS):
        try:
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": context.research_recursion_limit},
            )
        except (
            GraphRecursionError,
            ModelCallLimitExceededError,
            ToolCallLimitExceededError,
        ) as exc:
            raise AgentBudgetExceededError(
                "research agent budget exhausted before a valid decision"
            ) from exc
        except StructuredOutputError as exc:
            last_structured_error = exc
            continue

        try:
            structured_response = _structured_response(result)
            decision = AgentResearchDecision.model_validate(structured_response)
        except (ValidationError, _StructuredResponseError) as exc:
            last_structured_error = exc
            continue

        return _validate_and_materialize_result(
            decision,
            candidates=candidate_ledger,
            evidence=evidence_ledger,
            primary_external_id=primary_external_id,
        )

    raise AgentBudgetExceededError(
        "research agent structured response attempts exhausted after 2 attempts"
    ) from last_structured_error


class _StructuredResponseError(ValueError):
    pass


def _structured_response(result: object) -> object:
    if not isinstance(result, Mapping) or "structured_response" not in result:
        raise _StructuredResponseError("agent result is missing structured_response")
    return result["structured_response"]


def _trusted_candidates(
    state: DeepReadingState,
    context: DeepReadingContext,
) -> tuple[dict[str, PaperCandidate], str]:
    detail = context.task_store.get_conversation_detail(
        context.conversation_id,
        user_id=context.user_id,
    )
    if detail is None:
        raise ResearchContractError("conversation paper catalog is unavailable")

    primary_internal_id = _required_id(
        state.get("primary_paper_id"),
        "primary_paper_id",
    )
    stored_primary = detail.primary_paper
    if stored_primary.id != primary_internal_id:
        raise ResearchContractError(
            "state primary_paper_id does not match the conversation catalog"
        )

    by_internal_id = {
        paper.id: paper for paper in [stored_primary, *detail.active_papers]
    }
    active_internal_ids = state.get("active_paper_ids", [])
    if len(active_internal_ids) != len(set(active_internal_ids)):
        raise ResearchContractError("active_paper_ids must be unique")
    for internal_id in active_internal_ids:
        if internal_id not in by_internal_id:
            raise ResearchContractError(
                f"active paper is missing from the conversation catalog: {internal_id}"
            )

    trusted_records = [stored_primary]
    trusted_records.extend(by_internal_id[item] for item in active_internal_ids)
    candidates: dict[str, PaperCandidate] = {}
    for record in trusted_records:
        candidate = _candidate_from_record(record)
        existing = candidates.get(candidate.external_id)
        if existing is not None and existing != candidate:
            raise ResearchContractError(
                f"conflicting metadata for paper: {candidate.external_id}"
            )
        candidates[candidate.external_id] = candidate
    return candidates, _canonical_arxiv_id(
        stored_primary.external_id,
        "primary external ID",
    )


def _candidate_from_record(record: object) -> PaperCandidate:
    try:
        source = getattr(record, "source")
        if source != "arxiv":
            raise ResearchContractError(
                f"conversation paper source must be arxiv, got: {source!r}"
            )
        candidate = PaperCandidate(
            source=source,
            external_id=_canonical_arxiv_id(
                getattr(record, "external_id"),
                "conversation paper external ID",
            ),
            title=getattr(record, "title"),
            authors=list(getattr(record, "authors")),
            abstract=getattr(record, "abstract"),
            source_url=getattr(record, "source_url"),
        )
        return candidate
    except ResearchContractError:
        raise
    except (AttributeError, TypeError, ValidationError) as exc:
        raise ResearchContractError("invalid paper metadata in conversation catalog") from exc


def _build_search_tool(
    context: DeepReadingContext,
    candidates: dict[str, PaperCandidate],
) -> BaseTool:
    @tool("search_related_papers")
    def search_related_papers(query: str, limit: int) -> list[dict[str, object]]:
        """Search the structured arXiv catalog for related paper candidates."""
        cleaned_query = _required_id(query, "search query")
        _require_agent_limit(limit, "limit")
        _emit_tool_call(
            context,
            stage="research",
            name="search_related_papers",
            arguments={"query": _clip(cleaned_query), "limit": limit},
        )
        raw_results = context.paper_search(cleaned_query, limit)
        if len(raw_results) > limit:
            raise ResearchContractError("paper search returned more than the requested limit")

        found: list[PaperCandidate] = []
        seen: set[str] = set()
        for raw in raw_results:
            try:
                candidate = PaperCandidate.model_validate(raw)
            except ValidationError as exc:
                raise ResearchContractError(
                    "paper search returned invalid arXiv source metadata"
                ) from exc
            if candidate.source != "arxiv":
                raise ResearchContractError("paper search source must be arxiv")
            candidate = candidate.model_copy(
                update={
                    "external_id": _canonical_arxiv_id(
                        candidate.external_id,
                        "paper search external ID",
                    )
                }
            )
            if candidate.external_id in seen:
                raise ResearchContractError(
                    f"paper search returned duplicate ID: {candidate.external_id}"
                )
            seen.add(candidate.external_id)
            existing = candidates.get(candidate.external_id)
            candidates[candidate.external_id] = existing or candidate
            found.append(existing or candidate)

        _emit_tool_result(
            context,
            stage="research",
            name="search_related_papers",
            content=f"found {len(found)} candidate papers",
        )
        return [item.model_dump(mode="json") for item in found]

    return search_related_papers


def _build_prepare_tool(
    context: DeepReadingContext,
    candidates: Mapping[str, PaperCandidate],
    prepared: dict[str, PaperCandidate],
) -> BaseTool:
    @tool("prepare_paper")
    def prepare_paper(external_id: str) -> dict[str, str]:
        """Download and index one trusted primary, active, or searched paper."""
        normalized_id = _canonical_arxiv_id(external_id, "external_id")
        candidate = candidates.get(normalized_id)
        if candidate is None:
            raise ResearchContractError(
                f"paper ID is not allowed for preparation: {normalized_id}"
            )
        if normalized_id in prepared:
            return {"external_id": normalized_id, "status": "already_prepared"}

        downloaded = _call_mcp_json(
            context,
            name=_DOWNLOAD_TOOL,
            arguments={"arxiv_id": normalized_id},
            stage="prepare",
        )
        download_paper_id = _canonical_arxiv_id(
            downloaded.get("paper_id"),
            "download paper ID",
        )
        if download_paper_id != normalized_id:
            raise ResearchContractError(
                "download MCP paper ID does not match the requested paper ID"
            )
        text = downloaded.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ResearchContractError("download MCP payload is missing paper text")

        indexed_document = {
            "paper_id": normalized_id,
            "text": text,
        }
        build_result = _call_mcp_json(
            context,
            name=_BUILD_TOOL,
            arguments={"documents": [indexed_document]},
            stage="prepare",
        )
        indexed_ids = _indexed_paper_ids(build_result)
        if normalized_id not in indexed_ids:
            raise ResearchContractError(
                "build MCP paper IDs do not include the requested paper ID"
            )
        prepared[normalized_id] = candidate
        return {"external_id": normalized_id, "status": "prepared"}

    return prepare_paper


def _build_retrieval_tool(
    context: DeepReadingContext,
    prepared: Mapping[str, PaperCandidate],
    evidence: dict[str, EvidenceItem],
) -> BaseTool:
    @tool("retrieve_paper_evidence")
    def retrieve_paper_evidence(
        question: str,
        external_id: str,
        top_k_each: int,
        summary_k: int,
    ) -> dict[str, object]:
        """Retrieve planned evidence only from a paper prepared in this run."""
        cleaned_question = _required_id(question, "retrieval question")
        normalized_id = _canonical_arxiv_id(external_id, "external_id")
        _require_agent_limit(top_k_each, "top_k_each")
        _require_agent_limit(summary_k, "summary_k")
        candidate = prepared.get(normalized_id)
        if candidate is None:
            raise ResearchContractError(
                f"paper ID is not prepared for retrieval: {normalized_id}"
            )

        payload = _call_mcp_json(
            context,
            name=_RETRIEVAL_TOOL,
            arguments={
                "question": cleaned_question,
                "paper_id": normalized_id,
                "paper_title": candidate.title,
                "abstract": candidate.abstract or "",
                "top_k_each": top_k_each,
                "summary_k": summary_k,
            },
            stage="research",
        )
        parsed_items, summary_ids = _decode_evidence_pool(
            payload,
            candidate,
            question=cleaned_question,
            top_k_each=top_k_each,
            summary_k=summary_k,
        )
        new_items: list[EvidenceItem] = []
        for item in parsed_items:
            existing = evidence.get(item.id)
            if existing is not None and existing != item:
                raise ResearchContractError(
                    f"conflicting evidence for global ID: {item.id}"
                )
            if existing is None:
                new_items.append(item)
        evidence.update((item.id, item) for item in new_items)
        return {
            "evidence_items": [
                item.model_dump(mode="json") for item in parsed_items
            ],
            "summary_item_ids": summary_ids,
        }

    return retrieve_paper_evidence


def _call_mcp_json(
    context: DeepReadingContext,
    *,
    name: str,
    arguments: dict[str, object],
    stage: Literal["prepare", "research"],
) -> dict[str, object]:
    custom_tool = _required_mcp_tool(context.mcp_tools, name)
    _emit_tool_call(
        context,
        stage=stage,
        name=name,
        arguments=_event_arguments(arguments),
    )
    try:
        raw = custom_tool.handler(arguments)
    except MCPToolError as exc:
        raise ResearchContractError(
            f"{name} reported a deterministic tool failure"
        ) from exc
    try:
        if not isinstance(raw, str):
            raise ResearchContractError(f"{name} did not return JSON text")
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise ResearchContractError(f"{name} JSON payload must be an object")
    except json.JSONDecodeError as exc:
        _emit_tool_result(
            context,
            stage=stage,
            name=name,
            content="invalid JSON payload",
        )
        raise ResearchContractError(f"{name} did not return valid JSON") from exc
    _emit_tool_result(
        context,
        stage=stage,
        name=name,
        content="validated JSON object",
    )
    return decoded


def _required_mcp_tool(tools: Mapping[str, Tool], name: str) -> Tool:
    selected = tools.get(name)
    if selected is None:
        raise ResearchContractError(f"required MCP tool is unavailable: {name}")
    if selected.name != name:
        raise ResearchContractError(f"MCP tool map entry has mismatched name: {name}")
    return selected


def _indexed_paper_ids(payload: Mapping[str, object]) -> set[str]:
    ids: set[str] = set()
    found_list = False
    for field in ("fresh_papers", "cached_papers", "existing_papers"):
        value = payload.get(field)
        if value is None:
            continue
        found_list = True
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ResearchContractError(f"build MCP field {field} must be a string list")
        ids.update(
            _canonical_arxiv_id(item, f"build MCP {field} paper ID")
            for item in value
        )
    if not found_list:
        raise ResearchContractError("build MCP payload is missing indexed paper IDs")
    return ids


def _decode_evidence_pool(
    payload: Mapping[str, object],
    candidate: PaperCandidate,
    *,
    question: str,
    top_k_each: int,
    summary_k: int,
) -> tuple[list[EvidenceItem], list[str]]:
    pool = payload.get("evidence_pool")
    if not isinstance(pool, Mapping):
        raise ResearchContractError("planned retrieval payload is missing evidence_pool")
    raw_items = pool.get("items")
    if not isinstance(raw_items, list):
        raise ResearchContractError("evidence_pool.items must be a list")

    parsed: list[EvidenceItem] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise ResearchContractError("evidence_pool item must be an object")
        raw_evidence_id = _required_id(raw.get("id"), "evidence ID")
        if raw_evidence_id in seen:
            raise ResearchContractError(
                f"duplicate evidence ID: {raw_evidence_id}"
            )
        seen.add(raw_evidence_id)
        paper_id = _canonical_arxiv_id(
            raw.get("paper_id"),
            "evidence paper ID",
        )
        if paper_id != candidate.external_id:
            raise ResearchContractError(
                "evidence MCP paper ID does not match the requested paper ID"
            )
        evidence_id = _global_evidence_id(
            paper_external_id=paper_id,
            question=question,
            top_k_each=top_k_each,
            summary_k=summary_k,
            raw_evidence_id=raw_evidence_id,
        )
        score = _evidence_score(raw.get("best_score"))
        supports = _evidence_supports(raw.get("matched_queries"))
        try:
            parsed.append(
                EvidenceItem(
                    id=evidence_id,
                    paper_external_id=paper_id,
                    paper_title=candidate.title,
                    chunk_text=raw.get("chunk_text"),
                    score=score,
                    supports=supports,
                )
            )
        except ValidationError as exc:
            raise ResearchContractError(
                f"invalid evidence item returned by MCP: {evidence_id}"
            ) from exc

    raw_summary_ids = pool.get("summary_items")
    if not isinstance(raw_summary_ids, list) or not all(
        isinstance(item, str) for item in raw_summary_ids
    ):
        raise ResearchContractError("evidence_pool.summary_items must be a string list")
    if len(raw_summary_ids) != len(set(raw_summary_ids)):
        raise ResearchContractError("evidence_pool.summary_items contains duplicates")
    dangling = set(raw_summary_ids).difference(seen)
    if dangling:
        raise ResearchContractError(
            f"evidence_pool.summary_items references unknown ID: {sorted(dangling)[0]}"
        )
    return parsed, [
        _global_evidence_id(
            paper_external_id=candidate.external_id,
            question=question,
            top_k_each=top_k_each,
            summary_k=summary_k,
            raw_evidence_id=raw_id,
        )
        for raw_id in raw_summary_ids
    ]


def _evidence_score(value: object) -> float:
    if isinstance(value, bool):
        raise ResearchContractError("evidence score must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchContractError("evidence score must be numeric") from exc
    if not math.isfinite(score):
        raise ResearchContractError("evidence score must be finite")
    if score < 0:
        raise ResearchContractError("evidence score must be nonnegative")
    return min(score / 10.0, 1.0)


def _evidence_supports(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ResearchContractError("evidence matched_queries must be a list")
    supports: list[str] = []
    seen: set[str] = set()
    for match in value:
        if not isinstance(match, Mapping):
            raise ResearchContractError("evidence matched query must be an object")
        targets = match.get("targets")
        if not isinstance(targets, list) or not all(
            isinstance(target, str) and target.strip() for target in targets
        ):
            raise ResearchContractError("evidence query targets must be strings")
        for target in targets:
            if target not in seen:
                seen.add(target)
                supports.append(target)
    return supports


def _validate_and_materialize_result(
    decision: AgentResearchDecision,
    *,
    candidates: Mapping[str, PaperCandidate],
    evidence: Mapping[str, EvidenceItem],
    primary_external_id: str,
) -> ResearchResult:
    selected_ids = decision.selected_evidence_ids
    if len(selected_ids) != len(set(selected_ids)):
        raise ResearchContractError("selected evidence IDs must be unique")
    selected_items: list[EvidenceItem] = []
    for evidence_id in selected_ids:
        item = evidence.get(evidence_id)
        if item is None:
            raise ResearchContractError(
                f"selected evidence ID is not in the retrieval ledger: {evidence_id}"
            )
        selected_items.append(item)

    used_papers: list[PaperUse] = []
    used_external_ids: set[str] = set()
    evidence_claimed_by: dict[str, str] = {}
    for use in decision.paper_uses:
        external_id = _canonical_arxiv_id(
            use.external_id,
            "paper use external_id",
        )
        if external_id in used_external_ids:
            raise ResearchContractError(f"duplicate paper use ID: {external_id}")
        used_external_ids.add(external_id)
        candidate = candidates.get(external_id)
        if candidate is None:
            raise ResearchContractError(f"paper use ID is not authoritative: {external_id}")
        if not use.evidence_ids:
            raise ResearchContractError("paper use must include evidence IDs")
        if len(use.evidence_ids) != len(set(use.evidence_ids)):
            raise ResearchContractError("paper use evidence IDs must be unique")
        for evidence_id in use.evidence_ids:
            item = evidence.get(evidence_id)
            if item is None or evidence_id not in selected_ids:
                raise ResearchContractError(
                    f"paper use references unselected evidence ID: {evidence_id}"
                )
            if item.paper_external_id != external_id:
                raise ResearchContractError(
                    f"evidence {evidence_id} does not belong to paper {external_id}"
                )
            previous = evidence_claimed_by.get(evidence_id)
            if previous is not None and previous != external_id:
                raise ResearchContractError(
                    f"evidence {evidence_id} is claimed by multiple papers"
                )
            evidence_claimed_by[evidence_id] = external_id
        if external_id != primary_external_id:
            used_papers.append(
                PaperUse(
                    paper=candidate,
                    role=use.role,
                    evidence_ids=list(use.evidence_ids),
                )
            )

    for item in selected_items:
        if (
            item.paper_external_id != primary_external_id
            and item.id not in evidence_claimed_by
        ):
            raise ResearchContractError(
                f"selected related-paper evidence has no paper use: {item.id}"
            )
    try:
        return ResearchResult(
            evidence_items=selected_items,
            used_papers=used_papers,
            limitations=list(decision.limitations),
        )
    except ValidationError as exc:
        raise ResearchContractError("materialized research result is invalid") from exc


def _research_messages(
    state: DeepReadingState,
    *,
    primary_external_id: str,
    active_external_ids: Sequence[str],
) -> list[AnyMessage]:
    messages: list[AnyMessage] = [
        SystemMessage(
            content=(
                "Research the user's paper-reading question with the three provided "
                "tools. Prepare a paper before retrieving it. Select only evidence "
                "IDs returned in this run, and list paper_uses only for evidence that "
                "is selected. The primary paper is already associated with the "
                f"conversation ({primary_external_id}); active papers are: "
                f"{', '.join(active_external_ids)}."
            )
        )
    ]
    summary = state.get("conversation_summary")
    if summary is not None:
        messages.append(
            HumanMessage(
                content=(
                    "Confirmed conversation summary:\n"
                    + json.dumps(summary, ensure_ascii=False, sort_keys=True)
                )
            )
        )
    messages.extend(state.get("messages", []))
    return messages


def _required_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchContractError(f"{field_name} must be a non-blank string")
    return value.strip()


def _canonical_arxiv_id(value: object, field_name: str) -> str:
    raw = _required_id(value, field_name)
    normalized = normalize_arxiv_id(raw)
    if normalized is None:
        raise ResearchContractError(f"{field_name} is not a valid arXiv ID or URL")
    return normalized


def _global_evidence_id(
    *,
    paper_external_id: str,
    question: str,
    top_k_each: int,
    summary_k: int,
    raw_evidence_id: str,
) -> str:
    identity = json.dumps(
        [
            paper_external_id,
            question,
            top_k_each,
            summary_k,
            raw_evidence_id,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"evg_{hashlib.sha256(identity).hexdigest()[:24]}"


def _require_agent_limit(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not 1 <= value <= _MAX_AGENT_PAPERS:
        raise ResearchContractError(
            f"{field_name} must be between 1 and {_MAX_AGENT_PAPERS}"
        )


def _event_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    if "documents" in arguments:
        documents = arguments["documents"]
        if isinstance(documents, list):
            return {
                "documents": [
                    {
                        "paper_id": item.get("paper_id"),
                        "text_chars": len(str(item.get("text", ""))),
                    }
                    for item in documents
                    if isinstance(item, Mapping)
                ]
            }
    return {
        key: _clip(value) if isinstance(value, str) else value
        for key, value in arguments.items()
    }


def _emit_tool_call(
    context: DeepReadingContext,
    *,
    stage: Literal["prepare", "research"],
    name: str,
    arguments: Mapping[str, object],
) -> None:
    context.event_sink(
        "tool_call",
        {"stage": stage, "name": name, "arguments": dict(arguments)},
    )


def _emit_tool_result(
    context: DeepReadingContext,
    *,
    stage: Literal["prepare", "research"],
    name: str,
    content: str,
) -> None:
    context.event_sink(
        "tool_result",
        {"stage": stage, "name": name, "content": _clip(content)},
    )


def _clip(value: str) -> str:
    if len(value) <= _MAX_EVENT_TEXT:
        return value
    return value[: _MAX_EVENT_TEXT - 3].rstrip() + "..."
