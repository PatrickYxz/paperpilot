"""Primary-paper preparation node and MCP payload validation helpers."""
from __future__ import annotations

import json
from collections.abc import Mapping

from langgraph.runtime import Runtime

from paperpilot.papers import normalize_arxiv_id
from paperpilot.tools.mcp_client import MCPToolError

from ..research_agent import ResearchContractError
from ..state import DeepReadingState
from .binding import _validate_runtime_binding
from .context import DeepReadingContext
from .validation import _required_text

_DOWNLOAD_TOOL = "mcp__arxiv__download_paper"
_BUILD_TOOL = "mcp__colbert__build_index"


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
    try:
        raw = tool.handler(arguments)
    except MCPToolError as exc:
        raise ResearchContractError(
            f"{name} reported a deterministic tool failure"
        ) from exc
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
