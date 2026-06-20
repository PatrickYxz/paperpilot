"""colbert-mcp: ColBERT passage-level retrieval server.

Run with: python -m paperpilot.mcp_servers.colbert.server
The top-level MCP client launches it over stdio from paperpilot/mcp_servers.json.

This protocol layer validates input and routes to IndexManager. It does not
touch PyLate details directly.
"""
from __future__ import annotations

import uuid

from mcp.server.fastmcp import FastMCP

from paperpilot.mcp_servers.colbert.index_manager import (
    IndexManager,
    IndexNotFoundError,
)
from paperpilot.retrieval.llm_query_planner import plan_with_llm
from paperpilot.retrieval.planned_retrieval import run_planned_retrieval

mcp = FastMCP("colbert")
_manager: IndexManager | None = None


@mcp.tool()
def build_index(documents: list[dict]) -> dict:
    """Build isolated per-paper ColBERT indexes.

    Each paper_id gets one persisted index. Rebuilding an already indexed paper
    can hit either memory or disk cache and skip encoding.

    Args:
        documents: Non-empty list of dicts with paper_id (str) and text (str).

    Returns:
        dict with indexed_count, index_name, cached_papers, and fresh_papers.
    """
    return _build_index_impl(documents)


def _build_index_impl(documents: list[dict]) -> dict:
    if not documents:
        raise ValueError("documents must not be empty")
    for document in documents:
        if (
            not isinstance(document, dict)
            or "paper_id" not in document
            or "text" not in document
        ):
            raise ValueError(
                "each document must be dict with 'paper_id' and 'text': "
                f"{document!r}"
            )
    assert _manager is not None, "IndexManager not initialized"
    return _manager.build(documents)


@mcp.tool()
def search(query: str, paper_id: str, top_k: int = 5) -> list[dict]:
    """Search top-k chunks inside one paper's isolated index.

    Args:
        query: Natural language query.
        paper_id: The same paper_id previously passed to build_index.
        top_k: Number of chunks to return, default 5.

    Returns:
        list of dicts with paper_id, chunk_text, and score.
        Raises IndexNotFoundError if that paper has not been indexed.
    """
    return _search_impl(query, paper_id, top_k)


def _search_impl(query: str, paper_id: str, top_k: int) -> list[dict]:
    assert _manager is not None, "IndexManager not initialized"
    return _manager.search(query, paper_id, top_k)


@mcp.tool()
def planned_retrieval(
    question: str,
    paper_id: str,
    paper_title: str = "",
    abstract: str = "",
    top_k_each: int = 5,
    summary_k: int = 8,
) -> dict:
    """Plan and execute multiple retrieval queries for one paper.

    Args:
        question: User question to answer from the paper.
        paper_id: The same paper_id previously passed to build_index.
        paper_title: Optional paper title for query planning context.
        abstract: Optional abstract for query planning context.
        top_k_each: Number of chunks to retrieve for each planned query.
        summary_k: Number of deduped evidence chunks to include in summary.

    Returns:
        dict with query_plan_meta, summary_text, evidence_pool, and query_errors.
    """
    return _planned_retrieval_impl(
        question,
        paper_id,
        paper_title,
        abstract,
        top_k_each,
        summary_k,
    )


def _planned_retrieval_impl(
    question: str,
    paper_id: str,
    paper_title: str,
    abstract: str,
    top_k_each: int,
    summary_k: int,
) -> dict:
    assert _manager is not None, "IndexManager not initialized"
    plan, meta = plan_with_llm(
        question=question,
        paper_title=paper_title,
        abstract=abstract,
    )
    result = run_planned_retrieval(
        plan_id=f"plan-{uuid.uuid4().hex[:12]}",
        plan=plan,
        paper_id=paper_id,
        search=lambda query, pid, top_k: _manager.search(query, pid, top_k),
        top_k_each=top_k_each,
        summary_k=summary_k,
    )
    payload = result.to_dict()
    payload["query_plan_meta"] = meta
    return payload


if __name__ == "__main__":
    _manager = IndexManager()
    mcp.run(transport="stdio")
