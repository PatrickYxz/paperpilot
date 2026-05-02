"""graph-mcp: citation graph server backed by NetworkX and Semantic Scholar."""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from paperpilot.mcp_servers.graph.graph_manager import GraphManager

mcp = FastMCP("graph")
_manager: GraphManager | None = None


@mcp.tool()
def build_graph(arxiv_ids: list[str]) -> dict:
    """Build or extend the citation graph for downloaded arXiv papers.

    Args:
        arxiv_ids: Non-empty list of arXiv ids without version suffixes, for
            example ["2401.00001", "2401.00002"].

    Returns:
        dict with added_count, skipped_count, missing, total_nodes, total_edges.
    """
    return _build_graph_impl(arxiv_ids)


def _build_graph_impl(arxiv_ids: list[str]) -> dict:
    manager = _require_manager()
    return manager.build_graph(arxiv_ids)


@mcp.tool()
def get_neighbors(
    arxiv_id: str, direction: str = "both", limit: int = 10
) -> list[dict]:
    """Return citation neighbors for one paper.

    Args:
        arxiv_id: Paper id already present in the graph.
        direction: "references", "citations", or "both".
        limit: Maximum number of neighbors to return.

    Returns:
        list of dicts with arxiv_id, title, year, authors, edge.
    """
    return _get_neighbors_impl(arxiv_id, direction, limit)


def _get_neighbors_impl(
    arxiv_id: str, direction: str = "both", limit: int = 10
) -> list[dict]:
    manager = _require_manager()
    return manager.get_neighbors(arxiv_id, direction, limit)


@mcp.tool()
def get_shortest_path(from_id: str, to_id: str) -> dict:
    """Return a directed citation path from one paper to another."""
    return _get_shortest_path_impl(from_id, to_id)


def _get_shortest_path_impl(from_id: str, to_id: str) -> dict:
    manager = _require_manager()
    return manager.get_shortest_path(from_id, to_id)


@mcp.tool()
def get_common_citations(arxiv_ids: list[str], top_k: int = 10) -> list[dict]:
    """Return references cited by at least two input papers."""
    return _get_common_citations_impl(arxiv_ids, top_k)


def _get_common_citations_impl(arxiv_ids: list[str], top_k: int = 10) -> list[dict]:
    manager = _require_manager()
    return manager.get_common_citations(arxiv_ids, top_k)


def _require_manager() -> GraphManager:
    if _manager is None:
        raise RuntimeError("GraphManager not initialized")
    return _manager


if __name__ == "__main__":
    _manager = GraphManager(os.environ.get("PAPERPILOT_GRAPH_PATH", "data/graph/citation_graph.pkl"))
    mcp.run(transport="stdio")
