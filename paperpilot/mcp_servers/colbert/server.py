"""colbert-mcp: ColBERT passage-level retrieval server.

Run with: python -m paperpilot.mcp_servers.colbert.server
The top-level MCP client launches it over stdio from paperpilot/mcp_servers.json.

This protocol layer validates input and routes to IndexManager. It does not
touch PyLate details directly.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from paperpilot.mcp_servers.colbert.index_manager import (
    IndexManager,
    IndexNotFoundError,
)

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


if __name__ == "__main__":
    _manager = IndexManager()
    mcp.run(transport="stdio")
