"""vlm-mcp: Qwen-VL-Max page-level visual understanding.

Run: python -m paperpilot.mcp_servers.vlm.server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from paperpilot.mcp_servers.vlm.page_renderer import PageRenderer
from paperpilot.mcp_servers.vlm.qwen_client import QwenClient

mcp = FastMCP("vlm")
_renderer: PageRenderer | None = None
_qwen: QwenClient | None = None


@mcp.tool()
def understand_paper_page(arxiv_id: str, page_num: int, query: str) -> str:
    """View one page of an arXiv paper as an image and answer a visual query.

    Args:
        arxiv_id: arXiv id, e.g. "1706.03762".
        page_num: 1-based page index. The first page is 1.
        query: What to inspect, e.g. "describe Figure 3" or
            "what does the architecture diagram show".

    Returns:
        Plain text description from Qwen-VL-Max.
    """
    return _impl(arxiv_id, page_num, query)


def _impl(arxiv_id: str, page_num: int, query: str) -> str:
    if not isinstance(arxiv_id, str) or not arxiv_id.strip():
        raise ValueError("arxiv_id must be a non-empty string")
    if not isinstance(page_num, int) or page_num < 1:
        raise ValueError(f"page_num must be int >= 1, got {page_num!r}")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    assert _renderer is not None, "PageRenderer not initialized"
    assert _qwen is not None, "QwenClient not initialized"

    png_bytes = _renderer.get_page_png(arxiv_id, page_num)
    return _qwen.describe_page(png_bytes, query)


if __name__ == "__main__":
    _renderer = PageRenderer()
    _qwen = QwenClient()
    mcp.run(transport="stdio")
