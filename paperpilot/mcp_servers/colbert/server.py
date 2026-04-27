"""colbert-mcp: ColBERT 段落级语义检索 server。Day 6 起。

启动:python -m paperpilot.mcp_servers.colbert.server
通过 stdio 被 paperpilot.tools.mcp_client 拉起,manifest 见 paperpilot/mcp_servers.json。

协议层职责: 输入校验 + 路由到 IndexManager。**不接触 PyLate**。
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
    """对一组论文全文建立 ColBERT 索引(覆盖前一次)。

    调用示例: build_index(documents=[{"paper_id": "2401.12345", "text": "...全文..."}])
    download_paper 的返回值 (含 paper_id 和 text) 可直接封装进列表传入。

    Args:
        documents: list,每项 dict 含 paper_id (str) 与 text (str) 字段。
            非空,字段缺失会抛 ValueError。
            示例: [{"paper_id": "2401.12345", "text": "论文全文内容..."}]

    Returns:
        dict 含 indexed_count 与 index_name="paperpilot_current"。
    """
    return _build_index_impl(documents)


def _build_index_impl(documents: list[dict]) -> dict:
    if not documents:
        raise ValueError("documents must not be empty")
    for d in documents:
        if not isinstance(d, dict) or "paper_id" not in d or "text" not in d:
            raise ValueError(
                f"each document must be dict with 'paper_id' and 'text': {d!r}"
            )
    assert _manager is not None, "IndexManager not initialized"
    return _manager.build(documents)


@mcp.tool()
def search(query: str, top_k: int = 5) -> list[dict]:
    """在当前 ColBERT 索引上查询 top-k 段落。

    Args:
        query: 自然语言查询。
        top_k: 返回的段落数,默认 5。

    Returns:
        list,每项 dict 含 paper_id (str)、chunk_text (str)、score (float)。
        若索引不存在(未先调 build_index)则抛 IndexNotFoundError。
    """
    return _search_impl(query, top_k)


def _search_impl(query: str, top_k: int) -> list[dict]:
    assert _manager is not None, "IndexManager not initialized"
    return _manager.search(query, top_k)


if __name__ == "__main__":
    _manager = IndexManager()
    mcp.run(transport="stdio")
