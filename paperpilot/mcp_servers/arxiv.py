"""arxiv-mcp: arXiv 论文搜索 server。Day 5 起。

启动:python -m paperpilot.mcp_servers.arxiv
通过 stdio 被 paperpilot.tools.mcp_client 拉起,manifest 见
paperpilot/mcp_servers.json。
"""
from __future__ import annotations

import arxiv
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arxiv")
_client = arxiv.Client(page_size=50, delay_seconds=3)


@mcp.tool()
def search_papers(
    query: str,
    max_results: int = 10,
    category: str | None = None,
    sort_by: str = "relevance",
) -> str:
    """按关键词搜索 arXiv 论文,返回元数据列表。

    Args:
        query: 搜索关键词,如 "LoRA fine-tuning"。支持 arXiv 查询语法:
            au:作者名 / ti:标题 / abs:摘要 / cat:类目。
        max_results: 返回结果数,默认 10,上限 50。
        category: 限定 arXiv 类目,如 "cs.CL"/"cs.LG"。None 不限。
        sort_by: "relevance"(默认) / "submittedDate" / "lastUpdatedDate"。

    Returns:
        每篇论文一段,字段:arxiv_id / title / authors / published /
        primary_category / pdf_url / abstract,多篇用 --- 分隔。
    """
    max_results = min(max_results, 50)
    full_q = f"({query}) AND cat:{category}" if category else query
    sort_enum = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }.get(sort_by, arxiv.SortCriterion.Relevance)

    search = arxiv.Search(
        query=full_q, max_results=max_results, sort_by=sort_enum,
    )
    parts = [
        f"arxiv_id: {r.get_short_id()}\n"
        f"title: {r.title}\n"
        f"authors: {', '.join(a.name for a in r.authors)}\n"
        f"published: {r.published.date()}\n"
        f"primary_category: {r.primary_category}\n"
        f"pdf_url: {r.pdf_url}\n"
        f"abstract: {r.summary.strip()}"
        for r in _client.results(search)
    ]
    return "\n---\n".join(parts) if parts else f"No papers found for: {full_q}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
