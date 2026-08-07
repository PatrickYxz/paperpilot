"""arxiv-mcp: arXiv 论文搜索 server。Day 5 起。

启动:python -m paperpilot.mcp_servers.arxiv
通过 stdio 被 paperpilot.tools.mcp_client 拉起,manifest 见
paperpilot/mcp_servers.json。
"""
from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from pathlib import Path

import arxiv
import certifi
import fitz
from mcp.server.fastmcp import FastMCP

from paperpilot.papers import search_arxiv_candidates

mcp = FastMCP("arxiv")
_client = arxiv.Client(page_size=50, delay_seconds=3)


class ArxivNotFoundError(RuntimeError):
    """arxiv 返回 404 / 无效 id。"""


class PDFParseError(RuntimeError):
    """pymupdf 解析 PDF 字节失败。"""


_PAPERS_DIR = Path("data/papers")


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
        max_results: 返回结果数,默认 10,上限 20。
        category: 限定 arXiv 类目,如 "cs.CL"/"cs.LG"。None 不限。
        sort_by: "relevance"(默认) / "submittedDate" / "lastUpdatedDate"。

    Returns:
        每篇论文一段,字段:arxiv_id / title / authors / published /
        primary_category / pdf_url / abstract,多篇用 --- 分隔。
    """
    max_results = min(max_results, 20)
    full_q = f"({query}) AND cat:{category}" if category else query
    sort_enum = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }.get(sort_by, arxiv.SortCriterion.Relevance)
    parts = [
        f"arxiv_id: {candidate.external_id}\n"
        f"title: {candidate.title}\n"
        f"authors: {', '.join(candidate.authors)}\n"
        f"published: {candidate.published}\n"
        f"primary_category: {candidate.primary_category}\n"
        f"pdf_url: {candidate.pdf_url}\n"
        f"abstract: {candidate.abstract or ''}"
        for candidate in search_arxiv_candidates(
            full_q, max_results, client=_client, sort_by=sort_enum
        )
    ]
    return "\n---\n".join(parts) if parts else f"No papers found for: {full_q}"



@mcp.tool()
def download_paper(arxiv_id: str) -> dict:
    """下载 arXiv 论文 PDF 并提取 text。命中本地缓存时跳过下载。

    Args:
        arxiv_id: arXiv 标识符,如 "2401.12345" 或带版本 "2401.12345v2"。
            旧式 "cs.AI/0501001" 也允许,但调用方需保证 id 不含路径分隔符以外的特殊字符。

    Returns:
        dict 含 paper_id(原样返回)、text(纯文本,空白未规范化)。
    """
    return _download_paper_impl(arxiv_id)


def _download_paper_impl(arxiv_id: str) -> dict:
    """download_paper 的纯函数实现,绕过 FastMCP 装饰器,方便单测调用。"""
    cache_path = _PAPERS_DIR / f"{arxiv_id}.txt"
    if cache_path.exists():
        return {"paper_id": arxiv_id, "text": cache_path.read_text(encoding="utf-8")}

    pdf_bytes = _fetch_pdf(arxiv_id)
    text = _extract_text(arxiv_id, pdf_bytes)

    _PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return {"paper_id": arxiv_id, "text": text}


def _fetch_pdf(arxiv_id: str) -> bytes:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ArxivNotFoundError(f"arxiv paper not found: {arxiv_id}") from e
        raise


def _extract_text(arxiv_id: str, pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    except PDFParseError:
        raise
    except Exception as e:
        raise PDFParseError(f"failed to parse PDF for {arxiv_id}: {e}") from e


if __name__ == "__main__":
    mcp.run(transport="stdio")
