"""The analyze_paper_page agent tool: visual page understanding (book 4.4.1).

Text chunks lose figures, layouts, and table structure; this tool hands a
rendered page image to a VLM (mcp vlm server) and returns its answer. The
harness helpers are injected as callables, same as the memory/compute
tools, keeping this module free of research-agent imports.
"""
from __future__ import annotations

from typing import Callable

from langchain_core.tools import BaseTool, tool

_VLM_TOOL = "mcp__vlm__understand_paper_page"

_MESSAGE_TOOL_UNAVAILABLE = (
    "visual page analysis is unavailable in this run; answer from text "
    "evidence only and note the limitation"
)


def build_page_vision_tool(
    context,
    *,
    call_mcp_text: Callable[..., object],
    emit_tool_call: Callable[..., None],
    clip: Callable[[str], str],
    required_id: Callable[[object, str], str],
    require_agent_limit: Callable[[int, str], None],
    canonical_arxiv_id: Callable[[object, str], str],
) -> BaseTool:
    """Build the page-vision tool bound to one research run."""

    @tool("analyze_paper_page")
    def analyze_paper_page(external_id: str, page_num: int, query: str) -> str:
        """Analyze one page of a paper VISUALLY with a vision model.

        Use when the question is about a figure, diagram, table layout, or
        anything where seeing the page beats reading text chunks — e.g.
        "describe Figure 3", "what does the architecture diagram show",
        "read the numbers off Table 2". PRECONDITION: the paper must be
        downloaded (prepare_paper or the primary paper). This is slower
        than text retrieval (renders one page + vision call); prefer
        retrieve_paper_evidence for purely textual questions.

        Args:
            external_id: A downloaded paper's arXiv id, e.g. "1706.03762".
            page_num: 1-based page number (the first page is 1), e.g. 5.
            query: What to inspect on the page, phrased for a vision
                model, e.g. "describe the trend in Figure 2".

        Returns the vision model's plain-text answer about that page.
        """
        cleaned_query = required_id(query, "vision query")
        normalized_id = canonical_arxiv_id(external_id, "external_id")
        require_agent_limit(page_num, "page_num")
        emit_tool_call(
            context,
            stage="research",
            name="analyze_paper_page",
            arguments={
                "query": clip(cleaned_query),
                "external_id": normalized_id,
                "page_num": page_num,
            },
        )
        try:
            return str(
                call_mcp_text(
                    context,
                    name=_VLM_TOOL,
                    arguments={
                        "arxiv_id": normalized_id,
                        "page_num": page_num,
                        "query": cleaned_query,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            return f"{_MESSAGE_TOOL_UNAVAILABLE} ({type(exc).__name__})"

    return analyze_paper_page
