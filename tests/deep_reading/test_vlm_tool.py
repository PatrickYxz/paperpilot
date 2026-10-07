"""Page-vision agent tool tests with fakes (real VLM needs DashScope)."""
from __future__ import annotations

from unittest.mock import MagicMock

from paperpilot.deep_reading.research_agent import (
    _canonical_arxiv_id,
    _clip,
    _emit_tool_call,
    _require_agent_limit,
    _required_id,
)
from paperpilot.deep_reading.vlm_tool import build_page_vision_tool


def _tool(call_result="The figure shows attention scores rising."):
    context = MagicMock()
    context.user_id = "u1"
    calls = []

    def call_mcp_text(ctx, *, name, arguments):
        calls.append((name, arguments))
        return call_result

    tool = build_page_vision_tool(
        context,
        call_mcp_text=call_mcp_text,
        emit_tool_call=_emit_tool_call,
        clip=_clip,
        required_id=_required_id,
        require_agent_limit=_require_agent_limit,
        canonical_arxiv_id=_canonical_arxiv_id,
    )
    return tool, calls


def test_invokes_vlm_server_and_returns_text():
    tool, calls = _tool()
    out = tool.invoke({"external_id": "1706.03762", "page_num": 5,
                       "query": "describe Figure 3"})
    assert "attention scores" in out
    assert calls[0][0] == "mcp__vlm__understand_paper_page"
    assert calls[0][1]["page_num"] == 5


def test_description_states_when_to_prefer_text_retrieval():
    description = _tool()[0].description
    assert "VISUALLY" in description
    assert "prefer" in description and "retrieve_paper_evidence" in description


def test_failure_returns_fallback_not_exception():
    def boom(*a, **k):
        raise RuntimeError("dashscope quota")

    context = MagicMock()
    tool = build_page_vision_tool(
        context,
        call_mcp_text=boom,
        emit_tool_call=lambda *a, **k: None,
        clip=lambda v: v,
        required_id=lambda v, f: str(v),
        require_agent_limit=lambda v, f: None,
        canonical_arxiv_id=lambda v, f: str(v),
    )
    out = tool.invoke({"external_id": "1706.03762", "page_num": 1, "query": "q"})
    assert "unavailable" in out and "RuntimeError" in out
