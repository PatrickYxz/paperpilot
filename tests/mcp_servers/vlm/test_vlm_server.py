"""vlm-mcp protocol-layer tests. PageRenderer and QwenClient are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paperpilot.mcp_servers.vlm import server as vlm_server


@pytest.fixture
def mocks(monkeypatch):
    renderer = MagicMock()
    renderer.get_page_png.return_value = b"PNGBYTES"
    qwen = MagicMock()
    qwen.describe_page.return_value = "a beautiful figure"
    monkeypatch.setattr(vlm_server, "_renderer", renderer)
    monkeypatch.setattr(vlm_server, "_qwen", qwen)
    return {"renderer": renderer, "qwen": qwen}


def test_impl_routes_to_renderer_and_qwen(mocks):
    out = vlm_server._impl("1706.03762", 3, "describe Figure 1")
    assert out == "a beautiful figure"
    mocks["renderer"].get_page_png.assert_called_once_with("1706.03762", 3)
    mocks["qwen"].describe_page.assert_called_once_with(
        b"PNGBYTES",
        "describe Figure 1",
    )


def test_impl_rejects_empty_arxiv_id(mocks):
    with pytest.raises(ValueError, match="arxiv_id"):
        vlm_server._impl("", 1, "q")
    with pytest.raises(ValueError, match="arxiv_id"):
        vlm_server._impl("   ", 1, "q")
    mocks["renderer"].get_page_png.assert_not_called()


def test_impl_rejects_zero_or_negative_page_num(mocks):
    with pytest.raises(ValueError, match="page_num"):
        vlm_server._impl("p1", 0, "q")
    with pytest.raises(ValueError, match="page_num"):
        vlm_server._impl("p1", -3, "q")
    mocks["renderer"].get_page_png.assert_not_called()


def test_impl_rejects_empty_query(mocks):
    with pytest.raises(ValueError, match="query"):
        vlm_server._impl("p1", 1, "")
    with pytest.raises(ValueError, match="query"):
        vlm_server._impl("p1", 1, "   ")
    mocks["qwen"].describe_page.assert_not_called()
