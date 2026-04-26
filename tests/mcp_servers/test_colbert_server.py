"""colbert-mcp server.py 协议层单测。
mock IndexManager (避免拉 ColBERT 模型),只验证路由 + 输入校验。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paperpilot.mcp_servers.colbert import server as colbert_server
from paperpilot.mcp_servers.colbert.index_manager import IndexNotFoundError


@pytest.fixture
def mock_manager(monkeypatch):
    m = MagicMock()
    monkeypatch.setattr(colbert_server, "_manager", m)
    return m


def test_build_routes_to_manager(mock_manager):
    """build_index 调用透传给 IndexManager.build。"""
    mock_manager.build.return_value = {"indexed_count": 2, "index_name": "paperpilot_current"}
    docs = [
        {"paper_id": "p1", "text": "hello"},
        {"paper_id": "p2", "text": "world"},
    ]

    result = colbert_server._build_index_impl(docs)

    mock_manager.build.assert_called_once_with(docs)
    assert result == {"indexed_count": 2, "index_name": "paperpilot_current"}


def test_search_index_missing_raises(mock_manager):
    """IndexManager.search 抛 IndexNotFoundError 时,server 透传同类异常。"""
    mock_manager.search.side_effect = IndexNotFoundError("must call build_index first")

    with pytest.raises(IndexNotFoundError):
        colbert_server._search_impl("q", 5)


def test_build_empty_documents_raises_value_error(mock_manager):
    """空 documents → ValueError,不调 manager。"""
    with pytest.raises(ValueError, match="must not be empty"):
        colbert_server._build_index_impl([])
    mock_manager.build.assert_not_called()


def test_build_missing_field_raises_value_error(mock_manager):
    """document 缺 text 字段 → ValueError。"""
    bad = [{"paper_id": "p1"}]   # 缺 text
    with pytest.raises(ValueError, match="paper_id.*text"):
        colbert_server._build_index_impl(bad)
    mock_manager.build.assert_not_called()
