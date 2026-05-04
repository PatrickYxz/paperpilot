"""colbert-mcp server.py protocol-layer tests.

IndexManager is mocked so these tests only cover routing and validation.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paperpilot.mcp_servers.colbert import server as colbert_server
from paperpilot.mcp_servers.colbert.index_manager import IndexNotFoundError


@pytest.fixture
def mock_manager(monkeypatch):
    manager = MagicMock()
    monkeypatch.setattr(colbert_server, "_manager", manager)
    return manager


def test_build_routes_to_manager(mock_manager):
    mock_manager.build.return_value = {
        "indexed_count": 2,
        "index_name": "paperpilot_current",
        "cached_papers": [],
        "fresh_papers": ["p1", "p2"],
    }
    docs = [
        {"paper_id": "p1", "text": "hello"},
        {"paper_id": "p2", "text": "world"},
    ]

    result = colbert_server._build_index_impl(docs)

    mock_manager.build.assert_called_once_with(docs)
    assert result["fresh_papers"] == ["p1", "p2"]


def test_build_empty_documents_raises_value_error(mock_manager):
    with pytest.raises(ValueError, match="must not be empty"):
        colbert_server._build_index_impl([])
    mock_manager.build.assert_not_called()


def test_build_missing_field_raises_value_error(mock_manager):
    bad = [{"paper_id": "p1"}]
    with pytest.raises(ValueError, match="paper_id.*text"):
        colbert_server._build_index_impl(bad)
    mock_manager.build.assert_not_called()


def test_search_passes_paper_id_to_manager(mock_manager):
    mock_manager.search.return_value = [
        {"paper_id": "p1", "chunk_text": "x", "score": 0.5}
    ]

    out = colbert_server._search_impl("q", paper_id="p1", top_k=3)

    mock_manager.search.assert_called_once_with("q", "p1", 3)
    assert out[0]["paper_id"] == "p1"


def test_search_index_missing_raises(mock_manager):
    mock_manager.search.side_effect = IndexNotFoundError("no index for 'p1'")
    with pytest.raises(IndexNotFoundError):
        colbert_server._search_impl("q", paper_id="p1", top_k=5)
