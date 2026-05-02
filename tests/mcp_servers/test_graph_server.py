"""graph-mcp protocol layer tests."""
from __future__ import annotations

import pytest

from paperpilot.mcp_servers.graph import server
from paperpilot.mcp_servers.graph.graph_manager import NodeNotFoundError


class FakeGraphManager:
    def build_graph(self, arxiv_ids):
        return {"added_count": len(arxiv_ids), "skipped_count": 0, "missing": []}

    def get_neighbors(self, arxiv_id, direction="both", limit=10):
        return [{"arxiv_id": arxiv_id, "edge": direction, "limit": limit}]

    def get_shortest_path(self, from_id, to_id):
        return {"path": [{"arxiv_id": from_id}, {"arxiv_id": to_id}], "length": 1}

    def get_common_citations(self, arxiv_ids, top_k=10):
        return [{"arxiv_id": "shared", "cited_by_count": len(arxiv_ids), "top_k": top_k}]


def test_build_graph_impl(monkeypatch):
    monkeypatch.setattr(server, "_manager", FakeGraphManager())

    result = server._build_graph_impl(["2401.00001", "2401.00002"])

    assert result["added_count"] == 2


def test_get_neighbors_impl(monkeypatch):
    monkeypatch.setattr(server, "_manager", FakeGraphManager())

    result = server._get_neighbors_impl("2401.00001", "references", 3)

    assert result == [{"arxiv_id": "2401.00001", "edge": "references", "limit": 3}]


def test_get_shortest_path_impl(monkeypatch):
    monkeypatch.setattr(server, "_manager", FakeGraphManager())

    result = server._get_shortest_path_impl("a", "b")

    assert result["length"] == 1


def test_get_common_citations_impl(monkeypatch):
    monkeypatch.setattr(server, "_manager", FakeGraphManager())

    result = server._get_common_citations_impl(["a", "b"], 5)

    assert result == [{"arxiv_id": "shared", "cited_by_count": 2, "top_k": 5}]


def test_impl_requires_initialized_manager(monkeypatch):
    monkeypatch.setattr(server, "_manager", None)

    with pytest.raises(RuntimeError):
        server._build_graph_impl(["2401.00001"])


def test_impl_propagates_manager_errors(monkeypatch):
    class RaisingManager(FakeGraphManager):
        def get_neighbors(self, arxiv_id, direction="both", limit=10):
            raise NodeNotFoundError("missing")

    monkeypatch.setattr(server, "_manager", RaisingManager())

    with pytest.raises(NodeNotFoundError):
        server._get_neighbors_impl("missing")
