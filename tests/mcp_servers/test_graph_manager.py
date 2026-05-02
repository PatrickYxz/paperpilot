"""GraphManager unit tests."""
from __future__ import annotations

import pickle

import pytest

from paperpilot.mcp_servers.graph.graph_manager import (
    GraphManager,
    NodeNotFoundError,
)


def _paper(arxiv_id: str, title: str, refs=None, cits=None) -> dict:
    return {
        "externalIds": {"ArXiv": arxiv_id},
        "title": title,
        "year": 2024,
        "authors": [{"name": "Ada"}, {"name": "Ben"}],
        "references": refs or [],
        "citations": cits or [],
    }


def _neighbor(arxiv_id: str, title: str, year: int = 2023) -> dict:
    return {
        "externalIds": {"ArXiv": arxiv_id},
        "title": title,
        "year": year,
        "authors": [{"name": "Cy"}],
    }


class FakeSSClient:
    def __init__(self, mapping: dict[str, dict | None]):
        self.mapping = mapping
        self.calls: list[list[str]] = []

    def fetch_papers_batch(self, arxiv_ids: list[str]) -> list[dict | None]:
        self.calls.append(list(arxiv_ids))
        return [self.mapping.get(i) for i in arxiv_ids]


def test_build_graph_adds_main_neighbors_edges(tmp_path):
    ss = FakeSSClient(
        {
            "2401.1": _paper(
                "2401.1",
                "Main",
                refs=[_neighbor("2301.1", "Ref")],
                cits=[_neighbor("2501.1", "Citation", year=2025)],
            )
        }
    )
    manager = GraphManager(tmp_path / "citation_graph.pkl", ss_client=ss)

    result = manager.build_graph(["2401.1"])

    assert result["added_count"] == 1
    assert result["skipped_count"] == 0
    assert result["missing"] == []
    assert result["total_nodes"] == 3
    assert result["total_edges"] == 2
    refs = manager.get_neighbors("2401.1", "references")
    cits = manager.get_neighbors("2401.1", "citations")
    assert refs[0]["arxiv_id"] == "2301.1"
    assert refs[0]["edge"] == "references"
    assert cits[0]["arxiv_id"] == "2501.1"
    assert cits[0]["edge"] == "citations"


def test_build_graph_skips_downloaded_without_fetch(tmp_path):
    ss = FakeSSClient({"2401.1": _paper("2401.1", "Main")})
    manager = GraphManager(tmp_path / "citation_graph.pkl", ss_client=ss)
    manager.build_graph(["2401.1"])

    result = manager.build_graph(["2401.1"])

    assert result["added_count"] == 0
    assert result["skipped_count"] == 1
    assert ss.calls == [["2401.1"]]


def test_build_graph_upgrades_neighbor(tmp_path):
    ss = FakeSSClient(
        {
            "2401.1": _paper("2401.1", "Main", refs=[_neighbor("2301.1", "Ref")]),
            "2301.1": _paper("2301.1", "Ref Full"),
        }
    )
    manager = GraphManager(tmp_path / "citation_graph.pkl", ss_client=ss)
    manager.build_graph(["2401.1"])

    result = manager.build_graph(["2301.1"])

    assert result["added_count"] == 1
    assert ss.calls == [["2401.1"], ["2301.1"]]
    assert manager._graph.nodes["2301.1"]["node_type"] == "downloaded"
    assert manager._graph.has_edge("2401.1", "2301.1")


def test_build_graph_reports_missing_and_validates_input(tmp_path):
    manager = GraphManager(
        tmp_path / "citation_graph.pkl", ss_client=FakeSSClient({"missing": None})
    )

    assert manager.build_graph(["missing"])["missing"] == ["missing"]
    with pytest.raises(ValueError):
        manager.build_graph([])
    with pytest.raises(ValueError):
        manager.build_graph(["ok", 1])  # type: ignore[list-item]


def test_pickle_persists_across_manager_instances(tmp_path):
    path = tmp_path / "citation_graph.pkl"
    manager = GraphManager(path, ss_client=FakeSSClient({"2401.1": _paper("2401.1", "Main")}))
    manager.build_graph(["2401.1"])

    loaded = GraphManager(path, ss_client=FakeSSClient({}))

    assert loaded.get_neighbors("2401.1") == []


def test_get_shortest_path(tmp_path):
    manager = GraphManager(
        tmp_path / "citation_graph.pkl",
        ss_client=FakeSSClient(
            {
                "a": _paper("a", "A", refs=[_neighbor("b", "B")]),
                "b": _paper("b", "B", refs=[_neighbor("c", "C")]),
            }
        ),
    )
    manager.build_graph(["a", "b"])

    result = manager.get_shortest_path("a", "c")

    assert result["length"] == 2
    assert [p["arxiv_id"] for p in result["path"]] == ["a", "b", "c"]
    assert manager.get_shortest_path("c", "a") == {"path": [], "length": -1}
    with pytest.raises(NodeNotFoundError):
        manager.get_shortest_path("a", "missing")


def test_get_common_citations(tmp_path):
    common = _neighbor("shared", "Shared", year=2020)
    manager = GraphManager(
        tmp_path / "citation_graph.pkl",
        ss_client=FakeSSClient(
            {
                "a": _paper("a", "A", refs=[common, _neighbor("x", "X")]),
                "b": _paper("b", "B", refs=[common, _neighbor("y", "Y")]),
                "c": _paper("c", "C", refs=[_neighbor("y", "Y")]),
            }
        ),
    )
    manager.build_graph(["a", "b", "c"])

    result = manager.get_common_citations(["a", "b", "c"], top_k=2)

    assert [r["arxiv_id"] for r in result] == ["y", "shared"]
    assert result[0]["cited_by_count"] == 2
    with pytest.raises(ValueError):
        manager.get_common_citations(["a"])


def test_get_neighbors_both_sorts_by_year_and_limits(tmp_path):
    manager = GraphManager(
        tmp_path / "citation_graph.pkl",
        ss_client=FakeSSClient(
            {
                "a": _paper(
                    "a",
                    "A",
                    refs=[
                        _neighbor("old", "Old", year=2020),
                        _neighbor("new", "New", year=2024),
                    ],
                    cits=[_neighbor("cite", "Citation", year=2025)],
                )
            }
        ),
    )
    manager.build_graph(["a"])

    result = manager.get_neighbors("a", "both", limit=2)

    assert [r["arxiv_id"] for r in result] == ["cite", "new"]


def test_get_neighbors_invalid_direction_and_missing_node(tmp_path):
    manager = GraphManager(
        tmp_path / "citation_graph.pkl", ss_client=FakeSSClient({"a": _paper("a", "A")})
    )
    manager.build_graph(["a"])

    with pytest.raises(ValueError):
        manager.get_neighbors("a", "sideways")
    with pytest.raises(NodeNotFoundError):
        manager.get_neighbors("missing")


def test_get_common_citations_skips_missing_inputs(tmp_path):
    common = _neighbor("shared", "Shared")
    manager = GraphManager(
        tmp_path / "citation_graph.pkl",
        ss_client=FakeSSClient(
            {
                "a": _paper("a", "A", refs=[common]),
                "b": _paper("b", "B", refs=[common]),
            }
        ),
    )
    manager.build_graph(["a", "b"])

    result = manager.get_common_citations(["a", "missing", "b"])

    assert result[0]["arxiv_id"] == "shared"


def test_build_graph_ignores_neighbors_without_arxiv_id(tmp_path):
    no_arxiv = {"externalIds": {"DOI": "10/example"}, "title": "No arxiv"}
    manager = GraphManager(
        tmp_path / "citation_graph.pkl",
        ss_client=FakeSSClient({"a": _paper("a", "A", refs=[no_arxiv])}),
    )

    result = manager.build_graph(["a"])

    assert result["total_nodes"] == 1
    assert result["total_edges"] == 0


def test_build_graph_handles_string_authors(tmp_path):
    paper = _paper("a", "A")
    paper["authors"] = ["Ada", "Ben"]
    manager = GraphManager(
        tmp_path / "citation_graph.pkl", ss_client=FakeSSClient({"a": paper})
    )

    manager.build_graph(["a"])

    assert manager._graph.nodes["a"]["authors"] == ["Ada", "Ben"]


def test_load_rejects_non_graph_pickle(tmp_path):
    path = tmp_path / "citation_graph.pkl"
    with path.open("wb") as f:
        pickle.dump({"not": "a graph"}, f)

    with pytest.raises(ValueError):
        GraphManager(path, ss_client=FakeSSClient({}))


def test_downloaded_node_is_not_downgraded_by_neighbor_upsert(tmp_path):
    manager = GraphManager(
        tmp_path / "citation_graph.pkl",
        ss_client=FakeSSClient(
            {
                "a": _paper("a", "A"),
                "b": _paper("b", "B", refs=[_neighbor("a", "A as neighbor")]),
            }
        ),
    )
    manager.build_graph(["a"])
    manager.build_graph(["b"])

    assert manager._graph.nodes["a"]["node_type"] == "downloaded"


def test_build_graph_saves_when_all_inputs_are_skipped(tmp_path):
    path = tmp_path / "citation_graph.pkl"
    manager = GraphManager(path, ss_client=FakeSSClient({"a": _paper("a", "A")}))
    manager.build_graph(["a"])
    before = path.stat().st_mtime_ns

    manager.build_graph(["a"])

    assert path.exists()
    assert path.stat().st_mtime_ns >= before


def test_get_shortest_path_missing_source_or_target(tmp_path):
    manager = GraphManager(
        tmp_path / "citation_graph.pkl", ss_client=FakeSSClient({"a": _paper("a", "A")})
    )
    manager.build_graph(["a"])

    with pytest.raises(NodeNotFoundError):
        manager.get_shortest_path("missing", "a")
    with pytest.raises(NodeNotFoundError):
        manager.get_shortest_path("a", "missing")
