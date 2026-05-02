"""NetworkX citation graph manager for graph-mcp."""
from __future__ import annotations

import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

import networkx as nx

from paperpilot.mcp_servers.graph.ss_client import SSClient


class NodeNotFoundError(RuntimeError):
    """Requested paper is not present in the citation graph."""


class GraphManager:
    def __init__(
        self,
        graph_path: str | Path = "data/graph/citation_graph.pkl",
        ss_client: SSClient | None = None,
    ):
        self._graph_path = Path(graph_path)
        self._ss_client = ss_client or SSClient()
        self._graph = self._load()

    def build_graph(self, arxiv_ids: list[str]) -> dict:
        if not arxiv_ids or not all(isinstance(i, str) for i in arxiv_ids):
            raise ValueError("arxiv_ids must be a non-empty list[str]")

        skipped = [
            i
            for i in arxiv_ids
            if i in self._graph and self._graph.nodes[i].get("node_type") == "downloaded"
        ]
        to_fetch = [i for i in arxiv_ids if i not in skipped]
        missing: list[str] = []
        added_count = 0

        if to_fetch:
            papers = self._ss_client.fetch_papers_batch(to_fetch)
            for requested_id, paper in zip(to_fetch, papers):
                if paper is None:
                    missing.append(requested_id)
                    continue

                arxiv_id = self._paper_arxiv_id(paper) or requested_id
                if self._graph.nodes.get(arxiv_id, {}).get("node_type") != "downloaded":
                    added_count += 1
                self._upsert_node(arxiv_id, paper, "downloaded")

                for ref in paper.get("references") or []:
                    ref_id = self._paper_arxiv_id(ref)
                    if not ref_id:
                        continue
                    self._upsert_node(ref_id, ref, "neighbor")
                    self._graph.add_edge(arxiv_id, ref_id)

                for citation in paper.get("citations") or []:
                    citation_id = self._paper_arxiv_id(citation)
                    if not citation_id:
                        continue
                    self._upsert_node(citation_id, citation, "neighbor")
                    self._graph.add_edge(citation_id, arxiv_id)

        self._save()
        return {
            "added_count": added_count,
            "skipped_count": len(skipped),
            "missing": missing,
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
        }

    def get_neighbors(
        self, arxiv_id: str, direction: str = "both", limit: int = 10
    ) -> list[dict]:
        self._require_node(arxiv_id)
        if direction not in {"references", "citations", "both"}:
            raise ValueError("direction must be references, citations, or both")

        neighbors: list[tuple[str, str]] = []
        if direction in {"references", "both"}:
            neighbors.extend((node_id, "references") for node_id in self._graph.successors(arxiv_id))
        if direction in {"citations", "both"}:
            neighbors.extend((node_id, "citations") for node_id in self._graph.predecessors(arxiv_id))

        neighbors.sort(key=lambda item: (-self._node_year(item[0]), item[0]))
        return [
            {**self._node_dict(node_id), "edge": edge}
            for node_id, edge in neighbors[:limit]
        ]

    def get_shortest_path(self, from_id: str, to_id: str) -> dict:
        self._require_node(from_id)
        self._require_node(to_id)
        try:
            path = nx.shortest_path(self._graph, source=from_id, target=to_id)
        except nx.NetworkXNoPath:
            return {"path": [], "length": -1}

        return {
            "path": [
                {"arxiv_id": node_id, "title": self._graph.nodes[node_id].get("title", "")}
                for node_id in path
            ],
            "length": len(path) - 1,
        }

    def get_common_citations(self, arxiv_ids: list[str], top_k: int = 10) -> list[dict]:
        if len(arxiv_ids) < 2 or not all(isinstance(i, str) for i in arxiv_ids):
            raise ValueError("arxiv_ids must contain at least two str ids")

        refs: list[str] = []
        for arxiv_id in arxiv_ids:
            if arxiv_id not in self._graph:
                continue
            refs.extend(self._graph.successors(arxiv_id))

        counts = Counter(refs)
        candidates = [
            (node_id, count) for node_id, count in counts.items() if count >= 2
        ]
        candidates.sort(key=lambda item: (-item[1], -self._node_year(item[0]), item[0]))
        return [
            {**self._node_dict(node_id), "cited_by_count": count}
            for node_id, count in candidates[:top_k]
        ]

    def _load(self) -> nx.DiGraph:
        if not self._graph_path.exists():
            return nx.DiGraph()
        with self._graph_path.open("rb") as f:
            graph = pickle.load(f)
        if not isinstance(graph, nx.DiGraph):
            raise ValueError(f"invalid graph pickle: {self._graph_path}")
        return graph

    def _save(self) -> None:
        self._graph_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._graph_path.with_suffix(self._graph_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(self._graph, f)
        os.replace(tmp_path, self._graph_path)

    def _upsert_node(self, arxiv_id: str, paper: dict, node_type: str) -> None:
        current = self._graph.nodes.get(arxiv_id, {})
        if current.get("node_type") == "downloaded" and node_type == "neighbor":
            node_type = "downloaded"
        self._graph.add_node(
            arxiv_id,
            node_type=node_type,
            arxiv_id=arxiv_id,
            title=paper.get("title") or current.get("title") or "",
            year=paper.get("year") or current.get("year"),
            authors=self._authors(paper.get("authors")) or current.get("authors", []),
        )

    def _node_dict(self, node_id: str) -> dict:
        data = self._graph.nodes[node_id]
        return {
            "arxiv_id": node_id,
            "title": data.get("title", ""),
            "year": data.get("year"),
            "authors": data.get("authors", []),
        }

    def _require_node(self, arxiv_id: str) -> None:
        if arxiv_id not in self._graph:
            raise NodeNotFoundError("must call build_graph first or paper not in graph")

    def _node_year(self, node_id: str) -> int:
        year = self._graph.nodes[node_id].get("year")
        return year if isinstance(year, int) else -1

    @staticmethod
    def _paper_arxiv_id(paper: dict | None) -> str | None:
        if not paper:
            return None
        external_ids = paper.get("externalIds") or {}
        return external_ids.get("ArXiv")

    @staticmethod
    def _authors(authors: Iterable[dict | str] | None) -> list[str]:
        if not authors:
            return []
        result: list[str] = []
        for author in authors:
            if isinstance(author, str):
                result.append(author)
            elif isinstance(author, dict) and author.get("name"):
                result.append(author["name"])
        return result
