"""graph-mcp integration tests through MCPClient."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from paperpilot.tools.mcp_client import MCPClient, MCPStartupError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def _make_manifest(tmp_path: Path, graph_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "graph": {
                        "command": sys.executable,
                        "args": ["-m", "paperpilot.mcp_servers.graph.server"],
                        "env": {
                            "PAPERPILOT_SS_FIXTURE_DIR": str(FIXTURE_DIR),
                            "PAPERPILOT_GRAPH_PATH": str(graph_path),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _parse_tool_result(text: str):
    s = text.strip()
    if not s:
        return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        results = []
        idx = 0
        while idx < len(s):
            obj, end = decoder.raw_decode(s, idx)
            results.append(obj)
            idx = end + len(s[end:]) - len(s[end:].lstrip())
        return results


@pytest.mark.slow
def test_build_and_query(tmp_path):
    graph_path = tmp_path / "citation_graph.pkl"
    client = MCPClient(_make_manifest(tmp_path, graph_path))
    client.start()
    try:
        tools = {t.name: t for t in client.list_tools()}
        build = tools["mcp__graph__build_graph"]
        neighbors = tools["mcp__graph__get_neighbors"]
        shortest = tools["mcp__graph__get_shortest_path"]
        common = tools["mcp__graph__get_common_citations"]

        stats = json.loads(
            build.handler({"arxiv_ids": ["2401.00001", "2401.00002"]})
        )
        assert stats["added_count"] == 2
        assert stats["total_nodes"] >= 2
        assert graph_path.exists()

        refs = _parse_tool_result(
            neighbors.handler(
                {"arxiv_id": "2401.00002", "direction": "references", "limit": 5}
            )
        )
        assert isinstance(refs, list)
        assert refs
        ref_id = refs[0]["arxiv_id"]

        path = json.loads(
            shortest.handler({"from_id": "2401.00002", "to_id": ref_id})
        )
        assert path["length"] == 1
        assert [p["arxiv_id"] for p in path["path"]] == ["2401.00002", ref_id]

        common_refs = _parse_tool_result(
            common.handler({"arxiv_ids": ["2401.00001", "2401.00002"], "top_k": 3})
        )
        assert isinstance(common_refs, list)
    finally:
        client.close()


@pytest.mark.slow
def test_pickle_persists_across_restart(tmp_path):
    graph_path = tmp_path / "citation_graph.pkl"
    manifest = _make_manifest(tmp_path, graph_path)
    client = MCPClient(manifest)
    client.start()
    try:
        build = next(t for t in client.list_tools() if t.name == "mcp__graph__build_graph")
        build.handler({"arxiv_ids": ["2401.00002"]})
    finally:
        client.close()

    restarted = MCPClient(manifest)
    restarted.start()
    try:
        neighbors = next(
            t for t in restarted.list_tools() if t.name == "mcp__graph__get_neighbors"
        )
        refs = _parse_tool_result(
            neighbors.handler(
                {"arxiv_id": "2401.00002", "direction": "references", "limit": 1}
            )
        )
        assert refs
    finally:
        restarted.close()


@pytest.mark.slow
def test_corrupt_pickle_startup_fail(tmp_path):
    graph_path = tmp_path / "citation_graph.pkl"
    graph_path.write_bytes(b"not a pickle")

    client = MCPClient(_make_manifest(tmp_path, graph_path))

    with pytest.raises(MCPStartupError):
        client.start()
