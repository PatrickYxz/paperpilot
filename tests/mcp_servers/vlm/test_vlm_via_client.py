"""vlm-mcp integration test (slow).

Runs a real vlm-mcp process and calls real DashScope. Local command:
pytest -m slow tests/mcp_servers/vlm/test_vlm_via_client.py -v
Requires DASHSCOPE_API_KEY in environment or project .env.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperpilot.tools.mcp_client import MCPClient


def _make_manifest(tmp_path: Path) -> Path:
    """Generate a manifest with only the vlm server for test isolation."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "mcpServers": {
                "vlm": {
                    "command": "python",
                    "args": ["-m", "paperpilot.mcp_servers.vlm.server"],
                }
            }
        }),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.slow
def test_understand_first_page_of_attention_paper(tmp_path):
    """Real vlm-mcp process + real DashScope call."""
    client = MCPClient(_make_manifest(tmp_path))
    client.start()
    try:
        tool = next(
            t for t in client.list_tools()
            if t.name == "mcp__vlm__understand_paper_page"
        )
        result = tool.handler({
            "arxiv_id": "1706.03762",
            "page_num": 3,
            "query": "describe the model architecture diagram",
        })
        assert isinstance(result, str)
        assert len(result) > 50, f"too short for a real description: {result!r}"
        keywords = ("attention", "encoder", "decoder", "transformer")
        assert any(kw in result.lower() for kw in keywords), (
            f"none of {keywords} in result: {result!r}"
        )
    finally:
        client.close()
