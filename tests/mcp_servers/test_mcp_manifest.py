import json
from pathlib import Path


def test_colbert_server_runs_with_huggingface_offline_cache() -> None:
    manifest = json.loads(
        Path("paperpilot/mcp_servers.json").read_text(encoding="utf-8")
    )

    colbert = manifest["mcpServers"]["colbert"]

    assert colbert["env"]["HF_HUB_OFFLINE"] == "1"
