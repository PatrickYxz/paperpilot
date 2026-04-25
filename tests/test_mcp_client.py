"""MCPClient 单元测试。用 tests/fixtures/echo_server.py 做对端。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperpilot.tools.mcp_client import MCPClient

FIXTURE = Path(__file__).parent / "fixtures" / "echo_server.py"


def _make_manifest(tmp_path: Path) -> Path:
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({
        "mcpServers": {
            "echo": {"command": "python", "args": [str(FIXTURE)]}
        }
    }))
    return m


def test_namespacing_and_listing(tmp_path):
    """tool 名加 mcp__echo__ 前缀;list_tools 返回正确数量。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        names = {t.name for t in c.list_tools()}
        assert "mcp__echo__echo" in names
        assert "mcp__echo__boom" in names
    finally:
        c.close()


def test_tool_call_roundtrip(tmp_path):
    """handler 同步调用穿过线程边界,返回正确结果。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        echo_tool = next(t for t in c.list_tools() if t.name == "mcp__echo__echo")
        assert echo_tool.handler({"text": "hello"}).strip() == "hello"
    finally:
        c.close()


def test_server_error_raises_MCPToolError(tmp_path):
    """server 端 raise → CallToolResult.isError=True → handler 抛 MCPToolError。
       agent_loop 现有 except Exception 会接走,转成 is_error tool_result。"""
    from paperpilot.tools.mcp_client import MCPToolError
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        boom = next(t for t in c.list_tools() if t.name == "mcp__echo__boom")
        with pytest.raises(MCPToolError, match="intentional failure"):
            boom.handler({})
    finally:
        c.close()
