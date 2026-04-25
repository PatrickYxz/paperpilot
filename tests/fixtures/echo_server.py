"""测试专用微型 MCP server。

只供 tests/test_mcp_client.py 用作对端,不进生产路径。
启动:python tests/fixtures/echo_server.py(stdio 模式,等待父进程)。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def echo(text: str) -> str:
    """原样返回输入文本。供测试 happy path。

    Args:
        text: 要回显的文本。
    """
    return text


@mcp.tool()
def boom() -> str:
    """故意抛异常,供测试 isError 错误传播路径。"""
    raise RuntimeError("intentional failure for testing")


if __name__ == "__main__":
    mcp.run(transport="stdio")
