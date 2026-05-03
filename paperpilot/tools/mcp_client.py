"""MCPClient: 同步 facing 的 MCP 客户端,封装后台 asyncio 线程 + N 个 session。

设计依据:docs/superpowers/specs/2026-04-25-mcp-layer-architecture-design.md (§4)
- agent_loop 保持同步;async 复杂度关在本文件
- 每个 MCP tool 包成 paperpilot.core.adapter.Tool,handler 是 sync 闭包
- 错误:启动 hard-fail / 运行 soft-fail / 180s 超时为边界
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from paperpilot.core.adapter import Tool

MCP_TOOL_TIMEOUT = int(os.environ.get("MCP_TOOL_TIMEOUT", 180))
# colbert-mcp 启动时需加载 PyLate 模型 (~15s on CPU),因此 initialize 超时设 60s
_INITIALIZE_TIMEOUT = int(os.environ.get("MCP_INITIALIZE_TIMEOUT", 60))


class MCPStartupError(RuntimeError):
    """启动期任何失败(manifest 错 / spawn 失败 / initialize 超时)。"""


class MCPToolError(RuntimeError):
    """运行期 tool 调用失败(server 端 isError=True 或 RPC 错)。"""


class MCPToolTimeout(RuntimeError):
    """运行期 tool 调用超过 MCP_TOOL_TIMEOUT。"""


class MCPClient:
    def __init__(self, manifest_path: str | Path):
        self._manifest_path = Path(manifest_path)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[Tool] = []
        self._started = False
        self._closed = False

    def start(self) -> None:
        """同步阻塞,直到所有 server 起好。任何失败 raise MCPStartupError。"""
        if self._started:
            raise RuntimeError("MCPClient.start() called twice")

        loop_ready = threading.Event()

        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            loop_ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_runner, daemon=True, name="mcp-loop")
        self._thread.start()
        loop_ready.wait(timeout=5)
        if not self._loop or not self._loop.is_running():
            time.sleep(0.05)
        if not self._loop or not self._loop.is_running():
            raise MCPStartupError("background event loop failed to start")

        try:
            fut = asyncio.run_coroutine_threadsafe(self._async_init(), self._loop)
            fut.result()
        except Exception as e:
            self.close()
            if isinstance(e, MCPStartupError):
                raise
            raise MCPStartupError(str(e)) from e

        self._started = True

    async def _async_init(self) -> None:
        if not self._manifest_path.exists():
            raise MCPStartupError(f"manifest not found: {self._manifest_path}")

        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise MCPStartupError(f"manifest invalid JSON: {e}") from e

        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            raise MCPStartupError("manifest missing 'mcpServers' dict")

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for name, cfg in servers.items():
            if "__" in name:
                raise MCPStartupError(f"server name '{name}' must not contain '__'")
            if "command" not in cfg or "args" not in cfg:
                raise MCPStartupError(f"server '{name}' missing command/args")

            env = {**os.environ, **cfg.get("env", {})}
            params = StdioServerParameters(
                command=_resolve_command(cfg["command"]),
                args=list(cfg["args"]),
                env=env,
            )
            try:
                stdio = await self._exit_stack.enter_async_context(stdio_client(params))
                read, write = stdio
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await asyncio.wait_for(
                    session.initialize(), timeout=_INITIALIZE_TIMEOUT
                )
            except Exception as e:
                raise MCPStartupError(f"failed to start server '{name}': {e}") from e

            tools_resp = await session.list_tools()
            self._sessions[name] = session
            for t in tools_resp.tools:
                self._tools.append(Tool(
                    name=f"mcp__{name}__{t.name}",
                    description=t.description or "",
                    input_schema=t.inputSchema,
                    handler=self._make_handler(name, t.name),
                ))

    def _make_handler(self, server: str, tool: str):
        def handler(args: dict) -> str:
            assert self._loop is not None
            coro = self._sessions[server].call_tool(tool, args)
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            try:
                result = fut.result(timeout=MCP_TOOL_TIMEOUT)
            except concurrent.futures.TimeoutError as e:
                fut.cancel()
                raise MCPToolTimeout(
                    f"mcp__{server}__{tool} > {MCP_TOOL_TIMEOUT}s"
                ) from e
            text = "\n".join(
                b.text for b in result.content if hasattr(b, "text")
            )
            if getattr(result, "isError", False):
                raise MCPToolError(f"mcp__{server}__{tool} failed: {text}")
            return text
        return handler

    def list_tools(self) -> list[Tool]:
        if not self._started:
            raise RuntimeError("MCPClient.list_tools() before start()")
        return list(self._tools)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop and self._loop.is_running() and self._exit_stack:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._exit_stack.__aexit__(None, None, None), self._loop,
                )
                fut.result(timeout=5)
            except Exception:
                pass
        self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)


def _resolve_command(command: str) -> str:
    if command.lower() in {"python", "python.exe"}:
        return sys.executable
    return command
