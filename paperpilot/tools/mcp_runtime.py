"""Long-lived, process-local MCP client runtime."""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from paperpilot.core.adapter import Tool
from paperpilot.tools.mcp_client import (
    MCPClient,
    MCPToolTimeout,
    MCPTransportError,
)

DEFAULT_MANIFEST_PATH = Path(__file__).parents[1] / "mcp_servers.json"


class MCPClientLike(Protocol):
    def start(self) -> None: ...

    def list_tools(self) -> list[Tool]: ...

    def close(self) -> None: ...


class MCPRuntime:
    """Reuse one MCP client while serializing task-level tool leases."""

    def __init__(
        self,
        client_factory: Callable[[], MCPClientLike] | None = None,
    ) -> None:
        self._client_factory = client_factory or (
            lambda: MCPClient(DEFAULT_MANIFEST_PATH)
        )
        self._client: MCPClientLike | None = None
        self._lifecycle_lock = threading.RLock()
        self._lease_lock = threading.Lock()
        self._lease_unhealthy = False

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._client is not None:
                return
            client = self._client_factory()
            try:
                client.start()
            except Exception:
                client.close()
                raise
            self._client = client

    @contextmanager
    def lease_tools(self) -> Iterator[list[Tool]]:
        with self._lease_lock:
            self.start()
            assert self._client is not None
            self._lease_unhealthy = False
            tools = [self._guard_tool(tool) for tool in self._client.list_tools()]
            try:
                yield tools
            finally:
                if self._lease_unhealthy:
                    self._invalidate_client()

    def invalidate(self) -> None:
        with self._lease_lock:
            self._invalidate_client()

    def _invalidate_client(self) -> None:
        with self._lifecycle_lock:
            client, self._client = self._client, None
            if client is not None:
                client.close()

    def close(self) -> None:
        self.invalidate()

    def _guard_tool(self, tool: Tool) -> Tool:
        def handler(args: dict):
            try:
                return tool.handler(args)
            except (
                MCPToolTimeout,
                MCPTransportError,
                ConnectionError,
                EOFError,
                OSError,
            ):
                self._lease_unhealthy = True
                raise

        return Tool(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            handler=handler,
        )
