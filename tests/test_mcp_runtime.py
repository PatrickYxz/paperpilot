"""Process-local MCP runtime lifecycle tests."""
from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

from paperpilot.tools.mcp_client import MCPToolTimeout, MCPTransportError
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.tools.types import Tool


def _fake_tool(handler: Callable[[dict], object] | None = None) -> Tool:
    return Tool(
        name="mcp__fake__search",
        description="fake search",
        input_schema={"type": "object", "properties": {}},
        handler=handler or (lambda args: "result"),
    )


class FakeClient:
    def __init__(
        self,
        *,
        fail_start: bool = False,
        tool_error: Exception | None = None,
    ) -> None:
        self.fail_start = fail_start
        self.tool_error = tool_error
        self.start_count = 0
        self.close_count = 0

    def start(self) -> None:
        self.start_count += 1
        if self.fail_start:
            raise RuntimeError("startup failed")

    def list_tools(self) -> list[Tool]:
        if self.tool_error is not None:
            error = self.tool_error

            def fail(args):
                raise error

            return [_fake_tool(fail)]
        return [_fake_tool()]

    def close(self) -> None:
        self.close_count += 1


class FakeClientFactory:
    def __init__(self, *, fail_first_start: bool = False) -> None:
        self.fail_first_start = fail_first_start
        self.clients: list[FakeClient] = []

    def __call__(self) -> FakeClient:
        client = FakeClient(
            fail_start=self.fail_first_start and not self.clients,
        )
        self.clients.append(client)
        return client


def test_runtime_reuses_one_client_for_multiple_leases():
    factory = FakeClientFactory()
    runtime = MCPRuntime(factory)

    with runtime.lease_tools() as first:
        assert [tool.name for tool in first] == ["mcp__fake__search"]
    with runtime.lease_tools() as second:
        assert [tool.name for tool in second] == ["mcp__fake__search"]

    assert len(factory.clients) == 1
    assert factory.clients[0].start_count == 1
    assert factory.clients[0].close_count == 0


def test_runtime_invalidate_closes_client_and_next_lease_restarts():
    factory = FakeClientFactory()
    runtime = MCPRuntime(factory)
    with runtime.lease_tools():
        pass

    runtime.invalidate()
    with runtime.lease_tools():
        pass

    assert len(factory.clients) == 2
    assert factory.clients[0].close_count == 1
    assert factory.clients[1].start_count == 1


def test_runtime_close_is_idempotent():
    factory = FakeClientFactory()
    runtime = MCPRuntime(factory)
    with runtime.lease_tools():
        pass

    runtime.close()
    runtime.close()

    assert factory.clients[0].close_count == 1


def test_runtime_cleans_up_failed_start_and_can_retry():
    factory = FakeClientFactory(fail_first_start=True)
    runtime = MCPRuntime(factory)

    with pytest.raises(RuntimeError, match="startup failed"):
        with runtime.lease_tools():
            pass

    with runtime.lease_tools() as tools:
        assert [tool.name for tool in tools] == ["mcp__fake__search"]

    assert len(factory.clients) == 2
    assert factory.clients[0].close_count == 1
    assert factory.clients[1].start_count == 1


def test_runtime_discards_client_after_mcp_timeout():
    first = FakeClient(tool_error=MCPToolTimeout("tool timed out"))
    second = FakeClient()
    clients = iter([first, second])
    runtime = MCPRuntime(lambda: next(clients))

    with runtime.lease_tools() as tools:
        with pytest.raises(MCPToolTimeout, match="tool timed out"):
            tools[0].handler({})

    with runtime.lease_tools() as tools:
        assert tools[0].handler({}) == "result"

    assert first.close_count == 1
    assert second.start_count == 1


def test_runtime_discards_client_after_transport_failure():
    first = FakeClient(tool_error=MCPTransportError("stream closed"))
    second = FakeClient()
    clients = iter([first, second])
    runtime = MCPRuntime(lambda: next(clients))

    with runtime.lease_tools() as tools:
        with pytest.raises(MCPTransportError, match="stream closed"):
            tools[0].handler({})

    with runtime.lease_tools() as tools:
        assert tools[0].handler({}) == "result"

    assert first.close_count == 1
    assert second.start_count == 1


def test_runtime_close_waits_for_active_lease():
    factory = FakeClientFactory()
    runtime = MCPRuntime(factory)
    lease_entered = threading.Event()
    release_lease = threading.Event()
    close_finished = threading.Event()

    def hold_lease() -> None:
        with runtime.lease_tools():
            lease_entered.set()
            assert release_lease.wait(timeout=2)

    def close_runtime() -> None:
        runtime.close()
        close_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        lease_future = pool.submit(hold_lease)
        assert lease_entered.wait(timeout=1)
        close_future = pool.submit(close_runtime)
        assert not close_finished.wait(timeout=0.1)
        release_lease.set()
        lease_future.result(timeout=1)
        close_future.result(timeout=1)

    assert close_finished.is_set()
    assert factory.clients[0].close_count == 1
