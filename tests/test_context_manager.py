"""ContextManager tests."""
from __future__ import annotations

from paperpilot.core.adapter import Tool
from paperpilot.core.context_manager import ContextManager


def test_context_manager_counts_messages_system_and_tools():
    manager = ContextManager(
        window_tokens=100,
        expected_output_tokens=10,
        soft_ratio=0.5,
        hard_ratio=0.8,
        critical_ratio=0.95,
    )
    tool = Tool(
        name="search_user_document",
        description="search user document",
        input_schema={"type": "object"},
        handler=lambda args: "",
    )

    state = manager.inspect(
        system="system" * 10,
        tools=[tool],
        messages=[{"role": "user", "content": "x" * 120}],
    )

    assert state.estimated_tokens > 10
    assert state.soft_limit == 50
    assert state.hard_limit == 80
    assert state.critical_limit == 95


def test_context_state_marks_soft_and_critical_limits():
    manager = ContextManager(
        window_tokens=20,
        expected_output_tokens=1,
        soft_ratio=0.5,
        hard_ratio=0.75,
        critical_ratio=0.9,
    )

    state = manager.inspect(
        system="",
        tools=[],
        messages=[{"role": "user", "content": "x" * 80}],
    )

    assert state.over_soft_limit
    assert state.over_hard_limit
    assert state.over_critical_limit
    assert state.needs_compact
