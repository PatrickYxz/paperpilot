"""Real invocation of the search_user_memory agent tool.

Invokes through the same harness helpers the research agent injects,
guarding against silent import loss that module-level checks cannot catch.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from paperpilot.user_memory.agent_tool import build_user_memory_tool
from paperpilot.deep_reading.research_agent import _emit_tool_call, _clip, _required_id
from paperpilot.web.store.records import UserMemoryRecord


def _tool(context):
    return build_user_memory_tool(
        context,
        emit_tool_call=_emit_tool_call,
        required_id=_required_id,
        clip=_clip,
    )


def _context(records, summaries=()):
    context = MagicMock()
    context.user_id = "user-1"
    context.task_store.list_user_memories.return_value = records
    context.task_store.list_user_turn_summaries.return_value = list(summaries)
    context.context_management.enabled = False
    return context


def _record(memory_id="m1", content="User works on distillation."):
    return UserMemoryRecord(
        memory_id=memory_id,
        user_id="user-1",
        kind="fact",
        content=content,
        context={},
        source_conversation_id="conv-1",
        source_task_id="task-1",
        source_message_id="msg-1",
        support_span="span",
        created_at="2026-10-01T00:00:00Z",
        status="active",
    )


def test_tool_invoke_with_no_memories():
    tool = _tool(_context([]))
    assert "No stored user memories" in tool.invoke({"query": "anything"})


def test_tool_invoke_returns_matching_memory_and_notice():
    tool = _tool(
        _context([_record(content="User works on distillation.")])
    )
    out = tool.invoke({"query": "distillation"})
    assert "User works on distillation." in out
    assert "not instructions" in out
    assert "conv-1" in out


def test_tool_invoke_includes_conversation_digest_hits():
    class _Summary:
        user_message_id = "msg-9"
        conversation_id = "conv-9"
        question = "How does LoRA rank affect accuracy?"
        narrative = "Found rank 8 sufficient."
        created_at = "2026-09-01T00:00:00Z"

    tool = _tool(_context([], [_Summary()]))
    out = tool.invoke({"query": "lora rank"})
    assert "conv-9" in out
    assert "rank 8" in out


def test_tool_invoke_survives_store_failure():
    context = _context([])
    context.task_store.list_user_memories.side_effect = RuntimeError("db down")
    tool = _tool(context)
    assert "failed" in tool.invoke({"query": "q"})


def test_tool_renders_card_labels():
    record = _record()
    object.__setattr__(
        record, "context",
        {"topic": "research_focus", "subject": "user", "backstory": "stated when asking papers"},
    )
    tool = _tool(_context([record]))
    out = tool.invoke({"query": "distillation"})
    assert "[fact|research_focus|user]" in out
    assert "stated when asking papers" in out
