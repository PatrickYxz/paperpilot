"""ConversationSession tests."""
from __future__ import annotations

import json
from contextlib import contextmanager

from paperpilot.conversation import ConversationSession, run
from paperpilot.bulk_input import BulkPaperInputDetector
from paperpilot.core.adapter import ParsedResponse, Tool
from paperpilot.document_store import DocumentStore
from paperpilot.session_store import SessionStore


class FakeMcp:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class FakeClient:
    def __init__(self, text: str, seen_messages: list[list[dict]]) -> None:
        self.text = text
        self.seen_messages = seen_messages

    def call(self, messages, tools, *, system):
        self.seen_messages.append(list(messages))
        return ParsedResponse(
            text=self.text,
            tool_calls=[],
            usage={"total_tokens": 1},
            raw=None,
        )

    def append_assistant_turn(self, messages, response):
        messages.append({"role": "assistant", "content": response.text})

    def append_tool_results(self, messages, results):
        messages.append({
            "role": "user",
            "content": [
                {
                    "tool_use_id": result.id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in results
            ],
        })


class FakeRuntime:
    def __init__(self) -> None:
        self.lease_count = 0
        self.close_count = 0

    @contextmanager
    def lease_tools(self):
        self.lease_count += 1
        yield [
            Tool(
                name="mcp__fake__search",
                description="fake search",
                input_schema={"type": "object", "properties": {}},
                handler=lambda args: "result",
            )
        ]

    def close(self) -> None:
        self.close_count += 1


def test_session_reuses_messages_across_asks_and_resets_guardrail_each_turn():
    seen_messages: list[list[dict]] = []
    replies = iter(["reply-1", "reply-2"])

    def client_factory():
        return FakeClient(next(replies), seen_messages)

    def tool_builder(registry, todo_store, messages_ref, input_provider, on_event):
        return [], FakeMcp()

    session = ConversationSession(
        max_iter_per_turn=1,
        client_factory=client_factory,
        tool_builder=tool_builder,
        on_event=lambda kind, payload: None,
        input_provider=lambda question: "answer",
    )

    session.ask("first")
    session.ask("second")

    assert len(seen_messages) == 2
    assert seen_messages[0] == [{"role": "user", "content": "first"}]
    assert seen_messages[1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply-1"},
        {"role": "user", "content": "second"},
    ]
    assert session.messages[-1] == {"role": "assistant", "content": "reply-2"}


def test_session_close_closes_mcp():
    mcps: list[FakeMcp] = []

    def tool_builder(registry, todo_store, messages_ref, input_provider, on_event):
        mcp = FakeMcp()
        mcps.append(mcp)
        return [], mcp

    session = ConversationSession(
        client_factory=lambda: FakeClient("reply", []),
        tool_builder=tool_builder,
        on_event=lambda kind, payload: None,
    )

    session.start()
    session.close()
    session.close()

    assert mcps[0].closed == 1


def test_run_with_runtime_reuses_mcp_but_isolates_task_messages(monkeypatch):
    runtime = FakeRuntime()
    seen_messages: list[list[dict]] = []
    replies = iter(["reply-1", "reply-2"])

    monkeypatch.setattr(
        "paperpilot.conversation.LLMClient",
        lambda: FakeClient(next(replies), seen_messages),
    )

    run("first task", mcp_runtime=runtime)
    run("second task", mcp_runtime=runtime)

    assert runtime.lease_count == 2
    assert runtime.close_count == 0
    assert seen_messages == [
        [{"role": "user", "content": "first task"}],
        [{"role": "user", "content": "second task"}],
    ]


def test_session_reset_rebuilds_clean_conversation():
    mcps: list[FakeMcp] = []

    def tool_builder(registry, todo_store, messages_ref, input_provider, on_event):
        mcp = FakeMcp()
        mcps.append(mcp)
        return [], mcp

    session = ConversationSession(
        client_factory=lambda: FakeClient("reply", []),
        tool_builder=tool_builder,
        on_event=lambda kind, payload: None,
    )

    session.ask("before reset")
    old_messages = session.messages
    session.reset()

    assert len(mcps) == 2
    assert mcps[0].closed == 1
    assert session.messages == []
    assert session.messages is not old_messages


def test_named_session_loads_existing_messages_and_saves_after_ask(tmp_path):
    store = SessionStore(tmp_path)
    store.save("demo", [{"role": "user", "content": "restored"}])
    seen_messages: list[list[dict]] = []

    def tool_builder(registry, todo_store, messages_ref, input_provider, on_event):
        return [], FakeMcp()

    session = ConversationSession(
        session_name="demo",
        session_store=store,
        client_factory=lambda: FakeClient("reply", seen_messages),
        tool_builder=tool_builder,
        on_event=lambda kind, payload: None,
    )

    session.ask("next")

    assert seen_messages[0] == [
        {"role": "user", "content": "restored"},
        {"role": "user", "content": "next"},
    ]
    assert store.load("demo") == [
        {"role": "user", "content": "restored"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "reply"},
    ]


def test_reset_session_ignores_existing_messages(tmp_path):
    store = SessionStore(tmp_path)
    store.save("demo", [{"role": "user", "content": "old"}])

    def tool_builder(registry, todo_store, messages_ref, input_provider, on_event):
        return [], FakeMcp()

    session = ConversationSession(
        session_name="demo",
        session_store=store,
        reset_session=True,
        client_factory=lambda: FakeClient("reply", []),
        tool_builder=tool_builder,
        on_event=lambda kind, payload: None,
    )

    session.start()

    assert session.messages == []
    assert store.load("demo") == []


def test_named_session_reset_saves_empty_history(tmp_path):
    store = SessionStore(tmp_path)

    def tool_builder(registry, todo_store, messages_ref, input_provider, on_event):
        return [], FakeMcp()

    session = ConversationSession(
        session_name="demo",
        session_store=store,
        client_factory=lambda: FakeClient("reply", []),
        tool_builder=tool_builder,
        on_event=lambda kind, payload: None,
    )

    session.ask("before reset")
    session.reset()

    assert session.messages == []
    assert store.load("demo") == []


def test_unnamed_session_does_not_touch_session_store(tmp_path):
    store = SessionStore(tmp_path)

    def tool_builder(registry, todo_store, messages_ref, input_provider, on_event):
        return [], FakeMcp()

    session = ConversationSession(
        session_store=store,
        client_factory=lambda: FakeClient("reply", []),
        tool_builder=tool_builder,
        on_event=lambda kind, payload: None,
    )

    session.ask("in memory only")

    assert list(tmp_path.iterdir()) == []


def test_named_session_compact_saves_changed_messages(tmp_path):
    store = SessionStore(tmp_path)

    def tool_builder(registry, todo_store, messages_ref, input_provider, on_event):
        def compact_handler(args):
            messages_ref[:] = [{"role": "user", "content": "summary"}]
            return "compacted"

        return [
            Tool(
                name="compact_context",
                description="compact",
                input_schema={},
                handler=compact_handler,
            )
        ], FakeMcp()

    session = ConversationSession(
        session_name="demo",
        session_store=store,
        client_factory=lambda: FakeClient("reply", []),
        tool_builder=tool_builder,
        on_event=lambda kind, payload: None,
    )

    result = session.compact()

    assert result == "compacted"
    assert store.load("demo") == [{"role": "user", "content": "summary"}]


def test_session_saves_bulk_paper_input_outside_messages(tmp_path):
    seen_messages: list[list[dict]] = []
    events: list[tuple[str, dict]] = []
    long_text = (
        "Please compare this paper with 2402.13718.\n\n"
        "Abstract\nThis paper studies long-context retrieval.\n\n"
        "Introduction\n" + ("retrieval augmented generation " * 80) + "\n\n"
        "References\n[1] Prior work."
    )

    def tool_builder(registry, todo_store, messages_ref, input_provider, on_event):
        return [], FakeMcp()

    session = ConversationSession(
        client_factory=lambda: FakeClient("reply", seen_messages),
        tool_builder=tool_builder,
        document_store=DocumentStore(tmp_path),
        bulk_input_detector=BulkPaperInputDetector(min_chars=200),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    session.ask(long_text)

    first_user_content = seen_messages[0][0]["content"]
    assert "user_doc_id: userdoc-" in first_user_content
    assert "target_paper_id: 2402.13718" in first_user_content
    assert "retrieval augmented generation retrieval augmented generation" not in (
        first_user_content
    )
    assert any(kind == "bulk_input_saved" for kind, _ in events)
    assert list(tmp_path.glob("userdoc-*.txt"))
    search_tool = next(tool for tool in session.tools if tool.name == "search_user_document")
    doc_id = first_user_content.split("user_doc_id: ")[1].splitlines()[0]
    payload = json.loads(search_tool.handler({
        "doc_id": doc_id,
        "query": "long-context retrieval",
        "top_k": 1,
    }))
    assert payload["hits"]
