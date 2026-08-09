"""WorkflowRunner tests."""
from __future__ import annotations

import pytest

from paperpilot.papers import PaperCandidate
from paperpilot.web.task_store import TaskStore
from paperpilot.web.workflow import WorkflowRunner


PRIMARY_PAPER = PaperCandidate(
    external_id="2401.12345v1",
    title="Primary paper",
    authors=["Ada Lovelace"],
    abstract="Primary abstract.",
    source_url="https://arxiv.org/abs/2401.12345v1",
)


class FakeDeepReadingRunner:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.error = error

    def run(self, task_id: str) -> None:
        self.calls.append(task_id)
        if self.error is not None:
            raise self.error


def _conversation_task(store: TaskStore):
    user = store.create_user(
        username="alice",
        password_hash="hash",
        password_salt="salt",
    )
    conversation = store.create_conversation(user_id=user.id, paper=PRIMARY_PAPER)
    return store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content="Compare the paper's evidence.",
        depth="standard",
        expected_head_message_id=None,
    ).task


def test_simulated_workflow_writes_events_status_and_artifact(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="Run workflow", depth="deep")
    store.add_event(
        task_id=task.id,
        type="queued",
        stage="queue",
        message="Queued.",
        payload={"depth": task.depth, "simulated": True},
    )
    runner = WorkflowRunner(store, delay_seconds=0)

    runner.run_simulated(task.id)

    updated = store.get_task(task.id)
    events = store.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items
    artifacts = store.list_artifacts_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items

    assert updated is not None
    assert updated.status == "completed"
    assert events is not None
    assert [event.type for event in events] == [
        "queued",
        "started",
        "progress",
        "progress",
        "completed",
    ]
    assert artifacts is not None
    assert len(artifacts) == 1
    assert artifacts[0].kind == "result"
    assert artifacts[0].payload["depth"] == "deep"


def test_real_workflow_uses_real_runner_and_writes_result(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="Real query", depth="quick")
    runner = WorkflowRunner(
        store,
        delay_seconds=0,
        real_runner=lambda query: [{"role": "assistant", "content": f"answer: {query}"}],
    )

    runner.run_real(task.id)

    updated = store.get_task(task.id)
    events = store.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items
    artifacts = store.list_artifacts_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items

    assert updated is not None
    assert updated.status == "completed"
    assert events is not None
    assert [event.stage for event in events] == ["real_start", "real_complete"]
    assert artifacts is not None
    assert artifacts[0].title == "PaperPilot result"
    assert artifacts[0].content == "answer: Real query"
    assert artifacts[0].payload["execution_mode"] == "real"


def test_real_workflow_persists_agent_events_from_runner(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="Real query", depth="quick")

    def eventful_runner(query: str, *, on_event) -> list[dict]:
        on_event("turn", {"iteration": 1, "tool_calls": ["search"], "text": None})
        on_event("tool_call", {"name": "search", "arguments": {"query": query}})
        on_event("tool_result", {"name": "search", "content": "result text"})
        return [{"role": "assistant", "content": "answer"}]

    runner = WorkflowRunner(store, delay_seconds=0, real_runner=eventful_runner)

    runner.run_real(task.id)

    events = store.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items
    artifacts = store.list_artifacts_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items

    assert events is not None
    assert [event.stage for event in events] == [
        "real_start",
        "agent_turn",
        "tool_call",
        "tool_result",
        "real_complete",
    ]
    assert events[2].payload["tool_name"] == "search"
    assert events[3].payload["content_preview"] == "result text"
    assert artifacts is not None
    assert artifacts[0].content == "answer"


def test_real_workflow_failure_marks_task_failed(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="Real query", depth="quick")

    def failing_runner(query: str) -> list[dict]:
        raise RuntimeError("boom")

    runner = WorkflowRunner(store, delay_seconds=0, real_runner=failing_runner)

    runner.run_real(task.id)

    updated = store.get_task(task.id)
    events = store.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items
    artifacts = store.list_artifacts_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items

    assert updated is not None
    assert updated.status == "failed"
    assert events is not None
    assert events[-1].type == "failed"
    assert events[-1].stage == "failure"
    assert artifacts == []


def test_real_conversation_task_delegates_without_calling_legacy_runner(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = _conversation_task(store)
    deep_runner = FakeDeepReadingRunner()
    legacy_calls: list[str] = []
    runner = WorkflowRunner(
        store,
        delay_seconds=0,
        real_runner=lambda query: legacy_calls.append(query) or [],
        deep_reading_runner=deep_runner,
    )

    runner.run_real(task.id)

    assert deep_runner.calls == [task.id]
    assert legacy_calls == []
    assert store.get_task(task.id).status == "pending"
    assert store.list_artifacts_page(
        task.id,
        user_id=task.user_id,
        after_id=0,
        limit=100,
    ).items == []


def test_real_conversation_task_requires_configured_deep_reading_runner(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = _conversation_task(store)
    runner = WorkflowRunner(store, delay_seconds=0, real_runner=lambda _query: [])

    with pytest.raises(RuntimeError, match="deep-reading runner"):
        runner.run_real(task.id)

    assert store.get_task(task.id).status == "pending"


def test_legacy_initial_task_read_failure_is_caught_and_marks_failed(
    tmp_path,
    monkeypatch,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="Transient read failure")
    real_get_task = store.get_task
    get_calls = 0

    def fail_once(task_id: str):
        nonlocal get_calls
        get_calls += 1
        if get_calls == 1:
            raise RuntimeError("temporary read error")
        return real_get_task(task_id)

    monkeypatch.setattr(
        store,
        "get_task",
        fail_once,
    )
    runner = WorkflowRunner(store, delay_seconds=0, real_runner=lambda _query: [])

    runner.run_real(task.id)

    assert store.get_task(task.id).status == "failed"
    events = store.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items
    assert events[-1].type == "failed"
    assert events[-1].message == (
        "Real PaperPilot execution failed: temporary read error"
    )


def test_configured_deep_runner_route_read_failure_propagates_without_status_write(
    tmp_path,
    monkeypatch,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="Cannot classify this task")
    real_get_task = store.get_task
    get_calls = 0

    def fail_once(task_id: str):
        nonlocal get_calls
        get_calls += 1
        if get_calls == 1:
            raise RuntimeError("route database unavailable")
        return real_get_task(task_id)

    monkeypatch.setattr(store, "get_task", fail_once)
    deep_runner = FakeDeepReadingRunner()
    runner = WorkflowRunner(
        store,
        delay_seconds=0,
        real_runner=lambda _query: [],
        deep_reading_runner=deep_runner,
    )

    with pytest.raises(RuntimeError, match="route database unavailable"):
        runner.run_real(task.id)

    assert deep_runner.calls == []
    assert store.get_task(task.id).status == "pending"
    events = store.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    ).items
    assert events == []


def test_unknown_deep_reading_failure_propagates_without_legacy_failure_write(
    tmp_path,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = _conversation_task(store)
    deep_runner = FakeDeepReadingRunner(error=RuntimeError("checkpoint unavailable"))
    runner = WorkflowRunner(
        store,
        delay_seconds=0,
        real_runner=lambda _query: [],
        deep_reading_runner=deep_runner,
    )

    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        runner.run_real(task.id)

    assert store.get_task(task.id).status == "pending"
    events = store.list_events_page(
        task.id,
        user_id=task.user_id,
        after_id=0,
        limit=100,
    ).items
    assert [event.type for event in events] == ["queued"]
