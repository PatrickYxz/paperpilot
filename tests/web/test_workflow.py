"""WorkflowRunner tests."""
from __future__ import annotations

from paperpilot.web.task_store import TaskStore
from paperpilot.web.workflow import WorkflowRunner


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
