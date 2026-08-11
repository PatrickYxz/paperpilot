"""Conversation-task routing and process resource ownership tests."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import paperpilot.web.app as app_module
from paperpilot.deep_reading.runner import DeepReadingRunner
from paperpilot.papers import PaperCandidate
from paperpilot.web import worker_tasks
from paperpilot.web.app import create_app
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.task_store import TaskStore


PRIMARY_PAPER = PaperCandidate(
    external_id="2401.12345v1",
    title="Primary paper",
    authors=["Ada Lovelace"],
    abstract="Primary abstract.",
    source_url="https://arxiv.org/abs/2401.12345v1",
)


class FakeMCPRuntime:
    def __init__(
        self,
        close_order: list[str] | None = None,
        *,
        close_error: Exception | None = None,
    ) -> None:
        self.close_count = 0
        self.lease_calls = 0
        self.close_order = close_order
        self.close_error = close_error

    def lease_tools(self):  # pragma: no cover - Web reader must never start MCP
        self.lease_calls += 1
        raise AssertionError("Web checkpoint reader started MCP")

    def close(self) -> None:
        self.close_count += 1
        if self.close_order is not None:
            self.close_order.append("mcp")
        if self.close_error is not None:
            raise self.close_error


class FakeCheckpointRuntime:
    def __init__(
        self,
        close_order: list[str] | None = None,
        *,
        close_error: Exception | None = None,
    ) -> None:
        self.saver = object()
        self.close_count = 0
        self.close_order = close_order
        self.close_error = close_error

    def check_health(self) -> None:
        pass

    def close(self) -> None:
        self.close_count += 1
        if self.close_order is not None:
            self.close_order.append("checkpoint")
        if self.close_error is not None:
            raise self.close_error


class FakeDeepReadingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        self.calls.append((task_id, allow_running))
        return True


class RecordingExecutor:
    def __init__(self, close_order: list[str]) -> None:
        self.close_order = close_order
        self.is_shutdown = False

    def reserve(self):  # pragma: no cover - lifecycle test never submits
        raise AssertionError("unexpected submission")

    def submit(self, task_id: str):  # pragma: no cover
        raise AssertionError("unexpected submission")

    def shutdown(self) -> None:
        self.is_shutdown = True
        self.close_order.append("executor")


class TrackingOwnedStore(TaskStore):
    def __init__(self, db_path, close_order: list[str]) -> None:
        super().__init__(db_path)
        self.close_order = close_order
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.close_order.append("store")
        super().close()


def _new_conversation_task(store: TaskStore, *, suffix: str):
    user = store.get_user_by_username("alice")
    if user is None:
        user = store.create_user(
            username="alice",
            password_hash="hash",
            password_salt="salt",
        )
    paper = PaperCandidate(
        external_id=f"2401.1234{suffix}v1",
        title=f"Primary paper {suffix}",
        authors=["Ada Lovelace"],
        abstract="Primary abstract.",
        source_url=f"https://arxiv.org/abs/2401.1234{suffix}v1",
    )
    conversation = store.create_conversation(user_id=user.id, paper=paper)
    return store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content=f"Question {suffix}",
        depth="standard",
        expected_head_message_id=None,
    ).task


def test_worker_reuses_process_mcp_and_checkpoint_for_conversation_tasks(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "tasks.sqlite3"
    seed = TaskStore(db_path)
    first = _new_conversation_task(seed, suffix="1")
    second = _new_conversation_task(seed, suffix="2")
    seed.close()
    mcp_instances: list[FakeMCPRuntime] = []
    checkpoint_instances: list[FakeCheckpointRuntime] = []
    checkpoint_paths: list[Path] = []
    built: list[dict[str, object]] = []
    deep_runners: list[FakeDeepReadingRunner] = []

    def make_mcp() -> FakeMCPRuntime:
        runtime = FakeMCPRuntime()
        mcp_instances.append(runtime)
        return runtime

    def make_checkpoint(path) -> FakeCheckpointRuntime:
        checkpoint_paths.append(Path(path))
        runtime = FakeCheckpointRuntime()
        checkpoint_instances.append(runtime)
        return runtime

    class RecordingDeepRunner(FakeDeepReadingRunner):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            built.append(kwargs)
            deep_runners.append(self)

    config = WebRuntimeConfig(checkpoint_db_path=tmp_path / "worker-checkpoints.sqlite3")

    monkeypatch.setattr(worker_tasks, "_runtime", None)
    monkeypatch.setattr(worker_tasks, "_checkpoint_runtime", None, raising=False)
    monkeypatch.setattr(worker_tasks, "_runtime_factory", make_mcp)
    monkeypatch.setattr(
        worker_tasks,
        "_checkpoint_runtime_factory",
        make_checkpoint,
        raising=False,
    )
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: TaskStore(db_path))
    monkeypatch.setattr(worker_tasks.WebRuntimeConfig, "from_env", lambda: config)
    monkeypatch.setattr(worker_tasks, "DeepReadingRunner", RecordingDeepRunner)

    worker_tasks._execute_research_task(first.id)
    worker_tasks._execute_research_task(second.id)

    assert len(mcp_instances) == 1
    assert len(checkpoint_instances) == 1
    assert [runner.calls for runner in deep_runners] == [
        [(first.id, False)],
        [(second.id, False)],
    ]
    assert checkpoint_paths == [config.checkpoint_db_path]
    assert [
        (kwargs["mcp_runtime"], kwargs["checkpointer"])
        for kwargs in built
    ] == [
        (mcp_instances[0], checkpoint_instances[0].saver),
        (mcp_instances[0], checkpoint_instances[0].saver),
    ]
    check = TaskStore(db_path)
    assert check.get_task(first.id).status == "pending"
    assert check.get_task(second.id).status == "pending"
    check.close()


def test_worker_recovers_redelivered_running_conversation_but_skips_completed(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "tasks.sqlite3"
    seed = TaskStore(db_path)
    running = _new_conversation_task(seed, suffix="3")
    completed = _new_conversation_task(seed, suffix="4")
    assert seed.claim_task(running.id) is not None
    with seed.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE research_tasks SET status = 'completed' WHERE id = ?",
            (completed.id,),
        )
    seed.close()
    builds: list[str] = []
    invocations: list[tuple[str, bool]] = []
    executions: list[str] = []

    monkeypatch.setattr(worker_tasks, "_runtime", None)
    monkeypatch.setattr(worker_tasks, "_checkpoint_runtime", None, raising=False)
    monkeypatch.setattr(worker_tasks, "_runtime_factory", FakeMCPRuntime)
    config = WebRuntimeConfig(checkpoint_db_path=tmp_path / "checkpoints.sqlite3")
    monkeypatch.setattr(
        worker_tasks,
        "_checkpoint_runtime_factory",
        lambda _path: FakeCheckpointRuntime(),
        raising=False,
    )
    monkeypatch.setattr(worker_tasks, "_store_factory", lambda: TaskStore(db_path))
    monkeypatch.setattr(worker_tasks.WebRuntimeConfig, "from_env", lambda: config)

    def make_runner(**kwargs):
        store = kwargs["task_store"]
        task_id = running.id if len(builds) < 2 else completed.id
        builds.append(store.get_task(task_id).status)

        class RecoveringRunner:
            def run(self, task_id: str, *, allow_running: bool = False) -> bool:
                invocations.append((task_id, allow_running))
                claimed = store.claim_task(
                    task_id,
                    allow_running=allow_running,
                )
                if claimed is None:
                    return False
                executions.append(task_id)
                return True

        return RecoveringRunner()

    monkeypatch.setattr(worker_tasks, "DeepReadingRunner", make_runner)

    worker_tasks._execute_research_task(running.id)
    assert executions == []

    worker_tasks._execute_research_task(running.id, redelivered=True)
    worker_tasks._execute_research_task(completed.id, redelivered=True)

    assert invocations == [
        (running.id, False),
        (running.id, True),
        (completed.id, True),
    ]
    assert executions == [running.id]
    assert builds == ["running", "running", "completed"]


def test_late_retry_exhaustion_preserves_completed_conversation_without_event(
    tmp_path,
):
    store = TaskStore(tmp_path / "completed-race.sqlite3")
    task = _new_conversation_task(store, suffix="9")
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE research_tasks SET status = 'completed' WHERE id = ?",
            (task.id,),
        )
    runner = DeepReadingRunner(
        task_store=store,
        checkpointer=object(),
        mcp_runtime=object(),
    )

    runner.fail_retry_exhausted(
        task.id,
        backend="celery",
        attempts=4,
        max_retries=3,
        exc=ConnectionError("late secret failure"),
    )

    assert store.get_task(task.id).status == "completed"
    events = store.list_events_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=100,
    )
    assert events is not None
    assert [event for event in events.items if event.type == "failed"] == []
    store.close()


def test_worker_build_passes_one_custom_config_to_checkpoint_and_runner(
    tmp_path,
    monkeypatch,
):
    config = WebRuntimeConfig(
        checkpoint_db_path=tmp_path / "custom-checkpoints.sqlite3",
        summary_token_threshold=4321,
        summary_recent_turns=4,
        research_recursion_limit=30,
        research_model_call_limit=10,
        research_tool_call_limit=14,
        research_max_output_tokens=2048,
        research_model_retries=0,
    )
    builds: list[dict[str, object]] = []
    config_reads = 0
    checkpoint_paths: list[Path] = []
    checkpoint = FakeCheckpointRuntime()

    class RecordingDeepRunner:
        def __init__(self, **kwargs) -> None:
            builds.append(kwargs)

    store = object()
    mcp = FakeMCPRuntime()

    def read_config():
        nonlocal config_reads
        config_reads += 1
        return config

    def open_checkpoint(path):
        checkpoint_paths.append(Path(path))
        return checkpoint

    monkeypatch.setattr(worker_tasks, "_checkpoint_runtime", None)
    monkeypatch.setattr(worker_tasks, "_checkpoint_runtime_factory", open_checkpoint)
    monkeypatch.setattr(worker_tasks.WebRuntimeConfig, "from_env", read_config)
    monkeypatch.setattr(worker_tasks, "DeepReadingRunner", RecordingDeepRunner)

    runner = worker_tasks._build_deep_reading_runner(store, mcp)

    assert isinstance(runner, RecordingDeepRunner)
    assert config_reads == 1
    assert checkpoint_paths == [config.checkpoint_db_path]
    assert builds == [
        {
            "task_store": store,
            "checkpointer": checkpoint.saver,
            "mcp_runtime": mcp,
            "summary_token_threshold": 4321,
            "summary_recent_turns": 4,
            "research_recursion_limit": 30,
            "research_model_call_limit": 10,
            "research_tool_call_limit": 14,
            "research_max_output_tokens": 2048,
            "research_model_retries": 0,
        }
    ]


def test_worker_import_does_not_read_deep_reading_runtime_config() -> None:
    environment = dict(os.environ)
    environment["PAPERPILOT_SUMMARY_TOKEN_THRESHOLD"] = "0"
    environment["PAPERPILOT_SUMMARY_RECENT_TURNS"] = "0"
    environment["PAPERPILOT_RESEARCH_RECURSION_LIMIT"] = "0"
    environment["PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT"] = "0"
    environment["PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT"] = "0"
    environment["PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS"] = "0"
    environment["PAPERPILOT_RESEARCH_MODEL_RETRIES"] = "-1"

    result = subprocess.run(
        [sys.executable, "-c", "import paperpilot.web.worker_tasks"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr


def test_app_default_runtime_uses_injected_store_directory_and_closes_in_order(
    tmp_path,
    monkeypatch,
):
    store = TaskStore(tmp_path / "business" / "tasks.sqlite3")
    close_order: list[str] = []
    checkpoint = FakeCheckpointRuntime(close_order)
    mcp = FakeMCPRuntime(close_order)
    checkpoint_paths: list[Path] = []
    deep_builds: list[dict[str, object]] = []

    def open_checkpoint(path):
        checkpoint_paths.append(Path(path))
        return checkpoint

    class RecordingDeepRunner:
        def __init__(self, **kwargs) -> None:
            deep_builds.append(kwargs)

        def run(self, task_id: str) -> None:  # pragma: no cover
            raise AssertionError(f"unexpected model execution for {task_id}")

    monkeypatch.setattr(app_module.SqliteCheckpointRuntime, "open", open_checkpoint)
    monkeypatch.setattr(app_module, "MCPRuntime", lambda: mcp)
    monkeypatch.setattr(app_module, "DeepReadingRunner", RecordingDeepRunner)
    executor = RecordingExecutor(close_order)
    monkeypatch.setattr(
        app_module,
        "build_task_executor",
        lambda runner, *, config: executor,
    )
    config = WebRuntimeConfig(
        task_executor="thread",
        summary_token_threshold=2468,
        summary_recent_turns=5,
        research_recursion_limit=11,
        research_model_call_limit=10,
        research_tool_call_limit=18,
        research_max_output_tokens=3072,
        research_model_retries=0,
    )

    with TestClient(
        create_app(
            store,
            runtime_config=config,
        )
    ):
        pass

    assert checkpoint_paths == [tmp_path / "business" / "checkpoints.sqlite3"]
    assert deep_builds == [
        {
            "task_store": store,
            "checkpointer": checkpoint.saver,
            "mcp_runtime": mcp,
            "summary_token_threshold": 2468,
            "summary_recent_turns": 5,
            "research_recursion_limit": 11,
            "research_model_call_limit": 10,
            "research_tool_call_limit": 18,
            "research_max_output_tokens": 3072,
            "research_model_retries": 0,
        }
    ]
    assert close_order == ["executor", "checkpoint", "mcp"]
    assert checkpoint.close_count == 1
    assert mcp.close_count == 1
    store.close()


def test_injected_executor_creates_model_free_api_resources_and_closes_owned(
    tmp_path,
    monkeypatch,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    close_order: list[str] = []
    executor = RecordingExecutor(close_order)
    checkpoint = FakeCheckpointRuntime(close_order)
    mcp = FakeMCPRuntime(close_order)
    builds: list[dict[str, object]] = []
    monkeypatch.setattr(
        app_module.SqliteCheckpointRuntime,
        "open",
        lambda _path: checkpoint,
    )
    monkeypatch.setattr(app_module, "MCPRuntime", lambda: mcp)

    class RecordingDeepRunner:
        def __init__(self, **kwargs) -> None:
            builds.append(kwargs)

        def run(self, task_id: str) -> None:  # pragma: no cover
            raise AssertionError(f"Web API reader executed model task {task_id}")

    monkeypatch.setattr(app_module, "DeepReadingRunner", RecordingDeepRunner)

    with TestClient(
        create_app(
            store,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(task_executor="thread"),
        )
    ) as client:
        assert isinstance(client.app.state.deep_reading_runner, RecordingDeepRunner)
        assert client.get("/health/ready").status_code == 200

    assert len(builds) == 1
    assert builds[0]["task_store"] is store
    assert builds[0]["checkpointer"] is checkpoint.saver
    assert builds[0]["mcp_runtime"] is mcp
    assert mcp.lease_calls == 0
    assert close_order == ["executor", "checkpoint", "mcp"]
    store.close()


def test_app_construction_mcp_failure_closes_owned_checkpoint_and_store(
    tmp_path,
    monkeypatch,
):
    close_order: list[str] = []
    store = TrackingOwnedStore(tmp_path / "tasks.sqlite3", close_order)
    checkpoint = FakeCheckpointRuntime(close_order)
    monkeypatch.setattr(app_module, "TaskStore", lambda: store)
    monkeypatch.setattr(
        app_module.SqliteCheckpointRuntime,
        "open",
        lambda _path: checkpoint,
    )
    monkeypatch.setattr(
        app_module,
        "MCPRuntime",
        lambda: (_ for _ in ()).throw(RuntimeError("MCP construction failed")),
    )

    with pytest.raises(RuntimeError, match="MCP construction failed"):
        create_app(runtime_config=WebRuntimeConfig(task_executor="thread"))

    assert close_order == ["checkpoint", "store"]
    assert checkpoint.close_count == 1
    assert store.close_count == 1


def test_app_construction_deep_runner_failure_preserves_error_and_best_effort_closes(
    tmp_path,
    monkeypatch,
):
    close_order: list[str] = []
    store = TrackingOwnedStore(tmp_path / "tasks.sqlite3", close_order)
    checkpoint = FakeCheckpointRuntime(
        close_order,
        close_error=RuntimeError("checkpoint close failed"),
    )
    mcp = FakeMCPRuntime(close_order)
    monkeypatch.setattr(app_module, "TaskStore", lambda: store)
    monkeypatch.setattr(
        app_module.SqliteCheckpointRuntime,
        "open",
        lambda _path: checkpoint,
    )
    monkeypatch.setattr(app_module, "MCPRuntime", lambda: mcp)
    monkeypatch.setattr(
        app_module,
        "DeepReadingRunner",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("DeepReadingRunner construction failed")
        ),
    )

    with pytest.raises(RuntimeError, match="DeepReadingRunner construction failed"):
        create_app(runtime_config=WebRuntimeConfig(task_executor="thread"))

    assert close_order == ["checkpoint", "mcp", "store"]
    assert checkpoint.close_count == 1
    assert mcp.close_count == 1
    assert store.close_count == 1


def test_app_construction_executor_failure_closes_owned_runtime_and_store(
    tmp_path,
    monkeypatch,
):
    close_order: list[str] = []
    store = TrackingOwnedStore(tmp_path / "tasks.sqlite3", close_order)
    checkpoint = FakeCheckpointRuntime(close_order)
    mcp = FakeMCPRuntime(close_order)
    monkeypatch.setattr(app_module, "TaskStore", lambda: store)
    monkeypatch.setattr(
        app_module.SqliteCheckpointRuntime,
        "open",
        lambda _path: checkpoint,
    )
    monkeypatch.setattr(app_module, "MCPRuntime", lambda: mcp)
    monkeypatch.setattr(app_module, "DeepReadingRunner", lambda **_kwargs: object())
    monkeypatch.setattr(
        app_module,
        "build_task_executor",
        lambda runner, *, config: (_ for _ in ()).throw(
            RuntimeError("executor construction failed")
        ),
    )

    with pytest.raises(RuntimeError, match="executor construction failed"):
        create_app(runtime_config=WebRuntimeConfig(task_executor="thread"))

    assert close_order == ["checkpoint", "mcp", "store"]
    assert checkpoint.close_count == 1
    assert mcp.close_count == 1
    assert store.close_count == 1


def test_app_construction_fastapi_failure_closes_default_executor_then_resources(
    tmp_path,
    monkeypatch,
):
    close_order: list[str] = []
    store = TrackingOwnedStore(tmp_path / "tasks.sqlite3", close_order)
    checkpoint = FakeCheckpointRuntime(close_order)
    mcp = FakeMCPRuntime(close_order)
    executor = RecordingExecutor(close_order)
    monkeypatch.setattr(app_module, "TaskStore", lambda: store)
    monkeypatch.setattr(
        app_module.SqliteCheckpointRuntime,
        "open",
        lambda _path: checkpoint,
    )
    monkeypatch.setattr(app_module, "MCPRuntime", lambda: mcp)
    monkeypatch.setattr(app_module, "DeepReadingRunner", lambda **_kwargs: object())
    monkeypatch.setattr(
        app_module,
        "build_task_executor",
        lambda runner, *, config: executor,
    )
    monkeypatch.setattr(
        app_module,
        "FastAPI",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("FastAPI construction failed")
        ),
    )

    with pytest.raises(RuntimeError, match="FastAPI construction failed"):
        create_app(runtime_config=WebRuntimeConfig(task_executor="thread"))

    assert close_order == ["executor", "checkpoint", "mcp", "store"]
    assert executor.is_shutdown is True
    assert checkpoint.close_count == 1
    assert mcp.close_count == 1
    assert store.close_count == 1


def test_app_does_not_close_injected_deep_reading_resources(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    checkpoint = FakeCheckpointRuntime()
    mcp = FakeMCPRuntime()
    deep_runner = FakeDeepReadingRunner()
    close_order: list[str] = []
    executor = RecordingExecutor(close_order)

    with TestClient(
        create_app(
            store,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(task_executor="thread"),
            checkpoint_runtime=checkpoint,
            mcp_runtime=mcp,
            deep_reading_runner=deep_runner,
        )
    ) as client:
        assert client.app.state.deep_reading_runner is deep_runner

    assert close_order == ["executor"]
    assert checkpoint.close_count == 0
    assert mcp.close_count == 0
    store.close()


def test_celery_web_parent_owns_model_free_checkpoint_reader_resources(
    tmp_path,
    monkeypatch,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    close_order: list[str] = []
    checkpoint = FakeCheckpointRuntime(close_order)
    mcp = FakeMCPRuntime(close_order)
    builds: list[dict[str, object]] = []
    monkeypatch.setattr(
        app_module.SqliteCheckpointRuntime,
        "open",
        lambda _path: checkpoint,
    )
    monkeypatch.setattr(app_module, "MCPRuntime", lambda: mcp)

    class RecordingDeepRunner:
        def __init__(self, **kwargs) -> None:
            builds.append(kwargs)

        def run(self, task_id: str) -> None:  # pragma: no cover
            raise AssertionError(f"Celery Web parent executed model task {task_id}")

    monkeypatch.setattr(app_module, "DeepReadingRunner", RecordingDeepRunner)

    with TestClient(
        create_app(
            store,
            runtime_config=WebRuntimeConfig(task_executor="celery"),
        )
    ) as client:
        assert isinstance(client.app.state.deep_reading_runner, RecordingDeepRunner)
        assert client.get("/health/ready").status_code == 200

    assert len(builds) == 1
    assert builds[0]["task_store"] is store
    assert builds[0]["checkpointer"] is checkpoint.saver
    assert builds[0]["mcp_runtime"] is mcp
    assert mcp.lease_calls == 0
    assert checkpoint.close_count == 1
    assert mcp.close_count == 1
    store.close()
