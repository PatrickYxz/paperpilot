"""Real SQLite checkpoint lifecycle and history-fork contract tests."""
from __future__ import annotations

import multiprocessing
import operator
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

import paperpilot.web.checkpoint as checkpoint_module
from paperpilot.web.checkpoint import (
    SqliteCheckpointRuntime,
    resolve_checkpoint_db_path,
)


class _CounterState(TypedDict):
    value: Annotated[int, operator.add]


@dataclass
class _UnregisteredPayload:
    value: int


def _build_counter_graph(saver):
    builder = StateGraph(_CounterState)
    builder.add_node("increment", lambda _state: {"value": 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=saver)


def _write_counter_states(path: str, thread_id: str, start_event) -> None:
    if not start_event.wait(timeout=10):
        raise RuntimeError("parent did not release checkpoint writers")
    runtime = SqliteCheckpointRuntime.open(path)
    try:
        graph = _build_counter_graph(runtime.saver)
        config = {"configurable": {"thread_id": thread_id}}
        for _ in range(5):
            graph.invoke({"value": 1}, config)
    finally:
        runtime.close()


def test_resolve_checkpoint_db_path_defaults_under_current_working_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH", raising=False)

    assert resolve_checkpoint_db_path() == (
        tmp_path / "data/langgraph/checkpoints.sqlite3"
    )


def test_resolve_checkpoint_db_path_uses_environment_override(
    tmp_path, monkeypatch
) -> None:
    configured = tmp_path / "configured.sqlite3"
    monkeypatch.setenv(
        "PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH", str(configured)
    )

    assert resolve_checkpoint_db_path() == configured


def test_checkpoint_runtime_defaults_to_strict_msgpack(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LANGGRAPH_STRICT_MSGPACK", raising=False)

    runtime = SqliteCheckpointRuntime.open()
    try:
        assert runtime.path == tmp_path / "data/langgraph/checkpoints.sqlite3"
        assert runtime.path.exists()
        serialized = runtime.saver.serde.dumps_typed(_UnregisteredPayload(value=7))
        assert runtime.saver.serde.loads_typed(serialized) == {"value": 7}
    finally:
        runtime.close()


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "unexpected"])
def test_checkpoint_runtime_rejects_explicitly_disabled_strict_msgpack(
    tmp_path, monkeypatch, value
) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", value)

    with pytest.raises(ValueError, match="LANGGRAPH_STRICT_MSGPACK"):
        SqliteCheckpointRuntime.open(path)

    assert not path.exists()


def test_checkpoint_runtime_persists_after_close_and_reopen(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    first = SqliteCheckpointRuntime.open(path)
    graph = _build_counter_graph(first.saver)
    graph.invoke({"value": 1}, {"configurable": {"thread_id": "conv-1"}})
    first.close()

    second = SqliteCheckpointRuntime.open(path)
    try:
        assert _build_counter_graph(second.saver).get_state(
            {"configurable": {"thread_id": "conv-1"}}
        ).values["value"] == 2
    finally:
        second.close()


def test_checkpoint_runtime_close_is_idempotent(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    runtime = SqliteCheckpointRuntime.open(path)

    runtime.close()
    runtime.close()
    assert path.with_name("checkpoints.sqlite3.setup.lock").is_file()


def test_checkpoint_setup_failure_releases_lock_and_closes_connection(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    connections = []
    real_connect = sqlite3.connect
    real_setup = SqliteSaver.setup
    setup_attempts = 0

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def fail_first_setup(saver) -> None:
        nonlocal setup_attempts
        setup_attempts += 1
        if setup_attempts == 1:
            raise RuntimeError("injected setup failure")
        real_setup(saver)

    monkeypatch.setattr(checkpoint_module.sqlite3, "connect", tracking_connect)
    monkeypatch.setattr(SqliteSaver, "setup", fail_first_setup)

    with pytest.raises(RuntimeError, match="injected setup failure"):
        SqliteCheckpointRuntime.open(path)

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connections[0].execute("SELECT 1")

    recovered = SqliteCheckpointRuntime.open(path)
    recovered.close()


def test_checkpoint_runtime_health_fails_after_close(tmp_path) -> None:
    runtime = SqliteCheckpointRuntime.open(tmp_path / "checkpoints.sqlite3")
    runtime.check_health()
    runtime.close()

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        runtime.check_health()


def test_checkpoint_setup_cli_creates_sqlite_tables(tmp_path) -> None:
    path = tmp_path / "cli-checkpoints.sqlite3"
    project_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-m", "paperpilot.web.checkpoint", "--setup"],
        cwd=project_root,
        env={
            "LANGGRAPH_STRICT_MSGPACK": "true",
            "PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH": str(path),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"checkpoints", "writes"} <= tables


def test_historical_checkpoint_can_be_forked_without_replacing_old_branch(
    tmp_path,
) -> None:
    runtime = SqliteCheckpointRuntime.open(tmp_path / "checkpoints.sqlite3")
    try:
        graph = _build_counter_graph(runtime.saver)
        thread_config = {"configurable": {"thread_id": "conv-fork"}}

        assert graph.invoke({"value": 1}, thread_config)["value"] == 2
        fork_target = graph.get_state(thread_config)
        fork_target_id = fork_target.config["configurable"]["checkpoint_id"]

        assert graph.invoke({"value": 1}, thread_config)["value"] == 4
        old_branch_tip = graph.get_state(thread_config)

        fork_result = graph.invoke({"value": 10}, fork_target.config)
        assert fork_result["value"] == 13
        fork_tip = next(
            snapshot
            for snapshot in graph.get_state_history(thread_config)
            if snapshot.values.get("value") == 13
        )

        ancestor = fork_tip
        ancestor_ids = []
        while ancestor.parent_config is not None:
            ancestor_id = ancestor.parent_config["configurable"]["checkpoint_id"]
            ancestor_ids.append(ancestor_id)
            ancestor = graph.get_state(ancestor.parent_config)

        assert fork_target_id in ancestor_ids
        assert graph.get_state(old_branch_tip.config).values["value"] == 4
    finally:
        runtime.close()


def test_four_processes_can_write_independent_threads_to_one_database(
    tmp_path,
) -> None:
    path = tmp_path / "multiprocess-checkpoints.sqlite3"
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    processes = [
        context.Process(
            target=_write_counter_states,
            args=(str(path), f"process-{index}", start_event),
        )
        for index in range(4)
    ]

    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=30)

    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0, 0, 0]

    runtime = SqliteCheckpointRuntime.open(path)
    try:
        graph = _build_counter_graph(runtime.saver)
        for index in range(4):
            snapshot = graph.get_state(
                {"configurable": {"thread_id": f"process-{index}"}}
            )
            assert snapshot.values["value"] == 10
    finally:
        runtime.close()
