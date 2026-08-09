"""Durable DeepReadingRunner integration and recovery tests."""
from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from langchain.messages import HumanMessage
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError

import paperpilot.deep_reading.graph as graph_module
from paperpilot.core.adapter import Tool
from paperpilot.deep_reading.graph import build_deep_reading_graph
from paperpilot.deep_reading.research_agent import DeepReadingTaskError
from paperpilot.deep_reading.runner import (
    DeepReadingCheckpoint,
    DeepReadingRunner,
)
from paperpilot.papers import PaperCandidate
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.checkpoint import SqliteCheckpointRuntime
from paperpilot.web.task_store import TaskStore


PRIMARY = PaperCandidate(
    external_id="2401.12345v1",
    title="Primary paper",
    authors=["Ada Lovelace"],
    abstract="A primary-paper abstract.",
    source_url="https://arxiv.org/abs/2401.12345v1",
)


class _FakeMCPClient:
    def __init__(self) -> None:
        self.start_count = 0
        self.close_count = 0

    def start(self) -> None:
        self.start_count += 1

    def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name="mcp__arxiv__download_paper",
                description="fake download",
                input_schema={"type": "object"},
                handler=lambda args: (
                    '{"paper_id":"%s","text":"trusted paper text"}'
                    % args["arxiv_id"]
                ),
            ),
            Tool(
                name="mcp__colbert__build_index",
                description="fake build",
                input_schema={"type": "object"},
                handler=lambda args: (
                    '{"fresh_papers":["%s"]}'
                    % args["documents"][0]["paper_id"]
                ),
            ),
        ]

    def close(self) -> None:
        self.close_count += 1


class _RecordingModel:
    def __init__(self, calls: list[dict[str, Any]], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail

    def write(self, state: dict[str, Any], task_id: str) -> dict[str, object]:
        self.calls.append(
            {
                "task_id": task_id,
                "message_ids": [message.id for message in state["messages"]],
                "message_contents": [
                    message.content for message in state["messages"]
                ],
            }
        )
        if self.fail:
            raise DeepReadingTaskError("expected terminal model failure")
        return {
            "content": f"answer for {task_id}",
            "citations": [],
            "result_quality": "partial",
        }


class _ModelFactory:
    def __init__(self, calls: list[dict[str, Any]], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail
        self.factory_calls = 0

    def __call__(self) -> _RecordingModel:
        self.factory_calls += 1
        return _RecordingModel(self.calls, fail=self.fail)


@pytest.fixture(autouse=True)
def _fake_expensive_nodes(monkeypatch):
    def fake_research(state: dict[str, Any], runtime: Any) -> dict[str, object]:
        del state
        return {
            "research_result": {
                "evidence_items": [],
                "used_papers": [],
                "limitations": [f"bounded fake research for {runtime.context.task_id}"],
            }
        }

    def fake_writer(state: dict[str, Any], runtime: Any) -> dict[str, object]:
        return {
            "answer_draft": runtime.context.model.write(
                state, runtime.context.task_id
            )
        }

    monkeypatch.setattr(graph_module, "research_evidence", fake_research)
    monkeypatch.setattr(graph_module, "write_answer", fake_writer)


def _create_store_and_conversation(db_path):
    store = TaskStore(db_path)
    user = store.create_user(
        username="alice",
        password_hash="hash",
        password_salt="salt",
    )
    conversation = store.create_conversation(user_id=user.id, paper=PRIMARY)
    return store, user, conversation


def _new_turn(store, user, conversation, content: str):
    detail = store.get_conversation_detail(conversation.id, user_id=user.id)
    assert detail is not None
    return store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content=content,
        depth="standard",
        expected_head_message_id=detail.conversation.head_message_id,
    )


def _runner(store, checkpoint_runtime, mcp_runtime, model_factory):
    return DeepReadingRunner(
        task_store=store,
        checkpointer=checkpoint_runtime.saver,
        mcp_runtime=mcp_runtime,
        model_factory=model_factory,
        paper_search=lambda _query, _limit: [],
    )


def _artifact_count(store: TaskStore, task_id: str, user_id: str) -> int:
    batch = store.list_artifacts_page(
        task_id,
        user_id=user_id,
        after_id=0,
        limit=100,
    )
    assert batch is not None
    return len(batch.items)


def test_success_finalizes_business_head_and_exposes_frozen_checkpoint(tmp_path) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_client = _FakeMCPClient()
    mcp_runtime = MCPRuntime(lambda: mcp_client)
    calls: list[dict[str, Any]] = []
    runner = _runner(
        store,
        checkpoint_runtime,
        mcp_runtime,
        _ModelFactory(calls),
    )
    try:
        turn = _new_turn(store, user, conversation, "What does the paper show?")
        runner.run(turn.task.id)

        task = store.get_task(turn.task.id, user_id=user.id)
        assistant = store.get_task_message(turn.task.id, "assistant")
        detail = store.get_conversation_detail(conversation.id, user_id=user.id)
        assert task is not None and assistant is not None and detail is not None
        assert task.status == "completed"
        assert task.final_checkpoint_id
        assert detail.conversation.head_message_id == assistant.id
        assert detail.conversation.head_checkpoint_id == task.final_checkpoint_id
        assert _artifact_count(store, task.id, user.id) == 1

        checkpoint = runner.read_checkpoint(
            conversation.id, task.final_checkpoint_id
        )
        assert isinstance(checkpoint, DeepReadingCheckpoint)
        assert checkpoint.is_complete is True
        assert checkpoint.state["current_task_id"] == task.id
        assert checkpoint.state["current_user_message_id"] == turn.user_message.id
        assert checkpoint.state["published_message_id"] == assistant.id
        assert [message.id for message in checkpoint.state["messages"]][-2:] == [
            turn.user_message.id,
            assistant.id,
        ]
        with pytest.raises(FrozenInstanceError):
            checkpoint.checkpoint_id = "mutated"  # type: ignore[misc]
        assert runner.read_checkpoint(conversation.id, "missing") is None
        assert len(calls) == 1
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_restart_and_second_turn_preserve_first_turn_messages(tmp_path) -> None:
    business_path = tmp_path / "business.sqlite3"
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    store, user, conversation = _create_store_and_conversation(business_path)
    first_checkpoint_runtime = SqliteCheckpointRuntime.open(checkpoint_path)
    first_mcp_runtime = MCPRuntime(_FakeMCPClient)
    first_calls: list[dict[str, Any]] = []
    first_runner = _runner(
        store,
        first_checkpoint_runtime,
        first_mcp_runtime,
        _ModelFactory(first_calls),
    )
    first_turn = _new_turn(store, user, conversation, "first question")
    first_runner.run(first_turn.task.id)
    first_task = store.get_task(first_turn.task.id, user_id=user.id)
    first_assistant = store.get_task_message(first_turn.task.id, "assistant")
    assert first_task is not None and first_task.final_checkpoint_id
    assert first_assistant is not None
    first_checkpoint_id = first_task.final_checkpoint_id
    first_mcp_runtime.close()
    first_checkpoint_runtime.close()
    store.close()

    reopened_store = TaskStore(business_path)
    reopened_checkpoint_runtime = SqliteCheckpointRuntime.open(checkpoint_path)
    second_mcp_runtime = MCPRuntime(_FakeMCPClient)
    second_calls: list[dict[str, Any]] = []
    second_runner = _runner(
        reopened_store,
        reopened_checkpoint_runtime,
        second_mcp_runtime,
        _ModelFactory(second_calls),
    )
    try:
        second_turn = _new_turn(
            reopened_store, user, conversation, "second question"
        )
        assert second_turn.task.base_checkpoint_id == first_checkpoint_id
        second_runner.run(second_turn.task.id)

        assert len(second_calls) == 1
        assert second_calls[0]["message_ids"] == [
            first_turn.user_message.id,
            first_assistant.id,
            second_turn.user_message.id,
        ]
        assert second_calls[0]["message_contents"] == [
            "first question",
            f"answer for {first_turn.task.id}",
            "second question",
        ]
    finally:
        second_mcp_runtime.close()
        reopened_checkpoint_runtime.close()
        reopened_store.close()


def test_rollback_is_model_free_and_third_turn_forks_from_first_checkpoint(
    tmp_path,
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    calls: list[dict[str, Any]] = []
    model_factory = _ModelFactory(calls)
    runner = _runner(store, checkpoint_runtime, mcp_runtime, model_factory)
    try:
        first_turn = _new_turn(store, user, conversation, "first question")
        runner.run(first_turn.task.id)
        first_task = store.get_task(first_turn.task.id, user_id=user.id)
        first_assistant = store.get_task_message(first_turn.task.id, "assistant")
        assert first_task is not None and first_task.final_checkpoint_id
        assert first_assistant is not None

        second_turn = _new_turn(store, user, conversation, "second question")
        runner.run(second_turn.task.id)
        second_task = store.get_task(second_turn.task.id, user_id=user.id)
        second_assistant = store.get_task_message(second_turn.task.id, "assistant")
        assert second_task is not None and second_task.final_checkpoint_id
        assert second_assistant is not None
        second_checkpoint_id = second_task.final_checkpoint_id

        calls_before_rollback = len(calls)
        first_checkpoint = runner.read_checkpoint(
            conversation.id, first_task.final_checkpoint_id
        )
        assert first_checkpoint is not None
        switched = store.switch_conversation_head(
            conversation.id,
            user_id=user.id,
            expected_head_message_id=second_assistant.id,
            target_message_id=first_assistant.id,
            target_checkpoint_id=first_task.final_checkpoint_id,
            active_paper_ids=first_checkpoint.state["active_paper_ids"],
        )
        assert switched is not None
        assert len(calls) == calls_before_rollback
        assert model_factory.factory_calls == calls_before_rollback

        third_turn = _new_turn(store, user, conversation, "third question")
        assert third_turn.task.base_checkpoint_id == first_task.final_checkpoint_id
        runner.run(third_turn.task.id)
        third_task = store.get_task(third_turn.task.id, user_id=user.id)
        assert third_task is not None and third_task.final_checkpoint_id

        assert runner.read_checkpoint(
            conversation.id, second_checkpoint_id
        ) is not None
        graph = build_deep_reading_graph(checkpoint_runtime.saver)
        ancestor = graph.get_state(
            {
                "configurable": {
                    "thread_id": conversation.id,
                    "checkpoint_id": third_task.final_checkpoint_id,
                }
            }
        )
        ancestor_ids: list[str] = []
        while ancestor.parent_config is not None:
            parent_id = ancestor.parent_config["configurable"]["checkpoint_id"]
            ancestor_ids.append(parent_id)
            ancestor = graph.get_state(ancestor.parent_config)
        assert first_task.final_checkpoint_id in ancestor_ids
        assert second_checkpoint_id not in ancestor_ids
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_redelivery_after_publish_checkpoint_failure_does_not_duplicate_rows(
    tmp_path, monkeypatch
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    runner = _runner(
        store,
        checkpoint_runtime,
        mcp_runtime,
        _ModelFactory([]),
    )
    turn = _new_turn(store, user, conversation, "crash A")
    real_put = checkpoint_runtime.saver.put
    injected = False

    def fail_after_publish(config, checkpoint, metadata, new_versions):
        nonlocal injected
        channel_values = checkpoint.get("channel_values", {})
        if not injected and channel_values.get("published_message_id"):
            injected = True
            raise sqlite3.OperationalError("injected checkpoint write failure")
        return real_put(config, checkpoint, metadata, new_versions)

    monkeypatch.setattr(checkpoint_runtime.saver, "put", fail_after_publish)
    try:
        with pytest.raises(sqlite3.OperationalError, match="checkpoint write"):
            runner.run(turn.task.id)
        assert store.get_task_message(turn.task.id, "assistant") is not None
        assert _artifact_count(store, turn.task.id, user.id) == 1

        monkeypatch.setattr(checkpoint_runtime.saver, "put", real_put)
        runner.run(turn.task.id)
        task = store.get_task(turn.task.id, user_id=user.id)
        assert task is not None and task.status == "completed"
        assert task.final_checkpoint_id is not None
        assert store.get_task_message(turn.task.id, "assistant") is not None
        assert _artifact_count(store, turn.task.id, user.id) == 1
        recovered = runner.read_checkpoint(
            conversation.id, task.final_checkpoint_id
        )
        assert recovered is not None and recovered.is_complete is True
        assert recovered.state["published_message_id"] == store.get_task_message(
            turn.task.id, "assistant"
        ).id
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_redelivery_after_finalization_failure_uses_only_trusted_complete_snapshot(
    tmp_path, monkeypatch
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_client = _FakeMCPClient()
    mcp_runtime = MCPRuntime(lambda: mcp_client)
    calls: list[dict[str, Any]] = []
    model_factory = _ModelFactory(calls)
    runner = _runner(store, checkpoint_runtime, mcp_runtime, model_factory)
    turn = _new_turn(store, user, conversation, "crash B")
    real_finalize = store.finalize_conversation_task
    finalize_calls = 0

    def fail_first_finalize(**kwargs):
        nonlocal finalize_calls
        finalize_calls += 1
        if finalize_calls == 1:
            raise SQLAlchemyOperationalError(
                "finalize conversation",
                {},
                RuntimeError("injected finalization failure"),
            )
        return real_finalize(**kwargs)

    monkeypatch.setattr(store, "finalize_conversation_task", fail_first_finalize)
    try:
        with pytest.raises(SQLAlchemyOperationalError, match="finalization"):
            runner.run(turn.task.id)
        assistant = store.get_task_message(turn.task.id, "assistant")
        assert assistant is not None
        calls_after_crash = len(calls)
        factory_calls_after_crash = model_factory.factory_calls
        mcp_starts_after_crash = mcp_client.start_count

        runner.run(turn.task.id)
        task = store.get_task(turn.task.id, user_id=user.id)
        assert task is not None and task.status == "completed"
        assert len(calls) == calls_after_crash
        assert model_factory.factory_calls == factory_calls_after_crash
        assert mcp_client.start_count == mcp_starts_after_crash
        assert store.get_task_message(turn.task.id, "assistant").id == assistant.id
        assert _artifact_count(store, turn.task.id, user.id) == 1
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("current_task_id", "another-task"),
        ("current_user_message_id", "another-message"),
        ("graph_version", "another-version"),
        ("published_message_id", "another-assistant"),
    ],
)
def test_untrusted_recovery_snapshot_is_not_finalized(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    model_factory = _ModelFactory([])
    runner = _runner(store, checkpoint_runtime, mcp_runtime, model_factory)
    turn = _new_turn(store, user, conversation, "do not trust this snapshot")
    real_finalize = store.finalize_conversation_task
    monkeypatch.setattr(
        store,
        "finalize_conversation_task",
        lambda **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("leave a complete snapshot")
        ),
    )
    try:
        with pytest.raises(sqlite3.OperationalError):
            runner.run(turn.task.id)
        assistant = store.get_task_message(turn.task.id, "assistant")
        assert assistant is not None
        graph = build_deep_reading_graph(checkpoint_runtime.saver)
        snapshot = graph.get_state(
            {"configurable": {"thread_id": conversation.id}}
        )
        values = dict(snapshot.values)
        values[field] = bad_value
        graph.update_state(snapshot.config, values)
        monkeypatch.setattr(store, "finalize_conversation_task", real_finalize)
        calls_before = model_factory.factory_calls

        runner.run(turn.task.id)

        assert model_factory.factory_calls == calls_before + 1
        assert store.get_task(turn.task.id, user_id=user.id).status == "completed"
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_incomplete_matching_snapshot_is_not_used_for_recovery(
    tmp_path, monkeypatch
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    model_factory = _ModelFactory([])
    runner = _runner(store, checkpoint_runtime, mcp_runtime, model_factory)
    turn = _new_turn(store, user, conversation, "incomplete recovery")
    real_finalize = store.finalize_conversation_task
    monkeypatch.setattr(
        store,
        "finalize_conversation_task",
        lambda **_kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("leave task pending")
        ),
    )
    try:
        with pytest.raises(sqlite3.OperationalError):
            runner.run(turn.task.id)
        graph = build_deep_reading_graph(checkpoint_runtime.saver)
        complete = graph.get_state(
            {"configurable": {"thread_id": conversation.id}}
        )
        assert complete.next == ()
        graph.update_state(complete.config, {}, as_node="write_answer")
        incomplete = graph.get_state(
            {"configurable": {"thread_id": conversation.id}}
        )
        assert incomplete.next == ("publish_result",)
        calls_before = model_factory.factory_calls
        monkeypatch.setattr(store, "finalize_conversation_task", real_finalize)

        runner.run(turn.task.id)

        assert model_factory.factory_calls == calls_before + 1
        assert store.get_task(turn.task.id, user_id=user.id).status == "completed"
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_only_deep_reading_task_error_is_marked_failed(tmp_path, monkeypatch) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    failing_runner = _runner(
        store,
        checkpoint_runtime,
        mcp_runtime,
        _ModelFactory([], fail=True),
    )
    try:
        expected_turn = _new_turn(store, user, conversation, "expected failure")
        failing_runner.run(expected_turn.task.id)
        assert store.get_task(expected_turn.task.id, user_id=user.id).status == "failed"

        # A second conversation isolates an unknown exception from the active failed turn.
        another = store.create_conversation(user_id=user.id, paper=PRIMARY)
        unknown_turn = _new_turn(store, user, another, "unknown failure")

        def raise_unknown(_state, runtime):
            del runtime
            raise RuntimeError("unknown graph failure")

        monkeypatch.setattr(graph_module, "research_evidence", raise_unknown)
        unknown_runner = _runner(
            store,
            checkpoint_runtime,
            mcp_runtime,
            _ModelFactory([]),
        )
        with pytest.raises(RuntimeError, match="unknown graph failure"):
            unknown_runner.run(unknown_turn.task.id)
        assert store.get_task(unknown_turn.task.id, user_id=user.id).status == "pending"
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_import_and_completed_task_do_not_call_default_or_injected_model_factory(
    tmp_path,
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    model_factory = _ModelFactory([])
    runner = _runner(store, checkpoint_runtime, mcp_runtime, model_factory)
    try:
        turn = _new_turn(store, user, conversation, "complete once")
        runner.run(turn.task.id)
        assert model_factory.factory_calls == 1

        runner.run(turn.task.id)
        assert model_factory.factory_calls == 1
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()
