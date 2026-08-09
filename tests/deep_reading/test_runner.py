"""Durable DeepReadingRunner integration and recovery tests."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.messages import AIMessage, HumanMessage
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError as SQLAlchemyOperationalError

import paperpilot.deep_reading.graph as graph_module
import paperpilot.deep_reading.runner as runner_module
from paperpilot.core.adapter import Tool
from paperpilot.deep_reading.graph import build_deep_reading_graph
from paperpilot.deep_reading.research_agent import (
    DeepReadingTaskError,
    ResearchContractError,
)
from paperpilot.deep_reading.runner import (
    DeepReadingCheckpoint,
    DeepReadingRunner,
    build_deep_reading_model,
)
from paperpilot.deep_reading.schemas import ConversationSummary
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

PROJECT_ROOT = Path(__file__).parents[2]


class _FakeMCPClient:
    def __init__(self) -> None:
        self.start_count = 0
        self.close_count = 0
        self.list_tools_count = 0

    def start(self) -> None:
        self.start_count += 1

    def list_tools(self) -> list[Tool]:
        self.list_tools_count += 1
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


class _SummaryParserErrorModel:
    def __init__(self, parser_error: ValidationError) -> None:
        self.parser_error = parser_error
        self.invoke_count = 0
        self.structured_output_calls: list[tuple[type, bool]] = []

    def with_structured_output(
        self,
        schema: type,
        *,
        include_raw: bool = False,
    ) -> object:
        self.structured_output_calls.append((schema, include_raw))
        return self

    def invoke(self, _model_input: object) -> object:
        self.invoke_count += 1
        return {
            "raw": AIMessage(content="secret-token full-paper-text"),
            "parsed": None,
            "parsing_error": self.parser_error,
        }


class _ProviderFailureModel:
    def __init__(self, failure: ConnectionError) -> None:
        self.failure = failure
        self.invoke_count = 0
        self.structured_output_calls: list[tuple[type, bool]] = []

    def with_structured_output(
        self,
        schema: type,
        *,
        include_raw: bool = False,
    ) -> object:
        self.structured_output_calls.append((schema, include_raw))
        return self

    def invoke(self, _model_input: object) -> object:
        self.invoke_count += 1
        raise self.failure


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


def _execute_business_sql(
    store: TaskStore,
    statement: str,
    parameters: dict[str, object],
) -> None:
    with store.engine.begin() as connection:
        connection.execute(text(statement), parameters)


def test_deepseek_model_disables_provider_retries_and_bounds_output(
    monkeypatch,
) -> None:
    construction: list[dict[str, object]] = []

    class RecordingChatDeepSeek:
        def __init__(self, **kwargs) -> None:
            construction.append(kwargs)

    import langchain_deepseek

    monkeypatch.setattr(langchain_deepseek, "ChatDeepSeek", RecordingChatDeepSeek)

    model = build_deep_reading_model(research_max_output_tokens=4096)

    assert isinstance(model, RecordingChatDeepSeek)
    assert construction == [
        {
            "model": "deepseek-chat",
            "temperature": 0,
            "max_tokens": 4096,
            "max_retries": 0,
        }
    ]


def test_strict_checkpoint_serializer_rejects_custom_application_object(
    tmp_path,
) -> None:
    script = textwrap.dedent(
        """
        import sys
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        from paperpilot.web.checkpoint import SqliteCheckpointRuntime


        class UnsafeApplicationObject:
            pass


        class UnsafeState(TypedDict, total=False):
            value: object


        def persist_unsafe_object(_state: UnsafeState) -> UnsafeState:
            return {"value": UnsafeApplicationObject()}


        runtime = SqliteCheckpointRuntime.open(sys.argv[1])
        builder = StateGraph(UnsafeState)
        builder.add_node("unsafe", persist_unsafe_object)
        builder.add_edge(START, "unsafe")
        builder.add_edge("unsafe", END)
        graph = builder.compile(checkpointer=runtime.saver)
        config = {"configurable": {"thread_id": "strict-security"}}
        try:
            graph.invoke({"value": "safe"}, config)
        except TypeError as exc:
            if "not msgpack serializable: UnsafeApplicationObject" not in str(exc):
                raise AssertionError(f"unexpected serializer failure: {exc}") from exc
        else:
            raise AssertionError("unsafe application object was checkpointed")
        runtime.close()

        reopened = SqliteCheckpointRuntime.open(sys.argv[1])
        try:
            restored_graph = builder.compile(checkpointer=reopened.saver)
            snapshot = restored_graph.get_state(config)
            if isinstance(snapshot.values.get("value"), UnsafeApplicationObject):
                raise AssertionError("unsafe application object was recovered")
            if snapshot.values.get("value") != "safe":
                raise AssertionError(
                    f"unexpected value survived rejection: {snapshot.values!r}"
                )
        finally:
            reopened.close()
        print("strict serializer rejected and did not recover custom object")
        """
    )
    environment = os.environ.copy()
    environment["LANGGRAPH_STRICT_MSGPACK"] = "true"

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "strict.sqlite3")],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "strict serializer rejected and did not recover custom object"
    )
    assert result.stderr == ""


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


def test_runner_passes_custom_runtime_bounds_into_graph_context(
    tmp_path,
    monkeypatch,
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    seen_bounds: list[tuple[int, int, int, int, int, int, int]] = []

    def record_context(_state, runtime):
        context = runtime.context
        seen_bounds.append(
            (
                context.summary_token_threshold,
                context.summary_recent_turns,
                context.research_recursion_limit,
                context.research_model_call_limit,
                context.research_tool_call_limit,
                context.research_max_output_tokens,
                context.research_model_retries,
            )
        )
        return {
            "research_result": {
                "evidence_items": [],
                "used_papers": [],
                "limitations": ["bounded test"],
            }
        }

    monkeypatch.setattr(graph_module, "research_evidence", record_context)
    runner = DeepReadingRunner(
        task_store=store,
        checkpointer=checkpoint_runtime.saver,
        mcp_runtime=mcp_runtime,
        model_factory=_ModelFactory([]),
        paper_search=lambda _query, _limit: [],
        summary_token_threshold=1234,
        summary_recent_turns=3,
        research_recursion_limit=9,
        research_model_call_limit=8,
        research_tool_call_limit=14,
        research_max_output_tokens=2048,
        research_model_retries=0,
    )
    try:
        turn = _new_turn(store, user, conversation, "Use custom runtime bounds")
        runner.run(turn.task.id)

        assert seen_bounds == [(1234, 3, 9, 8, 14, 2048, 0)]
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


@pytest.mark.parametrize(
    ("invalid_kwargs", "message"),
    [
        ({"summary_token_threshold": 0}, "must be positive"),
        ({"summary_recent_turns": 0}, "must be positive"),
        ({"research_recursion_limit": 0}, "must be positive"),
        ({"research_model_call_limit": 1}, "structured-response attempts"),
        ({"research_tool_call_limit": 1}, "structured-response attempts"),
        ({"research_max_output_tokens": 0}, "must be positive"),
        ({"research_model_retries": -1}, "must be nonnegative"),
    ],
)
def test_runner_rejects_invalid_runtime_bounds(invalid_kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        DeepReadingRunner(
            task_store=object(),
            checkpointer=object(),
            mcp_runtime=object(),
            **invalid_kwargs,
        )


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
    ("field", "bad_value", "expected_error_code"),
    [
        ("current_task_id", "another-task", "final_checkpoint_invalid"),
        (
            "current_user_message_id",
            "another-message",
            "final_checkpoint_invalid",
        ),
        ("graph_version", "another-version", "graph_version_unsupported"),
        ("schema_version", 999, "schema_version_unsupported"),
        (
            "published_message_id",
            "another-assistant",
            "final_checkpoint_invalid",
        ),
    ],
)
def test_corrupt_complete_recovery_snapshot_is_terminal_before_model_or_mcp(
    tmp_path,
    monkeypatch,
    field,
    bad_value,
    expected_error_code,
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_client = _FakeMCPClient()
    mcp_runtime = MCPRuntime(lambda: mcp_client)
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
        leases_before = mcp_client.list_tools_count

        runner.run(turn.task.id)

        assert model_factory.factory_calls == calls_before
        assert mcp_client.list_tools_count == leases_before
        assert store.get_task(turn.task.id, user_id=user.id).status == "failed"
        events = store.list_events_page(
            turn.task.id,
            user_id=user.id,
            after_id=0,
            limit=100,
        )
        assert events is not None
        failure = [event for event in events.items if event.type == "failed"][0]
        assert failure.stage == "deep_reading_terminal"
        assert failure.payload["error_code"] == expected_error_code
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


@pytest.mark.parametrize(
    ("corruption", "expected_error_code"),
    [
        ("nonexistent_base", "checkpoint_missing"),
        ("cross_thread_base", "checkpoint_missing"),
        ("task_base_mismatch", "checkpoint_binding_invalid"),
        ("user_parent_mismatch", "checkpoint_binding_invalid"),
        ("unpaired_business_head", "checkpoint_binding_invalid"),
        ("incomplete_base", "checkpoint_incomplete"),
        ("base_graph_version", "graph_version_unsupported"),
        ("base_schema_version", "schema_version_unsupported"),
        ("base_published_head", "checkpoint_binding_invalid"),
    ],
)
def test_active_task_rejects_untrusted_business_base_before_model_or_mcp(
    tmp_path, corruption, expected_error_code
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    seed_mcp_runtime = MCPRuntime(_FakeMCPClient)
    seed_runner = _runner(
        store,
        checkpoint_runtime,
        seed_mcp_runtime,
        _ModelFactory([]),
    )
    seed_turn = _new_turn(store, user, conversation, "trusted seed")
    seed_runner.run(seed_turn.task.id)
    seed_task = store.get_task(seed_turn.task.id, user_id=user.id)
    seed_assistant = store.get_task_message(seed_turn.task.id, "assistant")
    assert seed_task is not None and seed_task.final_checkpoint_id is not None
    assert seed_assistant is not None

    other_checkpoint_id: str | None = None
    if corruption == "cross_thread_base":
        other_conversation = store.create_conversation(user_id=user.id, paper=PRIMARY)
        other_turn = _new_turn(store, user, other_conversation, "other thread")
        seed_runner.run(other_turn.task.id)
        other_task = store.get_task(other_turn.task.id, user_id=user.id)
        assert other_task is not None and other_task.final_checkpoint_id is not None
        other_checkpoint_id = other_task.final_checkpoint_id
    seed_mcp_runtime.close()

    turn = _new_turn(store, user, conversation, "must validate base")
    graph = build_deep_reading_graph(checkpoint_runtime.saver)
    seed_snapshot = graph.get_state(
        {
            "configurable": {
                "thread_id": conversation.id,
                "checkpoint_id": seed_task.final_checkpoint_id,
            }
        }
    )
    seed_update_config = {
        "configurable": dict(seed_snapshot.config["configurable"])
    }
    seed_update_config["configurable"].setdefault("checkpoint_ns", "")
    replacement_base: str | None = None
    if corruption == "nonexistent_base":
        replacement_base = "checkpoint-does-not-exist"
    elif corruption == "cross_thread_base":
        replacement_base = other_checkpoint_id
    elif corruption == "incomplete_base":
        incomplete_config = graph.update_state(
            seed_update_config,
            {},
            as_node="write_answer",
        )
        replacement_base = incomplete_config["configurable"]["checkpoint_id"]
        assert graph.get_state(incomplete_config).next == ("publish_result",)
    elif corruption in {
        "base_graph_version",
        "base_schema_version",
        "base_published_head",
    }:
        field, value = {
            "base_graph_version": ("graph_version", "conversation-v999"),
            "base_schema_version": ("schema_version", 999),
            "base_published_head": (
                "published_message_id",
                "another-assistant-message",
            ),
        }[corruption]
        corrupt_config = graph.update_state(
            seed_update_config,
            {field: value},
        )
        replacement_base = corrupt_config["configurable"]["checkpoint_id"]
        assert graph.get_state(corrupt_config).next == ()

    if replacement_base is not None:
        _execute_business_sql(
            store,
            "UPDATE research_tasks SET base_checkpoint_id = :checkpoint_id "
            "WHERE id = :task_id",
            {"checkpoint_id": replacement_base, "task_id": turn.task.id},
        )
        _execute_business_sql(
            store,
            "UPDATE conversations SET head_checkpoint_id = :checkpoint_id "
            "WHERE id = :conversation_id",
            {
                "checkpoint_id": replacement_base,
                "conversation_id": conversation.id,
            },
        )
    elif corruption == "task_base_mismatch":
        _execute_business_sql(
            store,
            "UPDATE research_tasks SET base_checkpoint_id = :checkpoint_id "
            "WHERE id = :task_id",
            {"checkpoint_id": "mismatched-task-base", "task_id": turn.task.id},
        )
    elif corruption == "user_parent_mismatch":
        _execute_business_sql(
            store,
            "UPDATE messages SET parent_message_id = NULL WHERE id = :message_id",
            {"message_id": turn.user_message.id},
        )
    elif corruption == "unpaired_business_head":
        _execute_business_sql(
            store,
            "UPDATE research_tasks SET base_checkpoint_id = NULL WHERE id = :task_id",
            {"task_id": turn.task.id},
        )
        _execute_business_sql(
            store,
            "UPDATE conversations SET head_checkpoint_id = NULL "
            "WHERE id = :conversation_id",
            {"conversation_id": conversation.id},
        )

    mcp_client = _FakeMCPClient()
    mcp_runtime = MCPRuntime(lambda: mcp_client)
    model_factory = _ModelFactory([])
    runner = _runner(
        store,
        checkpoint_runtime,
        mcp_runtime,
        model_factory,
    )
    try:
        runner.run(turn.task.id)

        task = store.get_task(turn.task.id, user_id=user.id)
        assert task is not None and task.status == "failed"
        assert store.get_task_message(turn.task.id, "assistant") is None
        assert model_factory.factory_calls == 0
        assert mcp_client.start_count == 0
        assert mcp_client.list_tools_count == 0
        events = store.list_events_page(
            turn.task.id,
            user_id=user.id,
            after_id=0,
            limit=100,
        )
        assert events is not None
        failures = [event for event in events.items if event.type == "failed"]
        assert len(failures) == 1
        failure = failures[0]
        assert failure.stage == "deep_reading_terminal"
        assert failure.payload["error_code"] == expected_error_code
        assert set(failure.payload) == {"error_code", "error_type"}
        rendered_failure = str(failure.to_dict())
        assert "checkpoint-does-not-exist" not in rendered_failure
        assert "conversation-v999" not in rendered_failure
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


def test_research_contract_error_is_terminal_and_public_event_is_safe(
    tmp_path,
    monkeypatch,
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
    turn = _new_turn(store, user, conversation, "terminal contract")

    def raise_contract(_state, runtime):
        del runtime
        raise ResearchContractError("secret-token full-paper-text")

    monkeypatch.setattr(graph_module, "research_evidence", raise_contract)
    try:
        runner.run(turn.task.id)

        task = store.get_task(turn.task.id, user_id=user.id)
        assert task is not None and task.status == "failed"
        events = store.list_events_page(
            turn.task.id,
            user_id=user.id,
            after_id=0,
            limit=100,
        )
        assert events is not None
        failures = [event for event in events.items if event.type == "failed"]
        assert len(failures) == 1
        failure = failures[0]
        assert failure.stage == "deep_reading_terminal"
        assert failure.payload == {
            "error_code": "research_contract_invalid",
            "error_type": "ResearchContractError",
        }
        assert "secret-token" not in str(failure.to_dict())
        assert "full-paper-text" not in str(failure.to_dict())
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_summary_parser_error_is_terminal_without_mcp_or_reexecution(
    tmp_path,
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_calls: list[str] = []

    class CountingMCPClient(_FakeMCPClient):
        def list_tools(self) -> list[Tool]:
            tools = super().list_tools()
            return [
                Tool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                    handler=lambda _args, name=tool.name: mcp_calls.append(name),
                )
                for tool in tools
            ]

    mcp_runtime = MCPRuntime(CountingMCPClient)
    with pytest.raises(ValidationError) as parser_exc:
        ConversationSummary.model_validate(
            {
                "confirmed_facts": ["secret-token full-paper-text", ""],
                "paper_findings": [],
                "comparison_context": [],
                "open_questions": [],
            }
        )
    model = _SummaryParserErrorModel(parser_exc.value)
    factory_calls: list[str] = []

    def model_factory() -> _SummaryParserErrorModel:
        factory_calls.append("model")
        return model

    runner = DeepReadingRunner(
        task_store=store,
        checkpointer=checkpoint_runtime.saver,
        mcp_runtime=mcp_runtime,
        model_factory=model_factory,
        paper_search=lambda _query, _limit: [],
        summary_token_threshold=1,
    )
    turn = _new_turn(
        store,
        user,
        conversation,
        "A long question that must enter the summary parser boundary.",
    )
    try:
        runner.run(turn.task.id)

        task = store.get_task(turn.task.id, user_id=user.id)
        assert task is not None and task.status == "failed"
        events = store.list_events_page(
            turn.task.id,
            user_id=user.id,
            after_id=0,
            limit=100,
        )
        assert events is not None
        failures = [event for event in events.items if event.type == "failed"]
        assert len(failures) == 1
        failure = failures[0]
        assert failure.stage == "deep_reading_terminal"
        assert failure.payload == {
            "error_code": "research_contract_invalid",
            "error_type": "ResearchContractError",
        }
        assert "secret-token" not in str(failure.to_dict())
        assert "full-paper-text" not in str(failure.to_dict())
        assert factory_calls == ["model"]
        assert model.invoke_count == 1
        assert model.structured_output_calls == [(ConversationSummary, True)]
        assert mcp_calls == []
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_summary_provider_connection_error_escapes_runner_with_identity(
    tmp_path,
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    failure = ConnectionError("provider connection reset")
    model = _ProviderFailureModel(failure)
    factory_calls: list[str] = []

    def model_factory() -> _ProviderFailureModel:
        factory_calls.append("model")
        return model

    runner = DeepReadingRunner(
        task_store=store,
        checkpointer=checkpoint_runtime.saver,
        mcp_runtime=mcp_runtime,
        model_factory=model_factory,
        paper_search=lambda _query, _limit: [],
        summary_token_threshold=1,
    )
    turn = _new_turn(
        store,
        user,
        conversation,
        "A long question that reaches the provider exactly once.",
    )
    try:
        with pytest.raises(ConnectionError) as exc_info:
            runner.run(turn.task.id)

        assert exc_info.value is failure
        task = store.get_task(turn.task.id, user_id=user.id)
        assert task is not None and task.status == "pending"
        events = store.list_events_page(
            turn.task.id,
            user_id=user.id,
            after_id=0,
            limit=100,
        )
        assert events is not None
        assert [event for event in events.items if event.type == "failed"] == []
        assert factory_calls == ["model"]
        assert model.invoke_count == 1
        assert model.structured_output_calls == [(ConversationSummary, True)]
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_invalid_task_status_records_stable_terminal_code(monkeypatch) -> None:
    failures: list[dict[str, object]] = []

    class RecordingStore:
        def fail_conversation_task(self, **kwargs):
            failures.append(kwargs)

    runner = DeepReadingRunner(
        task_store=RecordingStore(),  # type: ignore[arg-type]
        checkpointer=object(),
        mcp_runtime=object(),  # type: ignore[arg-type]
        model_factory=lambda: object(),
    )
    monkeypatch.setattr(
        runner,
        "_load_business_binding",
        lambda _task_id: (
            SimpleNamespace(id="task-status", status="cancelled"),
            SimpleNamespace(id="message-status"),
            SimpleNamespace(id="conversation-status"),
            "paper-status",
        ),
    )

    runner.run("task-status")

    assert failures == [
        {
            "task_id": "task-status",
            "message": "The task is not in an executable state.",
            "stage": "deep_reading_terminal",
            "payload": {
                "error_code": "task_status_invalid",
                "error_type": "TaskStatusError",
            },
        }
    ]


def test_missing_published_assistant_becomes_final_checkpoint_failure(
    tmp_path,
    monkeypatch,
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    runner = _runner(store, checkpoint_runtime, mcp_runtime, _ModelFactory([]))
    turn = _new_turn(store, user, conversation, "missing assistant")

    monkeypatch.setattr(
        graph_module,
        "publish_result",
        lambda _state, runtime: {"published_message_id": None},
    )
    try:
        runner.run(turn.task.id)

        task = store.get_task(turn.task.id, user_id=user.id)
        assert task is not None and task.status == "failed"
        events = store.list_events_page(
            turn.task.id,
            user_id=user.id,
            after_id=0,
            limit=100,
        )
        assert events is not None
        failure = [event for event in events.items if event.type == "failed"][0]
        assert failure.stage == "deep_reading_terminal"
        assert failure.payload["error_code"] == "final_checkpoint_invalid"
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


@pytest.mark.parametrize(
    "failure",
    [
        sqlite3.OperationalError("database locked"),
        ConnectionError("provider connection reset"),
    ],
)
def test_database_and_provider_errors_escape_for_task_retry(
    tmp_path,
    monkeypatch,
    failure,
) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    runner = _runner(store, checkpoint_runtime, mcp_runtime, _ModelFactory([]))
    turn = _new_turn(store, user, conversation, "transient infrastructure")

    def raise_infrastructure(_state, runtime):
        del runtime
        raise failure

    monkeypatch.setattr(graph_module, "research_evidence", raise_infrastructure)
    try:
        with pytest.raises(type(failure)) as exc_info:
            runner.run(turn.task.id)

        assert exc_info.value is failure
        assert store.get_task(turn.task.id, user_id=user.id).status == "pending"
        events = store.list_events_page(
            turn.task.id,
            user_id=user.id,
            after_id=0,
            limit=100,
        )
        assert events is not None
        assert [event for event in events.items if event.type == "failed"] == []
    finally:
        mcp_runtime.close()
        checkpoint_runtime.close()
        store.close()


def test_saver_error_escapes_for_task_retry(tmp_path, monkeypatch) -> None:
    store, user, conversation = _create_store_and_conversation(
        tmp_path / "business.sqlite3"
    )
    checkpoint_runtime = SqliteCheckpointRuntime.open(
        tmp_path / "checkpoints.sqlite3"
    )
    mcp_runtime = MCPRuntime(_FakeMCPClient)
    runner = _runner(store, checkpoint_runtime, mcp_runtime, _ModelFactory([]))
    turn = _new_turn(store, user, conversation, "saver failure")
    failure = OSError("checkpoint unavailable")

    monkeypatch.setattr(
        runner_module,
        "build_deep_reading_graph",
        lambda _checkpointer: (_ for _ in ()).throw(failure),
    )
    try:
        with pytest.raises(OSError) as exc_info:
            runner.run(turn.task.id)

        assert exc_info.value is failure
        assert store.get_task(turn.task.id, user_id=user.id).status == "pending"
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
        terminal_events = store.list_events_page(
            expected_turn.task.id,
            user_id=user.id,
            after_id=0,
            limit=100,
        )
        assert terminal_events is not None
        failure = [
            event for event in terminal_events.items if event.type == "failed"
        ][0]
        assert failure.stage == "deep_reading_terminal"
        assert failure.payload["error_code"] == "research_contract_invalid"
        assert "expected terminal model failure" not in str(failure.to_dict())

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
