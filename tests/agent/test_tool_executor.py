from __future__ import annotations

import json
import threading

import pytest

from paperpilot.agent.errors import (
    ToolExecutionBlocked,
    ToolExecutionFailure,
    ToolExecutionReplayRequired,
)
from paperpilot.agent.policy import RunPolicy
from paperpilot.agent.store import RunLeaseLostError, SQLiteRunStore, stable_tool_execution_id
from paperpilot.agent.tool_executor import ToolExecutor, ToolPolicyRegistry
from paperpilot.core.adapter import Tool, ToolCall, ToolResult
from paperpilot.web.task_store import TaskStore


@pytest.fixture
def store_and_run(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    task = TaskStore(db_path).create_task(question="Question")
    store = SQLiteRunStore(db_path)
    run = store.create_run(
        task_id=task.id,
        policy=RunPolicy.for_depth("standard"),
        initial_messages=[{"role": "user", "content": task.question}],
        runtime_state={"turn_count": 0, "tokens_used": 0},
    )
    assert store.claim_run(run.id, owner_id="worker-1", lease_seconds=30) is not None
    step = store.start_step(
        run_id=run.id,
        owner_id="worker-1",
        kind="tool",
        input_data={"tool_name": "mcp__colbert__search"},
    )
    return store, run, step


@pytest.fixture
def tool_registry():
    return ToolPolicyRegistry()


def _executor(store, *tools, clock=None):
    return ToolExecutor(store=store, tools=list(tools), monotonic=clock)


def test_tool_policy_registry_classifies_retryability_and_unknowns(tool_registry):
    assert tool_registry.classify("mcp__colbert__search") == "read_only"
    assert tool_registry.classify("mcp__colbert__build_index") == "idempotent_write"
    assert tool_registry.classify("paper_deep_read") == "non_retryable"
    assert tool_registry.classify("custom_mutation") == "non_retryable"


def test_tool_executor_returns_success_for_atomic_runtime_commit(store_and_run):
    store, run, step = store_and_run
    executor = _executor(
        store,
        Tool(
            name="mcp__colbert__search",
            description="search",
            input_schema={},
            handler=lambda arguments: "full result" + "x" * 1_200,
        ),
    )
    call = ToolCall(
        id="call-1",
        name="mcp__colbert__search",
        arguments={"query": "attention", "paper_id": "1706.03762"},
    )

    invocation = executor.execute(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker-1",
        tool_call=call,
        policy=RunPolicy.for_depth("standard"),
    )

    stored = store.get_tool_execution(stable_tool_execution_id(run.id, "call-1"))
    assert invocation.tool_result.content.startswith("full result")
    assert invocation.tool_result.is_error is False
    assert stored is not None
    assert stored.status == "started"
    assert len(invocation.result_preview) == 1_000
    assert invocation.execution_id == stored.id


def test_persisted_tool_arguments_hash_long_text_and_redact_secrets(store_and_run):
    store, run, step = store_and_run
    seen_arguments = []
    executor = _executor(
        store,
        Tool(
            name="mcp__colbert__build_index",
            description="index",
            input_schema={},
            handler=lambda arguments: seen_arguments.append(arguments) or "indexed",
        ),
    )
    call = ToolCall(
        id="build-1",
        name="mcp__colbert__build_index",
        arguments={
            "documents": [{"paper_id": "paper-1", "text": "x" * 10_000}],
            "api_key": "secret-value",
        },
    )

    invocation = executor.execute(
        run_id=run.id,
        step_id=step.id,
        owner_id="worker-1",
        tool_call=call,
        policy=RunPolicy.for_depth("standard"),
    )

    stored = store.get_tool_execution(stable_tool_execution_id(run.id, "build-1"))
    assert stored is not None
    rendered = json.dumps(stored.arguments, sort_keys=True)
    assert "secret-value" not in rendered
    assert '"char_count": 10000' in rendered
    assert '"sha256"' in rendered
    assert seen_arguments == [call.arguments]
    assert invocation.sanitized_arguments == stored.arguments


def test_web_policy_denies_ask_user_without_persisting_execution(store_and_run):
    store, run, step = store_and_run
    executor = _executor(
        store,
        Tool("ask_user", "ask", {}, lambda arguments: "answer"),
    )

    with pytest.raises(ToolExecutionFailure) as raised:
        executor.execute(
            run_id=run.id,
            step_id=step.id,
            owner_id="worker-1",
            tool_call=ToolCall("ask-1", "ask_user", {"question": "Continue?"}),
            policy=RunPolicy.for_depth("standard"),
        )

    assert raised.value.failure.failure_class == "validation"
    assert store.get_tool_execution(stable_tool_execution_id(run.id, "ask-1")) is None


def test_mcp_server_not_in_policy_is_denied_before_invoking_handler(store_and_run):
    store, run, step = store_and_run
    invoked = False

    def handler(arguments):
        nonlocal invoked
        invoked = True
        return "should not run"

    executor = _executor(store, Tool("mcp__private__search", "private", {}, handler))

    with pytest.raises(ToolExecutionFailure) as raised:
        executor.execute(
            run_id=run.id,
            step_id=step.id,
            owner_id="worker-1",
            tool_call=ToolCall("private-1", "mcp__private__search", {}),
            policy=RunPolicy.for_depth("standard"),
        )

    assert raised.value.failure.failure_class == "validation"
    assert invoked is False


def test_quick_policy_denies_subagent_before_persisting_execution(store_and_run):
    store, run, step = store_and_run
    executor = _executor(store, Tool("paper_deep_read", "read", {}, lambda arguments: "done"))

    with pytest.raises(ToolExecutionFailure) as raised:
        executor.execute(
            run_id=run.id,
            step_id=step.id,
            owner_id="worker-1",
            tool_call=ToolCall("read-1", "paper_deep_read", {"paper_id": "x"}),
            policy=RunPolicy.for_depth("quick"),
        )

    assert raised.value.failure.failure_class == "validation"
    assert store.get_tool_execution(stable_tool_execution_id(run.id, "read-1")) is None


def test_handler_failure_persists_bounded_failure_and_exposes_retryability(store_and_run):
    store, run, step = store_and_run
    executor = _executor(
        store,
        Tool(
            "mcp__colbert__search",
            "search",
            {},
            lambda arguments: (_ for _ in ()).throw(TimeoutError("x" * 2_500)),
        ),
    )

    with pytest.raises(ToolExecutionFailure) as raised:
        executor.execute(
            run_id=run.id,
            step_id=step.id,
            owner_id="worker-1",
            tool_call=ToolCall("timeout-1", "mcp__colbert__search", {}),
            policy=RunPolicy.for_depth("standard"),
        )

    stored = store.get_tool_execution(stable_tool_execution_id(run.id, "timeout-1"))
    assert raised.value.failure.failure_class == "timeout"
    assert raised.value.retryable is True
    assert len(str(raised.value)) == 2_500
    assert stored is not None
    assert stored.status == "failed"
    assert len(stored.failure_message) == 2_000


def test_non_retryable_tool_failure_is_not_retryable(store_and_run):
    store, run, step = store_and_run
    executor = _executor(
        store,
        Tool(
            "paper_deep_read",
            "read",
            {},
            lambda arguments: (_ for _ in ()).throw(TimeoutError("slow")),
        ),
    )

    with pytest.raises(ToolExecutionFailure) as raised:
        executor.execute(
            run_id=run.id,
            step_id=step.id,
            owner_id="worker-1",
            tool_call=ToolCall("deep-1", "paper_deep_read", {}),
            policy=RunPolicy.for_depth("standard"),
        )

    assert raised.value.classification == "non_retryable"
    assert raised.value.retryable is False


def test_expired_owner_cannot_record_started_or_failed_execution(store_and_run):
    store, run, step = store_and_run
    executor = _executor(
        store,
        Tool("mcp__colbert__search", "search", {}, lambda arguments: "result"),
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE agent_runs SET lease_expires_at = '2000-01-01T00:00:00.000000Z' WHERE id = ?",
            (run.id,),
        )

    with pytest.raises(RunLeaseLostError):
        executor.execute(
            run_id=run.id,
            step_id=step.id,
            owner_id="worker-1",
            tool_call=ToolCall("expired-1", "mcp__colbert__search", {}),
            policy=RunPolicy.for_depth("standard"),
        )

    assert store.get_tool_execution(stable_tool_execution_id(run.id, "expired-1")) is None


def test_old_attempt_cannot_write_failure_after_new_owner_claims(store_and_run):
    store, run, old_step = store_and_run
    call = ToolCall("late-1", "mcp__colbert__search", {})

    def late_handler(arguments):
        with store._connect() as conn:
            conn.execute(
                "UPDATE agent_runs SET lease_expires_at = '2000-01-01T00:00:00.000000Z' WHERE id = ?",
                (run.id,),
            )
        assert store.claim_run(run.id, owner_id="worker-2", lease_seconds=30) is not None
        store.start_step(
            run_id=run.id,
            owner_id="worker-2",
            kind="tool",
            input_data={"tool_name": call.name},
        )
        raise TimeoutError("late")

    executor = _executor(
        store,
        Tool(call.name, "search", {}, late_handler),
    )

    with pytest.raises(RunLeaseLostError):
        executor.execute(
            run_id=run.id,
            step_id=old_step.id,
            owner_id="worker-1",
            tool_call=call,
            policy=RunPolicy.for_depth("standard"),
        )

    stored = store.get_tool_execution(stable_tool_execution_id(run.id, call.id))
    assert stored is not None
    assert stored.step_id == old_step.id
    assert stored.status == "started"


@pytest.mark.parametrize("exception_type", (KeyboardInterrupt, SystemExit, GeneratorExit))
def test_process_control_exceptions_propagate_without_failing_execution(
    store_and_run, exception_type
):
    store, run, step = store_and_run
    call = ToolCall("control-1", "mcp__colbert__search", {})
    executor = _executor(
        store,
        Tool(
            call.name,
            "search",
            {},
            lambda arguments: (_ for _ in ()).throw(exception_type()),
        ),
    )

    with pytest.raises(exception_type):
        executor.execute(
            run_id=run.id,
            step_id=step.id,
            owner_id="worker-1",
            tool_call=call,
            policy=RunPolicy.for_depth("standard"),
        )

    stored = store.get_tool_execution(stable_tool_execution_id(run.id, call.id))
    assert stored is not None
    assert stored.status == "started"


def test_completed_execution_requires_checkpoint_replay_without_reinvoking_handler(store_and_run):
    store, run, first_step = store_and_run
    calls = 0

    def handler(arguments):
        nonlocal calls
        calls += 1
        return f"result-{calls}"

    call = ToolCall("same-call", "mcp__colbert__search", {})
    executor = _executor(store, Tool(call.name, "search", {}, handler))
    first = executor.execute(
        run_id=run.id,
        step_id=first_step.id,
        owner_id="worker-1",
        tool_call=call,
        policy=RunPolicy.for_depth("standard"),
    )
    store.complete_step_and_checkpoint(
        step_id=first_step.id,
        run_id=run.id,
        owner_id="worker-1",
        output_data={"content": first.tool_result.content},
        messages=[{"role": "user", "content": first.tool_result.content}],
        runtime_state={"turn_count": 1, "tokens_used": 1},
        tool_completion={
            "execution_id": first.execution_id,
            "tool_result": first.tool_result,
            "result_preview": first.result_preview,
            "duration_ms": first.duration_ms,
        },
    )
    second_step = store.start_step(
        run_id=run.id,
        owner_id="worker-1",
        kind="tool",
        input_data={"tool_name": call.name},
    )

    with pytest.raises(ToolExecutionReplayRequired) as raised:
        executor.execute(
            run_id=run.id,
            step_id=second_step.id,
            owner_id="worker-1",
            tool_call=call,
            policy=RunPolicy.for_depth("standard"),
        )

    assert raised.value.execution.step_id == first_step.id
    assert calls == 1


def test_non_retryable_failure_blocks_new_attempt_without_reinvoking_handler(store_and_run):
    store, run, first_step = store_and_run
    calls = 0

    def handler(arguments):
        nonlocal calls
        calls += 1
        raise TimeoutError("slow")

    call = ToolCall("deep-retry", "paper_deep_read", {})
    executor = _executor(store, Tool(call.name, "read", {}, handler))
    with pytest.raises(ToolExecutionFailure):
        executor.execute(
            run_id=run.id,
            step_id=first_step.id,
            owner_id="worker-1",
            tool_call=call,
            policy=RunPolicy.for_depth("standard"),
        )
    store.fail_step(
        first_step.id,
        run_id=run.id,
        owner_id="worker-1",
        error_data={"failure_class": "timeout"},
    )
    retry_step = store.start_step(
        run_id=run.id,
        owner_id="worker-1",
        kind="tool",
        input_data={"tool_name": call.name},
    )

    with pytest.raises(ToolExecutionBlocked) as raised:
        executor.execute(
            run_id=run.id,
            step_id=retry_step.id,
            owner_id="worker-1",
            tool_call=call,
            policy=RunPolicy.for_depth("standard"),
        )

    assert raised.value.execution.status == "failed"
    assert calls == 1


@pytest.mark.parametrize(
    ("tool_name", "expected_calls"),
    (
        ("mcp__colbert__search", 2),
        ("mcp__colbert__build_index", 2),
        ("paper_deep_read", 1),
    ),
)
def test_stale_execution_only_reinvokes_safe_classifications(
    store_and_run, tool_name, expected_calls
):
    store, run, first_step = store_and_run
    calls = 0

    def handler(arguments):
        nonlocal calls
        calls += 1
        return f"result-{calls}"

    call = ToolCall("stale-call", tool_name, {})
    executor = _executor(store, Tool(call.name, "tool", {}, handler))
    executor.execute(
        run_id=run.id,
        step_id=first_step.id,
        owner_id="worker-1",
        tool_call=call,
        policy=RunPolicy.for_depth("standard"),
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE agent_runs SET lease_expires_at = '2000-01-01T00:00:00.000000Z' WHERE id = ?",
            (run.id,),
        )
    assert store.claim_run(run.id, owner_id="worker-2", lease_seconds=30) is not None
    retry_step = store.start_step(
        run_id=run.id,
        owner_id="worker-2",
        kind="tool",
        input_data={"tool_name": call.name},
    )

    if expected_calls == 1:
        with pytest.raises(ToolExecutionBlocked):
            executor.execute(
                run_id=run.id,
                step_id=retry_step.id,
                owner_id="worker-2",
                tool_call=call,
                policy=RunPolicy.for_depth("standard"),
            )
    else:
        executor.execute(
            run_id=run.id,
            step_id=retry_step.id,
            owner_id="worker-2",
            tool_call=call,
            policy=RunPolicy.for_depth("standard"),
        )

    assert calls == expected_calls


def test_concurrent_duplicate_execute_invokes_handler_once(store_and_run):
    store, run, step = store_and_run
    handler_entered = threading.Event()
    release_handler = threading.Event()
    calls = 0

    def handler(arguments):
        nonlocal calls
        calls += 1
        handler_entered.set()
        assert release_handler.wait(timeout=5)
        return "result"

    call = ToolCall("concurrent-1", "mcp__colbert__search", {})
    executor = _executor(store, Tool(call.name, "search", {}, handler))
    first_result = []
    second_result = []

    def invoke(result):
        try:
            result.append(
                executor.execute(
                    run_id=run.id,
                    step_id=step.id,
                    owner_id="worker-1",
                    tool_call=call,
                    policy=RunPolicy.for_depth("standard"),
                )
            )
        except Exception as exc:
            result.append(exc)

    first_thread = threading.Thread(target=invoke, args=(first_result,))
    first_thread.start()
    assert handler_entered.wait(timeout=5)
    second_thread = threading.Thread(target=invoke, args=(second_result,))
    second_thread.start()
    second_thread.join(timeout=5)
    release_handler.set()
    first_thread.join(timeout=5)

    assert calls == 1
    assert len(first_result) == len(second_result) == 1
    assert isinstance(second_result[0], ToolExecutionBlocked)
    assert first_thread.is_alive() is False
    assert second_thread.is_alive() is False
