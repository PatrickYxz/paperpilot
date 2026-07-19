# PaperPilot Agent Runtime Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a durable PaperPilot Agent Runtime that persists runs, steps, checkpoints, tool executions, cancellation, retries, leases, and recovery while preserving the existing CLI, MCP, Web authentication, and task APIs.

**Architecture:** Keep the current synchronous ReAct and MCP stack, but move real Web execution behind a stepwise `AgentRuntime`. Persist each safe execution boundary in SQLite through an isolated `RunStore`; thread and Celery executors only schedule work, while the Runtime owns run state, retry, cancellation, lease, and artifact semantics.

**Tech Stack:** Python 3.12, dataclasses and typing, SQLite WAL, FastAPI/Pydantic, Celery/Redis (optional), Anthropic-compatible DeepSeek client, MCP stdio, pytest.

## Global Constraints

- Implement the approved design in `docs/codex-only-plans/2026-07-19-paperpilot-agent-runtime-control-plane-design.md`.
- Do not add a new Agent framework, migration framework, database, or runtime dependency.
- Keep SQLite and the in-process thread executor as local defaults; Celery remains opt-in.
- Preserve current authentication, user isolation, existing task response fields, simulated mode, CLI chat, and one-shot `conversation.run()` behavior.
- Restore only from committed safe boundaries; do not claim mid-call resume or exactly-once external side effects.
- Keep `ask_user` unavailable to Web Runtime policy until a durable `waiting_input` design exists; CLI behavior remains unchanged.
- A run policy is immutable after creation and is restored from `policy_json`, not rebuilt from current environment variables.
- Every state mutation that advances a run must verify `owner_id` and an unexpired lease.
- A completed step and its checkpoint must commit in one SQLite transaction.
- Do not retry tools unless their explicit classification permits it.
- All implementation work follows red-green-refactor and uses `./.venv/bin/python -m pytest`.
- Do not change the default `execution_mode="simulated"` in this phase.

---

## File Responsibility Map

**Create:**

- `paperpilot/agent/models.py`: immutable run, step, checkpoint, tool execution, outcome, and paging records.
- `paperpilot/agent/policy.py`: immutable `RunPolicy`, depth mapping, and retry delay calculation.
- `paperpilot/agent/errors.py`: normalized failure classification and retry decision helpers.
- `paperpilot/agent/store.py`: `RunStore` protocol and SQLite implementation using the existing task database path.
- `paperpilot/agent/tool_executor.py`: tool policy registry, stable logical call IDs, execution recording, and tool error conversion.
- `paperpilot/agent/driver.py`: serializable Driver state and one-safe-boundary-at-a-time ReAct execution.
- `paperpilot/agent/runtime.py`: run claim, heartbeat, cancellation, retry, checkpoint, completion, and failure orchestration.
- `paperpilot/agent/reconciler.py`: deterministic recovery scan and optional background reconciliation loop.
- `paperpilot/web/agent_projection.py`: idempotent task event/status/artifact projection adapters for AgentRuntime.
- `tests/agent/__init__.py`: Agent Runtime test package marker.
- `tests/agent/test_models.py`: model and serialization tests.
- `tests/agent/test_policy.py`: depth policy and retry delay tests.
- `tests/agent/test_store.py`: schema, lifecycle, lease, step, checkpoint, cancellation, and paging tests.
- `tests/agent/test_errors.py`: failure classification tests.
- `tests/agent/test_tool_executor.py`: permission, stable-ID, success, and retryability tests.
- `tests/agent/test_driver.py`: one-step LLM/tool/compact/finalize tests.
- `tests/agent/test_runtime.py`: completion, resume, lease, cancellation, retry, and artifact tests.
- `tests/agent/test_reconciler.py`: expired/due run recovery tests.
- `tests/agent/test_runtime_recovery.py`: network-free end-to-end crash-boundary scenarios.
- `tests/web/test_agent_projection.py`: task status, event, and final artifact projection tests.

**Modify:**

- `paperpilot/agent/__init__.py`: export stable Agent Runtime public types.
- `paperpilot/core/loop.py`: expose shared build-index repair and downloaded-document extraction helpers without changing legacy behavior.
- `paperpilot/conversation.py`: provide context-managed Agent resources while preserving existing sessions and `run()`.
- `paperpilot/web/task_store.py`: add idempotent result artifact columns/index and task status projection helpers.
- `paperpilot/web/workflow.py`: route real tasks through `AgentRuntime`; keep simulated workflow unchanged.
- `paperpilot/web/task_executor.py`: expose scheduling-compatible Runtime submission without owning run state.
- `paperpilot/web/worker_tasks.py`: use warm MCP resources with the shared Runtime and stop claiming real work at task level.
- `paperpilot/web/app.py`: inject `RunStore`, expose run/steps/cancel endpoints, and start/stop reconciler.
- `paperpilot/web/event_mapper.py`: map durable run/step/tool/retry/cancel events with bounded payloads.
- `paperpilot/web/static/app.js`: display run state and provide a cancel command for active real tasks.
- `paperpilot/web/static/styles.css`: style the compact run controls without adding nested cards.
- `tests/test_agent_loop.py`: guard shared helper behavior during extraction.
- `tests/test_conversation_session.py`: verify resource factory legacy and warm-MCP paths.
- `tests/web/test_task_store.py`: verify idempotent result projection.
- `tests/web/test_workflow.py`: verify real Runtime delegation and task projection.
- `tests/web/test_task_executor.py`: verify scheduling remains bounded and Runtime-agnostic.
- `tests/web/test_celery_worker.py`: verify duplicate delivery and warm MCP Runtime integration.
- `tests/web/test_event_mapper.py`: verify durable event mappings and secret bounds.
- `tests/web/test_web_app.py`: verify run APIs, cancellation, ownership, readiness, and compatibility.
- `README.md`: document the durable runtime, configuration, recovery boundary, and remaining risks.

---

## Phase 1A: Domain Model And Persistence

### Task 1: Define Stable Agent Runtime Domain Types And Policies

**Files:**

- Create: `paperpilot/agent/models.py`
- Create: `paperpilot/agent/policy.py`
- Modify: `paperpilot/agent/__init__.py`
- Create: `tests/agent/__init__.py`
- Create: `tests/agent/test_models.py`
- Create: `tests/agent/test_policy.py`

**Interfaces:**

- Produces `RunStatus`, `StepKind`, `StepStatus`, `FailureClass`, `ToolClassification` literal aliases.
- Produces frozen records `AgentRun`, `AgentStep`, `RunCheckpoint`, `ToolExecution`, `RunOutcome`, `StepPage`.
- Produces frozen `RunPolicy` with `for_depth(depth: str) -> RunPolicy`, `to_dict() -> dict`, and `from_dict(payload: dict) -> RunPolicy`.
- Later tasks consume `AgentRun.is_terminal`,
  `RunPolicy.retry_delay_seconds(retry_number, jitter)`, and all `to_dict()`
  methods.

`AgentRun` has the exact persisted fields from design section 6.1, with
`policy: RunPolicy` instead of raw JSON after row mapping. `RunOutcome` has
fields `run_id`, `status`, `final_step_id`, `final_text`, `failure_class`, and
`failure_message`, plus named constructors `completed`, `waiting_retry`,
`failed`, and `cancelled`. `StepPage` has `items`, `next_after_sequence`, and
`has_more`. `AgentRun.to_dict(public=False)` includes owner/lease fields only
when `public` is false. `RunOutcome.completed` takes `(run_id, final_step_id,
final_text)`; the other constructors take the normalized failure or cancel
fields required by their status.

- [ ] **Step 1: Write failing model and policy tests**

Add tests that pin exact status values, immutable records, policy round-trip, depth budgets, Web `ask_user` exclusion, and bounded retry delays:

```python
from dataclasses import FrozenInstanceError

import pytest

from paperpilot.agent.models import AgentRun, RUN_TERMINAL_STATUSES
from paperpilot.agent.policy import RunPolicy


def test_agent_run_terminal_state_is_immutable():
    run = AgentRun(
        id="run-1",
        task_id="task-1",
        status="completed",
        attempt=1,
        current_step=2,
        cancel_requested_at=None,
        retry_at=None,
        failure_class=None,
        failure_message=None,
        policy=RunPolicy.for_depth("standard"),
        owner_id=None,
        lease_expires_at=None,
        schema_version=1,
        created_at="2026-07-19T00:00:00+00:00",
        updated_at="2026-07-19T00:01:00+00:00",
        started_at="2026-07-19T00:00:01+00:00",
        finished_at="2026-07-19T00:01:00+00:00",
    )
    assert run.is_terminal is True
    assert "completed" in RUN_TERMINAL_STATUSES
    with pytest.raises(FrozenInstanceError):
        run.status = "running"


def test_standard_policy_round_trips_without_environment_lookup():
    policy = RunPolicy.for_depth("standard")
    restored = RunPolicy.from_dict(policy.to_dict())
    assert restored == policy
    assert restored.max_iterations == 8
    assert restored.token_budget == 50_000
    assert restored.max_retries == 2
    assert "ask_user" not in restored.allowed_builtin_tools


def test_retry_delay_is_bounded_and_deterministic_without_jitter():
    policy = RunPolicy.for_depth("deep")
    assert policy.retry_delay_seconds(1, jitter=0.0) == 2.0
    assert policy.retry_delay_seconds(2, jitter=0.0) == 8.0
    assert policy.retry_delay_seconds(3, jitter=0.0) == 8.0
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/agent/test_models.py tests/agent/test_policy.py -q
```

Expected: collection fails with `ModuleNotFoundError` for `paperpilot.agent.models` or `paperpilot.agent.policy`.

- [ ] **Step 3: Implement the domain records and immutable policy**

Use these exact status sets and policy fields:

```python
RUN_ACTIVE_STATUSES = frozenset({
    "pending", "running", "waiting_retry", "cancelling",
})
RUN_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
STEP_KINDS = frozenset({"llm", "tool", "subagent", "compact", "finalize"})
STEP_STATUSES = frozenset({"started", "completed", "failed", "cancelled"})
FAILURE_CLASSES = frozenset({
    "validation", "authentication", "rate_limit", "timeout",
    "transport", "capacity", "cancelled", "internal",
})
TOOL_CLASSIFICATIONS = frozenset({
    "read_only", "idempotent_write", "non_retryable",
})
SCHEMA_VERSION = 1
```

Define `RunPolicy` with exact fields:

```python
@dataclass(frozen=True)
class RunPolicy:
    max_iterations: int
    token_budget: int
    max_retries: int = 2
    retry_delays: tuple[float, ...] = (2.0, 8.0)
    allowed_builtin_tools: tuple[str, ...] = (
        "load_skill", "research_todo", "paper_deep_read",
        "compact_context", "search_user_document",
    )
    allowed_mcp_servers: tuple[str, ...] = (
        "arxiv", "colbert", "graph", "vlm",
    )
    allow_subagents: bool = True
    lease_seconds: int = 300
    lease_renewal_seconds: int = 60
    llm_timeout_seconds: int = 180
    tool_timeout_seconds: int = 180
```

Depth mapping must be `quick=(4, 20_000, False)`, `standard=(8, 50_000, True)`, and `deep=(12, 120_000, True)` for iterations, token budget, and subagent permission. Validate all numeric fields as positive, reject unknown depths, and serialize tuples as JSON arrays while restoring them as tuples.

- [ ] **Step 4: Export the public types and rerun GREEN**

Update `paperpilot/agent/__init__.py` to export only stable public types:

```python
from paperpilot.agent.models import (
    AgentRun,
    AgentStep,
    FailureClass,
    RunCheckpoint,
    RunOutcome,
    RunStatus,
    StepKind,
    ToolExecution,
)
from paperpilot.agent.policy import RunPolicy

__all__ = [
    "AgentRun", "AgentStep", "FailureClass", "RunCheckpoint",
    "RunOutcome", "RunPolicy", "RunStatus", "StepKind",
    "ToolExecution",
]
```

Run:

```bash
./.venv/bin/python -m pytest tests/agent/test_models.py tests/agent/test_policy.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add paperpilot/agent tests/agent
git commit -m "Add agent runtime domain model"
```

### Task 2: Create The SQLite Run Store And Schema

**Files:**

- Create: `paperpilot/agent/store.py`
- Create: `tests/agent/test_store.py`
- Modify: `paperpilot/agent/__init__.py`

**Interfaces:**

- Consumes `AgentRun`, `AgentStep`, `RunCheckpoint`, `ToolExecution`, and `RunPolicy` from Task 1.
- Produces `RunStore` protocol and `SQLiteRunStore(db_path, clock=utc_now)`.
- Produces `create_run(task_id, policy, initial_messages, runtime_state)`,
  `get_run`, `get_active_run_for_task`, `get_latest_run_for_task`, and
  `check_health`.
- Uses the same SQLite file as `TaskStore` but does not call private `TaskStore` methods.

- [ ] **Step 1: Write failing schema and run creation tests**

The test must create a real `research_tasks` row first, then verify the Agent tables, policy snapshot, checkpoint zero, and active-run uniqueness:

```python
import sqlite3

import pytest

from paperpilot.agent.policy import RunPolicy
from paperpilot.agent.store import ActiveRunExistsError, SQLiteRunStore
from paperpilot.web.task_store import TaskStore


def test_create_run_persists_policy_and_initial_checkpoint(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    task = TaskStore(db_path).create_task(question="Compare two papers")
    store = SQLiteRunStore(db_path)
    run = store.create_run(
        task_id=task.id,
        policy=RunPolicy.for_depth("standard"),
        initial_messages=[{"role": "user", "content": task.question}],
        runtime_state={"turn_count": 0, "tokens_used": 0},
    )
    checkpoint = store.get_latest_checkpoint(run.id)
    assert run.status == "pending"
    assert run.current_step == 0
    assert checkpoint is not None
    assert checkpoint.step_sequence == 0
    assert checkpoint.messages[0]["content"] == task.question
    assert store.get_active_run_for_task(task.id).id == run.id


def test_only_one_active_run_is_allowed_per_task(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    task = TaskStore(db_path).create_task(question="Question")
    store = SQLiteRunStore(db_path)
    arguments = {
        "task_id": task.id,
        "policy": RunPolicy.for_depth("quick"),
        "initial_messages": [{"role": "user", "content": "Question"}],
        "runtime_state": {"turn_count": 0, "tokens_used": 0},
    }
    store.create_run(**arguments)
    with pytest.raises(ActiveRunExistsError):
        store.create_run(**arguments)
```

- [ ] **Step 2: Run the focused store tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/agent/test_store.py -q
```

Expected: collection fails because `paperpilot.agent.store` does not exist.

- [ ] **Step 3: Implement schema initialization and row mapping**

`SQLiteRunStore._connect()` must set `row_factory=sqlite3.Row`,
`foreign_keys=ON`, `busy_timeout=30000`, and `synchronous=NORMAL`.
`get_active_run_for_task` filters to active statuses, while
`get_latest_run_for_task` returns the newest active or terminal run for API
history. Schema initialization must create:

```sql
CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    current_step INTEGER NOT NULL,
    cancel_requested_at TEXT,
    retry_at TEXT,
    failure_class TEXT,
    failure_message TEXT,
    policy_json TEXT NOT NULL,
    owner_id TEXT,
    lease_expires_at TEXT,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    FOREIGN KEY(task_id) REFERENCES research_tasks(id)
);

CREATE TABLE IF NOT EXISTS agent_steps (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    input_json TEXT NOT NULL,
    output_json TEXT,
    error_json TEXT,
    schema_version INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY(run_id) REFERENCES agent_runs(id),
    UNIQUE(run_id, sequence, attempt)
);

CREATE TABLE IF NOT EXISTS run_checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_sequence INTEGER NOT NULL,
    messages_json TEXT NOT NULL,
    runtime_state_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES agent_runs(id),
    UNIQUE(run_id, step_sequence)
);

CREATE TABLE IF NOT EXISTS tool_executions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    classification TEXT NOT NULL,
    status TEXT NOT NULL,
    result_preview TEXT,
    failure_class TEXT,
    failure_message TEXT,
    duration_ms INTEGER,
    schema_version INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    FOREIGN KEY(run_id) REFERENCES agent_runs(id),
    FOREIGN KEY(step_id) REFERENCES agent_steps(id)
);
```

Add indexes for `(run_id, sequence)`, `(run_id, step_sequence)`, `(task_id, created_at DESC)`, `(status, retry_at)`, and the partial active-run uniqueness constraint:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_one_active_task
ON agent_runs(task_id)
WHERE status IN ('pending', 'running', 'waiting_retry', 'cancelling');
```

`create_run` must start an explicit transaction, insert the run, insert checkpoint sequence 0 using `encode_messages`, and translate the unique-index `IntegrityError` into `ActiveRunExistsError`.

- [ ] **Step 4: Verify schema, foreign keys, and health**

Add tests that inspect `PRAGMA foreign_keys`, indexes, and all four table columns, then run:

```bash
./.venv/bin/python -m pytest tests/agent/test_store.py tests/web/test_task_store.py -q
```

Expected: Agent store and existing task store tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add paperpilot/agent/store.py paperpilot/agent/__init__.py tests/agent/test_store.py
git commit -m "Persist agent runs and checkpoints"
```

### Task 3: Add Leases, Steps, Checkpoints, Cancellation, And Tool Records

**Files:**

- Modify: `paperpilot/agent/store.py`
- Modify: `tests/agent/test_store.py`

**Interfaces:**

- Produces `claim_run(run_id, owner_id, lease_seconds) -> AgentRun | None`.
- Produces `renew_lease(run_id, owner_id, lease_seconds) -> bool` and `has_valid_lease(run_id, owner_id) -> bool`.
- Produces `start_step`, `complete_step_and_checkpoint`, `fail_step`, `request_cancel`, `schedule_retry`, `mark_completed`, `mark_failed`, `mark_cancelled`.
- Produces `start_tool_execution`, `complete_tool_execution`, `fail_tool_execution`, `get_tool_execution`.
- Produces `list_steps(run_id, after_sequence, limit) -> StepPage`,
  `list_pending_runs(limit) -> list[AgentRun]`,
  `list_cancelling_runs(limit) -> list[AgentRun]`, and
  `recover_due_runs(now, limit) -> list[AgentRun]`.

- [ ] **Step 1: Write failing lease and atomic-checkpoint tests**

Use an injected mutable clock and two owners:

```python
def test_only_current_lease_owner_can_commit_step(run_store, seeded_run, clock):
    claimed = run_store.claim_run(
        seeded_run.id, owner_id="owner-a", lease_seconds=30,
    )
    assert claimed is not None
    assert run_store.claim_run(
        seeded_run.id, owner_id="owner-b", lease_seconds=30,
    ) is None
    step = run_store.start_step(
        run_id=seeded_run.id,
        owner_id="owner-a",
        kind="llm",
        input_data={"message_count": 1},
    )
    with pytest.raises(RunLeaseLostError):
        run_store.complete_step_and_checkpoint(
            step_id=step.id,
            run_id=seeded_run.id,
            owner_id="owner-b",
            output_data={"tool_calls": []},
            messages=[{"role": "assistant", "content": "answer"}],
            runtime_state={"turn_count": 1, "tokens_used": 10},
        )


def test_completed_step_and_checkpoint_are_atomic(run_store, seeded_run):
    run_store.claim_run(seeded_run.id, owner_id="worker", lease_seconds=30)
    step = run_store.start_step(
        run_id=seeded_run.id,
        owner_id="worker",
        kind="llm",
        input_data={},
    )
    run_store.complete_step_and_checkpoint(
        step_id=step.id,
        run_id=seeded_run.id,
        owner_id="worker",
        output_data={"text": "answer"},
        messages=[{"role": "assistant", "content": "answer"}],
        runtime_state={"turn_count": 1, "tokens_used": 10},
    )
    assert run_store.get_run(seeded_run.id).current_step == 1
    assert run_store.get_latest_checkpoint(seeded_run.id).step_sequence == 1
```

- [ ] **Step 2: Write failing cancellation, retry, paging, and recovery tests**

Pin these behaviors:

```python
def test_cancel_request_is_idempotent(run_store, seeded_run):
    first = run_store.request_cancel(seeded_run.id)
    second = run_store.request_cancel(seeded_run.id)
    assert first.status == "cancelling"
    assert second.cancel_requested_at == first.cancel_requested_at


def test_recover_due_runs_releases_expired_owner_once(run_store, seeded_run, clock):
    run_store.claim_run(seeded_run.id, owner_id="dead", lease_seconds=10)
    clock.advance(seconds=11)
    recovered = run_store.recover_due_runs(now=clock.now(), limit=10)
    assert [run.id for run in recovered] == [seeded_run.id]
    assert recovered[0].status == "pending"
    assert run_store.recover_due_runs(now=clock.now(), limit=10) == []
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/agent/test_store.py -q
```

Expected: failures report missing lease, step, cancel, retry, paging, and recovery methods.

- [ ] **Step 4: Implement conditional state transitions and atomic commits**

All owner-sensitive updates must use the same condition:

```sql
WHERE id = ?
  AND status = 'running'
  AND owner_id = ?
  AND lease_expires_at > ?
```

`start_step` always uses `sequence = agent_runs.current_step + 1`. It sets
`attempt` to one more than the maximum existing attempt for that run/sequence,
so a crash before checkpoint creates a new attempt at the same logical
sequence, while a committed checkpoint advances the next sequence.

`claim_run` may claim `pending`, or reclaim `running` only when `lease_expires_at <= now`. `request_cancel` must set the timestamp once and move active states to `cancelling`; terminal states return unchanged. `complete_step_and_checkpoint` must:

1. execute `BEGIN IMMEDIATE`;
2. verify the run owner and lease;
3. update the started step to completed;
4. insert exactly one checkpoint at the same sequence;
5. update `agent_runs.current_step` and `updated_at`;
6. commit through the connection context.

When completing a tool step, the same method receives `tool_completion`
containing the stable tool execution ID, full `ToolResult`, bounded preview,
and duration. It updates the tool execution to completed in the same transaction
as the step and checkpoint. This is the only success path for a tool execution;
`ToolExecutor` must never mark success in a separate transaction.

`recover_due_runs` must atomically select expired `running` and due
`waiting_retry` rows. It transitions rows without a cancel request to `pending`,
transitions rows with `cancel_requested_at` to `cancelling`, clears
owner/lease/retry fields, increments run attempt for retry rows only, and
returns only rows changed by that call.
`list_pending_runs` returns a bounded oldest-first snapshot for startup recovery;
duplicate submissions are permitted because `claim_run` remains the correctness
boundary. `list_cancelling_runs` returns unowned or lease-expired cancelling
runs on every reconciliation pass. `claim_run` may claim such a cancelling row
without changing its status; Runtime sees the cancellation before planning a
new step and settles it to `cancelled`.

- [ ] **Step 5: Implement stable tool execution record mutations**

Use the LLM tool-use ID as the logical key source:

```python
def stable_tool_execution_id(run_id: str, tool_use_id: str) -> str:
    digest = hashlib.sha256(f"{run_id}:{tool_use_id}".encode("utf-8")).hexdigest()
    return f"tool_{digest[:32]}"
```

Starting the same logical ID with different normalized arguments must raise
`ToolExecutionConflictError`. Starting it with the same arguments returns the
existing record. A retry may move `step_id` to the new attempt only while status
is `started` or `failed`; a `completed` record is immutable. Completion is
performed only by `complete_step_and_checkpoint`, while failure updates are
idempotent for the same final values and reject conflicting final values.

- [ ] **Step 6: Run Agent store and TaskStore regression tests**

```bash
./.venv/bin/python -m pytest tests/agent/test_store.py tests/web/test_task_store.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add paperpilot/agent/store.py tests/agent/test_store.py
git commit -m "Add durable agent run lifecycle"
```

---

## Phase 1B: Stepwise Driver And Runtime

### Task 4: Normalize Failures And Execute Tools Under Policy

**Files:**

- Create: `paperpilot/agent/errors.py`
- Create: `paperpilot/agent/tool_executor.py`
- Create: `tests/agent/test_errors.py`
- Create: `tests/agent/test_tool_executor.py`

**Interfaces:**

- Consumes MCP exceptions from `paperpilot.tools.mcp_client` and execution records from Task 3.
- Produces `ClassifiedFailure`, `classify_exception(exc)`, `is_step_retryable(failure, classification)`.
- Produces `ToolPolicyRegistry.classify(tool_name) -> ToolClassification`.
- Produces `ToolExecutor.execute(run_id, step_id, tool_call, policy) -> ToolInvocationResult`.

- [ ] **Step 1: Write failing error-classification tests**

```python
from paperpilot.agent.errors import classify_exception
from paperpilot.tools.mcp_client import MCPToolTimeout, MCPTransportError


def test_transport_and_timeout_failures_are_normalized():
    transport = classify_exception(MCPTransportError("closed"))
    timeout = classify_exception(MCPToolTimeout("slow"))
    assert transport.failure_class == "transport"
    assert transport.transient is True
    assert timeout.failure_class == "timeout"
    assert timeout.transient is True


def test_value_error_is_validation_and_not_transient():
    failure = classify_exception(ValueError("invalid arguments"))
    assert failure.failure_class == "validation"
    assert failure.transient is False
```

Also test Anthropic-compatible errors by using small fake exceptions with `status_code` values 401, 429, and 503 so tests do not make network calls.

- [ ] **Step 2: Write failing ToolExecutor behavior tests**

Pin permission denial, read-only retryability, default non-retryable classification, stable ID reuse, full result return, and bounded database preview:

```python
def test_unknown_tool_defaults_to_non_retryable(tool_registry):
    assert tool_registry.classify("custom_mutation") == "non_retryable"


def test_tool_executor_returns_success_for_atomic_runtime_commit(executor, store):
    call = ToolCall(id="call-1", name="mcp__colbert__search", arguments={
        "query": "attention", "paper_id": "1706.03762",
    })
    invocation = executor.execute(
        run_id="run-1", step_id="step-1", tool_call=call,
        policy=RunPolicy.for_depth("standard"),
    )
    stored = store.get_tool_execution(stable_tool_execution_id("run-1", "call-1"))
    assert invocation.tool_result.content.startswith("full result")
    assert stored.status == "started"
    assert len(invocation.result_preview) <= 1000


def test_persisted_tool_arguments_hash_long_text_and_redact_secrets(executor, store):
    call = ToolCall(id="build-1", name="mcp__colbert__build_index", arguments={
        "documents": [{"paper_id": "paper-1", "text": "x" * 10_000}],
        "api_key": "secret-value",
    })
    executor.execute(
        run_id="run-1", step_id="step-1", tool_call=call,
        policy=RunPolicy.for_depth("standard"),
    )
    stored = store.get_tool_execution(stable_tool_execution_id("run-1", "build-1"))
    rendered = json.dumps(stored.arguments, sort_keys=True)
    assert "secret-value" not in rendered
    assert '"char_count": 10000' in rendered
    assert '"sha256"' in rendered
```

- [ ] **Step 3: Run the tests and verify RED**

```bash
./.venv/bin/python -m pytest tests/agent/test_errors.py tests/agent/test_tool_executor.py -q
```

Expected: collection fails for missing modules.

- [ ] **Step 4: Implement failure classification**

Classification order must be deterministic:

```python
if isinstance(exc, (MCPToolTimeout, TimeoutError)):
    return ClassifiedFailure("timeout", str(exc), transient=True)
if isinstance(exc, (MCPTransportError, ConnectionError, EOFError, OSError)):
    return ClassifiedFailure("transport", str(exc), transient=True)
if getattr(exc, "status_code", None) in {401, 403}:
    return ClassifiedFailure("authentication", str(exc), transient=False)
if getattr(exc, "status_code", None) == 429:
    return ClassifiedFailure("rate_limit", str(exc), transient=True)
if getattr(exc, "status_code", None) in {502, 503, 504}:
    return ClassifiedFailure("transport", str(exc), transient=True)
if isinstance(exc, (KeyError, TypeError, ValueError)):
    return ClassifiedFailure("validation", str(exc), transient=False)
return ClassifiedFailure("internal", f"{type(exc).__name__}: {exc}", transient=False)
```

Truncate persisted failure messages to 2000 characters without changing the raised exception's original text.

- [ ] **Step 5: Implement explicit tool policy and execution recording**

Classify exact names and suffixes:

```python
READ_ONLY_TOOL_SUFFIXES = (
    "__search", "__planned_retrieval", "__get_paper",
    "__get_citations", "__get_references", "__analyze_figure",
)
IDEMPOTENT_WRITE_TOOL_SUFFIXES = ("__build_index",)
READ_ONLY_BUILTINS = frozenset({
    "load_skill", "research_todo", "compact_context",
    "search_user_document",
})
```

`paper_deep_read` is `non_retryable` in Phase 1 because it mixes downloads,
index writes, searches, and nested LLM calls behind one handler. `ask_user` is
denied by Web `RunPolicy`. MCP tools are allowed only when the server segment
is present in `policy.allowed_mcp_servers`. ToolExecutor must persist `started`
before invoking `handler`; on success it returns the full
`ToolInvocationResult` for Runtime's atomic step/checkpoint commit, and on
failure it persists the failed execution and raises `ToolExecutionFailure`
carrying `ClassifiedFailure`, classification, and `retryable`.
Calling `paper_deep_read` when `policy.allow_subagents` is false is a
validation failure before any tool execution row is started.

Persisted arguments use a deterministic sanitizer. Keys matching
`api_key`, `authorization`, `cookie`, `password`, `secret`, or `token` become
`"[REDACTED]"`. Strings longer than 2000 characters become an object with
`char_count`, `sha256`, and a 200-character preview. The real handler receives
the original in-memory arguments. Step `input_data` uses the same sanitized
descriptor, and argument conflict checks compare the descriptor's canonical
JSON rather than raw secret or document text.

- [ ] **Step 6: Run tool and MCP regressions**

```bash
./.venv/bin/python -m pytest tests/agent/test_errors.py tests/agent/test_tool_executor.py tests/test_mcp_client.py tests/test_mcp_runtime.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add paperpilot/agent/errors.py paperpilot/agent/tool_executor.py tests/agent/test_errors.py tests/agent/test_tool_executor.py
git commit -m "Govern durable agent tool execution"
```

### Task 5: Build A Serializable One-Step ReAct Driver

**Files:**

- Create: `paperpilot/agent/driver.py`
- Modify: `paperpilot/core/loop.py`
- Modify: `tests/test_agent_loop.py`
- Create: `tests/agent/test_driver.py`

**Interfaces:**

- Consumes `LLMClient`, `Tool`, `ToolCall`, `ToolResult`, `ContextManager`, `RunPolicy`, and `ToolExecutor`.
- Produces serializable `DriverState`, `DriverAction`, and `DriverStepResult`.
- Produces `AgentLoopDriver.plan_next(state) -> DriverAction` and `execute(action, state) -> DriverStepResult`.
- Legacy `agent_loop` consumes extracted public helpers and must preserve its current tests exactly.

`DriverStepResult` contains `state`, `output_data`, `final_text`, and optional
`tool_completion`. `DriverState.runtime_payload()` returns every persisted
field except messages; messages are passed separately to the checkpoint store.

- [ ] **Step 1: Protect helper extraction with failing/characterization tests**

Move the current private build-index repair and downloaded-document extraction behavior behind public names without changing semantics:

```python
from paperpilot.core.loop import (
    extract_downloaded_document,
    should_repair_build_index_args,
)


def test_downloaded_document_helpers_preserve_existing_behavior():
    document = extract_downloaded_document(
        "mcp__arxiv__download_paper",
        '{"paper_id":"1706.03762","text":"full text"}',
    )
    call = ToolCall(id="build", name="mcp__colbert__build_index", arguments={})
    assert document == {"paper_id": "1706.03762", "text": "full text"}
    assert should_repair_build_index_args(call, [document]) is True
```

- [ ] **Step 2: Write failing Driver state and action tests**

Pin JSON round-trip, one LLM call per action, one tool call per action, multi-tool batching, and finalization:

```python
def test_driver_state_round_trip_preserves_pending_tool_batch():
    state = DriverState(
        messages=[{"role": "user", "content": "question"}],
        turn_count=1,
        tokens_used=12,
        downloaded_documents=[],
        todo_items=[{"content": "Read methods", "status": "in_progress"}],
        pending_tool_calls=[ToolCall(id="one", name="search", arguments={})],
        pending_tool_results=[],
        final_text=None,
    )
    assert DriverState.from_dict(state.to_dict()) == state


def test_driver_executes_one_tool_and_appends_batch_after_last_result(driver):
    state = driver_state_with_two_pending_tools()
    first = driver.execute(driver.plan_next(state), state)
    assert first.state.messages == state.messages
    assert len(first.state.pending_tool_results) == 1
    second = driver.execute(driver.plan_next(first.state), first.state)
    assert second.state.pending_tool_calls == []
    assert second.state.pending_tool_results == []
    assert second.state.messages[-1]["role"] == "user"
    assert len(second.state.messages[-1]["content"]) == 2
```

- [ ] **Step 3: Run the Driver tests and verify RED**

```bash
./.venv/bin/python -m pytest tests/agent/test_driver.py tests/test_agent_loop.py -q
```

Expected: Driver module collection fails while existing loop tests remain green.

- [ ] **Step 4: Implement Driver state and action planning**

`DriverState.to_dict()` must call `encode_messages`; `from_dict()` must call
`decode_messages`. Persist pending tool calls/results as plain dictionaries.
Persist `todo_items` as JSON-safe dictionaries. After each `research_todo`
tool action, Driver copies `resources.todo_store.items()` into the next state
so checkpoint recovery does not lose the multi-step research plan. Action
planning order is exact:

```python
def plan_next(self, state: DriverState) -> DriverAction:
    context = self.context_manager.inspect(
        system=self.system, tools=self.tools, messages=state.messages,
    )
    if state.final_text is not None:
        return DriverAction(kind="finalize", input_data={})
    if state.pending_tool_calls:
        call = state.pending_tool_calls[0]
        kind = "subagent" if call.name == "paper_deep_read" else "tool"
        return DriverAction(kind=kind, input_data={"tool_call": tool_call_dict(call)})
    if context.needs_compact:
        return DriverAction(kind="compact", input_data={
            "estimated_tokens": context.estimated_tokens,
        })
    return DriverAction(kind="llm", input_data={
        "message_count": len(state.messages),
    })
```

Before returning an LLM action, reject exhausted iteration/token budgets. If compaction still leaves the state over the critical limit, raise a non-retryable `ContextOverflowError`.

- [ ] **Step 5: Implement exactly-one-boundary execution**

LLM execution calls `client.call` once, records usage, appends the assistant turn, and either sets `final_text` or stores all returned tool calls. Tool execution repairs `build_index` arguments using extracted helpers, calls ToolExecutor once, records downloaded documents, removes one pending call, and appends all tool results only after the final pending call finishes. Compact execution calls `compact_messages` once. Finalize returns the existing final text without another LLM call.

- [ ] **Step 6: Run Driver, loop, context, compact, and message-codec tests**

```bash
./.venv/bin/python -m pytest tests/agent/test_driver.py tests/test_agent_loop.py tests/test_context_manager.py tests/builtin_tools/test_compact.py tests/test_message_codec.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add paperpilot/agent/driver.py paperpilot/core/loop.py tests/agent/test_driver.py tests/test_agent_loop.py
git commit -m "Add checkpointable agent loop driver"
```

### Task 6: Orchestrate Runs, Heartbeats, Retry, Cancellation, And Resume

**Files:**

- Create: `paperpilot/agent/runtime.py`
- Create: `tests/agent/test_runtime.py`

**Interfaces:**

- Consumes `SQLiteRunStore`, `AgentLoopDriver`, `RunPolicy`, `ClassifiedFailure`, and a context-managed resource factory.
- Produces `AgentRuntime.execute(run_id, owner_id) -> RunOutcome`.
- Produces `AgentRuntime.create_for_task(task_id, question, depth) -> AgentRun`.
- Produces `LeaseHeartbeat` that renews independently of a synchronous LLM/MCP call.
- Produces `RunResultSink.publish(run_id, step_id, content, metadata) -> None`
  and `RunEventSink.emit(task_id, kind, payload) -> None` protocols injected
  into Runtime.
- Emits domain callbacks with `run_id`, `step_id`, `sequence`, `attempt`, and `owner_id`.

- [ ] **Step 1: Write failing completion and resume tests**

```python
def test_runtime_completes_and_checkpoints_every_boundary(runtime_fixture):
    runtime, store, run = runtime_fixture(answer="final answer")
    outcome = runtime.execute(run.id, owner_id="worker-1")
    assert outcome.status == "completed"
    assert store.get_run(run.id).status == "completed"
    assert [step.kind for step in store.list_steps(run.id, 0, 100).items] == [
        "llm", "finalize",
    ]
    assert store.get_latest_checkpoint(run.id).step_sequence == 2


def test_runtime_resumes_after_committed_tool_without_reexecution(runtime_fixture):
    runtime, store, run, tool = runtime_fixture(crash_after_step=2)
    with pytest.raises(InjectedCrash):
        runtime.execute(run.id, owner_id="worker-dead")
    store.expire_lease_for_test(run.id)
    outcome = runtime.execute(run.id, owner_id="worker-new")
    assert outcome.status == "completed"
    assert tool.call_count == 1
```

- [ ] **Step 2: Write failing cancellation, retry, and lease-loss tests**

Pin these behaviors:

```python
def test_cancel_after_llm_call_discards_uncommitted_step(runtime_fixture):
    runtime, store, run = runtime_fixture(cancel_during_llm=True)
    outcome = runtime.execute(run.id, owner_id="worker")
    assert outcome.status == "cancelled"
    assert store.get_run(run.id).status == "cancelled"
    assert store.list_steps(run.id, 0, 100).items[-1].status == "cancelled"


def test_transient_failure_schedules_bounded_retry(runtime_fixture):
    runtime, store, run = runtime_fixture(failure=MCPTransportError("closed"))
    outcome = runtime.execute(run.id, owner_id="worker")
    assert outcome.status == "waiting_retry"
    assert store.get_run(run.id).retry_at is not None


def test_lost_lease_prevents_checkpoint_commit(runtime_fixture):
    runtime, store, run = runtime_fixture(steal_lease_after_action=True)
    with pytest.raises(RunLeaseLostError):
        runtime.execute(run.id, owner_id="worker-old")
    assert store.get_latest_checkpoint(run.id).step_sequence == 0
```

- [ ] **Step 3: Run Runtime tests and verify RED**

```bash
./.venv/bin/python -m pytest tests/agent/test_runtime.py -q
```

Expected: collection fails because `paperpilot.agent.runtime` does not exist.

- [ ] **Step 4: Implement Runtime creation, claim, and heartbeat**

`create_for_task` writes checkpoint zero with one user message and Driver state fields. `execute` must claim before opening external resources. `LeaseHeartbeat` uses a daemon thread, waits `lease_renewal_seconds`, calls `renew_lease`, records a lost-lease flag, and always joins on context exit. Tests inject a heartbeat factory rather than waiting real time.

- [ ] **Step 5: Implement the safe-boundary loop**

The loop order is fixed:

```python
while True:
    run = store.require_owned_run(run_id, owner_id)
    if run.cancel_requested_at is not None:
        return cancel_run(run, owner_id)
    state = load_driver_state(store.get_latest_checkpoint(run_id))
    action = driver.plan_next(state)
    step = store.start_step(
        run_id=run_id,
        owner_id=owner_id,
        kind=action.kind,
        input_data=action.input_data,
    )
    result = driver.execute(action, state, run_id=run_id, step_id=step.id)
    if heartbeat.lease_lost:
        raise RunLeaseLostError(run_id)
    if store.get_run(run_id).cancel_requested_at is not None:
        store.cancel_step(step.id, run_id=run_id, owner_id=owner_id)
        return cancel_run(store.get_run(run_id), owner_id)
    store.complete_step_and_checkpoint(
        step_id=step.id,
        run_id=run_id,
        owner_id=owner_id,
        output_data=result.output_data,
        messages=result.state.messages,
        runtime_state=result.state.runtime_payload(),
        tool_completion=result.tool_completion,
    )
    if action.kind == "finalize":
        result_sink.publish(
            run_id=run_id,
            step_id=step.id,
            content=result.final_text,
            metadata={"message_count": len(result.state.messages)},
        )
        return complete_run(run_id, step.id, result.final_text, owner_id)
```

For tool/subagent actions, `result.tool_completion` carries the full result into
the atomic tool/step/checkpoint commit; non-tool actions use `None`. Runtime must
not catch `RunLeaseLostError`. For other failures it persists the failed step,
classifies retryability, schedules retry only while
`run.attempt <= policy.max_retries`, and otherwise marks the run failed. The
finalize order is checkpoint, idempotent result publication, then terminal run
transition. A crash after publication replays finalize, receives the existing
artifact, and then marks the run completed.

- [ ] **Step 6: Add deterministic event assertions**

Ensure emitted events include ordered `run_claimed`, `step_started`, `step_completed`, `checkpoint_saved`, and `run_completed`; retry and cancel tests must assert `retry_scheduled` and `run_cancelled`. No event payload may contain full messages or full tool results.

- [ ] **Step 7: Run all Agent Runtime unit tests**

```bash
./.venv/bin/python -m pytest tests/agent -q
```

Expected: all Phase 1A and 1B tests pass.

- [ ] **Step 8: Commit Task 6**

```bash
git add paperpilot/agent/runtime.py tests/agent/test_runtime.py
git commit -m "Execute recoverable agent runs"
```

---

## Phase 1C: Conversation, Web, Thread, And Celery Integration

### Task 7: Provide Context-Managed Conversation Resources

**Files:**

- Modify: `paperpilot/conversation.py`
- Modify: `tests/test_conversation_session.py`
- Modify: `tests/test_mcp_runtime.py`

**Interfaces:**

- Produces frozen `AgentResources(system, tools, client, todo_store)`.
- Produces `open_agent_resources(messages_ref, on_event, mcp_runtime=None,
  input_provider=None, todo_items=None)` context manager.
- Runtime consumes the context manager; legacy `ConversationSession` and `run()` remain behaviorally unchanged.

- [ ] **Step 1: Write failing cold and warm resource tests**

```python
def test_open_agent_resources_closes_cold_mcp(monkeypatch):
    fake = FakeMCPClient()
    monkeypatch.setattr(conversation, "MCPClient", lambda path: fake)
    messages = [{"role": "user", "content": "question"}]
    with conversation.open_agent_resources(messages, lambda kind, payload: None) as resources:
        assert resources.system.startswith("你是 PaperPilot")
        assert any(tool.name == "mcp__echo__ping" for tool in resources.tools)
    assert fake.closed is True


def test_open_agent_resources_borrows_warm_runtime_without_closing_it():
    runtime = FakeMCPRuntime()
    with conversation.open_agent_resources([], lambda kind, payload: None, mcp_runtime=runtime):
        pass
    assert runtime.lease_count == 1
    assert runtime.closed is False


def test_open_agent_resources_restores_research_todos():
    items = [{"content": "Read methods", "status": "in_progress"}]
    with conversation.open_agent_resources(
        [], lambda kind, payload: None, todo_items=items,
    ) as resources:
        assert resources.todo_store.items() == items
```

- [ ] **Step 2: Run conversation tests and verify RED**

```bash
./.venv/bin/python -m pytest tests/test_conversation_session.py tests/test_mcp_runtime.py -q
```

Expected: failures report missing `open_agent_resources` and `AgentResources`.

- [ ] **Step 3: Implement the context manager using existing composition**

Cold path must start one `MCPClient`, compose tools through `_compose_tools`,
restore a `TodoStore` from `todo_items`, yield `AgentResources`, and close in
`finally`. Warm path must enter `mcp_runtime.lease_tools()`, compose borrowed
tools, and leave Runtime ownership unchanged. A Web input provider must raise
`RuntimeError("ask_user is unavailable in durable web runs")`; policy prevents
normal invocation, while the failure avoids stdin blocking if a model violates
policy.

- [ ] **Step 4: Preserve legacy session and one-shot tests**

```bash
./.venv/bin/python -m pytest tests/test_conversation_session.py tests/test_main_integration.py tests/test_mcp_runtime.py -q
```

Expected: all tests pass with no changes to CLI-visible commands.

- [ ] **Step 5: Commit Task 7**

```bash
git add paperpilot/conversation.py tests/test_conversation_session.py tests/test_mcp_runtime.py
git commit -m "Expose agent execution resources"
```

### Task 8: Route Real Workflows Through The Durable Runtime

**Files:**

- Modify: `paperpilot/web/task_store.py`
- Create: `paperpilot/web/agent_projection.py`
- Modify: `paperpilot/web/workflow.py`
- Modify: `paperpilot/web/worker_tasks.py`
- Modify: `paperpilot/web/task_executor.py`
- Modify: `tests/web/test_task_store.py`
- Create: `tests/web/test_agent_projection.py`
- Modify: `tests/web/test_workflow.py`
- Modify: `tests/web/test_task_executor.py`
- Modify: `tests/web/test_celery_worker.py`

**Interfaces:**

- Consumes `AgentRuntime`, `SQLiteRunStore`, and `open_agent_resources`.
- Produces `TaskStore.add_result_artifact_once(task_id, run_id, step_id, title, content, payload)`.
- Produces `TaskResultSink` and `TaskEventSink` adapters for AgentRuntime.
- `WorkflowRunner.run_real(task_id)` creates/resumes a run and delegates to the
  Runtime; injected sinks project state/events/artifacts while Runtime still
  owns transition order.
- Worker warm MCP Runtime is injected into the same runtime resource factory.

- [ ] **Step 1: Write failing idempotent artifact projection tests**

Add nullable `source_run_id` and `source_step_id` columns to `task_artifacts` through existing idempotent schema checks, plus a partial unique index on `source_run_id` for `kind='result'`:

Define `seeded_task_and_run` locally in `tests/web/test_agent_projection.py`; it
creates one TaskStore task, one SQLiteRunStore run with checkpoint zero, and
uses store transition helpers to place both rows in the requested statuses.
It returns `(task_store, run_store, task, run)` and performs no direct SQL.

```python
def test_result_artifact_is_idempotent_per_run(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="Question")
    first = store.add_result_artifact_once(
        task_id=task.id, run_id="run-1", step_id="step-2",
        title="PaperPilot result", content="answer", payload={"depth": "standard"},
    )
    second = store.add_result_artifact_once(
        task_id=task.id, run_id="run-1", step_id="step-2",
        title="PaperPilot result", content="answer", payload={"depth": "standard"},
    )
    assert second.id == first.id
    assert len(store.list_artifacts_page(task.id, user_id=None, after_id=0, limit=10).items) == 1


def test_result_sink_publishes_artifact_before_completed_status(tmp_path):
    store, run_store, task, run = seeded_task_and_run(
        tmp_path, task_status="running", run_status="running",
    )
    sink = TaskResultSink(store, run_store)
    sink.publish(
        run_id=run.id,
        step_id="step-2",
        content="answer",
        metadata={"message_count": 3},
    )
    assert store.get_task(task.id).status == "running"
    artifacts = store.list_artifacts_page(
        task.id, user_id=None, after_id=0, limit=10,
    ).items
    assert [artifact.content for artifact in artifacts] == ["answer"]


def test_completed_event_transitions_task_after_result_exists(tmp_path):
    store, run_store, task, run = seeded_task_and_run(
        tmp_path, task_status="running", run_status="running",
    )
    TaskResultSink(store, run_store).publish(
        run_id=run.id,
        step_id="step-2",
        content="answer",
        metadata={"message_count": 3},
    )
    event_sink = TaskEventSink(store, run_store)
    event_sink.emit(task.id, "run_completed", {
        "run_id": run.id, "task_id": task.id,
        "step_id": "step-2", "sequence": 2,
    })
    assert store.get_task(task.id).status == "completed"
```

- [ ] **Step 2: Write failing workflow and worker delegation tests**

```python
def test_real_workflow_uses_existing_active_run_and_runtime(tmp_path):
    task_store, run_store, task, run = seeded_active_run(tmp_path)
    runtime = FakeAgentRuntime(outcome=RunOutcome.completed(run.id, "step-2", "answer"))
    runner = WorkflowRunner(task_store, run_store=run_store, runtime=runtime)
    runner.run_real(task.id)
    assert runtime.calls == [(run.id, runtime.calls[0][1])]


def test_duplicate_celery_delivery_relies_on_run_claim_not_task_claim(monkeypatch):
    runtime = BlockingFakeAgentRuntime()
    configure_worker_runtime(monkeypatch, runtime)
    worker_tasks._execute_research_task("task-1", "real", redelivered=False)
    worker_tasks._execute_research_task("task-1", "real", redelivered=True)
    assert runtime.successful_claims == 1
```

- [ ] **Step 3: Run integration slices and verify RED**

```bash
./.venv/bin/python -m pytest tests/web/test_task_store.py tests/web/test_agent_projection.py tests/web/test_workflow.py tests/web/test_task_executor.py tests/web/test_celery_worker.py -q
```

Expected: failures report missing artifact idempotency and Runtime integration.

- [ ] **Step 4: Implement TaskStore projection support**

Schema upgrade must add missing columns with `ALTER TABLE`, create `idx_artifacts_result_run` as a partial unique index, return an existing identical result on conflict, and raise `ValueError` if the same run is projected with different step/content.

`TaskResultSink.publish` resolves `task_id` from RunStore and delegates to the
idempotent artifact method without changing task status. `TaskEventSink.emit`
stores the mapped event and applies task status projection. Before projecting
`run_completed`, it verifies that the run's result artifact exists; absence is
an internal consistency error and must not mark the task completed.

- [ ] **Step 5: Implement WorkflowRunner Runtime delegation**

For real tasks:

1. read the task;
2. get active run or create one with `RunPolicy.for_depth(task.depth)`;
3. call Runtime with a unique owner ID containing process, thread, and UUID components;
4. execute the run with `TaskResultSink` and `TaskEventSink` already injected;
5. let the event sink project `pending/waiting_retry` to task `pending`;
6. let the event sink project `running/cancelling` to task `running`;
7. let Runtime call the result sink before the `completed` event transitions task state;
8. project `failed/cancelled` to task `failed` with distinct event types.

Do not wrap `RunLeaseLostError` as task failure; another owner may be progressing the run.

- [ ] **Step 6: Update Celery and thread paths**

Simulated tasks retain current `TaskStore.claim_task` behavior. Real Celery tasks build `SQLiteRunStore`, borrow `_get_runtime()` through `open_agent_resources`, and call WorkflowRunner without task-level claim. Thread execution continues calling `WorkflowRunner.run_real`, which opens cold resources for that run. TaskExecutor still only admits and schedules `(task_id, execution_mode)`.

- [ ] **Step 7: Run Phase 1C integration tests**

```bash
./.venv/bin/python -m pytest tests/web/test_task_store.py tests/web/test_agent_projection.py tests/web/test_workflow.py tests/web/test_task_executor.py tests/web/test_celery_worker.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 8**

```bash
git add paperpilot/web/task_store.py paperpilot/web/agent_projection.py paperpilot/web/workflow.py paperpilot/web/worker_tasks.py paperpilot/web/task_executor.py tests/web/test_task_store.py tests/web/test_agent_projection.py tests/web/test_workflow.py tests/web/test_task_executor.py tests/web/test_celery_worker.py
git commit -m "Run web research through durable runtime"
```

### Task 9: Expose Run, Step, And Cancellation APIs With User Isolation

**Files:**

- Modify: `paperpilot/web/app.py`
- Modify: `tests/web/test_web_app.py`

**Interfaces:**

- Adds optional `run_store: RunStore | None` and `reconciler` injection to `create_app`.
- Real task creation persists checkpoint-zero AgentRun before queue submission;
  simulated task creation does not create a run.
- Adds `GET /api/tasks/{task_id}/run`.
- Adds `GET /api/tasks/{task_id}/run/steps?after_sequence=0&limit=50`.
- Adds `POST /api/tasks/{task_id}/cancel` with status 202 for active runs.
- Existing task endpoints and response shapes remain unchanged.

- [ ] **Step 1: Write failing authentication and ownership tests**

```python
def test_run_endpoints_require_authentication(client):
    assert client.get("/api/tasks/task-1/run").status_code == 401
    assert client.post("/api/tasks/task-1/cancel").status_code == 401


def test_other_user_cannot_read_or_cancel_run(app_client, users, owned_task):
    login(app_client, users.other)
    assert app_client.get(f"/api/tasks/{owned_task.id}/run").status_code == 404
    assert app_client.post(f"/api/tasks/{owned_task.id}/cancel").status_code == 404
```

- [ ] **Step 2: Write failing response, pagination, and idempotent cancel tests**

```python
def test_owner_reads_run_and_incremental_steps(authenticated_client, seeded_run):
    response = authenticated_client.get(f"/api/tasks/{seeded_run.task_id}/run")
    assert response.status_code == 200
    assert response.json()["id"] == seeded_run.id
    steps = authenticated_client.get(
        f"/api/tasks/{seeded_run.task_id}/run/steps?after_sequence=1&limit=1"
    ).json()
    assert steps == {
        "items": [expected_step_2_dict()],
        "next_after_sequence": 2,
        "has_more": True,
    }


def test_cancel_is_idempotent(authenticated_client, seeded_run):
    first = authenticated_client.post(f"/api/tasks/{seeded_run.task_id}/cancel")
    second = authenticated_client.post(f"/api/tasks/{seeded_run.task_id}/cancel")
    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["cancel_requested_at"] == first.json()["cancel_requested_at"]


def test_real_task_has_pending_run_before_executor_submission(app_fixture):
    app, client, executor, run_store = app_fixture(block_submission=True)
    response = client.post("/api/tasks", json={
        "question": "Compare the papers",
        "depth": "standard",
        "execution_mode": "real",
    })
    assert response.status_code == 201
    run = run_store.get_latest_run_for_task(response.json()["id"])
    assert run is not None
    assert run.status == "pending"
    assert run_store.get_latest_checkpoint(run.id).step_sequence == 0
```

- [ ] **Step 3: Run Web API tests and verify RED**

```bash
./.venv/bin/python -m pytest tests/web/test_web_app.py -q
```

Expected: new routes return 404 or fail because `create_app` lacks RunStore injection.

- [ ] **Step 4: Implement Pydantic response models and routes**

`AgentRunResponse` exposes all lifecycle fields except internal `owner_id`;
expose `lease_expires_at` only in development environment. `AgentStepResponse`
exposes bounded input/output/error dictionaries. Every route first calls
`TaskStore.get_task(task_id, user_id=user.id)` and returns 404 before querying
RunStore. Read and cancel routes use `get_latest_run_for_task` so terminal runs
remain inspectable. A task without a run returns 404 with
`detail="agent run not found"`.

In `create_task`, after `create_queued_task` and before `reservation.submit`,
create checkpoint-zero run only when `execution_mode == "real"`. If run creation
fails, release the reservation and conditionally fail the still-pending task.
If queue publication fails, conditionally mark both the unclaimed run and task
failed. A process crash after run creation and before publication leaves a
pending run that the startup reconciler can discover; this improves recovery
but does not claim a transactional SQLite/Redis outbox.

Cancel behavior:

```python
run = run_store.request_cancel(latest_run.id)
task_event_sink.emit(task.id, "cancel_requested", {
    "task_id": task.id,
    "run_id": run.id,
    "cancel_requested_at": run.cancel_requested_at,
})
status_code = 200 if run.is_terminal else 202
return JSONResponse(status_code=status_code, content=run.to_dict(public=True))
```

Emit `cancel_requested` only when the timestamp changed from `None`; terminal
or repeated cancel requests return the current run without adding duplicate
events.

- [ ] **Step 5: Verify existing API compatibility**

```bash
./.venv/bin/python -m pytest tests/web/test_web_app.py tests/web/test_auth.py tests/web/test_pagination.py -q
```

Expected: all tests pass; old task response snapshots remain unchanged.

- [ ] **Step 6: Commit Task 9**

```bash
git add paperpilot/web/app.py tests/web/test_web_app.py
git commit -m "Expose agent run control APIs"
```

### Task 10: Reconcile Expired, Retrying, And Pending Runs

**Files:**

- Create: `paperpilot/agent/reconciler.py`
- Create: `tests/agent/test_reconciler.py`
- Modify: `paperpilot/web/config.py`
- Modify: `paperpilot/web/app.py`
- Modify: `tests/web/test_config.py`
- Modify: `tests/web/test_runtime_config.py`
- Modify: `tests/web/test_web_app.py`

**Interfaces:**

- Produces `RunReconciler.reconcile_once(include_pending=False) -> ReconcileReport`
  and `start()/stop()`.
- Adds config `PAPERPILOT_RECONCILE_INTERVAL_SECONDS` default 15 and `PAPERPILOT_RECONCILE_BATCH_SIZE` default 50.
- Reconciler consumes `RunStore.recover_due_runs`, `list_pending_runs`,
  `list_cancelling_runs`, and `TaskExecutorLike.reserve/submit`.

`ReconcileReport` has `recovered_run_ids`, `pending_run_ids`,
`cancelling_run_ids`, `submitted_run_ids`, and `submission_failures`, all as
ordered lists of run IDs.

- [ ] **Step 1: Write failing deterministic reconciliation tests**

```python
def test_reconcile_once_submits_each_recovered_real_task_once(reconciler_fixture):
    reconciler, store, executor, expired, retry_due = reconciler_fixture()
    report = reconciler.reconcile_once()
    assert report.recovered_run_ids == [expired.id, retry_due.id]
    assert executor.submissions == [
        (expired.task_id, "real"),
        (retry_due.task_id, "real"),
    ]


def test_startup_pass_submits_preexisting_pending_run(reconciler_fixture):
    reconciler, store, executor, pending = reconciler_fixture(pending_only=True)
    report = reconciler.reconcile_once(include_pending=True)
    assert report.pending_run_ids == [pending.id]
    assert executor.submissions == [(pending.task_id, "real")]


def test_periodic_pass_submits_unowned_cancelling_run(reconciler_fixture):
    reconciler, store, executor, cancelling = reconciler_fixture(
        cancelling_only=True,
    )
    report = reconciler.reconcile_once(include_pending=False)
    assert report.cancelling_run_ids == [cancelling.id]
    assert executor.submissions == [(cancelling.task_id, "real")]


def test_reconcile_submission_failure_returns_run_to_waiting_retry(reconciler_fixture):
    reconciler, store, executor, expired, _ = reconciler_fixture(submit_error=True)
    report = reconciler.reconcile_once()
    assert report.submission_failures == [expired.id]
    assert store.get_run(expired.id).status == "waiting_retry"
```

- [ ] **Step 2: Write failing startup/shutdown and config tests**

Verify invalid zero/negative intervals fail config loading, `create_app` starts one injected reconciler on startup, and shutdown stops it before shutting down the executor.

- [ ] **Step 3: Run tests and verify RED**

```bash
./.venv/bin/python -m pytest tests/agent/test_reconciler.py tests/web/test_config.py tests/web/test_runtime_config.py tests/web/test_web_app.py -q
```

Expected: missing reconciler module and config fields.

- [ ] **Step 4: Implement one-pass recovery before background looping**

`reconcile_once(include_pending=False)` calls `recover_due_runs` atomically,
always loads `list_cancelling_runs`, and, when requested by startup, merges a
bounded `list_pending_runs` snapshot. It deduplicates by run ID, reserves
executor capacity per run, submits real mode,
and releases reservation on persistence errors. Capacity rejection reschedules
the run with failure class `capacity` and a delay equal to
`overload_retry_after_seconds`. Broker submission failure follows existing
guarded cleanup semantics and never overwrites a run already claimed by a
worker.

- [ ] **Step 5: Implement bounded background lifecycle**

The reconciler performs one `include_pending=True` pass at startup. Its daemon
thread then uses `Event.wait(interval)` and calls `include_pending=False`, so a
slow pending run is not repeatedly published every interval. Shutdown is
immediate, one pass failure is logged without exiting, and the thread joins
within `interval + 1` seconds. In test injection, `start` and `stop` are simple
call counters. `create_app` stores it on `app.state.run_reconciler`.

- [ ] **Step 6: Run reconciliation and Web runtime tests**

```bash
./.venv/bin/python -m pytest tests/agent/test_reconciler.py tests/web/test_config.py tests/web/test_runtime_config.py tests/web/test_web_app.py tests/web/test_task_executor.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 10**

```bash
git add paperpilot/agent/reconciler.py paperpilot/web/config.py paperpilot/web/app.py tests/agent/test_reconciler.py tests/web/test_config.py tests/web/test_runtime_config.py tests/web/test_web_app.py
git commit -m "Recover interrupted agent runs"
```

---

## Phase 1D: Events, Product Controls, Fault Injection, And Verification

### Task 11: Project Durable Events And Add Compact Run Controls

**Files:**

- Modify: `paperpilot/web/event_mapper.py`
- Modify: `paperpilot/web/static/app.js`
- Modify: `paperpilot/web/static/styles.css`
- Modify: `tests/web/test_event_mapper.py`
- Modify: `tests/web/test_web_app.py`

**Interfaces:**

- Runtime event payloads map to existing `task_events` with stable stages.
- Web task detail fetches run data for real tasks and shows one Cancel button for active runs.
- Cancel UI calls the new endpoint, disables while pending, and refreshes task updates.

- [ ] **Step 1: Write failing event mapping tests**

```python
def test_durable_step_event_keeps_correlation_and_bounds_payload():
    mapped = map_paperpilot_event("step_completed", {
        "run_id": "run-1", "step_id": "step-2", "sequence": 2,
        "attempt": 1, "owner_id": "worker-1", "output": "x" * 5000,
        "api_key": "must-not-leak",
    })
    assert mapped["stage"] == "agent_step"
    assert mapped["payload"]["run_id"] == "run-1"
    assert mapped["payload"]["step_id"] == "step-2"
    assert "must-not-leak" not in str(mapped)
    assert len(str(mapped["payload"])) < 2000
```

Cover run claim/resume, retry scheduled, cancel requested/cancelled, tool start/complete/fail, checkpoint, completion, and failure.

- [ ] **Step 2: Run mapper tests and verify RED**

```bash
./.venv/bin/python -m pytest tests/web/test_event_mapper.py -q
```

Expected: durable events fall through to generic `agent_event` mapping.

- [ ] **Step 3: Implement explicit bounded mappings**

Use stable stages `agent_run`, `agent_step`, `agent_tool`, `agent_retry`, `agent_cancel`, and `agent_checkpoint`. Construct payloads from an allowlist of correlation and metric fields; never copy the incoming dictionary wholesale. Keep the existing mappings for legacy loop events.

- [ ] **Step 4: Add a feature-complete Cancel control**

In `renderTaskDetail`, create a command row only when selected task status is `pending` or `running`. Use an icon/text button with accessible label `Cancel task`, call `POST /api/tasks/{id}/cancel`, disable immediately, show failure in the existing form message region, and call `drainTaskUpdates` after success. Do not add a nested card or instructional copy.

Add `loadRun(taskId)` that tolerates a 404 for simulated tasks and renders run status, attempt, current step, and retry time in the existing definition list. Escape every value through `escapeHtml`.

- [ ] **Step 5: Verify JavaScript syntax and Web tests**

```bash
node --check paperpilot/web/static/app.js
./.venv/bin/python -m pytest tests/web/test_event_mapper.py tests/web/test_web_app.py -q
```

Expected: Node exits 0 and all Web tests pass. If Node is unavailable, record that fact and verify through the local browser during Task 12; do not add Node to dependencies.

- [ ] **Step 6: Commit Task 11**

```bash
git add paperpilot/web/event_mapper.py paperpilot/web/static/app.js paperpilot/web/static/styles.css tests/web/test_event_mapper.py tests/web/test_web_app.py
git commit -m "Show durable agent run controls"
```

### Task 12: Prove Recovery Boundaries And Complete The Phase Audit

**Files:**

- Create: `tests/agent/test_runtime_recovery.py`
- Modify: `README.md`
- Modify: `docs/codex-only-plans/2026-07-19-paperpilot-agent-runtime-control-plane-plan.md`

**Interfaces:**

- Provides a network-free fault-injection harness proving every approved acceptance criterion.
- Documents actual commands, configuration, recovery boundaries, and residual risk.
- Marks plan checkboxes only after the corresponding evidence exists.

- [ ] **Step 1: Write the end-to-end recovery matrix before final implementation adjustments**

Use parametrized fake LLM/tools/store hooks:

```python
@pytest.mark.parametrize("crash_boundary", [
    "after_llm_checkpoint",
    "after_first_tool_checkpoint",
    "after_all_tool_results_checkpoint",
    "after_finalize_checkpoint",
    "after_result_artifact",
])
def test_runtime_recovers_without_duplicate_completed_work(
    recovery_harness, crash_boundary,
):
    first = recovery_harness.run_until_crash(crash_boundary)
    recovered = recovery_harness.resume_with_new_owner(first.run_id)
    assert recovered.status == "completed"
    assert recovery_harness.completed_tool_call_ids_are_unique()
    assert recovery_harness.result_artifact_count(first.run_id) == 1
```

Add separate tests for concurrent owners, cancellation during LLM/tool calls, exhausted retry budget, authentication failure, transport-triggered MCP invalidation, and duplicate Celery delivery.

- [ ] **Step 2: Run the recovery test and verify failures identify real gaps**

```bash
./.venv/bin/python -m pytest tests/agent/test_runtime_recovery.py -q
```

Expected: any failure must name a specific unmet boundary. Fix production code in the owning module and rerun its focused unit tests before rerunning this matrix; do not weaken assertions.

- [ ] **Step 3: Run all focused Agent and Web suites**

```bash
./.venv/bin/python -m pytest tests/agent tests/test_agent_loop.py tests/test_conversation_session.py tests/test_mcp_client.py tests/test_mcp_runtime.py tests/web -q
```

Expected: all focused tests pass.

- [ ] **Step 4: Run the complete test suite**

```bash
./.venv/bin/python -m pytest -q
```

Expected: zero failures. Record the exact pass/deselect/warning counts in the final report.

- [ ] **Step 5: Run static and repository checks**

```bash
./.venv/bin/python -m compileall -q paperpilot scripts
git diff --check
node --check paperpilot/web/static/app.js
```

Expected: all available commands exit 0. Do not claim Node verification if the executable is unavailable.

- [ ] **Step 6: Verify Celery registration and in-memory duplicate protection**

```bash
./.venv/bin/python -m pytest tests/web/test_celery_worker.py tests/agent/test_runtime_recovery.py -q
```

Expected: task registration, warm MCP reuse, redelivery, run claim, and idempotent artifact tests pass without Redis or external network access.

- [ ] **Step 7: Update README with verified behavior and remaining risks**

Document:

- run/step/checkpoint architecture;
- thread and Celery execution flow;
- lease, retry, cancellation, and reconciler configuration;
- safe-boundary recovery semantics;
- Web run/steps/cancel endpoints;
- `ask_user` Web exclusion;
- SQLite write-pressure and checkpoint-size limits;
- non-retryable mixed subagent calls;
- the unresolved SQLite/Redis dual-write window;
- exact current full-suite result from Step 4.

- [ ] **Step 8: Audit all ten design acceptance criteria**

For each criterion in design section 17, record the proving test name and command in the implementation plan's execution log. Any criterion without direct evidence remains incomplete and blocks phase completion.

- [ ] **Step 9: Commit Task 12**

```bash
git add tests/agent/test_runtime_recovery.py README.md docs/codex-only-plans/2026-07-19-paperpilot-agent-runtime-control-plane-plan.md
git commit -m "Verify durable agent runtime recovery"
```

---

## Plan Self-Review Checklist

- [x] Every design goal in sections 2 and 17 maps to at least one task and direct test.
- [x] No task changes authentication or user visibility without preserving current 401/404 behavior.
- [x] No task changes simulated mode or CLI behavior.
- [x] All persisted model names and statuses match Tasks 1-3 and later API/runtime use.
- [x] All retries pass through explicit failure and tool classifications.
- [x] Every completed step checkpoint is atomic and owner/lease guarded.
- [x] Cancellation is cooperative and does not claim to interrupt a synchronous call.
- [x] Artifact projection is idempotent after finalize crashes.
- [x] Thread and Celery paths share AgentRuntime.
- [x] Recovery tests cover every safe boundary and duplicate delivery.

## Execution Log

This section is populated during execution with commit hashes, focused test outputs, full-suite output, and the final requirement-by-requirement audit. It intentionally starts empty because no implementation task has run yet.
