# PaperPilot Warm MCP Runtime And Celery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Redis/Celery task boundary whose long-lived worker processes reuse one MCP runtime while preserving isolated state for every PaperPilot research task.

**Architecture:** FastAPI persists a task and submits only its ID and execution mode to Celery. Each Celery prefork child lazily creates one process-local `MCPRuntime`; `conversation.run()` borrows that runtime's MCP tools while rebuilding all task-local tools and state for every invocation.

**Tech Stack:** Python 3.12, FastAPI, Celery 5.x, Redis, stdio MCP, SQLite, pytest.

## Global Constraints

- Keep `thread` as the default task executor; enable Celery only with `PAPERPILOT_TASK_EXECUTOR=celery`.
- Do not change existing task/event/artifact response schemas.
- Do not migrate SQLite to PostgreSQL in this phase.
- Do not run concurrent tasks against one `MCPRuntime`; scale with Celery prefork processes.
- Do not initialize ColBERT inside `worker_process_init`; initialize lazily on the first real task.
- Preserve task-local messages, TodoStore, input provider, and event callback for every run.
- Use test-first red/green cycles for every production behavior.
- Because this checkout already contains approved uncommitted auth/concurrency changes in overlapping files, do not create intermediate commits or stage files unless the user explicitly asks.

---

### Task 1: Process-Local MCP Runtime

**Files:**
- Create: `paperpilot/tools/mcp_runtime.py`
- Create: `tests/test_mcp_runtime.py`

**Interfaces:**
- Consumes: `MCPClient`, `Tool`.
- Produces: `MCPRuntime.start()`, `MCPRuntime.lease_tools()`, `MCPRuntime.invalidate()`, and `MCPRuntime.close()`.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_runtime_reuses_one_client_for_multiple_leases():
    factory = FakeClientFactory()
    runtime = MCPRuntime(factory)

    with runtime.lease_tools() as first:
        assert [tool.name for tool in first] == ["mcp__fake__search"]
    with runtime.lease_tools() as second:
        assert [tool.name for tool in second] == ["mcp__fake__search"]

    assert factory.create_count == 1
    assert factory.client.start_count == 1
    assert factory.client.close_count == 0


def test_runtime_invalidate_closes_client_and_next_lease_restarts():
    factory = FakeClientFactory()
    runtime = MCPRuntime(factory)
    with runtime.lease_tools():
        pass

    runtime.invalidate()
    with runtime.lease_tools():
        pass

    assert factory.create_count == 2
    assert factory.clients[0].close_count == 1
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_mcp_runtime.py -q`

Expected: collection fails because `paperpilot.tools.mcp_runtime` does not exist.

- [ ] **Step 3: Implement the minimal runtime**

```python
class MCPRuntime:
    def __init__(self, client_factory=None):
        self._client_factory = client_factory or (lambda: MCPClient(MANIFEST_PATH))
        self._client = None
        self._lifecycle_lock = threading.RLock()
        self._lease_lock = threading.Lock()

    @contextmanager
    def lease_tools(self):
        with self._lease_lock:
            self.start()
            assert self._client is not None
            yield self._client.list_tools()

    def start(self):
        with self._lifecycle_lock:
            if self._client is not None:
                return
            client = self._client_factory()
            client.start()
            self._client = client

    def invalidate(self):
        with self._lifecycle_lock:
            client, self._client = self._client, None
            if client is not None:
                client.close()

    def close(self):
        self.invalidate()
```

- [ ] **Step 4: Verify GREEN and focused regression**

Run: `.venv/bin/python -m pytest tests/test_mcp_runtime.py tests/test_mcp_client.py -q`

Expected: all selected tests pass.

### Task 2: Borrow Runtime Tools Without Sharing Conversation State

**Files:**
- Modify: `paperpilot/conversation.py`
- Modify: `tests/test_conversation_session.py`

**Interfaces:**
- Consumes: `MCPRuntime.lease_tools()` from Task 1.
- Produces: `run(query, *, max_iter=8, on_event=None, mcp_runtime=None)`.

- [ ] **Step 1: Write failing shared-runtime conversation tests**

```python
def test_run_with_runtime_borrows_tools_without_closing_runtime(monkeypatch):
    runtime = FakeRuntime()
    seen_tools = []
    monkeypatch.setattr("paperpilot.conversation.agent_loop", fake_agent_loop(seen_tools))

    run("first", mcp_runtime=runtime)
    run("second", mcp_runtime=runtime)

    assert runtime.lease_count == 2
    assert runtime.close_count == 0
    assert seen_tools[0] is not seen_tools[1]


def test_run_without_runtime_keeps_one_shot_client_behavior(monkeypatch):
    resources = []
    monkeypatch.setattr("paperpilot.conversation._build_tools", fake_builder(resources))

    run("one shot")

    assert resources[0].close_count == 1
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_conversation_session.py -q`

Expected: the shared-runtime test fails because `run()` has no `mcp_runtime` parameter.

- [ ] **Step 3: Extract task-tool composition and add runtime injection**

```python
def _compose_tools(registry, todo_store, messages_ref, input_provider, on_event, mcp_tools):
    emit = on_event or _default_logger
    return [
        load_skill_tool(registry),
        research_todo_tool(todo_store),
        paper_deep_read_tool(client_factory=lambda: LLMClient(), mcp_tools=mcp_tools, on_event=emit),
        compact_context_tool(messages_ref=messages_ref, client_factory=lambda: LLMClient(), on_event=emit),
        search_user_document_tool(),
        ask_user_tool(input_provider=input_provider or _default_input_provider, on_event=emit),
        *mcp_tools,
    ]


def run(query, *, max_iter=8, on_event=None, mcp_runtime=None):
    if mcp_runtime is None:
        with ConversationSession(max_iter_per_turn=max_iter, on_event=on_event) as session:
            return session.ask(query)
    with mcp_runtime.lease_tools() as mcp_tools:
        tool_builder = _borrowed_tool_builder(mcp_tools)
        with ConversationSession(max_iter_per_turn=max_iter, on_event=on_event, tool_builder=tool_builder) as session:
            return session.ask(query)
```

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_conversation_session.py -q`

Expected: all selected tests pass and existing one-shot behavior remains intact.

### Task 3: Celery App And Long-Lived Worker Task

**Files:**
- Modify: `requirements.txt`
- Create: `paperpilot/web/celery_app.py`
- Create: `paperpilot/web/worker_tasks.py`
- Create: `tests/web/test_celery_worker.py`

**Interfaces:**
- Consumes: `MCPRuntime`, `WorkflowRunner`, `TaskStore`, `conversation.run`.
- Produces: Celery app `celery_app` and registered task name `paperpilot.web.execute_research_task`.

- [ ] **Step 1: Add the approved Celery/Redis dependency and install it**

```text
celery[redis]>=5.5,<6
```

Run: `.venv/bin/python -m pip install "celery[redis]>=5.5,<6"`

Expected: Celery and Redis Python client install successfully.

- [ ] **Step 2: Write failing Celery configuration and worker reuse tests**

```python
def test_celery_app_uses_json_and_fair_long_task_settings():
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True


def test_worker_reuses_runtime_for_two_real_tasks(monkeypatch):
    runtime_factory = FakeRuntimeFactory()
    monkeypatch.setattr(worker_tasks, "_runtime_factory", runtime_factory)
    monkeypatch.setattr(worker_tasks, "_store_factory", fake_store_factory)

    worker_tasks._execute_research_task("task_1", "real")
    worker_tasks._execute_research_task("task_2", "real")

    assert runtime_factory.create_count == 1
    assert runtime_factory.runtime.lease_count == 2


def test_simulated_worker_task_does_not_start_runtime(monkeypatch):
    runtime_factory = FakeRuntimeFactory()
    monkeypatch.setattr(worker_tasks, "_runtime_factory", runtime_factory)

    worker_tasks._execute_research_task("task_1", "simulated")

    assert runtime_factory.create_count == 0
```

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/python -m pytest tests/web/test_celery_worker.py -q`

Expected: collection fails because Celery modules do not exist.

- [ ] **Step 4: Implement Celery configuration and lazy worker runtime**

```python
celery_app = Celery(
    "paperpilot",
    broker=os.environ.get("PAPERPILOT_CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"),
    include=["paperpilot.web.worker_tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
)
```

```python
_runtime = None

def _get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = _runtime_factory()
    return _runtime

def _execute_research_task(task_id, execution_mode):
    store = _store_factory()
    if execution_mode == "simulated":
        WorkflowRunner(store).run_simulated(task_id)
        return
    runtime = _get_runtime()
    WorkflowRunner(
        store,
        real_runner=lambda query, on_event=None: conversation.run(
            query, on_event=on_event, mcp_runtime=runtime
        ),
    ).run_real(task_id)
```

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/web/test_celery_worker.py tests/web/test_workflow.py -q`

Expected: all selected tests pass.

### Task 4: Celery Executor And FastAPI Queue Boundary

**Files:**
- Modify: `paperpilot/web/task_executor.py`
- Modify: `paperpilot/web/app.py`
- Modify: `tests/web/test_task_executor.py`
- Modify: `tests/web/test_web_app.py`

**Interfaces:**
- Consumes: Celery task name from Task 3.
- Produces: `CeleryTaskExecutor`, `build_task_executor()`, and HTTP 503 queue failure behavior.

- [ ] **Step 1: Write failing executor selection and submission tests**

```python
def test_celery_executor_sends_task_id_and_mode():
    sender = FakeTaskSender()
    executor = CeleryTaskExecutor(sender)

    executor.submit("task_1", "real")

    assert sender.calls == [{
        "name": "paperpilot.web.execute_research_task",
        "args": ["task_1", "real"],
    }]


def test_build_task_executor_selects_celery(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_TASK_EXECUTOR", "celery")
    assert isinstance(build_task_executor(FakeRunner()), CeleryTaskExecutor)
```

- [ ] **Step 2: Write failing API queue-unavailable test**

```python
def test_create_task_marks_failed_when_queue_submission_fails(tmp_path):
    client, store = _client_with_executor(tmp_path, FailingExecutor())
    _register(client)

    response = client.post("/api/tasks", json={"question": "queued", "depth": "quick"})

    assert response.status_code == 503
    task = store.list_tasks()[0]
    assert task.status == "failed"
    assert store.list_events(task.id)[-1].stage == "queue"
```

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/python -m pytest tests/web/test_task_executor.py tests/web/test_web_app.py -q`

Expected: tests fail because Celery executor and queue failure handling do not exist.

- [ ] **Step 4: Implement executor protocol, selection, and failure handling**

```python
class CeleryTaskExecutor:
    def __init__(self, sender=None):
        self.sender = sender or celery_app

    def submit(self, task_id, execution_mode):
        return self.sender.send_task(
            "paperpilot.web.execute_research_task",
            args=[task_id, execution_mode],
        )

    def shutdown(self):
        return None


def build_task_executor(runner):
    backend = os.environ.get("PAPERPILOT_TASK_EXECUTOR", "thread").strip().lower()
    if backend == "celery":
        return CeleryTaskExecutor()
    if backend == "thread":
        return TaskExecutor(runner)
    raise ValueError(f"invalid PAPERPILOT_TASK_EXECUTOR: {backend!r}")
```

Wrap `executor.submit()` in `app.py`; on failure update the task to failed, append a queue failure event, and raise HTTP 503.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/web/test_task_executor.py tests/web/test_web_app.py -q`

Expected: all selected tests pass.

### Task 5: Shared Task Database And Redis Runtime Configuration

**Files:**
- Modify: `paperpilot/web/task_store.py`
- Create: `compose.yaml`
- Modify: `README.md`
- Modify: `tests/web/test_task_store.py`
- Create: `tests/web/test_runtime_config.py`

**Interfaces:**
- Consumes: environment settings in the design.
- Produces: `TaskStore()` environment-aware default path and documented local Redis/Celery commands.

- [ ] **Step 1: Write failing database path and configuration tests**

```python
def test_task_store_uses_environment_default_path(tmp_path, monkeypatch):
    db_path = tmp_path / "shared.sqlite3"
    monkeypatch.setenv("PAPERPILOT_TASK_DB_PATH", str(db_path))

    TaskStore().create_task(question="shared")

    assert db_path.exists()


def test_compose_enables_redis_aof():
    text = Path("compose.yaml").read_text(encoding="utf-8")
    assert "--appendonly" in text
    assert "yes" in text
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/web/test_task_store.py tests/web/test_runtime_config.py -q`

Expected: environment-path assertion fails and compose file is missing.

- [ ] **Step 3: Implement environment path and runtime files**

```python
def _default_task_db_path():
    return Path(os.environ.get("PAPERPILOT_TASK_DB_PATH", DEFAULT_TASK_DB_PATH))

class TaskStore:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path is not None else _default_task_db_path()
```

Create `compose.yaml` with Redis 7, `redis-server --appendonly yes --appendfsync everysec`, a named data volume, and a health check. Document exact API and worker start commands in `README.md`.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/web/test_task_store.py tests/web/test_runtime_config.py -q`

Expected: all selected tests pass.

### Task 6: Verification And Review

**Files:**
- Review all modified files from Tasks 1-5.

**Interfaces:**
- Consumes: completed implementation.
- Produces: verified behavior and an explicit list of remaining limits.

- [ ] **Step 1: Run the focused MCP/Celery/Web suite**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_mcp_runtime.py \
  tests/test_conversation_session.py \
  tests/web -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the full regression suite**

Run: `.venv/bin/python -m pytest -q`

Expected: all tests pass; only known pre-existing warnings remain.

- [ ] **Step 3: Run static diff checks**

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 4: Validate dependency and Celery registration**

Run:

```bash
.venv/bin/python -c "from paperpilot.web.celery_app import celery_app; print(sorted(celery_app.conf.include))"
.venv/bin/celery -A paperpilot.web.celery_app:celery_app inspect registered
```

Expected: local import succeeds. The inspect command may report no nodes when Redis/worker is not running; that is an environment limitation, not an import failure.

- [ ] **Step 5: Review scope and residual risk**

Confirm from the diff that:

- CLI one-shot `conversation.run()` remains backward compatible.
- API defaults to thread execution without Redis.
- Celery mode sends only JSON-safe task ID and execution mode.
- Worker runtime is lazy and process-local.
- Per-task state is rebuilt for every call.
- SQLite/PostgreSQL migration and remote MCP remain outside this change.

## Review Amendments

The completion review added the following in-scope hardening without changing
the approved storage schema:

- MCP timeout/connection failures invalidate the shared runtime after the
  active lease exits, and teardown waits for active leases.
- MCP SDK and AnyIO disconnects are translated to `MCPTransportError` at the
  client boundary so runtime self-healing covers real transport failures.
- `TaskStore.claim_task()` atomically prevents fresh duplicate messages from
  executing the same task; redelivered messages may recover `running` work.
- Celery task limits are shorter than Redis visibility timeout.
- Celery cancels late-ack long tasks when the broker connection is lost before
  allowing redelivery recovery.
- Celery uses the research task ID as its message task ID.
- Concurrent duplicate registration maps the database unique constraint to
  the existing HTTP 409 behavior.
- Redis binds to `127.0.0.1` in local compose configuration.
- Ambiguous publish failures use an atomic `pending -> failed` transition and
  never overwrite work already claimed by a worker.

The database/broker dual-write window remains explicit. Closing it requires a
transactional outbox or reconciler and a separately approved data/behavior
change.
