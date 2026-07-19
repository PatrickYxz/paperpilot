# PaperPilot SQLite Pagination And Incremental Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden PaperPilot's FastAPI read path for small concurrent workloads by enabling SQLite WAL/indexes, bounding all task-history reads, and replacing four-request full polling with one incremental updates request.

**Architecture:** `TaskStore` owns WAL connections, indexed keyset/watermark queries, and a single-transaction updates snapshot. A focused `pagination.py` module owns opaque task cursor encoding and validation; FastAPI maps store pages into explicit response envelopes, while the static frontend tracks event/artifact watermarks and polls only the aggregate updates endpoint.

**Tech Stack:** Python 3.12, FastAPI/Pydantic, standard-library SQLite/base64/json, vanilla JavaScript, pytest, FastAPI TestClient.

## Global Constraints

- Implement `docs/codex-only-plans/2026-07-14-paperpilot-sqlite-pagination-performance-design.md` exactly.
- Do not add Python or JavaScript runtime dependencies.
- Keep authentication, user isolation, task creation, single-task detail, Celery dispatch, task state transitions, and MCP Runtime reuse unchanged.
- Change task/event/artifact list API responses from arrays to bounded envelopes and update the static frontend in the same pass.
- Task order is `created_at DESC, id DESC`; page size defaults to 50 and must stay within 1..100.
- Event/artifact reads use `id > after_id`, ascending order, and the same 1..100 page-size boundary.
- `get_task_updates()` must read task, events, and artifacts inside one explicit SQLite read transaction.
- WAL improves reader/writer overlap but SQLite remains single writer; do not claim high-write scalability.
- Preserve the known SQLite/Redis dual-write gap; transactional outbox work remains out of scope.
- This checkout contains approved uncommitted auth/Celery/MCP work. Do not stage, commit, push, reset, or revert unless the user explicitly asks.
- Follow RED/GREEN TDD for every production behavior and run each named failing test before implementation.

---

### Task 1: Opaque Task Cursor Codec

**Files:**
- Create: `paperpilot/web/pagination.py`
- Create: `tests/web/test_pagination.py`

**Interfaces:**
- Consumes: `user_id`, optional status, `created_at`, and task id strings.
- Produces: `TaskCursor`, `InvalidTaskCursor`, `encode_task_cursor`, and `decode_task_cursor`.

- [ ] **Step 1: Write failing cursor tests**

Create round-trip, malformed payload, wrong user, and wrong status tests:

```python
def test_task_cursor_round_trip_binds_user_and_status():
    encoded = encode_task_cursor(
        user_id="user_1",
        status="running",
        created_at="2026-07-14T08:00:00+00:00",
        task_id="task_2",
    )
    assert decode_task_cursor(
        encoded,
        expected_user_id="user_1",
        expected_status="running",
    ) == TaskCursor(
        created_at="2026-07-14T08:00:00+00:00",
        task_id="task_2",
    )


@pytest.mark.parametrize("value", ["not-base64", "e30"])
def test_task_cursor_rejects_malformed_payload(value):
    with pytest.raises(InvalidTaskCursor, match="invalid task cursor"):
        decode_task_cursor(
            value,
            expected_user_id="user_1",
            expected_status=None,
        )


def test_task_cursor_rejects_different_request_context():
    encoded = encode_task_cursor(
        user_id="user_1",
        status="pending",
        created_at="2026-07-14T08:00:00+00:00",
        task_id="task_2",
    )
    with pytest.raises(InvalidTaskCursor):
        decode_task_cursor(
            encoded,
            expected_user_id="user_2",
            expected_status="pending",
        )
    with pytest.raises(InvalidTaskCursor):
        decode_task_cursor(
            encoded,
            expected_user_id="user_1",
            expected_status="completed",
        )
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/web/test_pagination.py -q`

Expected: collection fails because `paperpilot.web.pagination` does not exist.

- [ ] **Step 3: Implement the cursor codec**

Create `paperpilot/web/pagination.py` with:

```python
"""Opaque cursor helpers for bounded Web task pagination."""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import datetime

_CURSOR_VERSION = 1


@dataclass(frozen=True)
class TaskCursor:
    created_at: str
    task_id: str


class InvalidTaskCursor(ValueError):
    """Raised when a task cursor is invalid for the current request."""


def encode_task_cursor(
    *,
    user_id: str,
    status: str | None,
    created_at: str,
    task_id: str,
) -> str:
    payload = {
        "v": _CURSOR_VERSION,
        "u": user_id,
        "s": status,
        "t": created_at,
        "i": task_id,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_task_cursor(
    value: str,
    *,
    expected_user_id: str,
    expected_status: str | None,
) -> TaskCursor:
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(raw.decode())
        if not isinstance(payload, dict) or set(payload) != {"v", "u", "s", "t", "i"}:
            raise ValueError
        created_at = payload["t"]
        task_id = payload["i"]
        if (
            payload["v"] != _CURSOR_VERSION
            or payload["u"] != expected_user_id
            or payload["s"] != expected_status
            or not isinstance(created_at, str)
            or not isinstance(task_id, str)
            or not created_at
            or not task_id
        ):
            raise ValueError
        datetime.fromisoformat(created_at)
    except (
        binascii.Error,
        json.JSONDecodeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise InvalidTaskCursor("invalid task cursor") from exc
    return TaskCursor(created_at=created_at, task_id=task_id)
```

- [ ] **Step 4: Verify GREEN and syntax**

Run: `.venv/bin/python -m pytest tests/web/test_pagination.py -q`

Run: `.venv/bin/python -m compileall -q paperpilot/web/pagination.py tests/web/test_pagination.py`

Expected: all cursor tests pass and compileall exits 0.

---

### Task 2: SQLite WAL, Connection Pragmas, And Business Indexes

**Files:**
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_task_store.py`

**Interfaces:**
- Consumes: the existing `TaskStore(db_path)` schema initialization.
- Produces: WAL database state, configured business connections, and four named query indexes.

- [ ] **Step 1: Write failing runtime-setting and index tests**

```python
def test_store_enables_wal_and_connection_pragmas(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    with store._connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_store_creates_query_indexes(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    with store._connect() as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
    assert {
        "idx_tasks_user_created_id",
        "idx_tasks_user_status_created_id",
        "idx_events_task_id_id",
        "idx_artifacts_task_id_id",
    } <= names
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/web/test_task_store.py -q -k 'wal_and_connection or creates_query_indexes'`

Expected: journal/foreign-key assertions fail and indexes are absent.

- [ ] **Step 3: Configure every connection and initialize WAL**

Update `_connect()`:

```python
def _connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn
```

At the start of `_ensure_schema()` execute `PRAGMA journal_mode = WAL`, normalize the returned string, and raise `RuntimeError` if SQLite does not return `wal`. Do not run the journal-mode mutation for every business query.

- [ ] **Step 4: Add the four approved indexes after table migration**

```sql
CREATE INDEX IF NOT EXISTS idx_tasks_user_created_id
ON research_tasks(user_id, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_tasks_user_status_created_id
ON research_tasks(user_id, status, created_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_events_task_id_id
ON task_events(task_id, id);

CREATE INDEX IF NOT EXISTS idx_artifacts_task_id_id
ON task_artifacts(task_id, id);
```

- [ ] **Step 5: Verify GREEN and legacy migration compatibility**

Run: `.venv/bin/python -m pytest tests/web/test_task_store.py -q`

Expected: all TaskStore tests pass, including the legacy `task_events` schema migration.

### Task 3: Indexed Task Keyset Pagination

**Files:**
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_task_store.py`

**Interfaces:**
- Consumes: explicit `user_id | None`, optional status, limit, and optional `(before_created_at, before_id)` pair.
- Produces: `TaskPage(items, has_more)` and `TaskStore.list_tasks_page`.

- [ ] **Step 1: Write failing stable-pagination and isolation tests**

Create real users before setting `user_id` so enabled foreign keys remain valid. Force five tasks to the same timestamp and walk pages of two:

```python
def _create_test_user(store, username):
    return store.create_user(
        username=username,
        password_hash="hash",
        password_salt="salt",
    )


def test_list_tasks_page_uses_stable_keyset_with_equal_timestamps(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = _create_test_user(store, "alice")
    created = [
        store.create_task(question=f"task {index}", user_id=user.id)
        for index in range(5)
    ]
    with store._connect() as conn:
        conn.execute(
            "UPDATE research_tasks SET created_at = ? WHERE user_id = ?",
            ("2026-07-14T08:00:00+00:00", user.id),
        )

    first = store.list_tasks_page(user_id=user.id, limit=2)
    second = store.list_tasks_page(
        user_id=user.id,
        limit=2,
        before_created_at=first.items[-1].created_at,
        before_id=first.items[-1].id,
    )
    third = store.list_tasks_page(
        user_id=user.id,
        limit=2,
        before_created_at=second.items[-1].created_at,
        before_id=second.items[-1].id,
    )

    actual = [task.id for task in first.items + second.items + third.items]
    assert actual == sorted((task.id for task in created), reverse=True)
    assert len(set(actual)) == 5
    assert [first.has_more, second.has_more, third.has_more] == [True, True, False]
```

Add a second test with Alice pending/failed tasks and Bob pending task; assert Alice's `status="pending"` page contains only Alice's pending id.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/web/test_task_store.py -q -k 'list_tasks_page'`

Expected: fails because `list_tasks_page` does not exist.

- [ ] **Step 3: Add page type and bounded query**

```python
@dataclass(frozen=True)
class TaskPage:
    items: list[ResearchTask]
    has_more: bool
```

Implement:

```python
def list_tasks_page(
    self,
    *,
    user_id: str | None,
    limit: int,
    status: str | None = None,
    before_created_at: str | None = None,
    before_id: str | None = None,
) -> TaskPage:
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    if (before_created_at is None) != (before_id is None):
        raise ValueError("task page position requires created_at and id")

    clauses: list[str] = []
    params: list[object] = []
    if user_id is None:
        clauses.append("user_id IS NULL")
    else:
        clauses.append("user_id = ?")
        params.append(user_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if before_created_at is not None:
        clauses.append(
            "(created_at < ? OR (created_at = ? AND id < ?))"
        )
        params.extend([before_created_at, before_created_at, before_id])

    query = f"""
        SELECT *
        FROM research_tasks
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC, id DESC
        LIMIT ?
    """
    params.append(limit + 1)
    with self._connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    tasks = [_task_from_row(row) for row in rows]
    return TaskPage(
        items=tasks[:limit],
        has_more=len(tasks) > limit,
    )
```

The dynamic SQL joins only predefined clause strings; all request values remain bound parameters. Temporarily retain old unbounded methods until all callers migrate in Task 5.

- [ ] **Step 4: Prove both task query shapes use indexes**

Seed 200 user tasks and parameterize `status=None` / `status="pending"`. Run `EXPLAIN QUERY PLAN` against the exact SQL templates and assert:

```python
assert expected_index in details
assert "USE TEMP B-TREE" not in details
```

Expected indexes are `idx_tasks_user_created_id` and `idx_tasks_user_status_created_id` respectively.

- [ ] **Step 5: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/web/test_task_store.py -q`

Expected: all TaskStore tests pass and the query plans name the business indexes.

---

### Task 4: Event/Artifact Watermarks And Consistent Updates Snapshot

**Files:**
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_task_store.py`

**Interfaces:**
- Consumes: task id, explicit `user_id | None`, non-negative watermark, bounded limit.
- Produces: `TaskEventBatch`, `TaskArtifactBatch`, `TaskUpdates`, two page methods, and `get_task_updates()`.

- [ ] **Step 1: Write failing watermark tests**

```python
def test_event_and_artifact_pages_advance_watermarks(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    task = store.create_task(question="updates")
    events = [
        store.add_event(task_id=task.id, type="progress", message=f"event {i}")
        for i in range(3)
    ]
    artifacts = [
        store.add_artifact(
            task_id=task.id,
            kind="result",
            title=f"artifact {i}",
            content=f"content {i}",
        )
        for i in range(2)
    ]

    event_page = store.list_events_page(
        task.id,
        user_id=None,
        after_id=events[0].id,
        limit=1,
    )
    artifact_page = store.list_artifacts_page(
        task.id,
        user_id=None,
        after_id=0,
        limit=1,
    )

    assert event_page.items == [events[1]]
    assert event_page.next_after_id == events[1].id
    assert event_page.has_more is True
    assert artifact_page.items == [artifacts[0]]
    assert artifact_page.next_after_id == artifacts[0].id
    assert artifact_page.has_more is True
```

Add the ownership test:

```python
def test_task_updates_returns_one_owned_snapshot(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice")
    bob = _create_test_user(store, "bob")
    task = store.create_task(question="alice updates", user_id=alice.id)
    event = store.add_event(task_id=task.id, type="queued", message="queued")
    artifact = store.add_artifact(
        task_id=task.id,
        kind="result",
        title="result",
        content="answer",
    )

    updates = store.get_task_updates(
        task.id,
        user_id=alice.id,
        after_event_id=0,
        after_artifact_id=0,
        limit=50,
    )

    assert updates is not None
    assert updates.task == task
    assert updates.events.items == [event]
    assert updates.artifacts.items == [artifact]
    assert store.get_task_updates(
        task.id,
        user_id=bob.id,
        after_event_id=0,
        after_artifact_id=0,
        limit=50,
    ) is None
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/web/test_task_store.py -q -k 'pages_advance or owned_snapshot'`

Expected: page/update methods are missing.

- [ ] **Step 3: Add batch/update data types**

```python
@dataclass(frozen=True)
class TaskEventBatch:
    items: list[TaskEvent]
    next_after_id: int
    has_more: bool


@dataclass(frozen=True)
class TaskArtifactBatch:
    items: list[TaskArtifact]
    next_after_id: int
    has_more: bool


@dataclass(frozen=True)
class TaskUpdates:
    task: ResearchTask
    events: TaskEventBatch
    artifacts: TaskArtifactBatch
```

- [ ] **Step 4: Implement bounded connection-level readers**

Create `_read_event_batch` and `_read_artifact_batch`. The event helper is:

```python
def _read_event_batch(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    after_id: int,
    limit: int,
) -> TaskEventBatch:
    rows = conn.execute(
        """
        SELECT id, task_id, type, stage, message, payload_json, created_at
        FROM task_events
        WHERE task_id = ? AND id > ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (task_id, after_id, limit + 1),
    ).fetchall()
    items = [_event_from_row(row) for row in rows[:limit]]
    return TaskEventBatch(
        items=items,
        next_after_id=items[-1].id if items else after_id,
        has_more=len(rows) > limit,
    )
```

The artifact helper is:

```python
def _read_artifact_batch(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    after_id: int,
    limit: int,
) -> TaskArtifactBatch:
    rows = conn.execute(
        """
        SELECT id, task_id, kind, title, content, payload_json, created_at
        FROM task_artifacts
        WHERE task_id = ? AND id > ?
        ORDER BY id ASC
        LIMIT ?
        """,
        (task_id, after_id, limit + 1),
    ).fetchall()
    items = [_artifact_from_row(row) for row in rows[:limit]]
    return TaskArtifactBatch(
        items=items,
        next_after_id=items[-1].id if items else after_id,
        has_more=len(rows) > limit,
    )


def _validate_incremental_page(after_id: int, limit: int) -> None:
    if after_id < 0:
        raise ValueError("after_id must be non-negative")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
```

- [ ] **Step 5: Implement ownership lookup, page methods, and explicit read transaction**

Create the ownership helper and public page methods:

```python
def _select_owned_task_row(
    conn: sqlite3.Connection,
    task_id: str,
    user_id: str | None,
) -> sqlite3.Row | None:
    if user_id is None:
        return conn.execute(
            "SELECT * FROM research_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM research_tasks WHERE id = ? AND user_id = ?",
        (task_id, user_id),
    ).fetchone()


def list_events_page(
    self,
    task_id: str,
    *,
    user_id: str | None,
    after_id: int,
    limit: int,
) -> TaskEventBatch | None:
    _validate_incremental_page(after_id, limit)
    with self._connect() as conn:
        if _select_owned_task_row(conn, task_id, user_id) is None:
            return None
        return _read_event_batch(
            conn,
            task_id,
            after_id=after_id,
            limit=limit,
        )


def list_artifacts_page(
    self,
    task_id: str,
    *,
    user_id: str | None,
    after_id: int,
    limit: int,
) -> TaskArtifactBatch | None:
    _validate_incremental_page(after_id, limit)
    with self._connect() as conn:
        if _select_owned_task_row(conn, task_id, user_id) is None:
            return None
        return _read_artifact_batch(
            conn,
            task_id,
            after_id=after_id,
            limit=limit,
        )
```

Then implement the aggregate snapshot:

```python
def get_task_updates(
    self,
    task_id: str,
    *,
    user_id: str | None,
    after_event_id: int,
    after_artifact_id: int,
    limit: int,
) -> TaskUpdates | None:
    _validate_incremental_page(after_event_id, limit)
    _validate_incremental_page(after_artifact_id, limit)
    with self._connect() as conn:
        conn.execute("BEGIN")
        row = _select_owned_task_row(conn, task_id, user_id)
        if row is None:
            return None
        return TaskUpdates(
            task=_task_from_row(row),
            events=_read_event_batch(
                conn, task_id, after_id=after_event_id, limit=limit
            ),
            artifacts=_read_artifact_batch(
                conn, task_id, after_id=after_artifact_id, limit=limit
            ),
        )
```

The connection context commits the explicit read transaction on normal return and rolls it back on exceptions.

- [ ] **Step 6: Prove event/artifact index usage**

Use `EXPLAIN QUERY PLAN` for both exact watermark queries. Assert details contain `idx_events_task_id_id` / `idx_artifacts_task_id_id` and no `USE TEMP B-TREE`.

- [ ] **Step 7: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/web/test_task_store.py -q`

Expected: all TaskStore tests pass.

### Task 5: FastAPI Page Contracts And Complete Caller Migration

**Files:**
- Modify: `paperpilot/web/app.py`
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_web_app.py`
- Modify: `tests/web/test_workflow.py`
- Modify: `tests/web/test_celery_worker.py`
- Modify: `tests/web/test_task_store.py`

**Interfaces:**
- Consumes: cursor codec and TaskStore page/update methods.
- Produces: task/event/artifact page response models, updates response model, and four bounded FastAPI routes.

- [ ] **Step 1: Change API tests to the approved envelopes and add cursor/update tests**

Change task-list assertions to read `payload["items"]`, `next_cursor`, and `has_more`. Change event/artifact assertions to read `response.json()["items"]`. Bob's empty task list must equal:

```python
{
    "items": [],
    "next_cursor": None,
    "has_more": False,
}
```

Add tests that:

1. create three tasks and traverse them using `limit=2` plus `next_cursor`, proving all ids appear once;
2. reject malformed cursor and a cursor issued to another authenticated user with status 422 and `invalid task cursor` detail;
3. reject `limit=0`, `limit=101`, and negative watermarks with FastAPI 422;
4. call updates with `limit=2`, then call it again using returned event/artifact watermarks and prove returned ids are disjoint;
5. preserve 404 for missing and other-user updates.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/web/test_web_app.py -q`

Expected: envelope assertions fail and `/updates` returns 404.

- [ ] **Step 3: Add Pydantic response models**

Import cursor helpers and define:

```python
class TaskPageResponse(BaseModel):
    items: list[TaskResponse]
    next_cursor: str | None
    has_more: bool


class TaskEventPageResponse(BaseModel):
    items: list[TaskEventResponse]
    next_after_id: int
    has_more: bool


class TaskArtifactPageResponse(BaseModel):
    items: list[TaskArtifactResponse]
    next_after_id: int
    has_more: bool


class TaskUpdatesResponse(BaseModel):
    task: TaskResponse
    events: TaskEventPageResponse
    artifacts: TaskArtifactPageResponse
```

- [ ] **Step 4: Replace the task-list route**

Use `limit: int = Query(default=50, ge=1, le=100)` and optional cursor. Decode with current user/status, convert `InvalidTaskCursor` to `HTTPException(status_code=422, detail="invalid task cursor")`, call `list_tasks_page`, and encode a next cursor from the final returned task only when `has_more=True`.

The returned dictionary must be:

```python
{
    "items": [task.to_dict() for task in page.items],
    "next_cursor": next_cursor,
    "has_more": page.has_more,
}
```

- [ ] **Step 5: Replace event/artifact routes and add updates route**

All page routes use `after_id: int = Query(default=0, ge=0)` and the same bounded limit. Import `TaskEventBatch` and `TaskArtifactBatch`, then define shared mappers:

```python
def _event_page_dict(batch: TaskEventBatch) -> dict:
    return {
        "items": [event.to_dict() for event in batch.items],
        "next_after_id": batch.next_after_id,
        "has_more": batch.has_more,
    }


def _artifact_page_dict(batch: TaskArtifactBatch) -> dict:
    return {
        "items": [artifact.to_dict() for artifact in batch.items],
        "next_after_id": batch.next_after_id,
        "has_more": batch.has_more,
    }
```

Standalone routes return these mappings after converting `None` to the existing 404 response.

Add:

```python
@app.get(
    "/api/tasks/{task_id}/updates",
    response_model=TaskUpdatesResponse,
)
def get_task_updates(
    task_id: str,
    after_event_id: int = Query(default=0, ge=0),
    after_artifact_id: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    user: WebUser = Depends(require_user),
) -> dict:
    updates = store.get_task_updates(
        task_id,
        user_id=user.id,
        after_event_id=after_event_id,
        after_artifact_id=after_artifact_id,
        limit=limit,
    )
    if updates is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task": updates.task.to_dict(),
        "events": _event_page_dict(updates.events),
        "artifacts": _artifact_page_dict(updates.artifacts),
    }
```

Use private mapping helpers for page dictionaries so standalone and aggregate routes cannot drift.

- [ ] **Step 6: Migrate all internal test callers, then remove unbounded methods**

Replace direct store reads in workflow/Celery/store/API tests with bounded calls:

```python
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
```

For API-created tasks, assign `registered_user = _register(client)` and use `registered_user["id"]` as user id. Replace task lookup with `list_tasks_page(user_id=user_id, limit=100).items`. After `rg` finds no callers, delete `TaskStore.list_tasks`, `TaskStore.list_events`, and `TaskStore.list_artifacts`.

- [ ] **Step 7: Verify GREEN across Web modules**

Run: `.venv/bin/python -m pytest tests/web -q`

Expected: all Web tests pass with no bare-array API assumptions.

- [ ] **Step 8: Prove old unbounded reads are gone**

Run: `rg -n "store\.list_tasks\(|store\.list_events\(|store\.list_artifacts\(" paperpilot tests`

Expected: no matches.

---

### Task 6: Static Frontend Incremental Polling

**Files:**
- Modify: `paperpilot/web/static/app.js`
- Modify: `tests/web/test_web_app.py`

**Interfaces:**
- Consumes: `TaskPageResponse` and `TaskUpdatesResponse`.
- Produces: first-page rendering, local event/artifact caches, numeric watermarks, and one-request steady-state polling.

- [ ] **Step 1: Strengthen static asset contract and verify RED**

Add assertions:

```python
assert "payload.items" in js.text
assert "/updates" in js.text
assert "after_event_id" in js.text
assert "after_artifact_id" in js.text
assert 'requestJson(`/api/tasks/${encodedTaskId}/events`)' not in js.text
assert 'requestJson(`/api/tasks/${encodedTaskId}/artifacts`)' not in js.text
```

Run: `.venv/bin/python -m pytest tests/web/test_web_app.py::test_event_ui_static_assets_are_served -q`

Expected: fails because current detail polling still contains the event/artifact endpoint calls.

- [ ] **Step 2: Add race-safe selected-task state**

Replace current globals with:

```javascript
let selectedTaskId = null;
let selectedTask = null;
let selectedEvents = [];
let selectedArtifacts = [];
let eventAfterId = 0;
let artifactAfterId = 0;
let selectionVersion = 0;
let pollTimer = null;
```

Add:

```javascript
function resetSelectedTask(taskId) {
  selectionVersion += 1;
  selectedTaskId = taskId;
  selectedTask = null;
  selectedEvents = [];
  selectedArtifacts = [];
  eventAfterId = 0;
  artifactAfterId = 0;
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  return selectionVersion;
}
```

Call `resetSelectedTask(null)` from `showUnauthenticated()` so stale responses cannot render another user's data.

- [ ] **Step 3: Consume the task-page envelope**

```javascript
async function loadTasks() {
  if (workbench.hidden) {
    return;
  }
  const params = new URLSearchParams({ limit: "50" });
  if (statusFilter.value) {
    params.set("status", statusFilter.value);
  }
  const payload = await requestJson("/api/tasks?" + params.toString());
  renderTasks(payload.items);
}
```

- [ ] **Step 4: Implement cache merging and one updates request**

Add:

```javascript
function mergeById(existing, incoming) {
  const byId = new Map(existing.map((item) => [item.id, item]));
  for (const item of incoming) {
    byId.set(item.id, item);
  }
  return [...byId.values()].sort((left, right) => left.id - right.id);
}

async function requestTaskUpdates(taskId, version) {
  const encodedTaskId = encodeURIComponent(taskId);
  const params = new URLSearchParams({
    after_event_id: String(eventAfterId),
    after_artifact_id: String(artifactAfterId),
    limit: "100",
  });
  const payload = await requestJson(
    "/api/tasks/" + encodedTaskId + "/updates?" + params.toString(),
  );
  if (version !== selectionVersion || taskId !== selectedTaskId) {
    return null;
  }
  selectedTask = payload.task;
  selectedEvents = mergeById(selectedEvents, payload.events.items);
  selectedArtifacts = mergeById(selectedArtifacts, payload.artifacts.items);
  eventAfterId = payload.events.next_after_id;
  artifactAfterId = payload.artifacts.next_after_id;
  renderTaskDetail(selectedTask, selectedEvents, selectedArtifacts);
  return payload;
}
```

- [ ] **Step 5: Replace selection/polling flow and drain backlog**

Replace the old `loadTask` and `schedulePolling` functions with:

```javascript
async function loadTask(taskId) {
  if (workbench.hidden) {
    return;
  }
  const version = resetSelectedTask(taskId);
  await drainTaskUpdates(taskId, version);
  if (version === selectionVersion) {
    await loadTasks();
  }
}

async function drainTaskUpdates(taskId, version) {
  const previousStatus = selectedTask ? selectedTask.status : null;
  let payload = await requestTaskUpdates(taskId, version);
  if (!payload) {
    return;
  }
  while (payload.events.has_more || payload.artifacts.has_more) {
    payload = await requestTaskUpdates(taskId, version);
    if (!payload) {
      return;
    }
  }
  const currentStatus = payload.task.status;
  if (
    previousStatus
    && previousStatus !== currentStatus
    && currentStatus !== "pending"
    && currentStatus !== "running"
  ) {
    await loadTasks();
  }
  schedulePolling(taskId, version, currentStatus);
}

function schedulePolling(taskId, version, status) {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  if (status !== "pending" && status !== "running") {
    return;
  }
  pollTimer = setTimeout(() => {
    drainTaskUpdates(taskId, version).catch((error) => {
      setMessage(error.message, "error");
    });
  }, 1000);
}
```

The timeout callback calls only `drainTaskUpdates`; steady-state polling does not call task list, task detail, event, or artifact endpoints.

- [ ] **Step 6: Verify frontend contract and syntax**

Run: `.venv/bin/python -m pytest tests/web/test_web_app.py -q`

Run: `node --check paperpilot/web/static/app.js`

Expected: tests pass and Node exits 0. If Node is unavailable, record it and use browser loading later; do not add Node as a dependency.

### Task 7: Reproducible Store Benchmark And Runtime Documentation

**Files:**
- Create: `scripts/benchmark_web_task_store.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `TaskStore`, bounded task pages, and configured SQLite connections.
- Produces: machine-readable benchmark evidence plus runtime/API documentation.

- [ ] **Step 1: Create the benchmark CLI**

Support exact options:

```python
parser.add_argument("--tasks", type=int, default=2000)
parser.add_argument("--writes", type=int, default=200)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--page-size", type=int, default=50)
parser.add_argument("--db-path", type=Path)
```

Build the script around these exact helpers:

```python
def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 6)


def _run(db_path: Path, args: argparse.Namespace) -> dict:
    store = TaskStore(db_path)
    user = store.create_user(
        username=f"benchmark-{uuid.uuid4().hex}",
        password_hash="benchmark",
        password_salt="benchmark",
    )

    started = time.perf_counter()
    for index in range(args.tasks):
        store.create_task(question=f"seed task {index}", user_id=user.id)
    seed_seconds = time.perf_counter() - started

    started = time.perf_counter()
    page = store.list_tasks_page(user_id=user.id, limit=args.page_size)
    page_ms = (time.perf_counter() - started) * 1000
    page_json = json.dumps(
        [task.to_dict() for task in page.items],
        ensure_ascii=False,
    ).encode()

    def create_one(index: int) -> float:
        write_started = time.perf_counter()
        store.create_task(question=f"concurrent task {index}", user_id=user.id)
        return (time.perf_counter() - write_started) * 1000

    latencies: list[float] = []
    errors: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(create_one, index) for index in range(args.writes)]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    concurrent_wall_seconds = time.perf_counter() - started

    with store._connect() as conn:
        indexes = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'index' ORDER BY name"
            ).fetchall()
        ]
        unfiltered_plan = [
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM research_tasks "
                "WHERE user_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (user.id, args.page_size + 1),
            ).fetchall()
        ]
        filtered_plan = [
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM research_tasks "
                "WHERE user_id = ? AND status = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (user.id, "pending", args.page_size + 1),
            ).fetchall()
        ]
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
        foreign_keys = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
        busy_timeout = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
        synchronous = int(conn.execute("PRAGMA synchronous").fetchone()[0])

    return {
        "seed_tasks": args.tasks,
        "seed_seconds": round(seed_seconds, 6),
        "page_rows": len(page.items),
        "page_has_more": page.has_more,
        "page_ms": round(page_ms, 6),
        "page_json_bytes": len(page_json),
        "concurrent_writes": len(latencies),
        "concurrent_errors": errors,
        "concurrent_wall_seconds": round(concurrent_wall_seconds, 6),
        "write_p50_ms": _percentile(latencies, 0.50) if latencies else None,
        "write_p95_ms": _percentile(latencies, 0.95) if latencies else None,
        "write_max_ms": round(max(latencies), 6) if latencies else None,
        "journal_mode": journal_mode,
        "busy_timeout_ms": busy_timeout,
        "foreign_keys": foreign_keys,
        "synchronous": synchronous,
        "indexes": indexes,
        "query_plans": {
            "unfiltered": unfiltered_plan,
            "pending": filtered_plan,
        },
    }
```

Import `argparse`, `json`, `math`, `time`, `uuid`, `as_completed`, `ThreadPoolExecutor`, `Path`, and `TemporaryDirectory`. In `main()`, reject non-positive counts and page size above 100 with `parser.error`. Use the explicit path when provided; otherwise execute `_run(Path(tmp) / "benchmark.sqlite3", args)` inside `TemporaryDirectory()`. Print `json.dumps(result, indent=2, ensure_ascii=False)`. If every concurrent write fails, percentile fields remain `null` so the JSON still reports the actual errors.

Open a configured store connection to report:

- `journal_mode`, `foreign_keys`, `busy_timeout`, `synchronous`;
- all named indexes;
- `EXPLAIN QUERY PLAN` for task queries with and without status;
- seed time, bounded-page time/rows/bytes/has-more;
- concurrent success/error counts, wall time, P50/P95/max.

Print a single JSON object. Percentiles must sort observed latencies and use a deterministic nearest-rank index; do not depend on NumPy.

- [ ] **Step 2: Run the full benchmark and inspect invariants**

Run: `.venv/bin/python scripts/benchmark_web_task_store.py --tasks 2000 --writes 200 --workers 8 --page-size 50`

Expected:

- `page_rows == 50` and `page_has_more == true`;
- page bytes are materially below the 410,890-byte unbounded baseline;
- no concurrent write errors;
- WAL, foreign keys, 30000 ms busy timeout, NORMAL synchronous;
- both task indexes appear and no plan uses `USE TEMP B-TREE`.

Record timings, but do not assert a fixed millisecond threshold across machines.

- [ ] **Step 3: Document runtime behavior and benchmark usage**

Add `## Web Read Performance` after the Celery section in `README.md`. Explain:

- WAL/foreign keys/busy timeout and SQLite's remaining single-writer limit;
- task `{items, next_cursor, has_more}` envelope, default 50, maximum 100;
- event/artifact watermark pages and single-transaction `/updates`;
- one-second updates-only polling while pending/running;
- benchmark command and machine-dependent timing;
- PostgreSQL as the next database step when write concurrency exceeds this local design.

- [ ] **Step 4: Verify benchmark and docs**

Run: `.venv/bin/python -m compileall -q scripts/benchmark_web_task_store.py`

Run: `.venv/bin/python scripts/benchmark_web_task_store.py --tasks 100 --writes 20 --workers 4 --page-size 50`

Run: `git diff --check`

Expected: all commands exit 0; benchmark reports bounded page and no errors.

---

### Task 8: Full Verification, Browser Smoke, And Independent Review

**Files:**
- Review every file changed by Tasks 1-7.
- Do not modify unrelated worktree changes or create commits.

**Interfaces:**
- Consumes: complete implementation and test/benchmark evidence.
- Produces: verified code, runtime request-count evidence, review fixes, and residual-risk report.

- [ ] **Step 1: Run focused regression tests**

Run: `.venv/bin/python -m pytest tests/web/test_pagination.py tests/web/test_task_store.py tests/web/test_web_app.py tests/web/test_workflow.py tests/web/test_celery_worker.py -q`

Expected: all selected tests pass.

- [ ] **Step 2: Run complete suite and static checks**

Run: `.venv/bin/python -m pytest -q`

Run: `.venv/bin/python -m compileall -q paperpilot tests scripts/benchmark_web_task_store.py`

Run: `git diff --check`

Expected: full suite passes with only previously known warnings; static commands exit 0.

- [ ] **Step 3: Run final full benchmark**

Run: `.venv/bin/python scripts/benchmark_web_task_store.py --tasks 2000 --writes 200 --workers 8 --page-size 50`

Record page bytes, index names, concurrent errors, write P50/P95/max, and pragma values. Compare response bytes to the 410,890-byte baseline without claiming cross-machine latency guarantees.

- [ ] **Step 4: Start isolated local API server**

Run on an unused port:

```bash
PAPERPILOT_TASK_DB_PATH=/tmp/paperpilot-pagination-smoke.sqlite3 PAPERPILOT_TASK_EXECUTOR=thread .venv/bin/uvicorn paperpilot.web.app:app --host 127.0.0.1 --port 8010
```

If 8010 is occupied, use the next free port. Keep the process alive only through browser verification, then stop it.

- [ ] **Step 5: Verify browser behavior and network request count**

Using the in-app browser:

1. register a temporary user;
2. create a simulated task;
3. confirm task, event, and artifact rendering;
4. inspect requests during execution: steady-state polling sends one `/updates` request per interval and does not refetch task list, `/events`, or `/artifacts` each second;
5. confirm polling stops at completion and refreshes the task page once.

- [ ] **Step 6: Request independent code review**

Ask the reviewer to inspect cursor context validation, keyset correctness, explicit read transaction, SQLite pragmas, index use, response-model consistency, user isolation, frontend race handling, accidental unbounded paths, and missing tests. Address valid findings with RED/GREEN tests and rerun Steps 1-3.

- [ ] **Step 7: Perform final scope and risk audit**

Run: `git status --short --branch`

Run: `git diff --stat`

Run: `rg -n "list_tasks\(|list_events\(|list_artifacts\(" paperpilot tests`

Expected: no removed unbounded store calls and no unrelated reverts. Final report must state:

- SQLite remains single writer;
- polling remains one-second HTTP polling, not push;
- keyset traversal is not a cross-request snapshot under concurrent inserts/status changes;
- SQLite/Redis transactional outbox gap remains;
- real Redis/Docker smoke remains separate and may be unavailable locally.
