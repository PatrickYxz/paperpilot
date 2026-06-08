# PaperPilot Web Workflow A2 Implementation Plan

## Background And Goal

A1 created a local Web task workbench with persistent research tasks. A2 adds a
simulated workflow layer so each task has visible progress and event history.
This still does not run real PaperPilot deep-read. It prepares the state and UI
shape needed for later real workflow execution.

## Scope

In scope:

- Add a SQLite `task_events` table.
- Add repository methods for status updates and event recording.
- Add `GET /api/tasks/{task_id}/events`.
- Start a small FastAPI background simulation after task creation.
- Update the UI to poll task detail and events for pending/running tasks.
- Add focused tests for event persistence, status changes, and API behavior.

Out of scope:

- Real MCP calls.
- Real deep-read, report generation, candidate-paper confirmation, or evidence
  persistence.
- Retry semantics beyond displaying events.
- Distributed task queues.

## Data Model Additions

```sql
CREATE TABLE task_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  type TEXT NOT NULL,
  message TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES research_tasks(id)
);
```

Event types for A2:

- `queued`
- `started`
- `progress`
- `completed`
- `failed`

## API Additions

`GET /api/tasks/{task_id}/events`

Response:

```json
[
  {
    "id": 1,
    "task_id": "task_...",
    "type": "queued",
    "message": "Task queued.",
    "created_at": "..."
  }
]
```

`POST /api/tasks` still returns a task object. The created task starts as
`pending`, records a `queued` event, and schedules a local background simulation.

## Simulated Workflow

For A2 the simulation should:

1. Record `started`, update task to `running`.
2. Record one or two `progress` events.
3. Update task to `completed`, record `completed`.

The simulation is intentionally short and deterministic enough for tests. It is
not a real research workflow.

## Verification

Focused tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\web -q
```

Broader regression:

```powershell
.venv\Scripts\python.exe -m pytest tests\web tests\test_agent_loop.py tests\test_conversation_session.py tests\test_session_store.py tests\test_document_store.py tests\test_bulk_input.py tests\test_message_codec.py tests\test_context_manager.py tests\builtin_tools\test_ask_user.py tests\test_main_integration.py -q
```

Manual check:

- Open `http://127.0.0.1:8000/`.
- Create a task.
- Confirm status changes and event history appears without refreshing.

## Risks

- SQLite writes from background tasks must use short-lived connections.
- Existing task rows from A1 have no events; the UI must handle empty history.
- The previously running server process must be restarted to load code changes.
