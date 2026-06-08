# PaperPilot Web Task MVP A1 Implementation Plan

## Background And Goal

This plan implements the first slice of option A: a local Web research task
workbench skeleton. A1 does not run the real deep-read workflow. It only creates
and persists research tasks, exposes a small HTTP API, and provides a minimal
HTML/JavaScript UI for creating and viewing tasks.

The purpose is to give PaperPilot a durable product shell before wiring in the
existing agent, MCP, ColBERT, and report-generation workflows.

## Scope

In scope:

- Add a small SQLite-backed task repository.
- Add a FastAPI application with a minimal task API.
- Add static HTML/CSS/JavaScript pages for:
  - creating a research task,
  - listing historical tasks,
  - viewing one task detail.
- Add focused tests for the repository and API.

Out of scope for A1:

- Real deep-read execution.
- MCP calls from the Web API.
- Background task execution.
- Workflow state machine beyond simple task statuses.
- Candidate-paper confirmation.
- Report generation and evidence persistence.
- Login, auth, cloud deployment, Redis/Celery, PostgreSQL, React/Vue.

## Proposed API

Base path: `/api/tasks`

### Create Task

`POST /api/tasks`

Request:

```json
{
  "question": "What are the main retrieval strategies for long-context RAG?",
  "depth": "standard"
}
```

Response: `201 Created`

```json
{
  "id": "task_...",
  "question": "...",
  "depth": "standard",
  "status": "pending",
  "created_at": "...",
  "updated_at": "..."
}
```

Validation:

- `question` is required and non-empty.
- `depth` defaults to `standard`.
- Accepted `depth` values: `quick`, `standard`, `deep`.

### List Tasks

`GET /api/tasks`

Optional query:

- `status=pending|running|completed|failed`

Response:

```json
[
  {
    "id": "task_...",
    "question": "...",
    "depth": "standard",
    "status": "pending",
    "created_at": "...",
    "updated_at": "..."
  }
]
```

### Get Task

`GET /api/tasks/{task_id}`

Response: one task object, or `404` if missing.

## Proposed Data Model

SQLite database path:

```text
data/web/tasks.sqlite3
```

Table:

```sql
CREATE TABLE research_tasks (
  id TEXT PRIMARY KEY,
  question TEXT NOT NULL,
  depth TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

Allowed statuses for A1:

- `pending`
- `running`
- `completed`
- `failed`

Only `pending` is created by the A1 UI. The extra statuses are reserved for A2
and A3 so the UI and API shape do not need to change immediately.

## Proposed Files

New files:

- `paperpilot/web/__init__.py`
- `paperpilot/web/app.py`
- `paperpilot/web/task_store.py`
- `paperpilot/web/static/index.html`
- `paperpilot/web/static/styles.css`
- `paperpilot/web/static/app.js`
- `tests/web/test_task_store.py`
- `tests/web/test_web_app.py`

Modified files:

- `requirements.txt`
  - add `fastapi`
  - add `uvicorn`

No existing agent-loop, MCP, retrieval, eval, or conversation code should be
modified in A1 unless tests reveal a direct import issue.

## Implementation Steps

1. Add dependency entries for FastAPI and Uvicorn.
2. Implement `TaskStore` using stdlib `sqlite3`.
3. Implement FastAPI routes for create/list/get task.
4. Serve the static UI from the same FastAPI app.
5. Add tests for:
   - task creation and persistence,
   - list ordering,
   - status filtering,
   - missing task 404,
   - invalid create payload.
6. Run the focused tests.
7. If the environment allows, start the dev server and verify the page manually.

## Verification Commands

Focused tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\web tests\test_main_integration.py -q
```

Optional server run:

```powershell
.venv\Scripts\uvicorn.exe paperpilot.web.app:app --reload --host 127.0.0.1 --port 8000
```

Manual browser check:

- Open `http://127.0.0.1:8000/`.
- Create a task.
- Refresh the page.
- Confirm the task still appears.
- Open the task detail view.

## Risks And Confirmation Points

- This plan adds new dependencies: `fastapi` and `uvicorn`.
- This plan adds a new SQLite data file location under `data/web/`.
- The A1 API does not run deep-read. It only prepares the persistent task
  boundary needed for later workflow execution.
- If the current virtual environment does not have the new dependencies, package
  installation will require explicit approval.
