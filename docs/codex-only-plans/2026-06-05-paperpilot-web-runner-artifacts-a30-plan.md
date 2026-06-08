# PaperPilot Web Runner And Artifacts A3.0 Plan

## Background And Goal

A1 added persistent tasks, A2 added simulated workflow events, and A2.5 added
structured event fields. A3.0 prepares the boundary needed before connecting
real PaperPilot deep-read: workflow execution should live outside the FastAPI
route file, and task outputs should be persisted as artifacts instead of being
stored only in events.

A3.0 still does not call real LLM, MCP, ColBERT, or deep-read code.

## Scope

In scope:

- Add a `task_artifacts` SQLite table.
- Add `TaskArtifact` storage methods.
- Add `GET /api/tasks/{task_id}/artifacts`.
- Move the simulated workflow into `paperpilot/web/workflow.py`.
- Have the simulated workflow write one result artifact.
- Update the UI to display task artifacts.
- Add focused tests for artifact storage, API output, and runner behavior.

Out of scope:

- Real PaperPilot execution.
- MCP lifecycle management from the Web app.
- Candidate-paper confirmation.
- Evidence span persistence.
- Report generation from actual papers.
- Retry/resume behavior.

## Data Model

```sql
CREATE TABLE task_artifacts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  payload_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES research_tasks(id)
);
```

Artifact kinds for A3.0:

- `result`

Future likely kinds:

- `report`
- `evidence`
- `trace_summary`
- `candidate_papers`

## API Addition

`GET /api/tasks/{task_id}/artifacts`

Response:

```json
[
  {
    "id": 1,
    "task_id": "task_...",
    "kind": "result",
    "title": "Simulated research result",
    "content": "...",
    "payload": {"simulated": true},
    "created_at": "..."
  }
]
```

## Workflow Boundary

New file:

- `paperpilot/web/workflow.py`

The FastAPI app should schedule:

```python
runner.run_simulated(task.id)
```

The runner owns:

- status updates,
- event recording,
- artifact recording.

This keeps `app.py` focused on HTTP routing and dependency wiring.

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

- Restart the local server.
- Create a task.
- Confirm it completes.
- Confirm artifacts appear in the detail view.

## Risks

- Existing tasks will have no artifacts; UI must handle empty artifact lists.
- Artifacts should store compact results only. Large traces and evidence should
  be modeled in separate A3.x tables before real deep-read stores them.
