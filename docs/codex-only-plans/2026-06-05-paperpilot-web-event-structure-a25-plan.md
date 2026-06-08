# PaperPilot Web Event Structure A2.5 Plan

## Background And Goal

A2 added a simulated workflow and task event history. Before connecting real
deep-read, event records need enough structure to represent workflow stages and
small machine-readable payloads. A2.5 extends the event log without adding a
separate workflow-stage table yet.

## Scope

In scope:

- Add nullable `stage` and `payload_json` columns to `task_events`.
- Make schema creation compatible with both new and existing SQLite databases.
- Let `TaskEvent` expose `stage` and decoded `payload`.
- Record stage/payload in the simulated workflow.
- Return the new fields from `GET /api/tasks/{task_id}/events`.
- Show stage labels in the Web UI.
- Add tests for migration, payload round-trip, and API output.

Out of scope:

- Real deep-read execution.
- Candidate-paper confirmation.
- Report or evidence persistence.
- Separate workflow stage table.
- Retry semantics.

## Data Model Change

Existing table:

```sql
task_events(id, task_id, type, message, created_at)
```

A2.5 target:

```sql
task_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  type TEXT NOT NULL,
  stage TEXT,
  message TEXT NOT NULL,
  payload_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(task_id) REFERENCES research_tasks(id)
)
```

Migration approach:

- `CREATE TABLE IF NOT EXISTS` includes the new columns for fresh databases.
- Existing databases are inspected with `PRAGMA table_info(task_events)`.
- Missing columns are added with:
  - `ALTER TABLE task_events ADD COLUMN stage TEXT`
  - `ALTER TABLE task_events ADD COLUMN payload_json TEXT`

## Event Shape

API response:

```json
{
  "id": 1,
  "task_id": "task_...",
  "type": "progress",
  "stage": "prepare",
  "message": "Preparing research workflow state.",
  "payload": {
    "simulated": true
  },
  "created_at": "..."
}
```

The old fields remain, so existing UI/API consumers continue to work.

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
- Create a new task.
- Confirm event rows show stage names and still complete normally.

## Risks

- Existing A1/A2 smoke-test events will have `stage = null` and
  `payload_json = null`; the UI must handle that cleanly.
- Payloads should stay small. Large traces and evidence should be modeled
  separately in A3, not stored directly in `task_events`.
