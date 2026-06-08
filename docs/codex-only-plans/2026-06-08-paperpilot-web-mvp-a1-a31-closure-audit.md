# PaperPilot Web MVP A1-A3.1 Closure Audit

Date: 2026-06-08

## Goal

This document closes the current PaperPilot Web MVP slice before A3.2 work
starts. It records what belongs to the Web workbench track, what remains from
the broader dirty worktree, and which validation commands passed.

This is an audit and planning artifact only. No runtime behavior is changed by
this document.

## Current Web MVP State

The Web workbench has reached A3.1:

```text
A1   persistent task workbench
A2   simulated workflow events
A2.5 structured event stage/payload
A3.0 workflow runner boundary and task artifacts
A3.1 explicit real PaperPilot runner mode
```

The default execution path remains `simulated`. Real PaperPilot execution is
opt-in through `execution_mode = "real"`.

## Web MVP Files

Core Web package:

- `paperpilot/web/__init__.py`
- `paperpilot/web/app.py`
- `paperpilot/web/task_store.py`
- `paperpilot/web/workflow.py`
- `paperpilot/web/static/index.html`
- `paperpilot/web/static/styles.css`
- `paperpilot/web/static/app.js`

Web tests:

- `tests/web/test_task_store.py`
- `tests/web/test_web_app.py`
- `tests/web/test_workflow.py`

Dependency update:

- `requirements.txt`
  - `fastapi>=0.115.0`
  - `uvicorn>=0.30.0`

Web MVP plans:

- `docs/codex-only-plans/2026-06-05-paperpilot-audit-and-web-mvp-plan.md`
- `docs/codex-only-plans/2026-06-05-paperpilot-web-task-mvp-a1-plan.md`
- `docs/codex-only-plans/2026-06-05-paperpilot-web-workflow-a2-plan.md`
- `docs/codex-only-plans/2026-06-05-paperpilot-web-event-structure-a25-plan.md`
- `docs/codex-only-plans/2026-06-05-paperpilot-web-runner-artifacts-a30-plan.md`
- `docs/codex-only-plans/2026-06-05-paperpilot-web-real-runner-a31-plan.md`

Learning document:

- `docs/learning/2026-06-05-paperpilot-web-workbench-zero-basics.md`

## Data Model

Runtime database:

```text
data/web/tasks.sqlite3
```

This is ignored by the repository-level `data/` rule.

Tables introduced by the Web MVP:

```text
research_tasks
task_events
task_artifacts
```

Important design boundary:

- `research_tasks` is the durable task record.
- `task_events` is the progress and trace log.
- `task_artifacts` is the durable output/result store.

## API Surface

Current API endpoints:

```text
POST /api/tasks
GET  /api/tasks
GET  /api/tasks/{task_id}
GET  /api/tasks/{task_id}/events
GET  /api/tasks/{task_id}/artifacts
```

`POST /api/tasks` accepts:

```json
{
  "question": "...",
  "depth": "quick | standard | deep",
  "execution_mode": "simulated | real"
}
```

`execution_mode` defaults to `simulated`.

## Validation Results

Focused Web + agent/session regression:

```powershell
.venv\Scripts\python.exe -m pytest tests\web tests\test_agent_loop.py tests\test_conversation_session.py tests\test_session_store.py tests\test_document_store.py tests\test_bulk_input.py tests\test_message_codec.py tests\test_context_manager.py tests\builtin_tools\test_ask_user.py tests\test_main_integration.py -q
```

Result:

```text
71 passed, 1 deselected, 1 warning in 10.10s
```

Broader project regression excluding known proxy-environment issue:

```powershell
.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/mcp_servers/test_ss_client.py
```

Result:

```text
194 passed, 10 deselected, 1 warning in 76.79s
```

Warning:

```text
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
```

This warning does not currently block the Web MVP.

## Known Non-Web Dirty Worktree Items

The working tree still contains many files that predate or sit beside the Web
MVP track, including:

- CLI conversation/session enhancements:
  - `paperpilot/conversation.py`
  - `paperpilot/session_store.py`
  - `paperpilot/message_codec.py`
- User document and bulk-input support:
  - `paperpilot/document_store.py`
  - `paperpilot/bulk_input.py`
  - `paperpilot/builtin_tools/user_document.py`
- Ask-user and context management:
  - `paperpilot/builtin_tools/ask_user.py`
  - `paperpilot/core/context_manager.py`
- Agent-loop changes:
  - `paperpilot/core/loop.py`
  - `paperpilot/builtin_tools/compact.py`
  - `paperpilot/main.py`
- Personal/interview documents under `_personal/`.
- Older plan documents under `docs/codex-only-plans/`.

These should not be mixed blindly with the Web MVP changes if a future commit is
created.

## Known Risks

1. The repository still has a broad dirty worktree. A commit should be grouped
   carefully if the user asks for one.
2. The default full test suite can fail in `tests/mcp_servers/test_ss_client.py`
   when the environment contains a SOCKS proxy and the venv lacks `socksio`.
3. Real PaperPilot mode is opt-in but can still trigger model/MCP costs and
   environment dependencies when selected.
4. Existing smoke-test rows in `data/web/tasks.sqlite3` are local runtime data
   and should not be treated as source artifacts.

## Closure Recommendation

Before implementing A3.2, keep the following boundaries:

- Do not make real execution the default.
- Do not add evidence/report tables yet.
- Reuse `task_events.stage` and `task_events.payload_json` for trace mapping.
- Keep long tool outputs truncated before saving to the Web event log.
- Keep final output in `task_artifacts`.

