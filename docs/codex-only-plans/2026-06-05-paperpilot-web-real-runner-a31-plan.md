# PaperPilot Web Real Runner A3.1 Plan

## Background And Goal

A3.0 introduced a `WorkflowRunner` boundary and persisted task artifacts. A3.1
adds the first real PaperPilot execution path behind an explicit `real` mode.
The default task mode remains `simulated` so users can test the Web workbench
without accidentally starting LLM/MCP work.

## Scope

In scope:

- Add `execution_mode` to `POST /api/tasks`.
- Supported values:
  - `simulated` default
  - `real`
- Keep existing API clients working by defaulting missing `execution_mode` to
  `simulated`.
- Add `WorkflowRunner.run_real(task_id)`.
- `run_real` calls the existing PaperPilot one-shot runner for the task question.
- Persist the final assistant text as a `result` artifact.
- Record real-runner events and failure events.
- Add UI control to choose Simulated or Real PaperPilot.
- Add tests with an injected fake real runner function. Do not call a real LLM
  during tests.

Out of scope:

- Candidate-paper confirmation.
- Multi-stage real workflow.
- Evidence/report-specific tables.
- Real browser-driven testing of model output.
- New MCP server behavior.

## API Shape

Request:

```json
{
  "question": "...",
  "depth": "standard",
  "execution_mode": "real"
}
```

Response remains a task object.

Events for `real` mode:

- `queued / queue`
- `started / real_start`
- `completed / real_complete`
- or `failed / failure`

Artifact for `real` mode:

```json
{
  "kind": "result",
  "title": "PaperPilot result",
  "content": "<final assistant text>",
  "payload": {
    "execution_mode": "real"
  }
}
```

## Runner Boundary

`WorkflowRunner` should accept an injectable `real_runner` callable:

```python
Callable[[str], list[dict]]
```

Default real runner:

```python
paperpilot.conversation.run(query)
```

Tests use a fake callable to avoid API calls and MCP startup.

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

- Restart the server.
- Create a simulated task to verify default behavior still works.
- Do not run a real task unless model/API keys are configured and the user
  explicitly wants to spend the call.

## Risks

- Real mode may take much longer than simulated mode.
- Real mode depends on `.env`, model keys, and MCP server startup.
- The first A3.1 real result is one-shot final answer capture, not a full
  structured research workflow with candidate confirmation.
