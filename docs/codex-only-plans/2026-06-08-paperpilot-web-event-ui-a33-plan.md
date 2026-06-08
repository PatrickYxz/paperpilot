# PaperPilot Web Event UI A3.3 Plan

Date: 2026-06-08

## Background

A3.2 maps real PaperPilot `on_event` callbacks into `task_events`. The backend
now stores events such as:

- workflow events: `queue`, `real_start`, `real_complete`
- agent events: `agent_turn`
- tool events: `tool_call`, `tool_result`
- context events: `context_preflight`, `auto_compact`
- failure events: `guardrail`, `failure`

The current UI only renders each event as:

```text
type / stage message timestamp
```

That is technically correct, but it is hard to scan once real PaperPilot emits
tool calls and tool results.

## Goal

Improve the task detail event display so real runs are easier to inspect:

- classify each event as workflow, agent, tool, context, failure, or other;
- show a compact badge and stage label;
- show the message as the main line;
- show a small collapsible payload preview when payload exists;
- keep long payload previews bounded by backend truncation and frontend layout.

## Non-Goals

- Do not change the API.
- Do not change the database.
- Do not change `WorkflowRunner`.
- Do not trigger real PaperPilot execution.
- Do not add a frontend framework.

## Frontend Design

Event row shape:

```text
[TOOL] tool_call
Calling tool: mcp__colbert__search
timestamp
details toggle
payload preview
```

Classification rules:

```text
queue/start/prepare/deep_read_placeholder/complete/real_start/real_complete -> workflow
agent_turn -> agent
tool_call/tool_result -> tool
context_preflight/auto_compact -> context
failure/guardrail -> failure
otherwise -> other
```

Payload rendering:

- If payload is empty, do not show details.
- If payload exists, render a native `<details>` element.
- Render JSON with `JSON.stringify(payload, null, 2)`.
- Use `white-space: pre-wrap` and `overflow-wrap: anywhere`.

## Files To Modify

- `paperpilot/web/static/app.js`
- `paperpilot/web/static/styles.css`
- `tests/web/test_web_app.py`

## Tests

No JavaScript unit-test harness exists yet. For A3.3 keep tests lightweight:

- assert the index page serves the new event CSS class names or markup anchors;
- keep existing API tests unchanged;
- rely on HTTP/manual smoke for runtime UI behavior.

If the UI becomes more complex later, add a real browser/UI test layer.

## Verification

Focused:

```powershell
.venv\Scripts\python.exe -m pytest tests\web -q
```

Regression:

```powershell
.venv\Scripts\python.exe -m pytest tests\web tests\test_agent_loop.py tests\test_conversation_session.py tests\test_session_store.py tests\test_document_store.py tests\test_bulk_input.py tests\test_message_codec.py tests\test_context_manager.py tests\builtin_tools\test_ask_user.py tests\test_main_integration.py -q
```

Broader excluding known proxy issue:

```powershell
.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/mcp_servers/test_ss_client.py
```

## Acceptance Criteria

1. Event rows show category, stage, message, and time.
2. Tool events are visually distinguishable from workflow events.
3. Payload is available but collapsed by default.
4. Simulated mode still renders events correctly.
5. No backend API/schema change is introduced.

