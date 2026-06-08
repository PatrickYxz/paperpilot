# PaperPilot Web Real Runner Event Mapping A3.2 Plan

Date: 2026-06-08

## Background

A3.1 added an explicit real PaperPilot execution path:

```text
POST /api/tasks execution_mode=real
-> WorkflowRunner.run_real(task_id)
-> paperpilot.conversation.run(query)
-> final assistant text
-> task_artifacts kind=result
```

However, A3.1 only records coarse Web workflow events:

```text
queued / queue
started / real_start
completed / real_complete
failed / failure
```

The Web page still cannot show what the real PaperPilot agent did internally.
A3.2 should map PaperPilot `on_event` callbacks into `task_events` so a real run
becomes inspectable from the Web workbench.

## Goal

Show a useful, bounded trace of real PaperPilot execution in the Web task detail
view without changing the core agent loop or making real execution the default.

## Non-Goals

- Do not add new tables in A3.2.
- Do not store full raw traces or full tool results.
- Do not add candidate-paper confirmation yet.
- Do not implement retry/resume yet.
- Do not change MCP tool schemas.
- Do not make `execution_mode=real` the default.

## Existing Event Sources

The current PaperPilot loop and conversation stack already emit:

```text
turn
tool_arg_repair
tool_call
tool_result
guardrail_stop
context_preflight
auto_compact
context_overflow
bulk_input_saved
ask_user_prompt
ask_user_answer
```

Relevant code paths:

- `paperpilot/conversation.py`
  - `run(query, *, max_iter=8, on_event=None)`
  - `ConversationSession(..., on_event=...)`
- `paperpilot/core/loop.py`
  - `agent_loop(..., on_event=...)`
- `paperpilot/web/workflow.py`
  - `WorkflowRunner.run_real(task_id)`

## Proposed Design

Add a small mapper in the Web layer:

```text
paperpilot/web/event_mapper.py
```

Responsibilities:

1. Convert PaperPilot event kind + payload into Web event fields.
2. Truncate large text values before storage.
3. Preserve useful structured payload.
4. Avoid leaking full raw tool outputs into the UI.

The mapper should return:

```python
{
    "type": "...",
    "stage": "...",
    "message": "...",
    "payload": {...},
}
```

Then `WorkflowRunner.run_real()` can pass:

```python
on_event=lambda kind, payload: self._record_agent_event(task_id, kind, payload)
```

to the real runner.

## Event Mapping Rules

### `tool_call`

Input example:

```python
{"name": "mcp__arxiv__search_papers", "arguments": {"query": "..."}}
```

Web event:

```text
type: progress
stage: tool_call
message: Calling tool: mcp__arxiv__search_papers
payload:
  source_kind: tool_call
  tool_name: mcp__arxiv__search_papers
  arguments_preview: ...
```

### `tool_result`

Input example:

```python
{"name": "mcp__colbert__search", "content": "...long text..."}
```

Web event:

```text
type: progress
stage: tool_result
message: Tool result: mcp__colbert__search
payload:
  source_kind: tool_result
  tool_name: mcp__colbert__search
  content_preview: truncated text
  truncated: true | false
```

### `turn`

Input example:

```python
{"iteration": 2, "tool_calls": ["mcp__colbert__search"], "text": null}
```

Web event:

```text
type: progress
stage: agent_turn
message: Agent turn 2
payload:
  source_kind: turn
  iteration: 2
  tool_calls: [...]
  has_text: false
```

### `context_preflight`

Web event:

```text
type: progress
stage: context_preflight
message: Context preflight: hard
payload:
  estimated_tokens: ...
  window_tokens: ...
  stage: hard
```

### `auto_compact`

Web event:

```text
type: progress
stage: auto_compact
message: Context compacted.
payload:
  source_kind: auto_compact
  result_preview: ...
```

### `guardrail_stop` / `context_overflow`

Web event:

```text
type: failed
stage: guardrail
message: Guardrail stopped: ...
payload:
  source_kind: guardrail_stop
```

### Unknown event kinds

Unknown kinds should still be stored in bounded form:

```text
type: progress
stage: agent_event
message: Agent event: <kind>
payload:
  source_kind: <kind>
  payload_preview: ...
```

## Truncation Rules

Suggested constants:

```python
MAX_EVENT_TEXT_CHARS = 800
MAX_ARGUMENT_PREVIEW_CHARS = 600
MAX_PAYLOAD_PREVIEW_CHARS = 1000
```

Rationale:

- The UI should stay readable.
- SQLite should not become a full trace dump.
- Full evidence/report content should later go into dedicated artifacts or
  evidence tables, not event logs.

## Changes By File

New:

- `paperpilot/web/event_mapper.py`
- `tests/web/test_event_mapper.py`

Modify:

- `paperpilot/web/workflow.py`
  - inject `on_event` into the real runner call
  - record mapped events via `TaskStore.add_event`
- `tests/web/test_workflow.py`
  - verify fake real runner can emit events
  - verify mapped tool events are persisted
- `tests/web/test_web_app.py`
  - keep real-mode fake runner tests passing

No change expected:

- `paperpilot/core/loop.py`
- `paperpilot/conversation.py`
- MCP server modules
- SQLite schema

## Test Strategy

Focused mapper tests:

- `tool_call` maps to `progress/tool_call`.
- `tool_result` truncates long content.
- `turn` records iteration and tool calls.
- unknown events are bounded.

Workflow tests:

- Inject a fake real runner that accepts `on_event` and emits:
  - `tool_call`
  - `tool_result`
  - `turn`
- Assert `task_events` contains:
  - `real_start`
  - mapped agent events
  - `real_complete`
- Assert result artifact is still written.

Compatibility tests:

- Existing fake real runner that only accepts `query` should still work.
- This preserves the A3.1 runner injection interface.

## Real Runner Callable Compatibility

A3.1 currently uses:

```python
Callable[[str], list[dict]]
```

A3.2 should support both:

```python
real_runner(query)
real_runner(query, on_event=callback)
```

Implementation can use a small adapter:

```python
try:
    messages = self.real_runner(task.question, on_event=event_callback)
except TypeError:
    messages = self.real_runner(task.question)
```

However, avoid swallowing unrelated `TypeError` from inside the runner if
possible. A better implementation is to inspect the callable signature.

## Verification Commands

Focused:

```powershell
.venv\Scripts\python.exe -m pytest tests\web -q
```

Regression:

```powershell
.venv\Scripts\python.exe -m pytest tests\web tests\test_agent_loop.py tests\test_conversation_session.py tests\test_session_store.py tests\test_document_store.py tests\test_bulk_input.py tests\test_message_codec.py tests\test_context_manager.py tests\builtin_tools\test_ask_user.py tests\test_main_integration.py -q
```

Broader, excluding known proxy issue:

```powershell
.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/mcp_servers/test_ss_client.py
```

## Acceptance Criteria

A3.2 is done when:

1. Real-mode fake runner events are visible in `task_events`.
2. Long tool results are truncated before storage.
3. Result artifacts still work.
4. Existing simulated mode remains unchanged.
5. Existing A3.1 fake real runner tests remain valid.
6. No core agent-loop or MCP schema changes are required.

## Risks

1. Tool results can be large. Truncation must happen before writing to SQLite.
2. Some event payloads may contain non-JSON-safe objects. Mapper should coerce
   them into safe previews.
3. Real execution remains environment-sensitive. Tests should use fake runners.
4. If event volume becomes high, future A3.x may need pagination or event limits.

