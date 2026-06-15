# Eval Trace Full Tool Results Plan

Date: 2026-06-15

Codex only.

## Background And Goal

The evidence-selection case study found that JSONL traces only contain truncated `tool_result` content. The truncation happens in `paperpilot/core/loop.py` before the eval JSONL tracer receives the event.

This blocks evidence-selection diagnosis because traces show search queries and snippet heads, but not full retrieved chunks or full top-k results.

Goal: preserve full `tool_result` content in agent events so eval/debug traces can inspect retrieved evidence.

## Scope

- Change `agent_loop` event emission for `tool_result`.
- Keep the existing Web event mapper truncation unchanged.
- Add tests to prove full tool results are emitted and persisted.
- Do not change retrieval behavior, answer synthesis, scoring, or repair logic.

## Implementation

1. Replace `emit("tool_result", {"name": tc.name, "content": content[:200]})` with full `content`.
2. Add an agent-loop test with a long tool result.
3. Add or adjust JSONL tracer coverage if needed.
4. Run targeted tests:
   - `tests/test_agent_loop.py`
   - `tests/eval/test_jsonl_tracer.py`
   - `tests/web/test_event_mapper.py`

## Risk

Full tool results can be large in event callbacks. The Web path already maps them into bounded previews before storing task events. Eval traces are intentionally debug artifacts under ignored `data/`.
