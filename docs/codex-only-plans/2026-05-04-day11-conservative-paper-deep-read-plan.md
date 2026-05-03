# Day 11 Conservative paper_deep_read Plan

Codex only. Claude Code does not need to follow this file.

## Background And Goal

Claude Code generated a Day 11 plan for `paper_deep_read`, an L2 built-in tool
that spawns per-paper subagents and returns per-paper markdown summaries to the
main agent.

After reviewing the current implementation, the full concurrent version has a
known blocker: ColBERT currently uses a single global `paperpilot_current` index.
Parallel subagents that all call `build_index` can overwrite one another, so a
subagent may search the wrong paper even if its status is `ok`.

The conservative goal is to land the subagent architecture now while avoiding
that index race:

- Add `paper_deep_read` as a built-in tool alongside `load_skill` and
  `research_todo`.
- Keep each paper in an independent `agent_loop` context with its own
  `LLMClient` and `Guardrail`.
- Run subagents through `ThreadPoolExecutor`, but set the worker count to `1`
  for Day 11 so ColBERT build/search remains serial.
- Forward subagent events to the main tracer with `subagent_paper_id`.
- Add tests and smoke checks that expose search/paper mismatches.
- Use a higher subagent token budget than the main agent default because a
  downloaded paper's full text is inserted into that subagent's private context.

## Constraints

- Do not change `paperpilot/core/loop.py`, `adapter.py`, or `tools/mcp_client.py`.
- Avoid changing ColBERT MCP behavior unless smoke proves that serial
  `build_index` calls in one server session cannot work on Windows.
- Do not add dependencies.
- Do not integrate a new skill yet; Day 11 smoke directly prompts the model to
  call `paper_deep_read`.
- `paper_deep_read` tool results must be markdown strings, not structured lists.
- Avoid `TODO` / `FIXME` markers in touched source and smoke files.

## Implementation Steps

1. Add `paperpilot/builtin_tools/subagent.py`.
   - Define `paper_deep_read_tool(client_factory, mcp_tools, on_event)`.
   - Filter subagent tools to exactly:
     `mcp__arxiv__download_paper`, `mcp__colbert__build_index`,
     `mcp__colbert__search`.
   - Use `THREAD_POOL_SIZE = 1` for conservative Day 11 execution.
   - Preserve input order in rendered results.
   - Prefix forwarded events with `subagent_paper_id` under a lock.
   - Remove the invalid `paper_ids` argument from the ColBERT search guidance.

2. Add `tests/builtin_tools/test_subagent.py`.
   - Cover tool filtering, final text extraction, handler validation,
     `_run_one` success/error/max-iteration behavior, event forwarding,
     input-order rendering, metadata, and conservative worker count.

3. Integrate `paper_deep_read` in `paperpilot/main.py`.
   - Append `PAPER_DEEP_READ_NUDGE` to the system prompt.
   - Add `paper_deep_read_tool` between `research_todo` and MCP tools.
   - Pass the same `on_event` callback into both main loop and subagent tool.

4. Update `tests/test_main_integration.py`.
   - Add a fast system prompt assertion for `paper_deep_read`.
   - Extend the slow tool-list assertion to include `paper_deep_read`.

5. Add `scripts/day11_smoke.py`.
   - Prompt the main agent to call `paper_deep_read` for three known arXiv IDs.
   - Capture main and subagent tool events.
   - Assert `paper_deep_read` was called once with three IDs.
   - Assert the markdown result has three paper sections and at least two
     successful statuses.
   - Assert every captured subagent ColBERT search result refers only to the
     matching `subagent_paper_id`; this catches future accidental parallel
     index overwrite.

6. If the first smoke run shows Windows file-handle failures on consecutive
   ColBERT `build_index` calls, add a narrow `IndexManager` stabilization.
   - Release the previous PLAID index reference and chunk cache before
     overriding the fixed `paperpilot_current` path.
   - Keep the public MCP tool schema and returned data unchanged.

## Verification

- Run targeted unit tests for `tests/builtin_tools/test_subagent.py`.
- Run fast main integration tests.
- Run the slow tool-list integration test.
- Run the default suite excluding `test_graph_via_client.py`.
- Run `scripts/day11_smoke.py`.
- If practical, run Day 9 and Day 10 smokes for regression confidence.
- Scan touched Day 11 source/smoke files for `TODO|FIXME`.

## Risks And Follow-Ups

- This version validates subagent architecture, not true ColBERT parallel
  retrieval. The project should later add per-paper or per-index isolation in
  ColBERT before raising `THREAD_POOL_SIZE`.
- Consecutive `build_index` calls reuse one fixed index path. The conservative
  implementation may need explicit old-index release on Windows to avoid file
  handle errors.
- Smoke tests depend on external LLM and MCP behavior; one retry may be needed
  for transient arXiv or model-tool-call issues.
- Subagent runs can be token-heavy because Day 11 still passes full downloaded
  paper text through MCP tool results. Future PDF/context compaction work should
  reduce this cost.
- Day 12 should integrate `paper_deep_read` into a skill only after this
  conservative tool path is stable.
