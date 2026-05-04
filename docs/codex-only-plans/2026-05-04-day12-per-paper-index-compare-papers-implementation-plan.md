# Day 12 Codex Implementation Plan: Per-Paper ColBERT Index + Compare Papers

## Background And Goal

Claude Code already generated the Day 12 plan/spec, and Codex tightened the design before implementation. This execution follows the approved revised direction:

- Change ColBERT from one global `paperpilot_current` index to per-paper persisted indexes.
- Require `paper_id` on ColBERT search so retrieval cannot accidentally hit another paper.
- Remove the Day 11 serial worker limit from `paper_deep_read` by using three worker threads.
- Add `compare-papers` so multi-paper comparison naturally routes through `paper_deep_read`.
- Add a Day 12 smoke that proves the compare-papers path and subagent lifecycle overlap.

## Constraints

- Keep the change scoped to Day 12 files from the approved spec.
- Do not touch `loop.py`, `adapter.py`, or `mcp_client.py`.
- Do not add dependencies.
- Preserve original `paper_id` values in chunk ids and returned search results.
- Use filesystem-safe per-paper directory keys for ids such as `cs/0501001`.
- Run Python/pytest/smoke with elevation when the sandbox blocks `.venv\Scripts\python.exe`.

## Execution Steps

1. Rewrite `paperpilot/mcp_servers/colbert/index_manager.py`.
   - Add `_IndexState`, `_states`, `_paper_key`, `_paper_root`, `_index_path`, and `_chunks_path`.
   - Implement build paths: memory-hit, disk-hit, and cold build.
   - Implement `search(query, paper_id, top_k)` with lazy load and `IndexNotFoundError`.
   - Add `tests/mcp_servers/test_index_manager.py` with PyLate mocks.

2. Update the ColBERT MCP protocol layer.
   - Add `paper_id` to `search` tool and `_search_impl`.
   - Update build/search docstrings and test coverage in `tests/mcp_servers/test_colbert_server.py`.

3. Update `paper_deep_read`.
   - Set `THREAD_POOL_SIZE = 3`.
   - Update the subagent prompt so search always passes `paper_id`.
   - Emit `subagent_start` and `subagent_done` lifecycle events.
   - Update tests for worker count, prompt text, and lifecycle events.

4. Add skill and prompt wiring.
   - Create `paperpilot/skills/compare-papers.md`.
   - Update `paperpilot/skills/deep-read-paper.md`.
   - Add the `mcp__colbert__search` `paper_id` rule to `paperpilot/main.py`.
   - Extend main integration tests.

5. Update slow integration and smoke scripts.
   - Update `tests/mcp_servers/test_colbert_via_client.py` for the required `paper_id` search schema and per-paper behavior.
   - Create `scripts/day12_smoke.py` using subagent lifecycle overlap as concurrency evidence.

## Verification

- Targeted fast tests:
  - `pytest tests/mcp_servers/test_index_manager.py -v`
  - `pytest tests/mcp_servers/test_colbert_server.py -v`
  - `pytest tests/builtin_tools/test_subagent.py -v`
  - `pytest tests/test_main_integration.py -v -m "not slow"`
- Default suite:
  - `pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py`
- Slow/real integration:
  - `pytest tests/mcp_servers/test_colbert_via_client.py -v -m slow`
  - `pytest tests/test_main_integration.py::test_build_tools_contains_load_skill_research_todo_and_mcp_tools -v -m slow`
- Smokes:
  - `scripts/day9_smoke.py`
  - `scripts/day10_smoke.py`
  - `scripts/day11_smoke.py`
  - `scripts/day12_smoke.py`
- Hygiene:
  - grep for old Day 11 serial wording and `THREAD_POOL_SIZE = 1`.
  - grep touched files for `TODO` / `FIXME`.

## Risks

- Real PyLate may behave differently from mocks for disk loading. The slow ColBERT client tests and Day12 smoke are required before treating this complete.
- Three concurrent subagents may be too heavy on this machine. If Day12 smoke fails from resource pressure, reduce only after reporting the tradeoff.
- Existing cached index directories may have old global layout. The new per-paper logic should ignore them unless a per-paper directory and `chunks.json` both exist.
