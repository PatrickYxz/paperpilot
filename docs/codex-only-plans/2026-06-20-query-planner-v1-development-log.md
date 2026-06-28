# 2026-06-20 Query Planner v1 Development Log

Codex only. This log summarizes the work completed on June 20, 2026 for the
PaperPilot retrieval/query-planner direction.

## Background

The starting point was a QASPER recall problem: PaperPilot often had too many
retrieval misses or weakly targeted chunks. We discussed that Query Planner
should not just be prompt text, but a staged retrieval control layer:

1. Analyze the user question and produce a normalized plan.
2. Force execution of the planned retrieval queries.
3. Merge/deduplicate evidence while preserving source and query provenance.
4. Leave reranking as a later stage after planner behavior is observable.

The implementation scope chosen for today was Query Planner v1 without adding a
reranker yet.

## Completed Today

### Design and Research

- Added the formal Query Planner v1 design document.
- Saved GitHub/project research notes about similar retrieval planning systems.
- Wrote the implementation plan in `docs/codex-only-plans/`.
- Decided that the first product integration should be a new ColBERT MCP tool
  named `planned_retrieval`, plus an update to `deep-read-paper`.

Key files:

- `docs/superpowers/specs/2026-06-20-query-planner-v1-design.md`
- `docs/query_planner_github_research_20260620.md`
- `docs/codex-only-plans/2026-06-20-query-planner-v1-implementation-plan.md`

### Query Plan Schema and Validation

- Added structured query-plan dataclasses:
  - `QueryPlan`
  - `EvidenceRequirement`
  - `PlannedQuery`
  - `QueryConstraints`
  - `ExpansionHints`
- Added a minimal fallback plan for invalid or unavailable LLM planner output.
- Added defensive JSON parsing and validation for `query_plan_v1`.
- Ensured the literal user question is preserved as a fallback query.

Key files:

- `paperpilot/retrieval/query_plan.py`
- `paperpilot/retrieval/query_plan_validator.py`
- `tests/retrieval/test_query_plan_validator.py`

### Evidence Pool

- Added evidence pooling for planned retrieval results.
- Implemented exact dedupe:
  - Prefer `paper_id::chunk_id` when `chunk_id` exists.
  - Otherwise use normalized chunk text hash.
- Preserved provenance through `matched_queries`.
- Added summary selection:
  - First attempts to cover required evidence requirements.
  - Records missing requirements instead of blocking the answer.
  - Uses simple token-overlap diversity only for summary selection, while still
    preserving all deduped evidence in the pool.

Key files:

- `paperpilot/retrieval/evidence_pool.py`
- `tests/retrieval/test_evidence_pool.py`

### LLM Query Planner

- Added an LLM-backed planner prompt for `query_plan_v1`.
- Added `plan_with_llm(...)` with fallback behavior:
  - Valid JSON response becomes a validated `QueryPlan`.
  - Invalid JSON or planner call failure falls back to the literal query plan.
- Kept this isolated from ColBERT execution so the planner can be tested
  without starting MCP.

Key files:

- `paperpilot/retrieval/llm_query_planner.py`
- `tests/retrieval/test_llm_query_planner.py`

### Planned Retrieval Execution

- Added an execution layer that takes a `QueryPlan` and a paper-scoped search
  function.
- Executes all planned queries by priority.
- Converts raw search results into `RawSearchHit`.
- Records per-query errors and continues with remaining queries.
- Builds an `EvidencePool` and a human-readable summary.

Key files:

- `paperpilot/retrieval/planned_retrieval.py`
- `tests/retrieval/test_planned_retrieval.py`

### ColBERT MCP Integration

- Updated ColBERT search results to include stable `chunk_id`.
- Added `mcp__colbert__planned_retrieval`.
- The new tool:
  - Builds a query plan.
  - Executes multiple planned queries.
  - Returns `summary_text`, `evidence_pool`, `query_errors`, and
    `query_plan_meta`.
- Added a non-network server unit test and a slow MCP smoke test for the new
  tool.

Key files:

- `paperpilot/mcp_servers/colbert/index_manager.py`
- `paperpilot/mcp_servers/colbert/server.py`
- `tests/mcp_servers/test_colbert_server_planned_retrieval.py`
- `tests/mcp_servers/test_colbert_via_client.py`

### Deep Read Skill Integration

- Updated `deep-read-paper` so the default retrieval path is now:

  `download_paper -> build_index -> planned_retrieval -> optional follow-up search -> answer`

- Kept `mcp__colbert__search` as a targeted follow-up tool only when planned
  retrieval evidence is missing or insufficient.
- Preserved the final-answer contract:
  - `Short answer:`
  - `Evidence:`
  - optional `Notes:`

Key files:

- `paperpilot/skills/deep-read-paper.md`
- `tests/test_deep_read_skill.py`
- `tests/test_main_integration.py`

### Eval Metadata

- Updated eval baseline result shaping so traces can report planned retrieval
  metadata:
  - `planned_retrieval_used`
  - `planned_retrieval_stats`
  - `planned_retrieval_missing_requirements`
  - `planned_retrieval_query_plan_meta`
  - `planned_retrieval_query_errors`
- Kept existing answer repair and evidence selector behavior intact.

Key files:

- `paperpilot/eval/baselines.py`
- `tests/eval/test_baselines.py`

### ColBERT Server Startup Fix

- Investigated why the new slow MCP smoke initially failed.
- Root cause:
  - The ColBERT model cache exists locally.
  - In normal online mode, HuggingFace still performs a HEAD request for
    optional files such as `adapter_config.json`.
  - Current network/DNS access to HuggingFace fails, causing the server process
    to exit before MCP initialization.
- Verified that `HF_HUB_OFFLINE=1` fixes startup by forcing local cache usage.
- Added that env var to the real ColBERT MCP manifest and to test manifests.
- Added a manifest regression test.

Key files:

- `paperpilot/mcp_servers.json`
- `tests/mcp_servers/test_mcp_manifest.py`
- `tests/mcp_servers/test_colbert_via_client.py`

## Commits Created Today

- `b128d0c` Docs: add query planner v1 design
- `20632f6` Add query plan schema
- `9a1ba81` Add query plan validation
- `7e695e4` Add evidence pool selection
- `ad00621` Add LLM query planner
- `6a2413b` Add planned retrieval execution
- `70be9e7` Add planned retrieval MCP tool
- `b9f9b15` Update deep read skill for planned retrieval
- `c4599b3` Record planned retrieval eval metadata
- `b0a2822` Add planned retrieval MCP smoke test
- `1fdf697` Run ColBERT MCP with HuggingFace offline cache

Current branch:

- `codex/qasper-eval-upgrade`

## Verification Run Today

Focused Query Planner / ColBERT / skill suite:

```bash
.venv/bin/python -m pytest \
  tests/mcp_servers/test_mcp_manifest.py \
  tests/mcp_servers/test_colbert_server_planned_retrieval.py \
  tests/mcp_servers/test_index_manager.py \
  tests/retrieval/test_query_plan_validator.py \
  tests/retrieval/test_evidence_pool.py \
  tests/retrieval/test_llm_query_planner.py \
  tests/retrieval/test_planned_retrieval.py \
  tests/test_deep_read_skill.py \
  tests/test_main_integration.py \
  -q
```

Result:

```text
33 passed, 1 deselected
```

Planned retrieval MCP slow smoke:

```bash
.venv/bin/python -m pytest -m slow \
  tests/mcp_servers/test_colbert_via_client.py::test_planned_retrieval_tool_runs \
  -q
```

Result:

```text
1 passed
```

Earlier broader focused suite also passed:

```text
57 passed, 1 deselected
```

## Current Uncommitted Workspace State

These files were still dirty/untracked at the end of the session and were not
included in the final startup-fix commit:

- `scripts/day16_run_eval.py`
- `scripts/day24_rerun_paperpilot_cases.py`
- `scripts/day24_retrieval_recall_diagnosis.py`
- `docs/codex-only-plans/2026-06-20-query-planner-phase2-eval-integration-plan.md`
- `docs/codex-only-plans/2026-06-20-query-planner-v1-implementation-plan.md`
- `docs/query_planner_github_research_20260620.md`
- `docs/retrieval_recall_diagnosis_20260620_query_plan_v1_13cases.md`

Some of these are useful project notes/plans; the scripts appear to be from
earlier Phase 2/eval work and should be reviewed before staging.

## Suggested Next Session

1. Review and decide whether to commit the remaining docs/log artifacts.
2. Run real QASPER eval cases with the new `planned_retrieval` path.
3. Compare against previous Phase 2 prompt-guidance baseline:
   - retrieval count
   - deduped evidence count
   - missing requirements
   - answer quality
   - evidence selector rewrites
4. Inspect failure cases where:
   - `planned_retrieval_missing_requirements` is non-empty;
   - the evidence pool has relevant chunks but final answer still misses;
   - fallback planner is used too often due to LLM planner errors.
5. Only after observing those traces, revisit:
   - stronger dedupe,
   - requirement-aware follow-up search,
   - reranker design,
   - possible small-model reranker fine-tuning.

## Important Notes

- `HF_HUB_OFFLINE=1` assumes `lightonai/colbertv2.0` is already cached locally.
  On a fresh machine, cache/model warm-up must happen once before offline mode
  can work.
- `plan_with_llm` falls back safely if `DEEPSEEK_API_KEY` is missing or planner
  output is invalid.
- Evidence pool currently dedupes exact chunks and only uses lightweight overlap
  diversity for summary selection. It does not yet perform semantic reranking.
