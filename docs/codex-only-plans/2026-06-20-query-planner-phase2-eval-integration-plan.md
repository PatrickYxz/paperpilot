# Query Planner Phase 2 Eval Integration Plan

Date: 2026-06-20

Codex only.

## Background And Goal

Phase 1 added a deterministic Query Planner in `paperpilot/eval/query_planner.py`.
It can classify QASPER-style questions, infer the expected answer shape, and
generate structured search guidance.

Phase 2 should make that planner affect eval behavior without changing normal
PaperPilot user workflows.

Goal: add a new eval-only baseline, `paperpilot_query_plan_v1`, that injects a
compact retrieval guidance block into the PaperPilot prompt before the agent
runs.

## Constraints

- Only the eval path should opt into the new prompt.
- Do not modify `paperpilot.main.run`, the agent loop, MCP servers, ColBERT, or
  the `deep-read-paper` skill.
- Do not use QASPER oracle spans, highlighted evidence, gold answers, or manual
  calibration labels in the planner.
- Preserve the existing `paperpilot` baseline as-is.
- Keep traces for the new baseline separate from existing `paperpilot` traces.

## Implementation Steps

1. Update `paperpilot/eval/baselines.py`.
   - Add `use_query_plan` and optional `trace_id` parameters to `run_paperpilot`.
   - When enabled, call `plan_queries(case.question, title=case.paper_title, abstract=case.abstract)`.
   - Format the plan as a concise retrieval guidance block inside the prompt.
   - Return `query_plan_used`, `query_plan_version`, and `query_plan`.

2. Update `scripts/day16_run_eval.py`.
   - Add a new baseline choice: `paperpilot_query_plan_v1`.
   - Route it to `run_paperpilot(..., use_query_plan=True, trace_id=f"{case.case_id}__query_plan_v1")`.
   - Persist query-plan fields into result JSONL records.

3. Update focused tests.
   - Verify the planner-guided baseline injects planned searches into the prompt.
   - Verify query-plan fields are returned.
   - Verify the original `paperpilot` path remains unchanged.

## Verification

- Run focused tests:
  - `.venv/bin/python -m pytest tests/eval/test_query_planner.py tests/eval/test_baselines.py -q`
- Optionally run the broader eval unit set after implementation:
  - `.venv/bin/python -m pytest tests/eval -q`

## Risks

- The agent may ignore or partially follow the retrieval guidance.
- More planned searches can increase runtime or retrieve more distractors.
- If recall does not improve, the bottleneck may be chunking, PDF text quality,
  table handling, or arXiv/QASPER source mismatch rather than query wording.
