# Answer Synthesis Guard Repair V1 Implementation Plan

Date: 2026-06-14

Codex only.

## Background And Goal

The approved design is `docs/superpowers/specs/2026-06-14-answer-synthesis-guard-repair-v1-design.md`.

This implementation adds a conservative quality check and one-shot repair loop to the QASPER `deep-read-paper` evaluation path. It should improve final-answer cleanliness and evidence discipline without changing retrieval, semantic judging, or ordinary interactive PaperPilot behavior.

## Constraints

- Scope is limited to `paperpilot/eval/baselines.py::run_paperpilot`.
- Ordinary chat, compare-papers, subagent, web dashboard, ColBERT, MCP tools, and semantic judge are unchanged.
- Repair can call the LLM at most once.
- Repair cannot call tools or introduce new facts.
- Existing result consumers must still read `predicted`.
- Preserve `predicted_raw` so raw and repaired answers can be compared.

## Steps

1. Add `paperpilot/eval/answer_quality.py`.
   - Define issue codes and severity.
   - Parse `Short answer:`, `Evidence:`, and optional `Notes:`.
   - Flag internal process markers, missing sections, overlong short answers, unsupported numeric claims, list-dumping risk, and thin evidence.
   - Expose `evaluate_answer_quality()`, `should_repair_answer()`, and `build_repair_prompt()`.

2. Add tests for answer quality.
   - Severe issues: internal trace marker, missing sections, long short answer, unsupported number.
   - Risk-only issues: list dumping and thin evidence.
   - Repair trigger should only react to severe issues.

3. Update `paperpilot/eval/baselines.py`.
   - After raw PaperPilot final answer extraction, evaluate quality.
   - If severe issues exist, call the LLM once with a narrow repair prompt.
   - Evaluate the repaired answer.
   - Return `predicted_raw`, `predicted`, `answer_quality`, `answer_repaired`, `repair_answer_quality`, and `repair_error`.

4. Add focused tests for `run_paperpilot()`.
   - Mock the agent run so no tools or network are used.
   - Mock repair client so no real LLM call is used.
   - Verify repair happens for severe issues.
   - Verify repair does not happen for risk-only issues.

5. Tighten `paperpilot/skills/deep-read-paper.md`.
   - Remove or supersede final-answer instructions that force visible `Answer span candidates`.
   - Require direct `Short answer`, `Evidence`, optional `Notes`.
   - Add scope, checklist, numeric/entity, and no-process-trace constraints.

6. Update prompt integration tests.
   - Assert the deep-read skill contains new constraints.
   - Avoid brittle mojibake-specific assertions where possible.

## Verification

- Run targeted tests:
  - `python -m pytest tests/eval/test_answer_quality.py -q`
  - `python -m pytest tests/eval/test_baselines.py -q`
  - `python -m pytest tests/test_main_integration.py::test_deep_read_skill_mentions_targeted_search_and_short_answer -q`
- Run broader relevant tests:
  - `python -m pytest tests/eval tests/test_main_integration.py -q`

## Risks

- Repair can rewrite a wrong answer into a cleaner wrong answer. This is why raw and repaired answers are both preserved.
- Numeric support checks are simple string checks, not semantic verification.
- Some valid list answers may be flagged as risk-only list dumping; V1 records this but does not trigger repair.
