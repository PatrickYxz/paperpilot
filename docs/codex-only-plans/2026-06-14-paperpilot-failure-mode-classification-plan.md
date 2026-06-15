# PaperPilot Failure Mode Classification Plan

Date: 2026-06-14

Codex only. This plan records the next evaluation-system slice after the static eval dashboard.

## Background And Goal

The current QASPER evaluation stack now has three useful views:

- original strict keyword/span pass;
- LLM semantic audit labels;
- manually calibrated decisions over high-value disagreement cases.

The remaining gap is diagnosis. A score such as 66.0%, 71.3%, or 83.3% does not explain what to fix in PaperPilot. The next slice should classify reviewed non-perfect cases into concrete failure modes so that later accuracy work can target retrieval, answer synthesis, numeric handling, or scope control separately.

## Constraints

- Do not rerun the whole PaperPilot benchmark in this slice.
- Do not change the scoring formula or existing calibrated score.
- Treat `data/eval/semantic_calibration_candidates_20260614.jsonl` as the reviewed input source.
- Keep generated durable documentation under `docs/`.
- Add tests around classification rules so the report can be regenerated safely.
- Avoid broad frontend migration or React work in this slice.

## Failure Mode Taxonomy V1

Use a small multi-label taxonomy. A case can have more than one failure mode.

- `overbroad_scope`: answer includes extra methods, datasets, metrics, baselines, or unrelated context beyond the question.
- `missing_required_part`: answer covers some but not all required gold answer components.
- `wrong_numeric_or_fact`: answer contains a wrong number, score, label, dataset name, or factual claim.
- `strict_keyword_false_positive`: strict scorer passed because of keyword/span overlap, but semantic/manual review says the answer is materially wrong.
- `answer_format_noise`: answer is hard to judge because of verbose list-dumping, garbled text, or mixed reasoning/output formatting.
- `judge_or_gold_ambiguity`: remaining issue is mostly caused by narrow gold span, ambiguous QASPER wording, or judge over-penalization.
- `unknown`: fallback when notes and labels are insufficient.

## Execution Steps

1. Add a script, tentatively `scripts/day22_classify_failure_modes.py`.
   - Load reviewed calibration JSONL.
   - Select cases whose `review_decision` is `keep_partial` or `downgrade_to_incorrect`.
   - Optionally include `strict_pass + semantic_label in {incorrect, contradictory}` as strict false positives.
   - Assign multi-label failure modes using explicit rule helpers over `category`, `review_decision`, `semantic_label`, `strict_pass`, `review_notes`, `question`, `judge_reason`, and answer text.

2. Generate a Markdown report, tentatively `docs/semantic_failure_modes_20260614.md`.
   - Include counts by failure mode.
   - Include counts by review decision.
   - Include representative case IDs and notes for each mode.
   - Include concrete engineering implications for PaperPilot.

3. Add focused tests under `tests/eval/`.
   - Test that overbroad notes map to `overbroad_scope`.
   - Test that missing/missed/incomplete notes map to `missing_required_part`.
   - Test that wrong numbers map to `wrong_numeric_or_fact`.
   - Test that strict-pass severe semantic failures map to `strict_keyword_false_positive`.
   - Test Markdown report rendering.

4. Run targeted tests.
   - `python -m pytest tests/eval/test_classify_failure_modes_script.py -q`
   - If stable, run `python -m pytest tests/eval -q`

## Verification

- The script must run against the current reviewed JSONL without requiring network or LLM calls.
- The generated report should include all reviewed non-perfect cases exactly once in the detail table.
- Tests should cover the taxonomy mapping and report rendering.

## Risks And Follow-Ups

- Rule-based classification is not a final truth label. It is a practical triage layer for deciding what to fix next.
- Some cases may deserve manual correction after reading the generated report.
- Later, the web dashboard can expose failure-mode filters once this taxonomy stabilizes.
