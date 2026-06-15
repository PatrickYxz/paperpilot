# Evidence Selection Case Study Plan

Date: 2026-06-15

Codex only.

## Background And Goal

The accuracy tracker says the next bottleneck is likely evidence selection, not answer-format repair. The goal of this slice is to inspect three representative failure cases and decide whether each failure came from:

- retrieval not finding the gold evidence;
- retrieval finding both gold and distractor evidence, but synthesis choosing the wrong one;
- answer scope expanding beyond the question;
- numeric/entity verification failure;
- QASPER gold ambiguity.

This is a diagnostic slice. It should not change PaperPilot behavior.

## Cases

1. `qasper-1910.04601-q1` - WikiHop vs HotpotQA.
2. `qasper-1701.00185-q1` - overbroad clustering methods.
3. `qasper-1910.07181-q0` - wrong improvement percentages.

## Steps

1. Inspect available traces and calibration records for the three cases.
2. Extract:
   - question;
   - oracle spans;
   - manual review notes;
   - search queries;
   - retrieved text snippets;
   - raw and repaired answers where available.
3. Write a Markdown report under `docs/`.
4. Record which layer failed for each case.
5. Propose one narrow Evidence Selection V1 intervention after the diagnosis.

## Verification

- The script/report generation should be read-only with respect to runtime eval data.
- No behavior changes should be made in this slice.
- The report should cite concrete case IDs and retrieved evidence observations.
