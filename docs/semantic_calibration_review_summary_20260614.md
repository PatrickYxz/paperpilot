# Semantic Calibration Review Summary

Date: 2026-06-14

This is a first-pass manual calibration over the 31 high-value review cases:

- all 25 `partial` cases from the LLM semantic judge;
- all 6 `strict_pass + semantic incorrect/contradictory` cases.

The review uses a product-oriented core-coverage criterion:

> If PaperPilot covers the core gold answer and the extra content is not materially misleading, mark it as acceptable.

This is not the same as the original QASPER strict span-match criterion.

## Review Decisions

| decision | count |
|---|---:|
| `accept_as_correct` | 18 |
| `keep_partial` | 9 |
| `downgrade_to_incorrect` | 4 |
| `judge_error` | 0 |

## Calibrated Scores

| scoring view | count | rate |
|---|---:|---:|
| original strict pass | 99 / 150 | 66.0% |
| original LLM semantic correct-only | 107 / 150 | 71.3% |
| calibrated correct-only | 125 / 150 | 83.3% |
| calibrated weighted, `correct=1`, `partial=0.5` | 129.5 / 150 | 86.3% |

## Interpretation

The strict score is too harsh for product usefulness because it misses many answers that cover the gold answer in different wording or with additional context.

The raw LLM semantic score is also too harsh in several `partial` cases because it often penalizes extra supported context even when the core answer is present.

The calibrated score should be read as a core-coverage estimate, not as a final benchmark number. It says that PaperPilot often finds the right answer, but still has answer-quality problems:

- overbroad answers that include extra methods, datasets, or metrics;
- answers that miss one required subpart;
- wrong numbers in quantitative questions;
- a few strict-pass false positives where a keyword hit hides a wrong answer.

## Highest-Value Failure Modes

The remaining `keep_partial` and `downgrade_to_incorrect` cases point to these priorities:

1. Tighten final answers to the exact question scope.
2. Handle multi-part questions more explicitly.
3. Add special care for quantitative answers, because wrong numbers should not receive much credit.
4. Reduce answer list-dumping: include the direct answer first, then keep supporting context clearly separated.

## Files

- Machine-readable reviewed decisions: `data/eval/semantic_calibration_candidates_20260614.jsonl`
- Human-readable reviewed cases: `docs/semantic_calibration_candidates_20260614.md`
