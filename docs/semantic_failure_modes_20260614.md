# Semantic Failure Modes

Date: 2026-06-14

Source: `data\eval\semantic_calibration_candidates_20260614.jsonl`

This report classifies manually reviewed non-perfect calibration cases.
It is a diagnosis layer; it does not change the calibrated score.

## Scope

- Reviewed calibration candidates: 31
- Diagnostic cases: 13
- Decisions included: `downgrade_to_incorrect, keep_partial`

## Counts By Failure Mode

| failure_mode | count | meaning | engineering implication |
|---|---:|---|---|
| overbroad_scope | 4 | Answer includes extra methods, datasets, metrics, baselines, or unrelated context beyond the question. | Tighten answer synthesis: answer the asked scope first, then isolate optional context. |
| missing_required_part | 6 | Answer covers some gold content but misses at least one required component. | Add explicit checklist coverage for multi-part questions before final synthesis. |
| wrong_numeric_or_fact | 5 | Answer contains a wrong number, dataset name, method name, label, or central factual claim. | Add stricter numeric/entity verification against retrieved evidence before final output. |
| strict_keyword_false_positive | 3 | Strict scorer passed because of keyword/span overlap, but review says the answer is not fully correct. | Do not trust span hits alone; require answer-level semantic consistency for pass labels. |
| answer_format_noise | 1 | Answer is harder to judge because it is noisy, verbose, list-heavy, or mixes reasoning with final output. | Separate internal reasoning, candidate spans, final answer, and evidence more cleanly. |
| judge_or_gold_ambiguity | 0 | The main issue is ambiguous wording, narrow gold spans, or judge over-penalization. | Keep manual calibration available because some QASPER oracle spans are narrow or underspecified. |
| unknown | 0 | The reviewed fields are insufficient for a confident failure-mode assignment. | Needs manual inspection before it can drive product work. |

## Counts By Review Decision

| review_decision | count |
|---|---:|
| downgrade_to_incorrect | 4 |
| keep_partial | 9 |

## Case Details

| case_id | decision | semantic | strict | failure_modes | review_notes |
|---|---|---|---|---|---|
| `qasper-1701.00185-q1` | `keep_partial` | `partial` | pass | `overbroad_scope` | Gold asks for four popular clustering methods; answer includes them but adds separate baseline/neural methods that blur scope. |
| `qasper-1808.05902-q0` | `keep_partial` | `partial` | fail | `missing_required_part` | Mentions outperforming baselines and efficient SVI, but misses the specific SVI-best/converges-faster emphasis. |
| `qasper-1808.05902-q1` | `keep_partial` | `partial` | pass | `overbroad_scope`, `answer_format_noise` | Includes the six gold approaches but mixes in many related-work methods, making the scope noisy. |
| `qasper-1809.04960-q0` | `keep_partial` | `partial` | fail | `missing_required_part` | Answers paired-data sizes and Tencent data but does not clearly name both gold corpus descriptions. |
| `qasper-1907.02030-q0` | `keep_partial` | `partial` | fail | `overbroad_scope`, `wrong_numeric_or_fact` | Identifies F1/accuracy evaluation but uses different numbers and adds clustering formula not aligned with gold. |
| `qasper-1910.03042-q2` | `keep_partial` | `partial` | pass | `missing_required_part` | Covers several system components but misses Amazon Conversational Bot Toolkit and some exact gold components. |
| `qasper-1910.07181-q0` | `downgrade_to_incorrect` | `partial` | fail | `wrong_numeric_or_fact` | Question asks how much improvement; the central percentages differ from the gold answer. |
| `qasper-1911.03385-q0` | `keep_partial` | `partial` | pass | `missing_required_part` | Describes manual evaluation procedure, but does not state the gold criterion accuracy clearly. |
| `qasper-1911.03894-q1` | `downgrade_to_incorrect` | `partial` | fail | `overbroad_scope`, `wrong_numeric_or_fact` | Gold answer is XNLI TRANSLATE-TEST lag; prediction answers different POS subtasks. |
| `qasper-1911.03894-q2` | `keep_partial` | `partial` | fail | `missing_required_part` | Includes several correct improvements but misses the RoBERTa TRANSLATE-TEST lag required by gold. |
| `qasper-1804.10686-q1` | `downgrade_to_incorrect` | `incorrect` | pass | `wrong_numeric_or_fact`, `strict_keyword_false_positive` | Gold answer is Wiktionary; prediction answers WATLINK/RuThes/RuWordNet instead. |
| `qasper-1910.04601-q1` | `downgrade_to_incorrect` | `incorrect` | pass | `wrong_numeric_or_fact`, `strict_keyword_false_positive` | Gold says WikiHop; prediction says HotpotQA. |
| `qasper-2001.09899-q0` | `keep_partial` | `incorrect` | pass | `missing_required_part`, `strict_keyword_false_positive` | Mentions Randomwalk but omits Walktrap and Louvain clustering, so it is incomplete rather than fully correct. |

## What This Suggests Next

1. Prioritize answer-scope control before changing retrieval. Several partial cases found the relevant evidence but included extra or differently scoped material.
2. Add a final-answer checklist for questions asking for multiple methods, datasets, components, or metrics.
3. Add stricter numeric/entity verification for percentage, dataset-name, and method-name questions.
4. Keep strict score as a regression guard, but avoid treating it as the only product-quality metric.
