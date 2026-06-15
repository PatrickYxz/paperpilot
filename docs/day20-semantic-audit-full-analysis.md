# Day 20 Semantic Audit Full Analysis

Date: 2026-06-11

Baseline: `paperpilot`

Input files:

- `data/eval/results_paperpilot.jsonl`
- `data/eval/qasper_subset_enriched.jsonl`

Generated files:

- `data/eval/semantic_audit_paperpilot_full_20260611.jsonl`
- `data/eval/semantic_audit_summary_full_20260611.md`

## What Was Run

I ran the semantic audit over all 150 current QASPER subset cases, not just the 3-case smoke test.

The runner joined each PaperPilot result row with the enriched QASPER case by `case_id`, then asked the LLM judge to assign one of:

- `correct`
- `partial`
- `incorrect`
- `contradictory`
- `unverifiable`
- `judge_uncertain`

The strict keyword scorer was not changed. The semantic audit is an additional evaluation layer.

## Top-Line Numbers

| scoring view | count | rate |
|---|---:|---:|
| strict scorer pass | 99 / 150 | 66.0% |
| semantic `correct` only | 107 / 150 | 71.3% |
| semantic `correct + partial` | 132 / 150 | 88.0% |
| semantic weighted, `correct=1`, `partial=0.5` | 119.5 / 150 | 79.7% |

The important conclusion is not simply that the score increases. The important conclusion is that the strict keyword scorer has errors in both directions:

- it misses many semantically correct answers;
- it also passes some answers that are only partially correct or wrong.

## Label Counts

| semantic label | count |
|---|---:|
| correct | 107 |
| partial | 25 |
| incorrect | 15 |
| contradictory | 3 |
| unverifiable | 0 |
| judge_uncertain | 0 |

## Strict Score vs Semantic Label

| strict result | semantic label | count |
|---|---|---:|
| pass | correct | 77 |
| pass | partial | 16 |
| pass | incorrect | 5 |
| pass | contradictory | 1 |
| fail | correct | 30 |
| fail | partial | 9 |
| fail | incorrect | 10 |
| fail | contradictory | 2 |

## What This Means

### 1. The original 66.0% strict score is too pessimistic in some places

There are 30 cases where strict scorer says `fail`, but semantic judge says `correct`.

These are mostly cases where PaperPilot answered the question correctly but did not include the exact oracle span in the exact form expected by the keyword scorer.

Representative examples:

| case_id | question | semantic result |
|---|---|---|
| `qasper-1909.00694-q0` | What is the seed lexicon? | strict fail, semantic correct |
| `qasper-1908.06606-q2` | How is the clinical text structuring task defined? | strict fail, semantic correct |
| `qasper-2003.07433-q2` | Which clinically validated survey tools are used? | strict fail, semantic correct |
| `qasper-2001.05284-q1` | What are the series of simple models? | strict fail, semantic correct |

This confirms the earlier concern: pure oracle-span keyword matching can undercount real answer quality.

### 2. The original 66.0% strict score is also too optimistic in some places

There are 22 cases where strict scorer says `pass`, but semantic judge does not say `correct`.

That 22 breaks down into:

- 16 strict passes that are only `partial`;
- 5 strict passes that are `incorrect`;
- 1 strict pass that is `contradictory`.

The severe cases are the 6 strict passes labeled `incorrect` or `contradictory`.

Representative examples:

| case_id | question | semantic result |
|---|---|---|
| `qasper-2004.03685-q1` | What faithfulness criteria does they propose? | strict pass, semantic incorrect |
| `qasper-1910.04601-q1` | What dataset was used in the experiment? | strict pass, semantic incorrect |
| `qasper-1804.10686-q1` | Which corpus of synsets are used? | strict pass, semantic incorrect |
| `qasper-2001.09899-q0` | What are the state of the art measures? | strict pass, semantic incorrect |
| `qasper-1902.06843-q2` | What types of features are used from each data type? | strict pass, semantic contradictory |
| `qasper-2001.06286-q2` | What language tasks did they experiment on? | strict pass, semantic incorrect |

This confirms the other earlier concern: keyword passing can hide logically incomplete or wrong answers.

### 3. `partial` is the main unresolved scoring problem

There are 25 `partial` cases.

These are not all the same kind of problem. From the judge reasons, they include:

- answer contains some correct facts but misses one required part;
- answer includes the right item but also includes extra misleading items;
- answer gives the right direction but wrong number;
- answer is broadly related but not aligned to the exact question.

This means `partial` should not be silently merged into `correct`. It should either:

- remain a separate diagnostic label; or
- be used in a weighted score, for example `correct=1`, `partial=0.5`, others `0`.

For the current full run, weighted scoring gives 79.7%.

### 4. Empty answers exist, but they are not the main cause

There are 2 empty predictions:

| case_id | question | semantic result |
|---|---|---|
| `qasper-1909.00694-q1` | What are labels available in dataset for supervision? | incorrect |
| `qasper-1611.01400-q2` | what is the size of this built corpus? | incorrect |

This is a real PaperPilot failure mode, but it is not large enough to explain the whole accuracy gap.

### 5. The judge is useful, but should not be treated as ground truth yet

The semantic audit caught real issues that the strict scorer cannot express, but it is still an LLM judge.

Known limitations:

- The judge can be sensitive to prompt wording.
- The judge only sees selected full-paper context snippets, not necessarily the entire paper.
- `partial` is subjective and needs calibration.
- Some `contradictory` labels may need human review before being treated as confirmed false positives.

## Practical Interpretation

The current evidence supports three scoring views:

| view | use case | score |
|---|---|---:|
| strict keyword score | conservative historical baseline | 66.0% |
| semantic correct-only | stricter semantic pass rate | 71.3% |
| semantic weighted | more informative working score | 79.7% |

I would not use `correct + partial = 88.0%` as the headline score. It is too generous unless we manually confirm that most partials are acceptable for the product goal.

## Recommended Next Step

Before optimizing PaperPilot itself, manually audit a small but high-value calibration set:

1. Review all 6 severe strict false positives:
   - strict pass + semantic incorrect/contradictory.
2. Review all 16 strict pass + semantic partial cases.
3. Review 10 sampled strict fail + semantic correct cases.
4. Review all 2 empty predictions.

This is 34 cases total if we sample 10 from the 30 strict-fail-correct group.

The goal is to decide:

- whether the semantic judge labels are acceptable;
- whether `partial=0.5` is a reasonable score;
- which PaperPilot failure mode should be fixed first.

Based on the current numbers, the highest-impact PaperPilot fixes are likely:

1. Improve final-answer precision and completeness, especially multi-part questions.
2. Prevent keyword-hit false positives by making answers less overbroad and less list-dumpy.
3. Add guardrails for empty final answers.
4. Reduce wrong-number and wrong-dataset answers before spending time on retrieval tuning.
