# Phase 3 Verified Retrieval Case Results

Date: 2026-06-28

## Context

This note records real QASPER case runs for the Query Planner Phase 3
requirement-level evidence verifier.

Implementation state:

- `planned_retrieval` supports `verify_evidence=true`.
- The verifier runs at requirement level and returns
  `direct | partial | no` decisions.
- Eval records `evidence_verification_used`, verified summary counts, support
  counts, missing verified requirements, and conflicts.

Run command:

```bash
.venv/bin/python scripts/day24_rerun_paperpilot_cases.py \
  --case-id qasper-1910.04601-q1 \
  --case-id qasper-1701.00185-q1 \
  --case-id qasper-1910.07181-q0 \
  --use-query-plan \
  --verify-evidence \
  --out-path data/eval/paperpilot_phase3_verified_cases_20260628.jsonl \
  --trace-dir data/traces_phase3_verified_20260628
```

The first sandboxed attempt failed with `anthropic.APIConnectionError` because
the run needed network access for the LLM. The successful run was executed with
network permission.

## Batch 1 Summary

| Case | Type | Result | Verified summary | Direct / Partial / No | Main observation |
| --- | --- | --- | ---: | --- | --- |
| `qasper-1910.04601-q1` | dataset/entity | FAIL | 1 | 1 / 2 / 3 | Verifier selected `HotpotQA` as direct evidence, so dataset role ambiguity remains unresolved. |
| `qasper-1701.00185-q1` | list/method | PASS | 2 | 2 / 2 / 2 | Verifier narrowed evidence to relevant method-list regions, but conflict metadata shows list atom merging is still rough. |
| `qasper-1910.07181-q0` | numeric/comparison | FAIL | 2 | 0 / 9 / 7 | Verifier found only partial evidence; agent later expanded with extra searches and produced an overly broad numeric answer. |

## Detailed Notes

### `qasper-1910.04601-q1`

Question:

```text
What dataset was used in the experiment?
```

Observed answer:

```text
Short answer: HotpotQA
```

Verifier metadata:

- `evidence_verification_used=true`
- `verified_summary_count=1`
- support counts: `direct=1`, `partial=2`, `no=3`
- `missing_verified_requirements=[]`
- `verification_conflicts=[]`

The selected verified chunk states that the study uses HotpotQA. The verifier
marked this as `direct high`. This is the wrong behavior for the known failure:
the model still does not distinguish the dataset role carefully enough.

Interpretation:

- This is primarily a verifier prompt/schema weakness, not an MCP or evidence
  extraction failure.
- The verifier needs stronger role labels for dataset/entity questions:
  source dataset, constructed dataset, experiment/evaluation dataset,
  related-work dataset, candidate dataset.

### `qasper-1701.00185-q1`

Question:

```text
Which popular clustering methods did they experiment with?
```

Observed answer:

```text
K-means, Skip-thought Vectors, Recursive Neural Network, Paragraph Vector,
Average Embedding, Latent Semantic Analysis, Laplacian Eigenmaps, Locality
Preserving Indexing, and bidirectional RNN.
```

Verifier metadata:

- `evidence_verification_used=true`
- `verified_summary_count=2`
- support counts: `direct=2`, `partial=2`, `no=2`
- `missing_verified_requirements=[]`
- one conflict recorded for incompatible or non-identical list atoms

Interpretation:

- This is the best result in Batch 1.
- Verified evidence selection appears useful for narrowing a list/method case.
- The conflict shows a remaining design issue: for list questions, multiple
  direct chunks should often be merged as complementary evidence rather than
  treated as an answer conflict.

### `qasper-1910.07181-q0`

Question:

```text
How much is representaton improved for rare/medum frequency words compared to
standalone BERT and previous work?
```

Observed answer mixed WNLaMPro rare/medium MRR, RoBERTa numbers, and downstream
task accuracy improvements.

Verifier metadata:

- `evidence_verification_used=true`
- `verified_summary_count=2`
- support counts: `direct=0`, `partial=9`, `no=7`
- `missing_verified_requirements=[]`
- `verification_conflicts=[]`

Interpretation:

- The verifier correctly avoided claiming direct support, but selection still
  returned partial evidence.
- The agent then performed extra searches and broadened the final answer.
- Numeric/comparison questions need stricter behavior when no direct evidence
  exists: partial evidence should probably surface as insufficient or scoped
  evidence, not as a license to synthesize across multiple unrelated metrics.

## Current Verdict

The Phase 3 framework is working mechanically:

- verifier is called;
- metadata is recorded;
- eval extractor can prefer `verified_summary_items`;
- failures are now diagnosable at the evidence-decision level.

But it has not yet solved evidence pool selection quality.

Most important next fixes:

1. Strengthen dataset/entity verification around role disambiguation.
2. Add list-specific handling that treats complementary list atoms as coverage
   instead of generic conflict.
3. Make numeric partial evidence constrain the agent more strongly when no
   `direct` evidence exists.

## Batch 2 Plan

Run three additional cases to test whether the current behavior generalizes:

| Case | Reason |
| --- | --- |
| `qasper-1909.08402-q0` | numeric improvement question: "By how much do they outperform standard BERT?" |
| `qasper-1909.08402-q1` | dataset question from the same paper, useful for dataset/entity role behavior |
| `qasper-1908.06606-q1` | baseline/list question: "What baselines is the proposed model compared against?" |

Planned output:

- `data/eval/paperpilot_phase3_verified_cases_batch2_20260628.jsonl`
- `data/traces_phase3_verified_batch2_20260628/`

## Batch 2 Summary

Run command:

```bash
.venv/bin/python scripts/day24_rerun_paperpilot_cases.py \
  --case-id qasper-1909.08402-q0 \
  --case-id qasper-1909.08402-q1 \
  --case-id qasper-1908.06606-q1 \
  --use-query-plan \
  --verify-evidence \
  --out-path data/eval/paperpilot_phase3_verified_cases_batch2_20260628.jsonl \
  --trace-dir data/traces_phase3_verified_batch2_20260628
```

| Case | Type | Result | Verified summary | Direct / Partial / No | Main observation |
| --- | --- | --- | ---: | --- | --- |
| `qasper-1909.08402-q0` | numeric/improvement | FAIL | 1 | 1 / 4 / 1 | Verifier found one direct-looking evidence item, but answer-level selector later marked evidence insufficient. Numeric evidence still does not reliably preserve metric and comparator. |
| `qasper-1909.08402-q1` | dataset/entity | PASS | 1 | 1 / 1 / 4 | Dataset answer succeeded. This suggests dataset questions are not universally broken; the earlier HotpotQA failure is specifically a role-disambiguation problem. |
| `qasper-1908.06606-q1` | baseline/list | PASS | 0 | 0 / 0 / 6 | Verifier rejected all planned evidence, but the agent recovered via follow-up `colbert.search` and answered correctly. This means planned verified evidence was too conservative or missed the relevant baseline chunk. |

## Batch 2 Detailed Notes

### `qasper-1909.08402-q0`

Question:

```text
By how much do they outperform standard BERT?
```

Observed answer:

```text
Short answer: Evidence is insufficient.
```

Verifier metadata:

- `evidence_verification_used=true`
- `verified_summary_count=1`
- support counts: `direct=1`, `partial=4`, `no=1`
- `missing_verified_requirements=[]`
- `verification_conflicts=[]`

The selected verified chunk discusses improvement over standard BERT, but the
final answer selector still judged the retrieved passages insufficient.

Interpretation:

- The verifier selected a plausible chunk, but numeric answer extraction still
  failed.
- This reinforces that numeric cases need stricter structured evidence:
  metric, baseline, comparator, value, and task/dataset should be bundled before
  evidence is promoted as answer-ready.

### `qasper-1909.08402-q1`

Question:

```text
What dataset do they use?
```

Observed answer:

```text
GermEval 2019 shared task dataset, with 20,784 German books, blurbs, and metadata.
```

Verifier metadata:

- `evidence_verification_used=true`
- `verified_summary_count=1`
- support counts: `direct=1`, `partial=1`, `no=4`
- `missing_verified_requirements=[]`
- `verification_conflicts=[]`

Interpretation:

- This is a clean pass for dataset/entity retrieval.
- The earlier `HotpotQA` failure should be treated as role ambiguity, not as a
  blanket dataset-question failure.

### `qasper-1908.06606-q1`

Question:

```text
What baselines is the proposed model compared against?
```

Observed answer:

```text
QANet, BERT-Base
```

Verifier metadata:

- `evidence_verification_used=true`
- `verified_summary_count=0`
- support counts: `direct=0`, `partial=0`, `no=6`
- `missing_verified_requirements` recorded
- `verification_conflicts=[]`

The final answer succeeded only after follow-up `colbert.search`; planned
verified evidence did not provide selected summary evidence.

Interpretation:

- This is a pass for the overall agent workflow, but not for the Phase 3
  verified evidence pool.
- The verifier or planned candidates were too conservative for a baseline/list
  question.
- `verified_summary_items=[]` should probably trigger a clearer fallback path:
  either broaden candidate verification or surface missing evidence before the
  agent answers.

## Updated Verdict After Two Batches

Across six real cases:

- Passed final answer: 3 / 6
- Failed final answer: 3 / 6
- Verified retrieval clearly helped: `qasper-1701.00185-q1`,
  `qasper-1909.08402-q1`
- Verified retrieval failed by wrong direct judgment: `qasper-1910.04601-q1`
- Verified retrieval failed by weak/partial numeric handling:
  `qasper-1910.07181-q0`, `qasper-1909.08402-q0`
- Overall agent recovered despite verifier failure:
  `qasper-1908.06606-q1`

Current conclusion:

The Phase 3 implementation is useful as an observability layer and sometimes
improves evidence selection, but it is not yet a reliable reranker. The next
design/implementation pass should focus less on plumbing and more on the
verifier decision quality:

1. Add question-type-specific verifier rubrics.
2. For dataset/entity, require role classification and reject ambiguous dataset
   roles.
3. For numeric questions, require structured numeric atoms before allowing
   `direct`.
4. For list questions, support complementary list coverage and avoid treating
   all atom differences as conflicts.
5. When `verified_summary_items=[]`, make the fallback explicit instead of
   silently relying on agent follow-up search.

## Follow-Up Diagnosis and Fix

After reviewing the traces, two deterministic verifier-selection bugs were
identified.

### Root Cause 1: Candidate Truncation Dropped Summary Evidence

Case:

```text
qasper-1908.06606-q1
```

Before the fix:

- The correct Table IV chunk was present in `items` as `ev_4`.
- It was also present in `summary_items`.
- It was not sent to the verifier because `candidate_items_for_requirement`
  only took the top `verifier_candidate_k=6` items by ColBERT score.
- `ev_4` ranked below that cutoff, so `verified_summary_items=[]`.
- The final answer passed only because the agent performed a follow-up
  `colbert.search`.

Fix:

- Keep the top score-ranked candidates.
- Also append any requirement-targeting `summary_items` not already included.
- This ensures evidence already selected for display cannot be hidden from the
  verifier by the verifier candidate cap.

Verification:

- Added unit test:
  `test_candidate_items_include_summary_items_beyond_score_cap`.
- Re-ran `qasper-1908.06606-q1` as a single real-case smoke:
  - output: `data/eval/paperpilot_phase3_candidate_fix_smoke_20260628.jsonl`
  - trace: `data/traces_phase3_candidate_fix_smoke_20260628/`
  - result: PASS
  - `verified_summary_items=['ev_4']`
  - `direct_support_count=1`
  - no follow-up `mcp__colbert__search` was needed.

### Root Cause 2: Single List Evidence Was Misclassified as Conflict

Case:

```text
qasper-1908.06606-q1
```

After Root Cause 1 was fixed, verifier correctly marked `ev_4` as direct with
answer atoms:

```text
QANet, BERT-Base
```

However, conflict detection recorded a conflict because it treated multiple
answer atoms inside one high-confidence direct evidence item as incompatible.

Fix:

- Only record conflict when there are multiple high-confidence direct evidence
  items with different answer atoms.
- A single evidence item with multiple atoms is list coverage, not conflict.

Verification:

- Added unit test:
  `test_select_verified_summary_does_not_conflict_on_single_list_evidence`.

Post-fix verification:

```text
.venv/bin/python -m pytest tests/retrieval -q
31 passed

.venv/bin/python -m pytest -m "not slow" -q
331 passed, 11 deselected, 6 warnings
```
