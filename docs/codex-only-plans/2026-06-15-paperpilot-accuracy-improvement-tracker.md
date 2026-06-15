# PaperPilot Accuracy Improvement Tracker

Date: 2026-06-15

Codex only. This document tracks what has already been changed for PaperPilot accuracy work and what should be checked next.

## Current Goal

Improve PaperPilot's QASPER-style single-paper QA accuracy in a controlled way.

The current direction is not to chase one score blindly. The working goal is to separate the causes of bad answers into layers:

1. evaluation method;
2. output format and final-answer cleanliness;
3. answer scope control;
4. evidence selection;
5. retrieval/query behavior;
6. numeric/entity verification;
7. multi-part answer coverage.

## What Has Already Been Done

### 1. Original Strict Evaluation Preserved

The original QASPER-style scorer still exists. It checks whether the predicted answer contains any oracle span.

Known limitation:

- It can mark bad answers as pass when a misleading or incidental oracle keyword appears.
- It can mark useful paraphrased answers as fail when the exact oracle span is absent.

Current role:

- Keep it as a regression guard.
- Do not treat it as the only product-quality metric.

### 2. Enriched QASPER Cases Added

The evaluation data was enriched beyond question + oracle span.

Added/used metadata includes:

- gold answers;
- extractive spans;
- free-form answers;
- yes/no and unanswerable flags;
- evidence and highlighted evidence.

Reason:

- This lets later semantic judging compare PaperPilot's answer against gold evidence, not just keywords.

### 3. LLM Semantic Audit Added

A semantic audit layer was added over PaperPilot outputs.

Labels include:

- `correct`;
- `partial`;
- `incorrect`;
- `contradictory`;
- `unverifiable`;
- `judge_uncertain`.

Reason:

- This catches strict false positives and strict false negatives.
- It helps distinguish "found the right idea but worded differently" from "keyword hit but wrong answer."

### 4. Manual Calibration Layer Added

High-value disagreement cases were prepared for manual review.

Reviewed calibration candidate set:

- 31 total candidates;
- all `partial` semantic cases;
- all severe `strict_pass + semantic incorrect/contradictory` cases.

First-pass manual decisions:

- `accept_as_correct`: 18;
- `keep_partial`: 9;
- `downgrade_to_incorrect`: 4;
- `judge_error`: 0.

Current calibrated view:

- strict original: 99 / 150 = 66.0%;
- semantic correct-only: 107 / 150 = 71.3%;
- calibrated correct-only: 125 / 150 = 83.3%;
- calibrated weighted: 129.5 / 150 = 86.3%.

Important interpretation:

- The 83.3% calibrated score is a product-oriented core-coverage estimate, not a final benchmark truth.

### 5. Failure Mode Classification Added

The non-perfect calibrated cases were classified into failure modes.

Diagnostic cases:

- 13 total;
- 9 `keep_partial`;
- 4 `downgrade_to_incorrect`.

Failure mode counts:

- `missing_required_part`: 6;
- `wrong_numeric_or_fact`: 5;
- `overbroad_scope`: 4;
- `strict_keyword_false_positive`: 3;
- `answer_format_noise`: 1.

Main conclusion:

- Many failures are not pure retrieval failures.
- Several answers found useful evidence but failed in final synthesis, scope control, missing required parts, or numeric/entity correctness.

### 6. Web Evaluation Dashboard Added

The local FastAPI/static frontend now has an evaluation snapshot panel and calibration candidate browser.

Purpose:

- Make strict, semantic, and calibrated views easier to inspect.
- Let us browse calibration candidates without manually opening JSONL.

### 7. Answer Synthesis Guard Repair V1 Implemented

Implemented but not yet committed in the current worktree.

Changed areas:

- `paperpilot/skills/deep-read-paper.md`;
- `paperpilot/eval/answer_quality.py`;
- `paperpilot/eval/baselines.py`;
- `scripts/day16_run_eval.py`;
- focused tests.

What changed:

- The deep-read final answer prompt now requires:
  - `Short answer:`;
  - `Evidence:`;
  - optional `Notes:`.
- It no longer asks the final answer to expose `Answer span candidates`.
- It explicitly tells the model not to output process traces such as `Step 4`.
- The eval path now stores `predicted_raw` and can run one LLM repair pass when severe output-quality issues are found.

What repair currently does:

- cleans process traces;
- normalizes final answer format;
- reduces obvious output noise;
- checks simple numeric support between `Short answer` and `Evidence`.

What repair does not do:

- it does not know the gold answer;
- it does not re-retrieve evidence;
- it does not fix a wrong evidence choice;
- it does not guarantee semantic correctness.

Smoke-test result:

- A real one-case smoke ran successfully after clearing proxy variables and increasing MCP initialize timeout.
- Repair changed severe output quality to clean output quality.
- The tested case still answered `HotpotQA` when gold was `WikiHop`, so semantic correctness did not improve there.

Conclusion:

- B2 is useful as output cleanup and diagnostic preservation.
- B2 is not the main lever for further accuracy gains.

## Current Known Problems

### Environment Issues

1. Python venv requires escalation in the Codex sandbox.
   - Cause: `.venv` points to a base interpreter under AppData, outside sandbox read scope.
   - Current workaround: run Python commands with escalation.

2. MCP startup can fail with proxy-related SOCKS errors.
   - Cause: environment has SOCKS proxy variables but `socksio` is not installed.
   - Current workaround used for smoke: temporarily clear proxy variables.

3. ColBERT server may exceed the default MCP initialize timeout on first model load.
   - Current workaround used for smoke: set `MCP_INITIALIZE_TIMEOUT=180`.

These are environment blockers for reliable full eval runs, not answer-quality logic issues.

## What To Check Next

### Priority 1: Evidence Selection Diagnosis

Question:

- When retrieved evidence contains multiple candidate answers or distractor mentions, why does final synthesis choose the wrong one?

Example:

- `qasper-1910.04601-q1`
- Question asks: "What dataset was used in the experiment?"
- Gold: `WikiHop`
- PaperPilot answered: `HotpotQA`

Checks:

1. Inspect the trace for the case.
2. Check all retrieved chunks, not only the final answer.
3. Determine whether `WikiHop` was retrieved at all.
4. Determine whether `HotpotQA` came from a different section or a different paper/task context.
5. Decide whether the failure is:
   - retrieval did not retrieve gold evidence;
   - retrieval retrieved both but final synthesis chose the wrong evidence;
   - question/paper mapping confusion;
   - QASPER gold ambiguity.

Potential fix:

- Add evidence-candidate selection before final answer.
- Require the model to choose the answer sentence most directly aligned with the question, not merely a nearby dataset mention.

### Priority 2: Scope Control For List Questions

Question:

- How do we stop answers from adding related but unasked methods, datasets, metrics, or baselines?

Affected failure modes:

- `overbroad_scope`;
- `answer_format_noise`;
- some `wrong_numeric_or_fact`.

Checks:

1. Identify list-style questions:
   - "Which methods";
   - "What metrics";
   - "What components";
   - "What baselines";
   - "What datasets".
2. Inspect whether retrieved evidence has a narrow answer clause and broader context in the same paragraph.
3. Check if final answer copies the broader context instead of the narrow clause.

Potential fix:

- Add a question-type classifier for list questions.
- Add "scope boundary" instructions:
  - answer only the requested category;
  - put broader context in `Notes` only if needed;
  - do not promote nearby related items into the direct answer.

### Priority 3: Multi-Part Coverage Checklist

Question:

- When the gold answer has multiple required items, how do we make the model avoid missing one?

Affected failure mode:

- `missing_required_part`: 6 cases.

Checks:

1. Detect questions asking for multiple components.
2. In retrieved evidence, identify whether the required items appear in one chunk or multiple chunks.
3. Check if current search strategy retrieves enough chunks for all required items.
4. Check if synthesis drops items because it summarizes too aggressively.

Potential fix:

- Before final answer, require an internal checklist:
  - required answer slots;
  - supporting evidence for each slot;
  - missing slots explicitly marked as insufficient evidence.

Important:

- This checklist should not be exposed verbosely in final output unless needed.

### Priority 4: Numeric And Entity Verification

Question:

- How do we prevent wrong percentages, scores, dataset names, method names, or metric names?

Affected failure mode:

- `wrong_numeric_or_fact`: 5 cases.

Checks:

1. Detect numeric/entity questions:
   - "how much";
   - "what percentage";
   - "what score";
   - "what dataset";
   - "what metric";
   - "what method".
2. Compare numbers/entities in short answer against evidence.
3. Inspect whether multiple similar numbers appear in retrieved chunks.
4. Check whether the model picks a nearby but wrong number.

Potential fix:

- Add numeric/entity verification before final answer.
- For numeric answers, require exact evidence quote containing the same number.
- For dataset/method names, require the evidence sentence to contain both the entity and the relation asked by the question.

### Priority 5: Retrieval Query Strategy

Question:

- Are search queries retrieving the right evidence, or only semantically adjacent distractors?

Checks:

1. For each failure case, list actual search queries from trace.
2. Check whether queries include:
   - the exact question;
   - key entity terms;
   - likely section names;
   - oracle-like terms if they appear in question/gold metadata during offline diagnosis only.
3. Compare retrieved chunks against gold evidence.

Potential fix:

- For dataset/metric/method questions, add targeted query variants:
  - `"dataset experiment uses"`;
  - `"evaluation metric"`;
  - `"baseline systems"`;
  - `"Table" + metric name`;
  - `"experiment setup"`.

### Priority 6: Evaluation Runner Robustness

Question:

- Can we run small and full evals reliably without manual environment hacks?

Checks:

1. Decide whether to install `socksio` or clear proxy variables in eval scripts.
2. Decide whether `MCP_INITIALIZE_TIMEOUT=180` should be default for eval runs.
3. Make smoke scripts explicit about environment assumptions.

Potential fix:

- Add a documented smoke runner that sets safe eval environment variables.
- Keep dependency or proxy changes separate from model/prompt changes.

## Proposed Implementation Order

### Step A: Commit Current B2 Work

Reason:

- B2 has passing tests and a real one-case smoke.
- It should be saved before starting evidence-selection changes.

Recommended commit scope:

- implementation plan;
- `paperpilot/eval/answer_quality.py`;
- `paperpilot/eval/baselines.py`;
- `scripts/day16_run_eval.py`;
- `paperpilot/skills/deep-read-paper.md`;
- `tests/eval/test_answer_quality.py`;
- `tests/eval/test_baselines.py`;
- `tests/test_main_integration.py`.

Do not mix in unrelated calibration/failure-mode files unless explicitly deciding to commit those separately.

### Step B: Build Evidence Selection Case Study

Start with 3 cases:

1. `qasper-1910.04601-q1` - WikiHop vs HotpotQA.
2. `qasper-1701.00185-q1` - overbroad clustering methods.
3. `qasper-1910.07181-q0` - wrong improvement percentages.

Output:

- a short report under `docs/`;
- a trace summary script if needed;
- no behavior changes yet.

### Step C: Design Evidence Selection V1

Only after case study, choose one narrow intervention:

- evidence candidate reranking;
- final synthesis prompt changes;
- question-type-specific verification;
- search query strategy update.

### Step D: Implement One Narrow Fix

Do not fix all failure modes at once.

Candidate first fix:

- dataset/method/metric/entity relation verification.

Reason:

- `wrong_numeric_or_fact` and strict false positives are high-risk and easy to inspect.

### Step E: Smoke Then Re-Evaluate

Run order:

1. unit tests;
2. 3-case smoke;
3. 13 diagnostic case replay;
4. 20-case eval slice;
5. full 150-case eval only after smoke results look sane.

## Tracking Table

| area | status | current result | next action |
|---|---|---|---|
| strict scorer | done | preserved as regression guard | keep, do not rely on alone |
| enriched QASPER cases | done | gold answers/evidence preserved | use for diagnostics |
| semantic audit | done | LLM labels available | use with calibration |
| manual calibration | done | 31 cases reviewed | keep extending if new eval shifts |
| failure mode classification | done | 13 diagnostic cases classified | use to choose fixes |
| web eval dashboard | done | summary and candidate browser added | later add failure-mode filters |
| B2 answer repair | implemented, uncommitted | cleans format, not semantic errors | commit, then move on |
| evidence selection diagnosis | case study done | 3 representative cases inspected; trace capture now preserves full tool results for future eval/debug runs | design Evidence Selection V1 |
| scope control | partial | prompt tightened | diagnose list questions |
| multi-part checklist | not implemented | failure mode count 6 | design after case study |
| numeric/entity verification | partial | simple answer/evidence number check | design stronger relation check |
| eval environment robustness | unresolved | proxy/socksio and timeout issues | decide dependency/config fix |

## Verification Already Run

Unit/integration tests after B2:

- targeted tests: 12 passed;
- broader eval + main integration tests: 80 passed, 1 deselected.

Smoke tests:

- 3-case real smoke attempt failed due MCP environment issues.
- 3-case offline repair smoke completed.
- 1-case real agent smoke completed after clearing proxy variables and increasing MCP initialize timeout.

## Key Decision

Do not continue expanding repair until evidence selection is diagnosed.

Repair can make bad answers cleaner. It cannot choose the correct answer if the raw draft already selected the wrong evidence. The next accuracy work should inspect retrieval and evidence selection before adding more output cleanup.

## 2026-06-15 Evidence Selection Case Study Update

Case-study report:

- `docs/evidence_selection_case_study_20260615.md`

Script:

- `scripts/day23_evidence_selection_case_study.py`

Findings:

1. `qasper-1910.04601-q1` is a retrieval/evidence-selection miss. PaperPilot selected HotpotQA evidence while the calibration gold evidence says WikiHop. Repair cleaned the answer but could not fix the wrong evidence choice.
2. `qasper-1701.00185-q1` is mainly a scope-control failure. The relevant evidence contains the narrow answer and broader neighboring context; PaperPilot promoted the broader context into the direct answer.
3. `qasper-1910.07181-q0` is a numeric verification failure. The answer used percentage values that differ from calibration gold, so the next fix needs exact numeric relation verification.

New blocker discovered:

- Current JSONL traces store truncated tool-result content. They are enough to inspect search queries and snippet heads, but not enough to analyze full top-k retrieved evidence. Before a robust Evidence Selection V1, trace capture should preserve full or at least larger retrieved chunks for eval/debug runs.

2026-06-15 follow-up:

- Trace capture was updated so `agent_loop` emits full `tool_result.content` to callbacks.
- `paperpilot.eval.jsonl_tracer` now receives and persists complete tool results.
- The Web event mapper still truncates content into bounded previews before storing UI events, so the frontend remains protected from oversized tool results.
