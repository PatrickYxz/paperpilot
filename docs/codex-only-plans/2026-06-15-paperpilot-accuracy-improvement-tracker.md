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
| evidence selection diagnosis | case study done | 3 representative cases inspected; trace capture now preserves full tool results for future eval/debug runs | Evidence Selection V1 implemented; evaluate on diagnostic cases |
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

## 2026-06-15 Evidence Selection V1 Implementation Update

Implemented files:

- `paperpilot/eval/evidence_selection.py`
- `paperpilot/eval/baselines.py`
- `scripts/day16_run_eval.py`
- `tests/eval/test_evidence_selection.py`
- `tests/eval/test_baselines.py`

What changed:

1. Added an eval/runtime wrapper that checks whether PaperPilot's current answer is semantically supported by PaperPilot's own retrieved chunks.
2. The selector does not use QASPER gold answers, oracle spans, calibration labels, or manual review decisions.
3. The selector can return:
   - `keep_answer`;
   - `replace_answer`;
   - `insufficient_evidence`;
   - `selector_uncertain`.
4. `run_paperpilot()` now returns:
   - `evidence_selection`;
   - `evidence_rewritten`;
   - `evidence_selection_error`.
5. `scripts/day16_run_eval.py` now persists those selector fields into result JSONL files.
6. Trace extraction now supports the real ColBERT result format, where multiple JSON objects may be concatenated in one tool-result string.
7. Rewrite behavior is conservative:
   - invalid selector JSON keeps the current answer;
   - selector call failure keeps the current answer;
   - low-confidence `replace_answer` or `insufficient_evidence` keeps the current answer;
   - rewrite only happens when action is explicit and confidence is at least medium.

Validation:

- `tests/eval`: 85 passed.
- `tests/test_main_integration.py tests/test_agent_loop.py tests/eval/test_jsonl_tracer.py`: 16 passed, 1 deselected.
- Real 3-case agent smoke completed and wrote `data/eval/evidence_selector_smoke_20260615_184411.jsonl`.
- After fixing real trace parsing, selector replay over those smoke traces wrote `data/eval/evidence_selector_replay_20260615_185233.jsonl`.

Smoke/replay findings:

1. `qasper-1701.00185-q1` improved in the intended way: the selector rewrote the overbroad list to the four directly requested clustering methods.
2. `qasper-1910.07181-q0` was narrowed by removing extra downstream-task material, but it still kept `58%` and `37%` because those numbers are directly supported by retrieved chunks. This wrapper cannot know the QASPER oracle says `50%` and `31%`.
3. `qasper-1910.04601-q1` was narrowed to `HotpotQA` because retrieved chunks directly state that the study uses HotpotQA. This wrapper cannot force the QASPER oracle `WikiHop` when PaperPilot's retrieved evidence supports another answer.

Important conclusion:

- Evidence Selection V1 is useful for answer narrowing and support checking.
- It does not solve failures where retrieved evidence directly supports a benchmark-disagreeing answer.
- The next accuracy lever should inspect retrieval/query strategy and QASPER oracle ambiguity, not add more final-answer repair.

## 2026-06-16 Retrieval Recall Diagnosis Update

Plan:

- `docs/codex-only-plans/2026-06-16-retrieval-recall-diagnosis-plan.md`

Script:

- `scripts/day24_retrieval_recall_diagnosis.py`

Report:

- `docs/retrieval_recall_diagnosis_20260616.md`
- `docs/retrieval_recall_diagnosis_20260616_13cases.md`

What the diagnostic checks:

1. Whether QASPER oracle/gold evidence appears in `data/eval/qasper_subset_enriched.jsonl` full text.
2. Whether the current PaperPilot trace retrieved that gold evidence.
3. Whether a separate ColBERT diagnostic index built from QASPER enriched `full_text` can retrieve gold evidence with gold-oriented probe queries.

Important scoring distinction:

- A bare oracle span mention is not counted as gold-evidence recall.
- Gold-evidence recall requires the highlighted evidence or evidence-head text, not just a nearby entity/string mention.

Findings on the three representative cases:

1. `qasper-1701.00185-q1`
   - QASPER full text contains gold evidence.
   - Current PaperPilot trace recalls the gold evidence.
   - Failure is therefore not retrieval recall. It is answer synthesis / scope control.
   - Evidence Selection V1 addresses this by narrowing the answer to the four requested clustering methods.

2. `qasper-1910.07181-q0`
   - QASPER full text contains the `50%` / `31%` gold evidence.
   - Current PaperPilot trace does not recall that gold evidence.
   - A QASPER-full-text diagnostic ColBERT index can retrieve it with gold-oriented queries.
   - Failure is mainly current query strategy / retrieval targeting, not final synthesis.

3. `qasper-1910.04601-q1`
   - QASPER full text contains the `WikiHop` gold evidence.
   - Current trace mentions `WikiHop` only as a bare/background string, but does not retrieve the gold sentence `Our study uses WikiHop ...`.
   - Diagnostic QASPER-full-text index can retrieve the gold evidence.
   - The current runtime trace instead centers on `HotpotQA`, suggesting a retrieval/data-source/version mismatch issue rather than a pure final-answer synthesis issue.

Conclusion:

- Retrieval recall is a real blocker for at least the numeric/entity conflict cases.
- Scope-control failures still exist, but those are better handled by Evidence Selection V1.
- The next implementation should improve query generation and/or align eval retrieval with QASPER full text before adding more answer repair.

13-case expansion:

- Expanded from 3 representative cases to 13 non-perfect calibrated cases.
- All 13 cases contain the expected QASPER gold/oracle evidence in `data/eval/qasper_subset_enriched.jsonl` full text.
- A QASPER-full-text diagnostic ColBERT index can retrieve the gold evidence for 12/13 cases. The remaining case, `qasper-1701.00185-q1`, already has the gold evidence in the current PaperPilot trace.
- However, only 3/13 current traces have parseable retrieved chunks under the current trace parser:
  - `qasper-1701.00185-q1`: 15 chunks, gold evidence recalled.
  - `qasper-1910.07181-q0`: 28 chunks, gold evidence not recalled, diagnostic probe can retrieve it.
  - `qasper-1910.04601-q1`: 15 chunks, only a bare `WikiHop`/oracle mention appears, gold evidence itself is not recalled, diagnostic probe can retrieve it.
- The other 10/13 traces are old/truncated search-result traces. They contain ColBERT search calls, but the tool result content is cut off before valid JSON, so `extract_retrieved_chunks()` correctly returns 0 chunks. These cases should be treated as "current trace evidence unavailable", not as definitive current retrieval misses.

Updated conclusion from the larger sample:

- The larger sample strengthens the hypothesis that retrieval/query targeting is a major accuracy bottleneck, because gold evidence is present in the QASPER enriched source and retrievable from a QASPER-full-text diagnostic index.
- The exact distribution of retrieval failures versus synthesis failures is still not reliable until the 10 truncated-trace cases are rerun with current full trace capture.
- Next practical step: rerun those 10 cases with full traces, rerun the recall diagnosis, then decide whether to tune query generation, retrieval source alignment, or synthesis/scope control first.

10-case rerun with fresh traces:

- Rerun script:
  - `scripts/day24_rerun_paperpilot_cases.py`
- Rerun output:
  - `data/eval/paperpilot_rerun_10_traces_20260616.jsonl`
  - `data/eval/paperpilot_rerun_1907_retry_20260616.jsonl`
  - `data/traces_rerun_20260616/`
- Final rerun diagnosis report:
  - `docs/retrieval_recall_diagnosis_20260616_rerun_10.md`

Operational note:

- The first rerun failed before answering because `ALL_PROXY=socks5://127.0.0.1:7890` made `httpx` require the missing `socksio` package when starting the graph MCP server.
- The successful rerun cleared only `ALL_PROXY` / `all_proxy` for the command while keeping the normal HTTP/HTTPS proxy variables.
- One case, `qasper-1907.02030-q0`, timed out in the first successful 10-case batch and was rerun separately.

Fresh 10-case diagnosis:

- 10/10 cases contain gold evidence in QASPER enriched `full_text`.
- 10/10 cases are retrievable by the QASPER-full-text diagnostic probe.
- 5/10 fresh traces recall full gold evidence:
  - `qasper-1808.05902-q1`
  - `qasper-1809.04960-q0`
  - `qasper-1907.02030-q0`
  - `qasper-1911.03385-q0`
  - `qasper-1804.10686-q1`
- 3/10 fresh traces mention only an oracle/answer span but miss the full gold evidence:
  - `qasper-1808.05902-q0`
  - `qasper-1910.03042-q2`
  - `qasper-2001.09899-q0`
- 2/10 fresh traces miss gold evidence even though the diagnostic probe can retrieve it:
  - `qasper-1911.03894-q1`
  - `qasper-1911.03894-q2`

Combined with the earlier 3 parseable traces:

- 6/13 cases recall full gold evidence in the current/fresh PaperPilot trace.
- 4/13 cases only mention oracle/answer spans but miss the full supporting evidence.
- 3/13 cases look like clearer current-query retrieval misses where diagnostic probe can retrieve the gold evidence.

Updated conclusion after rerun:

- The problem is not "all failures are retrieval recall failures".
- Retrieval is still a major bottleneck: 7/13 inspected cases do not recall the full QASPER gold evidence in the PaperPilot trace.
- But synthesis/scope control is also a major bottleneck: 6/13 already recall full gold evidence and still can fail or answer too broadly.
- The next implementation should split retrieval improvements into two subproblems:
  1. query targeting for cases that completely miss the gold evidence;
  2. evidence-span targeting for cases that retrieve a nearby oracle string but not the full supporting evidence.
