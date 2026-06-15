# Evidence Selection V1 Design Plan

Date: 2026-06-15

Codex only. This plan records the approved Evidence Selection V1 design for PaperPilot accuracy work.

## Background

PaperPilot already has an answer-quality wrapper in the eval path:

- it stores `predicted_raw`;
- it checks final-answer format and simple support issues;
- it can run one repair pass for severe output-quality problems;
- it preserves the repaired output as `predicted`.

That repair layer is useful, but it mostly fixes answer shape:

- missing `Short answer:`;
- missing `Evidence:`;
- process traces such as `Step 4`;
- numbers in `Short answer:` not repeated in `Evidence:`.

It does not reliably answer the deeper question:

> Is PaperPilot's answer semantically supported by PaperPilot's own retrieved evidence?

The representative failures show why this matters:

1. `qasper-1910.04601-q1`: PaperPilot answered `HotpotQA`, but calibration says the correct dataset is `WikiHop`.
2. `qasper-1701.00185-q1`: PaperPilot included the important clustering methods but also added broader neighboring methods.
3. `qasper-1910.07181-q0`: PaperPilot selected percentage values that do not match the calibrated gold interpretation.

The agreed direction is therefore not another generic prompt tightening. V1 should add a wrapper-level evidence semantic support check between raw PaperPilot output and final evaluated prediction.

## Core Idea

The wrapper should check:

```text
Does PaperPilot's answer follow from PaperPilot's retrieved chunks?
```

It should not check:

```text
Does PaperPilot's answer match the QASPER gold answer?
```

That distinction is important:

- final benchmark judging compares `predicted answer` against `gold answer / gold evidence`;
- Evidence Selection V1 compares `predicted answer` against `PaperPilot retrieved evidence`.

Runtime/eval evidence selection must not use QASPER oracle spans, gold answers, manual labels, or calibration decisions.

## Proposed Flow

Current eval flow:

```text
question
 -> PaperPilot agent
 -> raw final answer
 -> format/support repair
 -> predicted
```

Proposed V1 flow:

```text
question
 -> PaperPilot agent
 -> raw final answer
 -> format/support repair if needed
 -> extract retrieved chunks from trace
 -> evidence semantic support selector
 -> keep, rewrite, or mark evidence insufficient
 -> predicted
```

The selector is a second wrapper step after basic answer-quality repair.

Reason:

- format repair should still clean malformed answers first;
- evidence selection should inspect the cleaned answer and the retrieved chunks;
- the two responsibilities should stay separate.

## Selector Responsibility

The selector receives:

- `question`;
- `raw_answer` or repaired answer candidate;
- retrieved chunks from `colbert.search` tool results;
- optional lightweight answer-quality diagnostics.

The selector returns structured JSON:

```json
{
  "question_type": "entity",
  "answer_supported": false,
  "selected_answer": "WikiHop",
  "supporting_evidence": "Our study uses WikiHop ...",
  "rejected_candidates": [
    {
      "candidate": "HotpotQA",
      "reason": "Mentioned as related/source context, but not the direct dataset used in the experiment."
    }
  ],
  "action": "replace_answer",
  "confidence": "medium"
}
```

The allowed `action` values should be:

- `keep_answer`: current answer is directly supported by retrieved evidence;
- `replace_answer`: retrieved evidence supports a better answer than the current answer;
- `insufficient_evidence`: retrieved evidence does not directly support a reliable answer;
- `selector_uncertain`: selector output is unclear or confidence is too low, so keep the current answer but record the uncertainty.

## Question Types

The selector should classify the question into one or more types:

- `entity`: dataset, corpus, method, metric, model, baseline, component, task, or source;
- `numeric`: percentage, score, count, improvement, difference, rank, table value;
- `list`: methods, datasets, metrics, baselines, components, factors, findings;
- `general`: explanation, summary, definition, motivation, conclusion.

This classification is not final judging. It only changes how evidence support is checked.

## Semantic Support Rules

### Entity Questions

The evidence must express the relation asked by the question.

Good support:

```text
Question: What dataset was used in the experiment?
Evidence: Our study uses WikiHop ...
```

Weak support:

```text
Evidence: HotpotQA is an actively used multi-hop QA dataset.
```

The weak support contains a dataset name, but it does not directly answer "what dataset was used in the experiment."

### Numeric Questions

The selected evidence must contain the same number used in the answer and must express the same comparison or measurement relation.

It is not enough for the number to appear somewhere in a nearby table if the question asks about a derived improvement or comparison.

### List Questions

Every item in the direct answer must belong to the requested category.

If the question asks:

```text
Which popular clustering methods did they experiment with?
```

The selector should reject items that are only nearby baselines, related neural variants, or dimensionality-reduction components unless the evidence clearly governs them under the asked category.

### General Questions

The selected evidence should directly support the claim, not merely discuss the same topic.

If the retrieved evidence is only background or related work, the selector should choose `insufficient_evidence` or `selector_uncertain`.

## Rewrite Behavior

When `action == "keep_answer"`:

- keep the candidate answer;
- store the selector result for diagnostics.

When `action == "replace_answer"`:

- generate a final answer only from `selected_answer` and `supporting_evidence`;
- do not add facts not present in the selector result;
- output the same final format:

```text
Short answer: <selected answer>

Evidence: <supporting evidence>

Notes: <optional caveat>
```

When `action == "insufficient_evidence"`:

- output a conservative answer:

```text
Short answer: Evidence is insufficient.

Evidence: The retrieved passages do not directly answer the question.
```

When `action == "selector_uncertain"`:

- keep the candidate answer;
- record selector uncertainty;
- do not let uncertain selector output rewrite the answer.

## Proposed Code Changes

### New Module: `paperpilot/eval/evidence_selection.py`

Responsibilities:

1. Extract `colbert.search` results from JSONL traces.
2. Normalize retrieved chunks into a compact structure:

```python
{
    "query": "...",
    "rank": 1,
    "paper_id": "...",
    "chunk_text": "..."
}
```

3. Build selector prompts.
4. Parse selector JSON.
5. Apply selector action to produce a final prediction.

Important:

- JSON parsing must be defensive.
- Failed selector calls should not crash the eval run.
- If selector output is invalid, keep the candidate answer and record `selector_uncertain`.

### Existing Module: `paperpilot/eval/baselines.py`

Extend `run_paperpilot()`:

1. run PaperPilot as today;
2. run existing answer-quality repair as today;
3. read trace chunks from `trace_path`;
4. run evidence selector unless there was an agent error;
5. return new diagnostic fields:

```python
{
    "predicted_raw": "...",
    "predicted": "...",
    "answer_quality": {...},
    "answer_repaired": true,
    "evidence_selection": {...},
    "evidence_rewritten": true,
    "evidence_selection_error": null
}
```

### Existing Script: `scripts/day16_run_eval.py`

Include selector fields in JSONL output:

- `evidence_selection`;
- `evidence_rewritten`;
- `evidence_selection_error`.

## Prompt Boundary

The selector prompt should be strict:

- use only retrieved chunks;
- do not use prior knowledge;
- do not use QASPER gold;
- do not judge benchmark correctness;
- decide whether the current answer is directly supported by the retrieved chunks;
- return JSON only.

The final rewrite prompt, if needed, should also be strict:

- do not call tools;
- do not add new facts;
- use only `selected_answer` and `supporting_evidence`;
- keep the final answer short.

## Testing Plan

### Unit Tests

Add tests for:

1. trace chunk extraction from JSONL tool results;
2. selector prompt includes question, answer, and retrieved chunks;
3. invalid selector JSON becomes `selector_uncertain`;
4. `keep_answer` keeps prediction;
5. `replace_answer` rewrites prediction from selected evidence;
6. `insufficient_evidence` returns conservative answer;
7. selector failure records error and keeps current prediction.

### Existing Tests To Keep Passing

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_answer_quality.py tests\eval\test_baselines.py tests\test_main_integration.py -q
```

### Real Smoke

After unit tests pass, run the three representative cases with full trace capture:

- `qasper-1910.04601-q1`;
- `qasper-1701.00185-q1`;
- `qasper-1910.07181-q0`.

Use the known eval environment workaround:

- clear proxy environment variables;
- set `MCP_INITIALIZE_TIMEOUT=180`.

## Success Criteria

V1 is successful if:

- the eval JSONL records selector decisions;
- selector decisions are readable and debuggable;
- malformed selector output does not break eval;
- at least one representative failure is improved or made more conservative;
- no existing answer repair behavior regresses.

## Risks

- The selector is still LLM-based, so it can make mistakes.
- It may over-rewrite correct answers if confidence/action gating is too aggressive.
- It depends on trace quality and complete retrieved chunks.
- It adds extra eval cost and latency.

Mitigation:

- default to keeping the current answer on invalid or uncertain selector output;
- only rewrite when action is explicit and supporting evidence is non-empty;
- preserve all diagnostics for manual review.

## Implementation Order

1. Add `paperpilot/eval/evidence_selection.py`.
2. Add unit tests for extraction, parsing, and action application.
3. Wire selector into `run_paperpilot()`.
4. Extend `scripts/day16_run_eval.py` output fields.
5. Run targeted tests.
6. Run 3-case smoke.
7. Decide whether to replay 13 diagnostic cases.

