# Semantic Audit Runner V1 Plan

Date: 2026-06-11

## Background

PaperPilot already has the first QASPER eval-upgrade slice on branch
`codex/qasper-eval-upgrade`:

- enriched QASPER loader
- enriched subset writer
- enriched subset profile
- semantic audit schema

The existing strict scorer is still unchanged. The current strict score checks
only whether the predicted answer contains an oracle extractive span. The next
step is to add an offline semantic audit runner that asks an LLM judge whether
PaperPilot's answer is semantically correct when compared with QASPER's gold
answer metadata and gold evidence.

## Goal

Build `semantic audit runner v1`:

```text
data/eval/qasper_subset_enriched.jsonl
data/eval/results_paperpilot.jsonl
  -> LLM judge
  -> data/eval/semantic_audit_paperpilot.jsonl
  -> data/eval/semantic_audit_summary.md
```

This should answer:

```text
Is the current 66.0% strict span score too strict, too generous, or mostly reliable?
```

## Non-Goals

This slice will not:

- change `paperpilot/eval/scorer.py`
- change `scripts/day16_run_eval.py`
- change PaperPilot prompts or deep-read behavior
- judge PaperPilot's retrieved trace evidence yet
- merge this branch into `main`
- commit generated `data/eval/*.jsonl` or `data/eval/*.md` artifacts by default

## Design

### Inputs

Primary inputs:

```text
data/eval/qasper_subset_enriched.jsonl
data/eval/results_paperpilot.jsonl
```

The runner should support another baseline later, but v1 defaults to:

```text
--baseline paperpilot
```

Each audit case is built by joining on `case_id`:

```text
enriched case:
  question
  oracle_spans
  answers[].free_form_answer
  answers[].yes_no
  answers[].unanswerable
  answers[].evidence
  answers[].highlighted_evidence

result row:
  predicted
  passed
  error
```

### Judge Input

The LLM judge should see:

```text
Question
Gold extractive spans
Gold free-form answers
Gold yes/no or unanswerable flags
Gold evidence paragraphs
Gold highlighted evidence
PaperPilot predicted answer
Strict scorer result
```

It should not see the full paper text by default. The point is to compare the
answer against QASPER's gold answer/evidence, not to redo full-paper reading.

### Judge Output

The judge must return strict JSON:

```json
{
  "semantic_label": "correct",
  "confidence": "high",
  "reason": "The answer states the same fact as the gold evidence."
}
```

Allowed `semantic_label` values are already defined in
`paperpilot/eval/semantic_audit.py`:

```text
correct
partial
incorrect
contradictory
unverifiable
judge_uncertain
```

Allowed confidence values for v1:

```text
high
medium
low
```

### Label Meaning

Use these definitions in the judge prompt:

- `correct`: The predicted answer directly and completely answers the question,
  and is consistent with the gold answer/evidence.
- `partial`: The predicted answer is partly correct but misses a required
  answer part, is too vague, or only answers one item from a multi-item answer.
- `incorrect`: The predicted answer does not answer the question or gives the
  wrong fact.
- `contradictory`: The predicted answer contradicts the gold answer/evidence.
- `unverifiable`: The gold evidence is insufficient for the judge to decide.
- `judge_uncertain`: The judge cannot confidently assign the other labels.

### Output JSONL

Write:

```text
data/eval/semantic_audit_paperpilot.jsonl
```

Each row should follow `SemanticAuditRecord.to_dict()`:

```json
{
  "case_id": "qasper-1909.00694-q1",
  "baseline": "paperpilot",
  "strict_pass": false,
  "semantic_label": "incorrect",
  "confidence": "high",
  "reason": "The prediction is empty.",
  "used_gold_evidence": true,
  "audit_model": "deepseek-chat",
  "audit_version": "v1"
}
```

### Summary Markdown

Write:

```text
data/eval/semantic_audit_summary.md
```

The summary should include:

```text
total audited cases
label counts
strict pass/fail x semantic label matrix
estimated strict false positives
estimated strict false negatives
judge uncertain count
```

Definitions:

```text
strict false positive:
  strict_pass = true and semantic_label in {partial, incorrect, contradictory}

strict false negative:
  strict_pass = false and semantic_label = correct
```

### Resume And Cost Controls

The runner must support:

```text
--limit N
--resume
```

Default behavior:

- `--limit` omitted means run all remaining cases.
- `--resume` skips case IDs already present in output JSONL.
- Results are appended one case at a time.

This matters because LLM judging costs money and can be interrupted.

### Testability

LLM calls should be behind a small function/class that can be replaced with a
fake judge in tests. Unit tests must not call a real LLM.

## Proposed Files

### Create: `paperpilot/eval/semantic_judge.py`

Responsibilities:

- format judge prompt
- parse strict JSON response
- convert judge response into `SemanticAuditRecord`
- expose a `SemanticJudge` class that uses `LLMClient`
- expose pure helper functions for tests

### Create: `scripts/day20_run_semantic_audit.py`

Responsibilities:

- load enriched cases
- load baseline result rows
- join by `case_id`
- skip existing output rows when `--resume` is set
- call `SemanticJudge`
- append `semantic_audit_<baseline>.jsonl`

### Create: `scripts/day20_summarize_semantic_audit.py`

Responsibilities:

- load semantic audit JSONL
- compute label counts
- compute strict/semantic matrix
- compute false positive / false negative counts
- write `semantic_audit_summary.md`

### Create Tests

```text
tests/eval/test_semantic_judge.py
tests/eval/test_run_semantic_audit_script.py
tests/eval/test_summarize_semantic_audit_script.py
```

## Implementation Tasks

### Task 1: Semantic Judge Prompt And Parser

Implement pure helpers first:

- build prompt from enriched case + result row
- extract JSON from model text
- validate label/confidence
- create `SemanticAuditRecord`

Tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval\test_semantic_judge.py -q
```

### Task 2: Audit Runner Script

Implement:

```text
scripts/day20_run_semantic_audit.py
```

CLI:

```powershell
.venv\Scripts\python.exe scripts\day20_run_semantic_audit.py --baseline paperpilot --limit 5 --resume
```

Tests should use a fake judge and temp JSONL files.

### Task 3: Audit Summary Script

Implement:

```text
scripts/day20_summarize_semantic_audit.py
```

CLI:

```powershell
.venv\Scripts\python.exe scripts\day20_summarize_semantic_audit.py --baseline paperpilot
```

Tests should use small fixture audit rows.

### Task 4: Smoke Run With Limit

After unit tests pass, run a small real judge smoke:

```powershell
.venv\Scripts\python.exe scripts\day20_run_semantic_audit.py --baseline paperpilot --limit 3 --resume
.venv\Scripts\python.exe scripts\day20_summarize_semantic_audit.py --baseline paperpilot
```

This requires `DEEPSEEK_API_KEY`. If the key is missing, skip the real LLM smoke
and report that unit tests passed but live audit was not run.

## Validation

Required tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\eval -q
.venv\Scripts\python.exe -m pytest tests\eval tests\test_main_integration.py -q
```

Required safety check:

```powershell
git diff 850ed14 -- paperpilot/eval/scorer.py
```

Expected:

```text
no diff
```

## Risks

1. LLM judge can be inconsistent. Mitigation: preserve strict score and store
   judge output separately.
2. Prompt can bias the judge toward oracle spans. Mitigation: include gold
   evidence and label definitions, and explicitly ask whether the prediction
   answers the question.
3. Gold evidence may be incomplete. Mitigation: allow `unverifiable` and
   `judge_uncertain`.
4. Cost can grow. Mitigation: `--limit`, `--resume`, append-only JSONL.

## Open Confirmation

Before implementation, confirm:

1. Use DeepSeek through the existing `LLMClient` for the judge.
2. Run only a small live smoke first, such as `--limit 3`.
3. Keep generated audit JSONL and summary uncommitted by default.
