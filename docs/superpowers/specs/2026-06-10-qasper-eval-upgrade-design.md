# QASPER Eval Upgrade Design

Date: 2026-06-10

## Background

PaperPilot currently reports a QASPER result of:

```text
paperpilot: 99 / 150 = 66.0%
```

That number is useful, but its meaning is narrow. The current scorer checks
whether the predicted answer contains any gold `extractive_spans` string. It is
a strict substring recall metric over a 150-case extractive subset, not a full
QASPER official evaluation and not a complete measurement of real research QA
quality.

QASPER itself is a full-paper grounded QA dataset. Questions are written after
reading the title and abstract, then answered from the full paper by annotators
who provide answers and supporting evidence. The original data contains richer
answer metadata than PaperPilot currently keeps:

```text
extractive_spans
free_form_answer
yes_no
unanswerable
evidence
highlighted_evidence
```

The eval upgrade should therefore keep the current strict metric for historical
comparison, while adding enough structure to audit false positives, false
negatives, evidence retrieval, and later system-level research quality.

## Goals

1. Preserve the existing strict span score and the current `passed` field.
2. Restore QASPER answer and evidence metadata into an enriched eval subset.
3. Define a semantic audit layer that checks whether strict pass/fail agrees
   with answer meaning.
4. Define an evidence-selection audit layer that can later distinguish
   retrieval failures from synthesis failures.
5. Keep the first implementation slice narrow and non-invasive.

## Non-Goals

The first implementation slice will not:

- Change PaperPilot answer generation prompts.
- Change the deep-read workflow.
- Replace the current `is_pass()` scorer.
- Overwrite historical `results_*.jsonl` files.
- Treat LLM judge output as the only truth.
- Implement the full system-level research QA audit.

## Current Eval Pipeline

The current relevant files are:

```text
paperpilot/eval/qasper_loader.py
paperpilot/eval/scorer.py
paperpilot/eval/baselines.py
scripts/day16_prepare_eval.py
scripts/day16_run_eval.py
scripts/day16_summarize.py
```

Current data flow:

```text
qasper-train-v0.3.json
  -> scripts/day16_prepare_eval.py
  -> data/eval/qasper_subset.jsonl
  -> scripts/day16_run_eval.py
  -> data/eval/results_<baseline>.jsonl
  -> scripts/day16_summarize.py
  -> data/eval/summary.md
```

Current `EvalCase` keeps:

```text
case_id
arxiv_id
paper_title
abstract
full_text
question
oracle_spans
```

Current `is_pass()` returns true when any normalized oracle span appears in the
normalized predicted answer.

## Four-Layer Evaluation Model

### Layer 1: strict_span_score

Purpose:

- Keep the existing exact-span baseline.
- Preserve the README 66.0% number as a comparable historical metric.
- Provide a deterministic, cheap regression signal.

Input:

```text
predicted answer
oracle extractive spans
```

Output:

```text
strict_pass: true | false
```

Design rule:

- Do not change the existing `passed` field in `results_*.jsonl`.
- If future scripts add fields, use additive names such as `strict_pass` rather
  than changing the existing meaning.

### Layer 2: answer_semantic_audit

Purpose:

- Audit whether strict pass/fail agrees with semantic correctness.
- Detect strict-pass false positives.
- Detect strict-fail false negatives.
- Identify partial answers that hit one gold span but miss other required
  answer parts.

Input:

```text
case_id
question
gold answer metadata
gold evidence / highlighted evidence
predicted answer
strict_pass
```

Output labels:

```text
correct
partial
incorrect
contradictory
unverifiable
judge_uncertain
```

Important design decision:

- The semantic audit should not judge from oracle spans alone.
- It should use QASPER gold evidence or highlighted evidence when available.
- This keeps the audit aligned with QASPER's original full-paper grounded QA
  task, instead of reducing it to a stronger keyword check.

Expected summary:

```text
strict pass + semantic correct
strict pass + semantic partial
strict pass + semantic incorrect / contradictory
strict fail + semantic correct
strict fail + semantic partial
strict fail + semantic incorrect
judge uncertain
```

This layer answers:

```text
Is the current 66.0% strict score too harsh, too generous, or mostly reliable?
```

### Layer 3: evidence_selection_audit

Purpose:

- Determine whether PaperPilot found the gold evidence.
- Separate retrieval failures from synthesis failures.
- Explain why a final answer passed or failed.

Input:

```text
QASPER gold evidence / highlighted evidence
PaperPilot retrieved evidence from traces
tool_calls
predicted answer
semantic audit label
```

Output labels:

```text
evidence_found
evidence_missing
evidence_irrelevant
answer_supported
answer_unsupported
```

This layer should eventually answer:

```text
Did ColBERT retrieve the right evidence?
Did the agent use that evidence correctly?
```

This layer is not part of the first implementation slice. It depends on trace
normalization and evidence extraction from tool results.

### Layer 4: system_level_research_audit

Purpose:

- Evaluate PaperPilot as a real research workstation, not only as a QASPER QA
  system.

Future dimensions:

```text
multi-paper comparison
evidence synthesis
report writing
citation reliability
uncertainty handling
user-facing traceability
```

This layer is intentionally out of scope for the QASPER eval upgrade's first
slice.

## First Implementation Slice

The first slice should prepare the data and reporting foundation without
changing the current scoring result.

### 1. Add Enriched Answer Structures

Introduce data structures that can represent QASPER's richer answer metadata.

Proposed fields:

```text
annotation_id
extractive_spans
free_form_answer
yes_no
unanswerable
evidence
highlighted_evidence
```

These can be represented as dataclasses in `paperpilot/eval/qasper_loader.py`
or a new adjacent module if the loader becomes too large.

### 2. Add an Enriched Subset Writer

Add a script or mode that writes:

```text
data/eval/qasper_subset_enriched.jsonl
```

Each row should include the current fields plus answer metadata:

```json
{
  "case_id": "qasper-1909.00694-q1",
  "arxiv_id": "1909.00694",
  "paper_title": "...",
  "abstract": "...",
  "full_text": "...",
  "question": "...",
  "oracle_spans": ["..."],
  "answers": [
    {
      "extractive_spans": ["..."],
      "free_form_answer": "...",
      "yes_no": null,
      "unanswerable": false,
      "evidence": ["..."],
      "highlighted_evidence": ["..."]
    }
  ]
}
```

The existing `qasper_subset.jsonl` should stay valid so old scripts keep
working.

### 3. Add Dataset Profiling

Add a small offline profiler that reads `qasper_subset_enriched.jsonl` and
summarizes:

```text
question count
answer type distribution
number of answers per question
extractive span count distribution
evidence paragraph count distribution
highlighted evidence availability
rough question type distribution
```

Output:

```text
data/eval/qasper_subset_profile.md
```

This gives a factual basis for choosing semantic judge inputs.

### 4. Define Semantic Audit Format

Define the JSONL output format before implementing an LLM judge:

```text
data/eval/semantic_audit_<baseline>.jsonl
```

Proposed row:

```json
{
  "case_id": "qasper-1909.00694-q1",
  "baseline": "paperpilot",
  "strict_pass": true,
  "semantic_label": "partial",
  "confidence": "medium",
  "reason": "The answer mentions one gold label but misses the other.",
  "used_gold_evidence": true,
  "audit_model": null,
  "audit_version": "v1"
}
```

First slice may provide only schema helpers and tests. The actual LLM judge can
be implemented in a later slice after reviewing the enriched subset profile.

## Data Flow After First Slice

```text
qasper-train-v0.3.json
  -> prepare enriched subset
  -> data/eval/qasper_subset_enriched.jsonl
  -> profile enriched subset
  -> data/eval/qasper_subset_profile.md

existing:
qasper_subset.jsonl
  -> run_eval
  -> results_<baseline>.jsonl
  -> summary.md

later:
qasper_subset_enriched.jsonl + results_<baseline>.jsonl
  -> semantic audit
  -> semantic_audit_<baseline>.jsonl
  -> semantic_audit_summary.md
```

## Error Handling

- If QASPER source data is missing, scripts should fail with the existing clear
  "run prepare first" style message.
- If a QA entry has no extractive spans but has other answer types, the enriched
  writer should preserve it only if the selected subset policy includes
  non-extractive answers. For the first slice, keep the current extractive-only
  subset policy and record skipped-answer counts in the profile.
- If evidence fields are missing, write empty arrays and include availability
  counts in the profile.
- If old result files do not contain new fields, audit scripts should derive
  `strict_pass` from `passed`.

## Testing Strategy

Focused tests should cover:

1. QASPER loader preserves answer metadata from fixtures.
2. Existing `load_qasper_cases()` behavior remains unchanged.
3. Enriched JSONL rows are serializable and include current fields.
4. Profile summary counts answer types and evidence availability.
5. Semantic audit schema validates allowed labels.
6. Current scorer tests continue to pass unchanged.

Suggested test files:

```text
tests/eval/test_qasper_loader.py
tests/eval/test_qasper_enriched.py
tests/eval/test_semantic_audit_schema.py
```

## Risks

1. Enriched rows may become large because they include full text and evidence.
   This is acceptable under `data/eval/`, which is already treated as runtime
   eval data.
2. LLM judge can be unstable if introduced too early. The first slice should
   define schema and data readiness before making judge calls.
3. Changing `is_pass()` would break historical comparability. Keep it unchanged.
4. Gold evidence may not fully represent all valid reasoning paths. Semantic
   audit output should be presented as an audit layer, not an absolute truth.

## Acceptance Criteria

The first implementation slice is complete when:

- Existing strict eval scripts still work.
- Existing scorer behavior is unchanged.
- An enriched QASPER subset can be generated from the same source file.
- Enriched rows preserve answer metadata and gold evidence fields.
- A profile markdown file summarizes the enriched subset.
- Semantic audit labels and JSONL schema are defined and test-covered.
- No PaperPilot answer-generation behavior changes.

## References

- QASPER paper: https://arxiv.org/abs/2105.03011
- QASPER dataset card: https://huggingface.co/datasets/allenai/qasper
- Current notes: `docs/learning/2026-06-10-qasper-original-eval-and-paperpilot-scorer.md`
- Current loader: `paperpilot/eval/qasper_loader.py`
- Current scorer: `paperpilot/eval/scorer.py`
- Current runner: `scripts/day16_run_eval.py`
