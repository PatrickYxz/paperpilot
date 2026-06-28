# Query Planner Phase 3: Requirement Evidence Verifier Design

Date: 2026-06-28

## Context

Query Planner v1 added a structured plan, forced multi-query retrieval, evidence
pool deduplication, and `mcp__colbert__planned_retrieval`. Recent QASPER problem
case reruns showed that this changed retrieval from ad hoc search into an
observable pipeline, but it did not yet make `summary_items` reliably answer
the question.

Observed failures:

- Dataset/entity questions retrieved related regions, but mixed source datasets,
  constructed datasets, and candidate datasets. Example: `HotpotQA`, `R4C`, and
  `WikiHop` all appeared in plausible evidence.
- List questions found comparison regions, but summary evidence did not reliably
  cover the complete governed list.
- Numeric comparison questions retrieved tables and comparison text, but answer
  synthesis mixed metrics, subsets, comparators, and downstream-task numbers.

The root issue is that current `summary_items` are selected by query target,
ColBERT score, and simple diversity. A chunk inherits a requirement target
because a query targeted that requirement; there is no verification that the
chunk directly supports the requirement.

## Goal

Add an optional requirement-level evidence verification layer that can rerank
and select evidence by direct support for each `EvidenceRequirement`.

The goal is not to increase raw recall. The goal is to make the evidence shown
to the agent cleaner, more requirement-grounded, and easier to evaluate.

Phase 3 should support this comparison:

```text
planned retrieval v1:
  ColBERT score + weak summary selection

planned retrieval v1.5:
  ColBERT retrieval + evidence pool + requirement verifier + verified summary
```

## Non-Goals

- Do not train or fine-tune a small reranker in this phase.
- Do not replace ColBERT retrieval.
- Do not make verified retrieval the default production path immediately.
- Do not remove existing `summary_items`; the raw v1 behavior remains available.
- Do not make the final answer generator depend on gold answers or oracle spans.

## Design Decision

Use option C:

```text
Implement the verifier as a real retrieval component,
but keep it disabled by default in planned_retrieval.
Enable it explicitly in eval runs.
```

This avoids baking an eval-only feature into scripts while also avoiding a
surprise latency/cost increase for normal agent runs.

## Architecture

Current flow:

```text
QueryPlan
  -> planned ColBERT searches
  -> raw hits
  -> evidence pool dedupe
  -> weak summary selection
  -> agent
```

Phase 3 flow:

```text
QueryPlan
  -> planned ColBERT searches
  -> raw hits
  -> evidence pool dedupe
  -> weak summary selection
  -> optional requirement verifier
  -> verified summary selection
  -> agent/eval
```

The verifier is optional. When disabled, the existing behavior and payload shape
remain compatible. When enabled, `planned_retrieval` returns additional
verification metadata and may replace the summary text with verified evidence.

## Components

### `paperpilot/retrieval/evidence_verifier.py`

New module responsible for:

- building verifier prompts;
- parsing verifier JSON;
- normalizing verifier decisions;
- scoring decisions for reranking;
- selecting verified evidence by requirement.

It should not call ColBERT or MCP directly. It operates on `QueryPlan` and
`EvidencePool`.

### `EvidenceVerificationDecision`

Dataclass-like structure:

```python
EvidenceVerificationDecision(
    requirement_id: str,
    evidence_id: str,
    support: str,        # "direct" | "partial" | "no"
    confidence: str,     # "high" | "medium" | "low"
    answer_atoms: list[str],
    risks: list[str],
    reason: str,
)
```

Meanings:

- `direct`: the chunk directly supports the requirement.
- `partial`: the chunk supports part of the requirement but is incomplete,
  ambiguous, or needs another chunk.
- `no`: the chunk is related but does not support the requirement.
- `answer_atoms`: entities, methods, numbers, metrics, comparators, or list
  items directly supported by the chunk.
- `risks`: concise warnings such as `source_dataset_not_answer`,
  `mentions_related_work`, `mixed_metric`, or `partial_list`.

### `EvidenceVerificationResult`

Dataclass-like structure:

```python
EvidenceVerificationResult(
    enabled: bool,
    method: str,
    decisions: list[EvidenceVerificationDecision],
    verified_summary_items: list[str],
    missing_verified_requirements: list[dict],
    conflicts: list[dict],
    stats: dict[str, int],
)
```

`verified_summary_items` contains evidence ids ordered for display to the agent.

### `VerificationClient`

The first verifier implementation uses the existing `LLMClient` interface. Tests
should use fake clients. The verifier should be written so a future cross-encoder
or small-model verifier can be swapped in behind the same decision schema.

## Verifier Prompt

The prompt evaluates one requirement against a small candidate set, not the whole
paper. The candidate set should usually be the top deduped evidence items that
target the requirement, capped to control cost.

Inputs:

- user question;
- question type and answer shape;
- requirement id and description;
- relevant query plan constraints;
- candidate evidence ids, matched queries, score, and chunk text.

The prompt must ask for JSON only.

Output schema:

```json
{
  "decisions": [
    {
      "requirement_id": "req_1",
      "evidence_id": "ev_3",
      "support": "direct",
      "confidence": "high",
      "answer_atoms": ["WikiHop"],
      "risks": [],
      "reason": "The chunk states that WikiHop is the dataset used."
    }
  ]
}
```

Verifier rules:

- Judge support for the requirement, not general topical relevance.
- Use only the candidate chunk text and query plan metadata.
- Do not use oracle spans or benchmark gold answers.
- Mark `partial` when the chunk is relevant but missing a required relation,
  comparator, metric, or list coverage.
- Mark `no` when the chunk is only related work, background, nearby context, or
  a different entity role.
- For numeric questions, the metric/subset/comparator/value relation must be
  present in the same chunk or explicitly linked neighboring evidence.
- For list questions, extract only list items governed by the question phrase.
- For dataset/entity questions, distinguish source dataset, constructed dataset,
  evaluation dataset, related-work dataset, and candidate dataset when possible.

## Reranking and Selection

The verifier decision is the primary ranking signal. ColBERT score is a
tie-breaker.

Initial scoring:

```text
direct + high      -> 100
direct + medium    -> 80
direct + low       -> 65
partial + high     -> 60
partial + medium   -> 45
partial + low      -> 30
no                 -> 0
```

Final score:

```text
verified_score = support_confidence_score + normalized_colbert_score * 10
```

The exact constants can be changed later, but the invariant is:

```text
semantic support outranks retrieval similarity
```

### Requirement-Level Selection

Selection must be grouped by requirement, not global top-k.

For each required requirement:

1. Select up to 2 `direct` evidence items, highest verified score first.
2. If no direct evidence exists, select up to 1 `partial` evidence item.
3. If only `no` evidence exists, select none and record
   `missing_verified_requirements`.
4. Record conflicts when high-confidence direct evidence for the same
   requirement contains incompatible `answer_atoms`.

After per-requirement selection, fill remaining `summary_k` slots with
high-scoring direct evidence not already selected. Do not add `no` evidence to
verified summaries.

### Question-Type Notes

Dataset/entity:

- Prefer chunks that state the relation asked by the question.
- Record conflict if multiple answer atoms appear in direct evidence with
  different roles.

List/method:

- Prefer chunks that contain a governed list.
- Multiple partial chunks can be selected when together they cover different
  list atoms.
- Do not treat sub-variants as separate list items unless the requirement asks
  for variants.

Numeric/comparison:

- Prefer chunks where metric, subset, comparator, and value appear together.
- Do not mix numbers across unrelated datasets, downstream tasks, or model
  variants.
- `answer_atoms` should include structured strings such as
  `WNLaMPro-medium`, `BERTbase`, `58%`, `Attentive Mimicking`.

## Payload Changes

When verification is disabled:

- Existing `planned_retrieval` payload remains unchanged.

When enabled:

```json
{
  "summary_text": "... verified evidence summary ...",
  "evidence_pool": {
    "items": [...],
    "summary_items": ["ev_1", "ev_2"],
    "verified_summary_items": ["ev_5", "ev_7"],
    "verification": {
      "enabled": true,
      "method": "llm_requirement_verifier_v1",
      "decisions": [...],
      "missing_verified_requirements": [],
      "conflicts": [],
      "stats": {
        "decision_count": 12,
        "direct_count": 4,
        "partial_count": 3,
        "no_count": 5
      }
    }
  }
}
```

`summary_items` remains the original v1 selection. `verified_summary_items`
records the verifier-based selection. `summary_text` may use verified evidence
when the tool is called with `verify_evidence=true`.

## MCP Tool Interface

Extend `mcp__colbert__planned_retrieval`:

```python
def planned_retrieval(
    question: str,
    paper_id: str,
    paper_title: str = "",
    abstract: str = "",
    top_k_each: int = 5,
    summary_k: int = 8,
    verify_evidence: bool = False,
    verifier_candidate_k: int = 6,
) -> dict:
```

Defaults preserve existing behavior.

When `verify_evidence=false`, no verifier is called.

When `verify_evidence=true`, the tool verifies requirement candidates and
returns verification metadata.

## Eval Integration

Extend rerun scripts with:

```text
--verify-evidence
```

Eval runs with this flag should instruct the agent or MCP call path to enable
verification. Because the main agent chooses tool arguments, the safest v1.5 eval
path is to add explicit prompt guidance in `run_paperpilot` when the eval option
is enabled:

```text
When calling mcp__colbert__planned_retrieval, set verify_evidence=true.
```

Result JSONL should include:

- `evidence_verification_used`
- `verified_summary_count`
- `direct_support_count`
- `partial_support_count`
- `unsupported_count`
- `missing_verified_requirements`
- `verification_conflicts`

## Error Handling

Verifier failure must not fail retrieval.

If verifier call or parsing fails:

- return original evidence pool and summary;
- include `verification.enabled=true`;
- include `verification_error`;
- set `verified_summary_items=[]`;
- leave `summary_items` unchanged.

If some requirements fail verification:

- record missing or partial support;
- do not fabricate evidence;
- allow agent/eval to answer conservatively.

## Cost Controls

To avoid unbounded LLM calls:

- cap candidates per requirement with `verifier_candidate_k`;
- default `verifier_candidate_k=6`;
- verify only deduped evidence items that target the requirement;
- truncate chunk text in verifier prompts to a fixed character budget;
- run one verifier call per requirement, not one call per chunk.

## Testing

Unit tests:

- parse valid verifier JSON;
- invalid verifier JSON becomes conservative `no` or records verifier error;
- direct/high outranks partial/high and high ColBERT score;
- per-requirement selection keeps coverage across requirements;
- `no` evidence is never selected into `verified_summary_items`;
- conflicts are recorded when direct evidence has incompatible answer atoms;
- `planned_retrieval(... verify_evidence=False)` keeps existing payload behavior;
- `planned_retrieval(... verify_evidence=True)` includes verification payload.

Eval tests:

- rerun script records verification metadata.
- evidence extractor can read verified summary items if present.

Smoke tests:

- Run the three known problem cases:
  - `qasper-1910.04601-q1`
  - `qasper-1701.00185-q1`
  - `qasper-1910.07181-q0`
- Compare v1 vs v1.5:
  - selected evidence atoms;
  - missing verified requirements;
  - conflicts;
  - final answer behavior.

## Success Criteria

Phase 3 is successful if the three known problem cases show a better evidence
selection story even before full benchmark improvements:

- Dataset case: verifier identifies dataset role ambiguity instead of blindly
  promoting the most similar dataset mention.
- List case: verifier either selects the governed comparison list or records
  partial/missing coverage instead of presenting noisy method variants as a
  complete list.
- Numeric case: verifier selects evidence where metric, subset, comparator, and
  value stay together, or marks the answer evidence insufficient.

Full QASPER pass-rate improvement is a later measurement, not the primary gate
for this spec.

## Rollout

1. Implement verifier schema, parser, and selection as pure retrieval code.
2. Integrate optional verifier hook into planned retrieval.
3. Add MCP flags with defaults disabled.
4. Add eval flag and JSONL metadata.
5. Re-run the three problem cases with v1.5.
6. Decide whether to enable verifier in normal `deep-read-paper` after observing
   latency, cost, and evidence quality.

## Open Follow-Up After Phase 3

If LLM verification improves evidence selection, save verifier decisions as
candidate training data for a future small reranker:

```text
(question, requirement, chunk) -> direct / partial / no
```

Only after this data exists should we evaluate a cross-encoder or fine-tuned
small model.
