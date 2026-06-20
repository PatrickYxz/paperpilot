# Query Planner v1 Design

Date: 2026-06-20

## Background

PaperPilot's current query planner experiment is Phase 2 prompt guidance. It builds a deterministic `QueryPlan` in `paperpilot/eval/query_planner.py`, formats the plan into natural-language instructions in `paperpilot/eval/baselines.py`, and relies on the agent to decide whether to run the suggested searches.

That is useful as a smoke test, but it is not a strong query planner baseline. The planned searches are not guaranteed to execute, retrieved evidence is not pooled as structured data, and query provenance is only indirectly recoverable from traces.

The next design should move query planning into the real PaperPilot deep-read workflow, while keeping the first implementation smaller than a full ColBERT retrieval refactor.

## Goals

- Introduce an LLM JSON query planner for paper QA.
- Validate planner output through a strict schema and fallback path.
- Force planned queries to execute once `planned_retrieval` is invoked.
- Collect retrieval results into a structured `EvidencePool`.
- Preserve query-to-evidence provenance.
- Return a compact, requirement-aware evidence summary for answer generation.
- Keep reranking as a later extension point.

## Non-Goals

- Do not build a reranker in v1.
- Do not fine-tune a small ranking model in v1.
- Do not implement semantic or fuzzy deduplication in v1.
- Do not refactor ColBERT indexing or chunking in v1.
- Do not add table/caption-specific retrieval in v1.
- Do not force planner execution from `agent_loop` in v1.

## Architecture

Query Planner v1 is a retrieval pre-stage in the real deep-read workflow.

```text
User question + paper metadata
  -> LLM QueryPlanner
  -> PlanValidator
  -> planned_retrieval tool
      -> execute literal query
      -> execute planned queries
      -> collect raw ColBERT chunks
      -> exact dedupe
      -> requirement-aware summary selection
  -> deep-read answer workflow
      -> answer from EvidencePool summary
      -> optional follow-up search if needed
```

The first integration point is a new `planned_retrieval` tool plus an update to the `deep-read-paper` skill. The skill should instruct the agent to call `planned_retrieval` after downloading the paper and building the index.

This design intentionally avoids modifying `agent_loop` at first. The v1 guarantee is:

- if `planned_retrieval` is called, all valid planned queries execute;
- deep-read guidance should make `planned_retrieval` the normal first retrieval step;
- if later evaluation shows the agent often skips the tool, a v2 can move planner triggering into the loop or run layer.

## QueryPlan Schema

The planner should output JSON with this shape:

```json
{
  "version": "query_plan_v1",
  "question": "original user question",
  "question_type": "dataset_used | method_list | metric_result | comparison | yes_no | definition | evidence_location | other",
  "answer_shape": "single_entity | list | number | comparison | yes_no | freeform",
  "intent_summary": "short summary of what must be answered",
  "focus_terms": ["key entities, methods, datasets, metrics"],
  "constraints": {
    "needs_numbers": false,
    "needs_comparison": false,
    "needs_table_or_figure": false,
    "polarity": "neutral | negative | contrastive"
  },
  "evidence_requirements": [
    {
      "id": "req_1",
      "description": "what evidence must directly show",
      "required": true
    }
  ],
  "queries": [
    {
      "id": "q_lit",
      "role": "literal",
      "query": "original user question",
      "targets": ["req_1"],
      "priority": 1
    },
    {
      "id": "q_1",
      "role": "focused_rewrite",
      "query": "dataset used experiment corpus benchmark",
      "targets": ["req_1"],
      "priority": 2
    }
  ],
  "avoid": [
    "related work mentions not used by the paper itself"
  ],
  "expansion_hints": {
    "neighbor_window": 1,
    "prefer_tables": false,
    "prefer_captions": false
  }
}
```

### Field Semantics

- `question_type` describes the user's intent.
- `answer_shape` describes the expected answer form.
- `intent_summary` gives a short plain-language interpretation of the question.
- `focus_terms` captures entities, methods, datasets, metrics, tasks, or other terms that searches should preserve.
- `constraints` captures retrieval-relevant constraints such as numeric evidence, comparison, table/figure needs, and polarity.
- `evidence_requirements` lists the evidence types needed for a trustworthy answer.
- `queries` contains executable retrieval units.
- `queries[].targets` maps a query to one or more evidence requirements.
- `avoid` captures common false-positive traps.
- `expansion_hints` records future retrieval hints. In v1 these hints may be stored without full execution.

The important distinction is that `queries` are executable retrieval actions, while `evidence_requirements` describe what must be covered by the returned evidence.

## Plan Validation

`PlanValidator` validates and normalizes planner output before execution.

Rules:

- JSON must parse into an object.
- `version` must be `query_plan_v1`.
- At least one `evidence_requirement` must exist.
- At least one literal query must exist.
- Query count is capped at 6.
- Query text must be non-empty and within a conservative length limit.
- Query ids and requirement ids must be unique after normalization.
- All `queries[].targets` must point to existing requirements.
- Invalid or missing `targets` are removed if the query is still useful.
- If no valid queries remain, fallback is used.

Fallback plan:

```text
version = query_plan_v1
question_type = other
answer_shape = freeform
one evidence requirement = direct evidence answering the question
one literal query = original question
```

The fallback is intentionally simple. Its job is to keep the retrieval pipeline operational, not to be clever.

## PlannedRetrievalExecutor

`PlannedRetrievalExecutor` receives a validated `QueryPlan`, `paper_id`, and retrieval settings.

Responsibilities:

- execute the literal query;
- execute planned queries in priority order;
- call the existing single-query ColBERT search for each query;
- record per-query success or failure;
- collect results into a raw evidence list;
- attach query provenance to every retrieved chunk.

For each result, the executor should record:

- `source_query_id`
- `source_query_role`
- `source_query`
- `source_rank`
- `source_score`
- `target_requirements`

The current ColBERT search result returns `paper_id`, `chunk_text`, and `score`. A future small interface improvement should return `chunk_id` too, because `IndexManager` internally already has chunk ids like `paper_id::chunk_i`. This is recommended but not required for v1.

## EvidencePool

`EvidencePool` is the structured container for all planned retrieval results.

Example shape:

```json
{
  "plan_id": "trace-or-generated-id",
  "question": "original user question",
  "query_plan": {},
  "items": [
    {
      "id": "ev_1",
      "paper_id": "arxiv:1910.04601",
      "chunk_text": "retrieved chunk text",
      "best_score": 23.7,
      "matched_queries": [
        {
          "query_id": "q_1",
          "query": "WikiHop dataset used experiment",
          "role": "focused_rewrite",
          "rank": 1,
          "score": 23.7,
          "targets": ["req_1"]
        }
      ]
    }
  ],
  "summary_items": ["ev_1", "ev_4"],
  "missing_requirements": [],
  "stats": {
    "query_count": 4,
    "raw_result_count": 20,
    "deduped_count": 12
  }
}
```

### Deduplication

Pool-level deduplication is exact-only in v1.

Rules:

- normalize whitespace in `chunk_text`;
- build a key from `paper_id + normalized_chunk_text_hash`;
- merge items with the same key;
- preserve all query provenance in `matched_queries`;
- keep the highest score as `best_score`.

No fuzzy, semantic, or high-overlap deduplication is performed at the pool level in v1.

### Summary Selection

The full pool keeps all deduped evidence items. The summary shown to the answering model should be smaller and more diverse.

Summary selection rules:

1. Iterate through required evidence requirements.
2. For each required requirement, select one best candidate evidence item.
3. Avoid selecting a candidate whose token overlap with already selected summary items is above the diversity threshold.
4. If no candidate exists for a requirement, record it in `missing_requirements`.
5. After requirement coverage, fill remaining summary slots by best score while preserving diversity.

The overlap check is summary-level suppression only. Suppressed items remain in `EvidencePool.items`.

`missing_requirements` shape:

```json
{
  "requirement_id": "req_dataset",
  "description": "dataset where the result was measured",
  "reason": "no_targeted_query | no_candidates"
}
```

Reasons:

- `no_targeted_query`: no valid query targeted the requirement.
- `no_candidates`: targeted queries ran, but no usable evidence item was available for the requirement.

Missing requirements do not block answer generation. They should make the answer workflow conservative about unsupported parts of the question.

## planned_retrieval Tool

The tool signature should be conceptually:

```text
planned_retrieval(question, paper_id, paper_title?, abstract?, top_k_each?, summary_k?)
```

The tool should:

1. call the LLM planner;
2. validate or fallback the plan;
3. execute planned retrieval;
4. build the evidence pool;
5. return compact evidence for the agent;
6. include full structured data in traces.

The tool response should include two layers:

- `summary_text`: compact evidence summary for the answering model;
- `evidence_pool`: structured JSON for traces, eval, and later reranking.

Example `summary_text`:

```text
Planned retrieval completed.
Queries executed: 5
Evidence chunks: 14 raw, 9 deduped
Missing requirements: none

Top evidence:
[ev_1] Found by q_1, q_lit. Score 23.71.
<chunk text>

[ev_4] Found by q_3. Score 20.04.
<chunk text>
```

## Answer Workflow

The `deep-read-paper` skill should be updated so that planned retrieval is the default first retrieval step:

```text
load_skill
download_paper
build_index
planned_retrieval
answer from EvidencePool summary
optional follow-up colbert.search
final answer
```

Rules for answer behavior:

- Prefer evidence from the `planned_retrieval` summary.
- Use follow-up `colbert.search` only to fill gaps or clarify missing requirements.
- If `missing_requirements` is non-empty, answer only the supported parts confidently.
- Do not invent missing details.
- If evidence is insufficient for part of the question, say so.

## Error Handling

- Planner LLM call fails: use fallback plan.
- Planner returns invalid JSON: attempt one repair or extraction pass; if still invalid, use fallback.
- Validator rejects all queries: use fallback plan.
- One planned query fails: record query error and continue remaining queries.
- All planned queries fail: return an empty evidence pool with query errors.
- Missing requirements: record and continue.
- Empty evidence pool: agent may run follow-up search or report insufficient evidence.

## Trace and Evaluation

Traces should include:

- raw planner response;
- validated `QueryPlan`;
- fallback status and reason;
- executed query list;
- per-query search errors;
- raw result count;
- deduped result count;
- `EvidencePool`;
- summary item ids;
- missing requirements.

Initial eval should focus on retrieval quality before answer quality:

- gold evidence enters full pool;
- gold evidence enters summary;
- missing requirement count;
- number of executed queries;
- answer correctness after planned retrieval;
- comparison against Phase 2 prompt-guidance traces.

## Testing Plan

Unit tests:

- valid plan schema passes validation;
- malformed JSON falls back;
- missing literal query is repaired or fallback is used;
- query count is capped;
- invalid targets are removed or rejected;
- exact duplicate chunks are merged;
- merged chunks preserve `matched_queries`;
- requirement-aware summary selects coverage before global score fill;
- missing requirements are recorded;
- overlap suppression affects summary only, not full pool.

Integration tests:

- `planned_retrieval` executes multiple planned queries;
- per-query provenance appears in evidence items;
- tool response includes `summary_text` and structured `evidence_pool`;
- trace includes query plan and evidence pool.

Evaluation:

- rerun the QASPER 13-case recall diagnosis;
- compare Phase 2 prompt-guidance against planned retrieval;
- inspect cases where gold evidence enters full pool but not summary;
- inspect cases where requirements are missing.

## Future Extensions

After v1 is validated, possible next steps are:

- force planner execution from `agent_loop` or `paperpilot.main.run` for deep-read tasks;
- return `chunk_id` from ColBERT search;
- add neighbor expansion around selected chunks;
- add table/caption-aware retrieval;
- add heuristic reranking;
- add LLM-based evidence grading;
- add a cross-encoder or fine-tuned small reranker.

