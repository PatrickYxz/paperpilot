# Query Planner Roadmap Plan

Date: 2026-06-18

Codex only: this document records the working plan for Codex-driven PaperPilot retrieval improvements.

## Background

Recent QASPER evaluation work shows that PaperPilot's accuracy problem is not only a final-answer synthesis problem.

From the 13 inspected cases:

- 6/13 cases already recall the full QASPER gold evidence in the PaperPilot trace.
- 4/13 cases mention only an oracle or answer span but miss the full supporting evidence.
- 3/13 cases look like clearer query retrieval misses where a QASPER-full-text diagnostic probe can retrieve the gold evidence.

This means retrieval is a major bottleneck, but the retrieval issue has at least two different shapes:

1. The query misses the gold evidence entirely.
2. The query reaches nearby answer terms but does not retrieve the full supporting evidence span.

The next development focus should therefore move from final-answer repair to query planning and evidence-targeted retrieval.

## Goal

Build a Query Planner that turns a raw user or eval question into a structured retrieval plan.

The planner should not merely make the query prettier. It should normalize the question into an explicit retrieval task:

- what kind of question this is;
- what answer shape is expected;
- which terms or entities must be preserved;
- which evidence patterns should be searched for;
- which broad but misleading evidence should be avoided;
- what search queries should be issued;
- whether the downstream retrieval step should expand neighboring chunks.

## Non-Goals

- Do not use QASPER oracle spans, highlighted evidence, or gold answers inside the runtime planner.
- Do not tune directly against a single benchmark answer by leaking oracle information.
- Do not replace the existing ColBERT index in V1.
- Do not introduce a new retrieval dependency in V1.
- Do not attempt a large agent-loop rewrite before proving planner recall improvements.

## Design Principles

1. Preserve question intent.
   - Negative questions must stay negative.
   - Comparison questions must preserve the comparison target.
   - Dataset, method, metric, and evaluation questions should not collapse into generic paper-summary queries.

2. Search for evidence, not just topic relevance.
   - A chunk that discusses the same paper section is not enough.
   - The retrieval plan should bias toward text that directly supports the expected answer.

3. Make the plan inspectable.
   - The plan must be traceable in JSON.
   - Eval reports should show planned queries and which query retrieved each chunk.

4. Keep V1 conservative.
   - Prefer deterministic templates plus lightweight validation.
   - Use an LLM planner only behind a fallback and schema validation.
   - If the planner output is invalid, fall back to rule-based templates.

## Proposed Plan Schema

V1 should use a small explicit schema:

```json
{
  "question_type": "negative_or_exception",
  "answer_shape": "list",
  "focus_terms": ["CamemBERT", "tasks"],
  "constraints": {
    "polarity": "negative",
    "needs_numbers": false,
    "needs_comparison": true,
    "needs_table": false
  },
  "must_find": [
    "tasks where CamemBERT does not improve",
    "exceptions to overall improvement",
    "lags behind previous state-of-the-art"
  ],
  "avoid": [
    "general CamemBERT performance improvements without exception details"
  ],
  "queries": [
    {
      "role": "literal",
      "query": "Which tasks does CamemBERT not improve on?"
    },
    {
      "role": "negative_exception",
      "query": "CamemBERT does not improve on tasks"
    },
    {
      "role": "contrast",
      "query": "CamemBERT lags behind previous state-of-the-art tasks"
    },
    {
      "role": "evidence_pattern",
      "query": "except for POS tagging Sequoia Spoken CamemBERT lags behind"
    }
  ],
  "expansion_hints": {
    "neighbor_window": 1,
    "prefer_tables": false,
    "prefer_captions": false
  }
}
```

The exact field names can change during implementation, but the planner should keep these capabilities.

## Question Types for V1

V1 should support these types:

- `dataset_used`
  - Typical cues: dataset, corpus, benchmark, data used, experiment used.
  - Retrieval bias: "we use", "dataset", "corpus", "benchmark", "experiments".

- `method_or_baseline_list`
  - Typical cues: which methods, baselines, approaches, models.
  - Retrieval bias: "baseline", "compared with", "methods", "approaches", "models".

- `metric_or_result`
  - Typical cues: accuracy, F1, score, result, how much, percentage.
  - Retrieval bias: "Table", "results", "score", "accuracy", "F1", "precision", "recall".

- `comparison_or_improvement`
  - Typical cues: better than, compared to, improvement, outperforms.
  - Retrieval bias: "outperforms", "compared with", "previous state-of-the-art", "improvement".

- `negative_or_exception`
  - Typical cues: not, does not, fail, except, worse, lag behind.
  - Retrieval bias: "except", "does not improve", "lags behind", "fails to outperform".

- `advantage_or_contribution`
  - Typical cues: advantage, contribution, proposed model, benefit.
  - Retrieval bias: "advantage", "contribution", "outperforms", "faster", "efficient".

- `evaluation_protocol`
  - Typical cues: how evaluate, manual evaluation, criteria, measured.
  - Retrieval bias: "evaluation", "annotators", "criteria", "scale", "measured by".

- `definition_or_description`
  - Typical cues: what is, describe, how does, architecture.
  - Retrieval bias: literal question plus architecture or definition terms.

## Implementation Phases

### Phase 1: Query Planner V1

Add a planner module that can be tested independently.

Candidate location:

- `paperpilot/eval/query_planner.py` for eval-first implementation.

Expected functions:

- `plan_queries(question: str, title: str = "", abstract: str = "") -> QueryPlan`
- `classify_question_type(question: str) -> str`
- `infer_answer_shape(question: str) -> str`
- `validate_query_plan(plan: QueryPlan, question: str) -> QueryPlan`

V1 implementation approach:

1. Use deterministic rules to classify the question.
2. Extract focus terms conservatively from the question.
3. Generate 4-6 query variants from type-specific templates.
4. Validate that required signals are present.
5. Return a structured plan.

No LLM planner is required in the first implementation. An LLM-based planner can be added after the deterministic baseline is measurable.

### Phase 2: Eval Integration

Wire the planner into PaperPilot eval runs without changing user-facing behavior first.

Candidate integration points:

- `paperpilot/eval/baselines.py`
- `scripts/day16_run_eval.py`

Expected behavior:

- For PaperPilot eval cases, build a query plan before the agent starts retrieval.
- Inject the plan into the prompt as retrieval guidance.
- Record the plan in trace/result output.

Important constraint:

- The planner must not receive oracle spans or highlighted evidence.

### Phase 3: Recall Diagnosis Comparison

Run the same recall diagnosis on the 13 inspected cases.

Measure:

- full gold evidence recalled;
- oracle-only hits;
- current-query misses;
- number of queries per case;
- number of retrieved chunks per case;
- whether answer accuracy improved or regressed.

Initial success target:

- Raise full gold evidence recall from 6/13 to at least 8/13 on the inspected set.
- Reduce oracle-only cases from 4/13 if possible.
- Avoid large increases in irrelevant chunk volume.

### Phase 4: Evidence Expansion

If planner V1 improves complete misses but many oracle-only cases remain, add evidence expansion.

Candidate behavior:

- When a chunk contains a focus term or likely oracle-like answer span, include neighboring chunks.
- For table/caption-like hits, include nearby caption/table context.
- Use `expansion_hints.neighbor_window` from the query plan.

This phase targets cases where retrieval touches the right local area but misses the full supporting evidence.

### Phase 5: Evidence Reranking

If the planner retrieves enough candidate evidence but final answers still drift, add a pre-answer evidence selector.

Candidate behavior:

- Score each retrieved chunk against `must_find`, `avoid`, and `answer_shape`.
- Keep only chunks that directly support the requested answer shape.
- Trigger a second retrieval pass when no chunk directly supports the question.

This should be considered after V1 recall data exists.

## Validation Plan

Unit tests:

- Classification tests for each supported question type.
- Query generation tests for negative, dataset, metric, comparison, and evaluation questions.
- Validation tests for invalid or too-short plans.
- Tests that oracle/gold evidence is never required by planner APIs.

Smoke tests:

- Run planner over the 13 diagnosis cases.
- Confirm the output plan is valid JSON-like structured data.
- Confirm negative questions include negative/exception-oriented queries.
- Confirm metric questions include result/table/score-oriented queries.

Evaluation tests:

- Rerun the 13 inspected cases with planner guidance.
- Generate a new retrieval recall diagnosis report.
- Compare against:
  - `docs/retrieval_recall_diagnosis_20260616.md`
  - `docs/retrieval_recall_diagnosis_20260616_rerun_10.md`

## Risks

- The planner may increase recall but also increase irrelevant retrieved evidence.
- Query templates may overfit English QASPER wording.
- LLM answer synthesis may still ignore the best retrieved evidence.
- Negative questions are especially fragile because a broad performance query can retrieve the opposite of what is needed.
- Too many queries can slow eval runs and add noise.

## Current Recommendation

Start with deterministic Query Planner V1.

Do not add an LLM planner yet. The first measurable question is whether structured query expansion alone improves recall on the 13 inspected cases.

If V1 improves recall, then add evidence expansion.

If V1 does not improve recall, inspect:

- whether query terms are still too broad;
- whether index chunk boundaries hide evidence;
- whether table/caption handling is the true bottleneck;
- whether the runtime arXiv source differs from QASPER full text.

## Immediate Next Task

Implement Phase 1:

1. Add `paperpilot/eval/query_planner.py`.
2. Add focused tests.
3. Add a small script to print plans for selected case ids.
4. Run planner over the 13 diagnosis cases and inspect the generated plans before wiring it into agent execution.
