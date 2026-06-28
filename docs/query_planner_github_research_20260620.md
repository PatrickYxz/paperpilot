# Query Planner GitHub Research Notes

Date: 2026-06-20

This note summarizes the GitHub research done before continuing the PaperPilot query planner design. The goal was to check how adjacent open-source RAG, paper QA, and research-agent projects handle query planning, forced retrieval execution, evidence pooling, and ranking.

## Current PaperPilot Context

PaperPilot's Phase 2 query planner is currently a prompt-guidance experiment:

- `paperpilot/eval/query_planner.py` builds a deterministic `QueryPlan`.
- `paperpilot/eval/baselines.py` formats that plan into natural-language retrieval guidance.
- The agent is still expected to decide whether and how to call `colbert.search`.
- The ColBERT retrieval tool itself still exposes a simple single-query interface: `search(query, paper_id, top_k)`.

This means Phase 2 does not yet prove a real query-planning architecture. It only tests whether adding planning guidance to the prompt influences the model's tool-use behavior.

The GitHub research below strongly suggests that mature systems usually make generated queries or sub-questions part of the executable retrieval pipeline, instead of leaving them as optional prompt advice.

## Projects Reviewed

### LangChain MultiQueryRetriever

Source:

- https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/retrievers/multi_query.py

Relevant mechanism:

- The retriever uses an LLM chain to generate multiple query variants for a user question.
- It optionally includes the original query.
- It then executes the underlying retriever once per generated query.
- It flattens all retrieved documents and returns a unique union.
- Async execution uses `asyncio.gather`; sync execution loops through each query.

Important design point:

LangChain does not merely tell the final answer model to consider alternative searches. The generated queries become actual retriever calls.

PaperPilot implication:

This maps directly to a `PlannedRetrievalExecutor` concept:

1. planner emits `planned_queries`;
2. executor calls `colbert.search` for each query;
3. evidence pool deduplicates and records which query found each chunk.

This is the closest lightweight pattern for PaperPilot's next step.

### LlamaIndex Query Transform and SubQuestionQueryEngine

Sources:

- https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/indices/query/query_transform/base.py
- https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/query_engine/sub_question_query_engine.py

Relevant mechanisms:

- `BaseQueryTransform` explicitly runs before the query is sent to the index.
- `HyDEQueryTransform` generates a hypothetical answer/document and uses it as an embedding string, optionally keeping the original query too.
- `DecomposeQueryTransform` generates a new query from the original query and index summary.
- `SubQuestionQueryEngine` breaks complex questions into sub-questions, runs each sub-question against a target query engine, gathers answers and sources, then synthesizes a final response.

Important design point:

LlamaIndex separates query transformation from response synthesis. A transformed query is not simply descriptive metadata; it becomes the query object used by retrieval.

PaperPilot implication:

PaperPilot's query planner schema should separate:

- question analysis: answer shape, constraints, focus terms;
- executable retrieval units: literal query, rewritten queries, sub-questions;
- evidence requirements: what kind of evidence would satisfy the answer.

This suggests that the planner output should include fields that the executor can run directly, not only explanatory guidance for the LLM.

### Future-House PaperQA

Sources:

- https://github.com/Future-House/paper-qa/blob/d7675d7b7eddeb3535e8c260399c5bbeeb818c50/src/paperqa/docs.py
- https://github.com/Future-House/paper-qa/blob/d7675d7b7eddeb3535e8c260399c5bbeeb818c50/src/paperqa/settings.py
- https://github.com/Future-House/paper-qa/blob/d7675d7b7eddeb3535e8c260399c5bbeeb818c50/src/paperqa/agents/tools.py

Relevant mechanisms:

- `AnswerSettings` has explicit knobs such as `evidence_k`, `evidence_retrieval`, `evidence_relevance_score_cutoff`, `answer_max_sources`, and `max_concurrent_requests`.
- `Docs.retrieve_texts` performs MMR search over the internal text index. It requests `k`, fetches `2 * k`, filters deleted docs, and returns top matches.
- `Docs.aget_evidence` retrieves candidate text chunks, summarizes them against the question, filters out irrelevant or failed contexts, deduplicates contexts, and appends them to the session.
- `Docs.aquery` lazily gathers evidence if no contexts exist, then serializes selected contexts into the final QA prompt.
- Agent tools are explicit: `paper_search`, `gather_evidence`, `gen_answer`, `reset`, and `complete`.
- `GatherEvidence.gather_evidence` accepts a specific question, calls `Docs.aget_evidence`, and reports how many pieces of evidence were added.

Important design point:

PaperQA treats evidence gathering as an explicit stage and tool, separate from final answer generation. It also has configuration for how many evidence pieces to retrieve and how many sources to use in the final answer.

PaperPilot implication:

PaperPilot should likely add an explicit evidence gathering layer rather than relying on the model to organically collect enough chunks through free-form tool calls.

Potential mapping:

- `QueryPlanner` decides what evidence is needed.
- `PlannedRetrievalExecutor` gathers evidence.
- `EvidencePool` deduplicates and keeps provenance.
- answer synthesis receives a curated evidence context.

PaperQA also supports the idea that reranking or evidence scoring belongs after retrieval and before final answer synthesis.

### Stanford OVAL STORM

Sources:

- https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/README.md
- https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/knowledge_storm/storm_wiki/engine.py
- https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/knowledge_storm/interface.py

Relevant mechanisms:

- STORM splits long-form article generation into a pre-writing research stage and a writing stage.
- The research stage collects references and generates an outline before article writing.
- STORM uses perspective-guided question asking and simulated conversation to improve the breadth and depth of research questions.
- `STORMWikiRunnerArguments` exposes retrieval-related parameters such as `max_search_queries_per_turn`, `search_top_k`, `retrieve_top_k`, and `max_thread_num`.
- `STORMWikiRunner.run_knowledge_curation_module` runs the knowledge curation module and persists `conversation_log.json` and `raw_search_results.json`.
- The generic `Retriever.retrieve` accepts either a string query or a list of queries, executes each query through the retrieval backend, and attaches the originating query into the retrieved information metadata.

Important design point:

STORM explicitly distinguishes research/knowledge curation from writing. It also records the query that produced each retrieved information item.

PaperPilot implication:

For QASPER-style paper QA, PaperPilot does not need STORM's full multi-perspective conversation system yet. But it should borrow two principles:

1. do evidence collection before answer generation;
2. preserve provenance from query to retrieved evidence.

This supports making query provenance a first-class field in PaperPilot's evidence pool.

### Microsoft GraphRAG

Source:

- https://github.com/microsoft/graphrag/blob/main/README.md

Relevant mechanisms:

- GraphRAG is described as a data pipeline and transformation suite for extracting structured data from unstructured text using LLMs.
- It is more focused on graph-style indexing and structured memory than per-question multi-query planning.
- The README emphasizes that indexing can be expensive and that prompt tuning is often necessary.

Important design point:

GraphRAG is less directly applicable to the current PaperPilot query planner problem, because PaperPilot is currently debugging per-paper passage recall and evidence selection rather than graph construction.

PaperPilot implication:

GraphRAG is better treated as a long-term reference for richer document structure, not as the immediate Phase 3 template.

## Cross-Project Patterns

The common pattern across the closest projects is:

1. Generate or transform queries before retrieval.
2. Execute those generated queries programmatically.
3. Merge the results.
4. Deduplicate evidence.
5. Preserve source/provenance.
6. Select or rank evidence before synthesis.
7. Only then generate the final answer.

The key contrast with PaperPilot Phase 2 is that Phase 2 stops after step 1 and compresses the plan into prompt guidance.

## Design Implications for PaperPilot

### Do not treat Phase 2 as a strong baseline

Phase 2 is useful as a smoke test, but it should not be considered a serious query planner baseline because:

- planned searches are not guaranteed to run;
- the retrieval tool remains single-query and model-driven;
- there is no evidence pool abstraction;
- source query provenance is only indirectly available through traces;
- evidence deduplication and reranking are not part of the planner path.

### Prefer a real product-path planner over eval-only patching

The next architecture should target the real PaperPilot workflow, not just eval scripts.

The cleanest next layer is:

```text
User question
  -> QueryPlanner
  -> PlannedRetrievalExecutor
  -> EvidencePool
  -> Answer workflow
```

The current ColBERT tool can remain unchanged initially. The executor can call the existing single-query tool repeatedly. This avoids a broad retrieval-server refactor while still making planned retrieval real.

### Keep reranking as a later stage

Reranking is likely useful, but it should be added after the planner and evidence pool boundaries are stable.

Suggested progression:

1. implement structured planning and forced multi-query execution;
2. implement evidence pooling, dedupe, and provenance;
3. run QASPER recall diagnostics;
4. decide whether a reranker is needed;
5. only then consider a small reranker model or fine-tuned reranker.

This avoids optimizing ranking before confirming that the right candidate evidence is being retrieved into the pool.

## Candidate PaperPilot Architecture

### QueryPlanner

Purpose:

- analyze the user question;
- classify the expected answer shape;
- identify entities, constraints, focus terms, and evidence requirements;
- generate executable search queries.

Likely schema fields:

- `question_type`
- `answer_shape`
- `focus_terms`
- `constraints`
- `must_find`
- `avoid`
- `planned_queries`
- `evidence_requirements`
- `expansion_hints`

The user preference from the design discussion is to move toward an LLM JSON planner rather than a purely deterministic planner.

### PlannedRetrievalExecutor

Purpose:

- run the literal question query;
- run each planned query;
- control `top_k` per query;
- call the current retrieval tool repeatedly;
- collect all retrieved chunks into a single pool.

It should not depend on the model choosing to execute searches. The executor owns that responsibility.

### EvidencePool

Purpose:

- normalize retrieved chunks;
- deduplicate repeated chunks;
- keep source query provenance;
- keep original score and rank;
- optionally preserve query role and evidence requirement tags.

Candidate item fields:

- `paper_id`
- `chunk_id` if available
- `chunk_text`
- `score`
- `source_query`
- `source_query_role`
- `source_rank`
- `matched_requirements`
- `duplicate_group_id`

### Reranker

Purpose:

- reorder the evidence pool after multi-query retrieval.

This should remain optional in the initial planner design. It can later be implemented as:

- heuristic reranking;
- LLM-based evidence selector;
- cross-encoder reranker;
- fine-tuned small reranker model.

## Recommendation

The next serious design should follow Option B from the discussion:

Build query planning into the real PaperPilot product path, not only eval.

Recommended near-term scope:

1. define an LLM JSON `QueryPlan` schema;
2. validate planner output and provide deterministic fallback;
3. create `PlannedRetrievalExecutor` that force-executes planned queries;
4. create `EvidencePool` with dedupe and provenance;
5. feed curated evidence into the answer workflow;
6. keep reranking as a documented extension point, not part of the first implementation.

This matches the strongest patterns seen in LangChain, LlamaIndex, PaperQA, and STORM while keeping the first implementation smaller than a full ColBERT tool refactor.

