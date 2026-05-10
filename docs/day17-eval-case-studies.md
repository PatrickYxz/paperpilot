# Day 17/18 eval case studies

Day 16 produced the first complete 150-case QASPER eval. Day 18 then reran the
PaperPilot row after tightening the deep-read workflow around multi-query
retrieval and explicit answer span candidates.

| Baseline | Pass rate | Pass / Total | Error |
|---|---:|---:|---:|
| abstract_only | 2.0% | 3 / 150 | 0 |
| full_text | 40.7% | 61 / 150 | 0 |
| paperpilot | 66.0% | 99 / 150 | 0 |

The headline is stronger than Day 16 but still specific: after the Day 18
prompt/evidence pass, PaperPilot beats the raw full-text baseline by 25.3
points under a strict substring scorer. The traces are still the main asset:
they show where the agent searched, what evidence it saw, and why failures
remain.

## Aggregate takeaways

- PaperPilot-only wins: 40 cases.
- Full-text-only wins: 2 cases.
- Both pass: 59 cases.
- Neither pass: 49 cases.
- Compared with the Day 16 PaperPilot run: 37 cases flipped from fail to pass,
  6 flipped from pass to fail, 62 stayed pass, and 45 stayed fail.
- Current PaperPilot failure buckets: `synthesis_miss` 50,
  `colbert_searched_low` 1.

## Case 1: PaperPilot wins by retrieving a precise definition

- Case: `qasper-1705.09665-q1`
- Paper: `1705.09665`, "Community Identity and User Engagement in a
  Multi-Community Landscape"
- Question: "How do the authors measure how temporally dynamic a community is?"
- Oracle span: "the average volatility of all utterances"
- Result: `paperpilot=pass`, `full_text=fail`
- Trace: `data/traces/qasper-1705.09665-q1.jsonl`

Trace shape:

```text
load_skill(deep-read-paper)
-> mcp__arxiv__download_paper(1705.09665)
-> mcp__colbert__build_index(paper_id=1705.09665)
-> mcp__colbert__search x4
```

Why it matters: this is the intended PaperPilot story. The agent loads the
workflow, indexes the paper, searches targeted evidence, and returns a grounded
answer that contains the QASPER evidence span.

## Case 2: PaperPilot wins on a compact factual list

- Case: `qasper-1906.00378-q2`
- Question: "Which languages are used in the multi-lingual caption model?"
- Oracle spans include "German-English, French-English, and Japanese-English"
- Result: `paperpilot=pass`, `full_text=fail`
- Trace: `data/traces/qasper-1906.00378-q2.jsonl`

Trace shape:

```text
load_skill(deep-read-paper)
-> mcp__arxiv__download_paper(1906.00378)
-> mcp__colbert__build_index(paper_id=1906.00378)
-> mcp__colbert__search x3
```

This case benefits from the Day 18 answer-span rule: the answer must expose the
short list before explanation, which helps the strict scorer and also makes the
answer easier to audit.

## Case 3: Full-text still wins on a short label answer

- Case: `qasper-1909.00694-q1`
- Question: "What are labels available in dataset for supervision?"
- Oracle spans: "negative", "positive"
- Result: `full_text=pass`, `paperpilot=fail`
- Trace: `data/traces/qasper-1909.00694-q1.jsonl`

Trace shape:

```text
load_skill(deep-read-paper)
-> mcp__arxiv__download_paper(1909.00694)
-> mcp__colbert__build_index(paper_id=1909.00694)
-> mcp__colbert__search x9
```

This is not an under-search problem. The agent searched heavily, but the final
answer did not reduce the evidence to the exact two-label form required by the
oracle. That points to answer extraction and final-span compression, not simply
more retrieval.

## Case 4: Retrieval happened, but synthesis missed the requested range

- Case: `qasper-1809.04960-q2`
- Question: "How many comments were used?"
- Oracle span: "from 50K to 4.8M"
- Result: `paperpilot=fail`, `full_text=fail`
- Trace: `data/traces/qasper-1809.04960-q2.jsonl`

Trace shape:

```text
load_skill(deep-read-paper)
-> mcp__arxiv__download_paper(1809.04960)
-> mcp__colbert__build_index(paper_id=1809.04960)
-> mcp__colbert__search x4
```

The answer surfaced nearby numeric evidence, but missed the exact range. This
is still the most important remaining failure pattern: retrieval can be present
while final synthesis fails to preserve the shortest numeric answer span.

## Recommended next technical work

Do not treat the 66.0% result as the end of the RAG work. The next narrow
improvement should focus on answer extraction after retrieval:

- add a stricter final-span selection step for short labels, lists, and numeric
  ranges;
- preserve exact table/range wording when the question asks "how many", "which
  labels", "what methods", or similar atomic facts;
- keep the strict substring scorer as the reproducible metric, but add a small
  semantic audit report for false negatives.
