# Day 17 eval case studies

Day 16 produced a complete 150-case QASPER eval:

| Baseline | Pass rate | Pass / Total | Error |
|---|---:|---:|---:|
| abstract_only | 2.0% | 3 / 150 | 0 |
| full_text | 40.7% | 61 / 150 | 0 |
| paperpilot | 45.3% | 68 / 150 | 0 |

The headline is deliberately modest: PaperPilot is better than a raw full-text
dump by 4.7 points, but the real value of this eval is that every PaperPilot
case has a trace. That makes wins and failures inspectable instead of anecdotal.

## Aggregate takeaways

- PaperPilot-only wins: 21 cases.
- Full-text-only wins: 14 cases.
- Both pass: 47 cases.
- Neither pass: 68 cases.
- PaperPilot failure buckets: `synthesis_miss` 56, `colbert_searched_low` 25,
  `no_colbert_search` 1.

## Case 1: PaperPilot wins by retrieving a precise definition

- Case: `qasper-1705.09665-q1`
- Paper: `1705.09665`, "Community Identity and User Engagement in a Multi-Community Landscape"
- Question: "How do the authors measure how temporally dynamic a community is?"
- Oracle span: "the average volatility of all utterances"
- Result: `paperpilot=pass`, `full_text=fail`
- Trace: `data/traces/qasper-1705.09665-q1.jsonl`

PaperPilot followed the intended chain:

```text
load_skill(deep-read-paper)
-> mcp__arxiv__download_paper(1705.09665)
-> mcp__colbert__build_index(paper_id=1705.09665)
-> mcp__colbert__search x3
```

The final answer explains word-level volatility, extends it to utterances, and
states that community dynamicity is computed from the average volatility of all
utterances. The full-text baseline gave a reasonable explanation, but missed the
exact oracle phrase, so the strict scorer marked it as failed.

Why it matters: this is the intended PaperPilot story. The agent loads the
workflow, indexes the paper, searches targeted evidence, and returns a grounded
answer that contains the QASPER evidence span.

## Case 2: PaperPilot wins on a task definition buried in the paper

- Case: `qasper-1908.06606-q2`
- Paper: `1908.06606`
- Question: "How is the clinical text structuring task defined?"
- Oracle spans include the CTS definition and the QA-CTS contrast.
- Result: `paperpilot=pass`, `full_text=fail`
- Trace: `data/traces/qasper-1908.06606-q2.jsonl`

Trace shape:

```text
load_skill(deep-read-paper)
-> mcp__arxiv__download_paper(1908.06606)
-> mcp__colbert__build_index(paper_id=1908.06606)
-> mcp__colbert__search x3
```

The PaperPilot answer explicitly cites the clinical text structuring definition
and contrasts traditional CTS with QA-CTS. This is a good case study because the
answer is definition-heavy and benefits from targeted retrieval rather than a
single full-text prompt.

## Case 3: Full-text wins where PaperPilot under-searches

- Case: `qasper-1909.08402-q1`
- Question: "What dataset do they use?"
- Oracle spans: "2019 GermEval shared task on hierarchical text classification",
  "GermEval 2019 shared task"
- Result: `full_text=pass`, `paperpilot=fail`
- Trace: `data/traces/qasper-1909.08402-q1.jsonl`
- Failure bucket: `colbert_searched_low`

Trace shape:

```text
load_skill(deep-read-paper)
-> mcp__arxiv__download_paper(1909.08402)
-> mcp__colbert__build_index(paper_id=1909.08402)
-> mcp__colbert__search x2
```

PaperPilot's answer identified the dataset as the 2019 GermEval shared task, but
the scorer required an exact English oracle span and counted it as failed. This
case shows two useful limitations at once:

- the agent sometimes stops at only two ColBERT searches, below the intended
  deep-read threshold;
- strict substring scoring can undercount semantically correct Chinese answers.

Why it matters: the eval is honest but conservative. It is useful for ranking
systems, but not every fail is an objectively wrong answer.

## Case 4: Retrieval happened, but synthesis missed the requested number

- Case: `qasper-1809.04960-q2`
- Question: "How many comments were used?"
- Oracle span: "from 50K to 4.8M"
- Result: `paperpilot=fail`, `full_text=fail`
- Trace: `data/traces/qasper-1809.04960-q2.jsonl`
- Failure bucket: `synthesis_miss`

Trace shape:

```text
load_skill(deep-read-paper)
-> mcp__arxiv__download_paper(1809.04960)
-> mcp__colbert__build_index(paper_id=1809.04960)
-> mcp__colbert__search x3
```

The agent performed the full retrieval path, but the final answer summarized the
dataset broadly and did not include the exact requested range. This is the most
important failure class in Day 16: retrieval can be present while final synthesis
still misses the answer span.

## Recommended next technical work

Do not treat Day 16 as proof that the current RAG workflow is finished. The
numbers suggest a narrower next step:

- enforce or strongly nudge at least three targeted ColBERT searches for
  deep-read questions;
- carry compact evidence snippets into the final answer and require the final
  response to include the shortest answer span before explanation;
- add a semantic/LLM judge audit on failed substring cases, but keep substring
  scoring as the main reproducible metric.

These are Day 18-style improvements. Day 17's job is to make the existing result
presentable and auditable.
