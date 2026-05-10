# PaperPilot

PaperPilot is a Python research agent for searching, reading, and synthesizing
academic papers. It uses an Anthropic-compatible ReAct loop with MCP tools for
arXiv download, per-paper ColBERT retrieval, citation graph lookup, multimodal
figure understanding, and concurrent paper reading.

**Current status**: evaluation-complete prototype with one post-eval
improvement pass. The core agent loop, tool stack, deep-read workflow,
report-writing workflow, and a third-party QASPER eval harness are implemented
and validated locally.

## What It Does

- Searches and downloads arXiv papers through an `arxiv` MCP server.
- Builds isolated per-paper ColBERT indexes and searches by `paper_id`.
- Uses skills such as `deep-read-paper`, `compare-papers`, and
  `write-research-report` to make long workflows explicit.
- Runs concurrent paper deep reads through a built-in subagent tool.
- Compacts long conversations with a protocol-safe `compact_context` tool.
- Produces structured Markdown research reports from multiple papers.
- Evaluates deep-read recall on a QASPER subset with reproducible baselines.

## Architecture

```text
User query
  -> paperpilot.main.run(...)
  -> synchronous ReAct loop
  -> built-in tools
       load_skill / research_todo / paper_deep_read / compact_context
  -> MCP tools
       arxiv / colbert / graph / vlm
  -> DeepSeek via Anthropic-compatible messages API
  -> final answer or Markdown report
```

The top-level loop stays synchronous. `MCPClient` hides the async MCP sessions
behind synchronous tool handlers, so the agent loop can reason over a simple
tool list while each MCP server owns its specialized runtime.

## Day 16/18 Evaluation

PaperPilot was evaluated on a third-party AI2 QASPER extractive-QA subset:
50 papers x 3 questions = 150 cases. The eval compares three baselines with the
same DeepSeek model. The abstract-only and full-text rows are the original
Day 16 baselines; the PaperPilot row was rerun after the Day 18 evidence-span
and multi-search prompt improvements:

| Baseline | Pass | Fail | Error | Pass Rate | Avg latency |
|---|---:|---:|---:|---:|---:|
| abstract_only | 3 | 147 | 0 | 2.0% | 0.8s |
| full_text | 61 | 89 | 0 | 40.7% | 3.5s |
| paperpilot | 99 | 51 | 0 | 66.0% | 107.9s |

Interpretation:

- Abstract-only answering is not enough for detailed paper QA.
- Full text gives the largest jump, which is expected for extractive recall.
- After the Day 18 prompt/evidence-span pass, PaperPilot beats full-text dump by
  25.3 points under the strict substring scorer.
- The improvement came from making the deep-read workflow search more
  consistently and forcing answers to surface compact evidence span candidates
  before explanation.
- The remaining failures are now mostly synthesis misses after retrieval, not
  tool startup, arXiv download, or missing ColBERT calls.

See [the eval summary](data/eval/summary.md) and
[Day 17 case studies](docs/day17-eval-case-studies.md).

## Reproducing The Eval

Install dependencies and configure keys:

```powershell
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
Copy-Item .env.example .env
# Fill DEEPSEEK_API_KEY. For optional tools, the graph client reads
# SEMANTIC_SCHOLAR_API_KEY and the VLM client reads DASHSCOPE_API_KEY.
```

Prepare the QASPER subset after downloading
`qasper-train-v0.3.json` into `data/eval/qasper-source/`:

```powershell
.venv\Scripts\python.exe scripts\day16_prepare_eval.py
```

Run the three baselines:

```powershell
.venv\Scripts\python.exe scripts\day16_run_eval.py --baseline abstract_only
.venv\Scripts\python.exe scripts\day16_run_eval.py --baseline full_text
.venv\Scripts\python.exe scripts\day16_run_eval.py --baseline paperpilot
.venv\Scripts\python.exe scripts\day16_summarize.py
```

Runtime artifacts are written under `data/eval/` and `data/traces/`. Large
JSONL result files and traces are intentionally ignored by git; the committed
summary is the portable eval artifact.

## Tech Stack

- **LLM**: DeepSeek through Anthropic-compatible messages API
- **Agent loop**: local synchronous ReAct loop
- **Tool protocol**: MCP over stdio
- **Retrieval**: PyLate/ColBERT per-paper indexes
- **Citation graph**: NetworkX + Semantic Scholar Graph API
- **PDF parsing**: PyMuPDF
- **VLM**: Qwen-VL via DashScope
- **Eval**: QASPER extractive QA, JSONL traces, deterministic substring scorer

## Current Limitations

- QASPER scoring is strict substring matching, so semantically correct answers
  can be counted as false negatives when wording differs.
- PaperPilot is much slower than full-text dump on this eval because it runs the
  complete agent workflow and writes per-case traces.
- The main remaining failure mode is answer synthesis: the trace often contains
  relevant evidence, but the final answer may miss the exact oracle wording,
  numeric range, or phrase boundary required by the strict scorer.
- `data/eval/results_*.jsonl` and `data/traces/*.jsonl` are local artifacts and
  are not committed.

## Validation

Known local checks for the Day 18 state:

```powershell
.venv\Scripts\python.exe -m pytest tests -q `
  --ignore=tests/mcp_servers/test_colbert_via_client.py `
  --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py `
  --ignore=tests/test_per_paper_index_slow.py
```

Current fast-suite result: `137 passed, 4 deselected`.

## License

MIT
