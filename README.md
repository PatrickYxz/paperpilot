# PaperPilot

PaperPilot is a Python research agent for searching, reading, and synthesizing
academic papers. It uses an Anthropic-compatible ReAct loop with MCP tools for
arXiv download, per-paper ColBERT retrieval, citation graph lookup, multimodal
figure understanding, and concurrent paper reading.

**Current status**: evaluation-complete prototype. The core agent loop,
tool stack, deep-read workflow, report-writing workflow, and a third-party
QASPER eval harness are implemented and validated locally.

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

## Day 16 Evaluation

PaperPilot was evaluated on a third-party AI2 QASPER extractive-QA subset:
50 papers x 3 questions = 150 cases. The eval compares three baselines with the
same DeepSeek model:

| Baseline | Pass | Fail | Error | Pass Rate | Avg latency |
|---|---:|---:|---:|---:|---:|
| abstract_only | 3 | 147 | 0 | 2.0% | 0.8s |
| full_text | 61 | 89 | 0 | 40.7% | 3.5s |
| paperpilot | 68 | 82 | 0 | 45.3% | 110.9s |

Interpretation:

- Abstract-only answering is not enough for detailed paper QA.
- Full text gives the largest jump, which is expected for extractive recall.
- PaperPilot beats full-text dump by 4.7 points, but not by a large margin; the
  honest takeaway is a small end-to-end RAG/agent advantage plus traceable
  failure attribution, not a decisive win.
- PaperPilot failures are mostly synthesis misses after retrieval, followed by
  too few ColBERT searches.

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
- The main Day 16 failure mode is answer synthesis, not tool startup or arXiv
  download; this suggests the next technical work should improve retrieval
  query planning, evidence packing, or answer grounding.
- `data/eval/results_*.jsonl` and `data/traces/*.jsonl` are local artifacts and
  are not committed.

## Validation

Known local checks for the Day 16/17 state:

```powershell
.venv\Scripts\python.exe -m pytest tests -q `
  --ignore=tests/mcp_servers/test_colbert_via_client.py `
  --ignore=tests/mcp_servers/vlm/test_vlm_via_client.py `
  --ignore=tests/test_per_paper_index_slow.py
```

Expected result after Day 16: `134 passed, 4 deselected`.

## License

MIT
