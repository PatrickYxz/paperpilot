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

## Web Workbench With Celery

The Web workbench defaults to its in-process thread executor, so local usage
does not require Redis. For a broker-backed queue and long-lived worker processes
that reuse one MCP runtime per process, start Redis and select the Celery
executor before starting the API:

```bash
docker compose up -d redis

export PAPERPILOT_TASK_EXECUTOR=celery
export PAPERPILOT_CELERY_BROKER_URL=redis://127.0.0.1:6379/0
export PAPERPILOT_TASK_DB_PATH="$PWD/data/web/tasks.sqlite3"
```

The default worker limits are a 3-hour soft limit, a 3-hour 5-minute hard
limit, and a 4-hour Redis visibility timeout. Override them together when a
deployment needs different limits; the visibility timeout must remain greater
than the hard limit:

```bash
export PAPERPILOT_TASK_SOFT_TIME_LIMIT_SECONDS=10800
export PAPERPILOT_TASK_TIME_LIMIT_SECONDS=11100
export PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS=14400
```

Start a worker in one terminal. Each prefork child lazily loads its MCP runtime
on its first real research task and reuses it for later tasks. Start with two
children and adjust only after measuring model memory use:

```bash
.venv/bin/celery -A paperpilot.web.celery_app:celery_app worker \
  --loglevel=INFO --concurrency=2
```

Start the API in another terminal with the same environment variables and
working directory so the API and workers share the same SQLite database:

```bash
.venv/bin/uvicorn paperpilot.web.app:app \
  --host 127.0.0.1 --port 8000 --workers 2 --no-access-log
```

Celery delivery is **at-least-once**. PaperPilot atomically claims pending
tasks so duplicate messages cannot run the same task concurrently; a message
marked as redelivered may reclaim a running task after worker loss. One
reliability window remains: SQLite task creation and Redis publication are not
one transaction. If the API process exits after the database commit but before
broker publication, a pending task can be left without a queue message. A
transactional outbox or reconciler is required before treating this deployment
as lossless across process crashes. If publication returns an ambiguous error,
the API only changes `pending` to `failed`; it never overwrites work that a
worker has already claimed as `running` or `completed`.

## Web Runtime Protection

For the local, in-process thread executor, configure the runtime before
starting the API:

```bash
export PAPERPILOT_TASK_EXECUTOR=thread
export PAPERPILOT_THREAD_WORKERS=2
export PAPERPILOT_THREAD_QUEUE_CAPACITY=4
export PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS=1
export PAPERPILOT_LOG_LEVEL=INFO
export PAPERPILOT_LOG_FORMAT=json
export PAPERPILOT_SLOW_REQUEST_MS=1000
export PAPERPILOT_ENV=development
```

Thread admission capacity is `workers + queue capacity` per Uvicorn process.
The defaults therefore allow 6 unfinished tasks per Uvicorn process: 2 running
and 4 queued. With two Uvicorn workers and the defaults, the theoretical
aggregate capacity across both processes is 12, but load distribution is not
guaranteed to be even.
When that per-process capacity is full, task creation returns `503` with a
`Retry-After` header and creates no task or event database rows.

`/health/live` has no dependency probe. `/health/ready` checks SQLite with
`SELECT 1` and only whether the executor has begun shutdown; Redis, MCP, and
temporary executor saturation do not fail readiness. PaperPilot accepts an
incoming `X-Request-ID` only when it matches `[A-Za-z0-9._:-]` and is at most
128 characters; otherwise it generates a new request ID.

When PaperPilot structured access logging is active, use `--no-access-log` to
disable duplicate Uvicorn access logs:

```bash
./.venv/bin/uvicorn paperpilot.web.app:app \
  --host 127.0.0.1 --port 8000 --workers 2 --no-access-log
```

Run the deterministic admission benchmark after changing thread-capacity
settings. It checks status and database-row counts, reports latency summaries
without enforcing machine-specific timing thresholds, and verifies that a
released runner restores capacity:

```bash
./.venv/bin/python scripts/benchmark_web_admission.py \
  --workers 2 --queue-capacity 4 --requests 12
```

Prometheus/OpenTelemetry and cross-process rate limiting remain separate
deployment-topology work.

## Web Read Performance

The Web task store configures SQLite in WAL mode and enables foreign-key
enforcement, a 30-second busy timeout, and `synchronous=NORMAL` on each
connection. WAL allows API reads to continue while a worker commits a write,
but SQLite still has only one writer at a time. It remains a local, shared-file
design rather than a general high-write-concurrency database.

All three list APIs return bounded pagination envelopes. Task lists use
`{items, next_cursor, has_more}` with keyset cursors; event and artifact lists
use `{items, next_after_id, has_more}` with numeric watermarks. The default page
size is 50 and the maximum is 100. `/api/tasks/{task_id}/updates` reads the
current task plus event and artifact pages in one explicit SQLite read
transaction, so the combined response comes from one consistent snapshot.

While a task is `pending` or `running`, the Web client polls only the updates
endpoint once per second. It advances the event and artifact watermarks instead
of repeatedly downloading complete histories or the full task list.

Run the reproducible store benchmark from the repository root:

```bash
.venv/bin/python scripts/benchmark_web_task_store.py \
  --tasks 2000 --writes 200 --workers 8 --page-size 50
```

The command uses a temporary database unless `--db-path` is provided and emits
one JSON object with bounded-page size and latency, SQLite settings and indexes,
query plans, and concurrent-write latency and errors. Timings are
machine-dependent and are intended for same-machine comparisons, not fixed
performance thresholds.

Move the task store to PostgreSQL when sustained concurrent writes, multiple
application hosts, or lock contention exceed this local SQLite design. WAL and
`busy_timeout` reduce local contention; they do not remove SQLite's
single-writer boundary.

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
