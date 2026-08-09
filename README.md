# PaperPilot

PaperPilot is a Python research agent for searching, reading, and synthesizing
academic papers. It uses an Anthropic-compatible ReAct loop with MCP tools for
arXiv download, per-paper ColBERT retrieval, citation graph lookup, multimodal
figure understanding, and concurrent paper reading.

**Current status**: the original evaluation-complete ReAct prototype remains
available, and the Web product now also has a typed Conversation runtime with
durable LangGraph checkpoints, continuous follow-up, branch alternatives, and
rollback. The two paths intentionally coexist during migration.

## What It Does

- Searches and downloads arXiv papers through an `arxiv` MCP server.
- Builds isolated per-paper ColBERT indexes and searches by `paper_id`.
- Uses skills such as `deep-read-paper`, `compare-papers`, and
  `write-research-report` to make long workflows explicit.
- Runs concurrent paper deep reads through a built-in subagent tool.
- Compacts long conversations with a protocol-safe `compact_context` tool.
- Produces structured Markdown research reports from multiple papers.
- Evaluates deep-read recall on a QASPER subset with reproducible baselines.
- Persists owned paper-reading Conversations, messages, tasks, branches, and
  rollback points through the Web API.

## Architecture

The Web Conversation path uses LangGraph only for orchestration, LangChain for
model/tool interaction and structured output, and Pydantic for typed contracts:

```text
HTTP Conversation API -> business SQLite -> queued research Task
  -> Worker -> DeepReadingRunner -> LangGraph fixed workflow
  -> bounded LangChain model/tools -> checkpoint SQLite
  -> published Message + authoritative Conversation head in business SQLite
```

The older `paperpilot.main.run(...)` path remains a synchronous ReAct loop.
`MCPClient` still hides async MCP sessions behind synchronous tool handlers for
that legacy path. The Web endpoint `/api/tasks` also remains the legacy
workbench path; new multi-turn product work goes through `/api/conversations`.

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

## Conversation Runtime And SQLite Ownership

The Conversation runtime deliberately uses two SQLite files with different
authorities:

- `PAPERPILOT_TASK_DB_PATH` is the business database. SQLAlchemy/Alembic owns
  users, sessions, papers, conversations, conversation-paper membership,
  messages, research tasks, events, artifacts, ownership, and the current
  Conversation head.
- `PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH` is the LangGraph database.
  `SqliteSaver` owns durable graph state, including the `checkpoints` and
  `writes` tables. It is not a user, Conversation, or authorization database.

The API and every Worker must receive the same two explicit paths, while each
process owns separate business connections, checkpoint connections, and
runtimes. Never share a live SQLite connection across Web/Worker processes.
The Web (including Celery-backed Web) constructs a lazy `MCPRuntime` only so
its `DeepReadingRunner` can read validated checkpoints for rollback; checkpoint
reads do not start MCP tools or a model. A Worker starts and reuses its MCP
runtime only when it executes actual Conversation work.

LangGraph Store-backed user profile/memory is intentionally not implemented in
this vertical slice. Conversation data and graph checkpoints are durable, but
cross-Conversation user memory is future work. Automated tests use bounded fake
models, MCP tools, and paper search; a real model/MCP smoke remains an explicit
manual integration check outside the test suite.

## Installation, Migration, And Startup Order

PaperPilot does not load `.env` automatically. For a Celery-backed deployment,
create and edit `.env`, set both SQLite paths, and set
`PAPERPILOT_TASK_EXECUTOR=celery` in `.env`. Export that file into the current
bash/zsh shell and start Redis before migration and startup. Run the following
prerequisites from the repository root:

```bash
cp .env.example .env
# Edit .env before continuing; replace placeholders and set the Celery executor.
set -a
source .env
set +a
docker compose up -d redis

uv pip sync requirements-lock.txt --python .venv/bin/python
./.venv/bin/python -m alembic -c alembic.ini upgrade head
LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m paperpilot.web.checkpoint --setup
```

After dependency sync, migration, and checkpoint setup succeed, open two
bash/zsh terminals in the repository root and load the same `.env` in each.
Start the Worker in terminal 1; it intentionally remains in the foreground.

```bash
set -a
source .env
set +a
./.venv/bin/celery -A paperpilot.web.celery_app:celery_app worker --loglevel=INFO
```

Once the Worker is ready, start the API in terminal 2:

```bash
set -a
source .env
set +a
./.venv/bin/uvicorn paperpilot.web.app:app --host 127.0.0.1 --port 8000
```

The five required dependency, migration, setup, Worker, and API commands above
are shown in their execution order even though the two long-running processes
use separate terminals.

Alembic head is `20260807_0002`. It manages the original Web tables plus
`papers`, `conversations`, `messages`, `conversation_papers`, and the
Conversation/checkpoint columns and indexes on `research_tasks`. Production
must run Alembic before starting multiple processes; `TaskStore` migration on
open remains only a local compatibility convenience. The migrations retain
unknown legacy rows and objects, and downgrade intentionally refuses to delete
business data.

For backup or restore, stop the API and all workers so neither SQLite file can
advance. Back up both files as one matched pair and verify both copies:

```bash
BACKUP_DIR="data/backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
sqlite3 "$PAPERPILOT_TASK_DB_PATH" ".backup '$BACKUP_DIR/business.sqlite3'"
sqlite3 "$PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH" ".backup '$BACKUP_DIR/checkpoints.sqlite3'"
sqlite3 "$BACKUP_DIR/business.sqlite3" "PRAGMA integrity_check;"
sqlite3 "$BACKUP_DIR/checkpoints.sqlite3" "PRAGMA integrity_check;"
```

Both integrity checks must print `ok`. Restore both members of the same pair
while the API and workers remain stopped; restoring only one can break the
Message/Task-to-checkpoint binding used by rollback:

```bash
RESTORE_DIR="data/backups/<selected-timestamp>"
sqlite3 "$RESTORE_DIR/business.sqlite3" ".backup '$PAPERPILOT_TASK_DB_PATH'"
sqlite3 "$RESTORE_DIR/checkpoints.sqlite3" ".backup '$PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH'"
```

## Web Workbench With Celery

The Web workbench defaults to its in-process thread executor, so local usage
does not require Redis. For broker-backed execution, use the same business and
checkpoint paths in the API and worker environments:

```bash
export PAPERPILOT_TASK_EXECUTOR=celery
export PAPERPILOT_CELERY_BROKER_URL=redis://127.0.0.1:6379/0
export PAPERPILOT_TASK_DB_PATH="$PWD/data/web/tasks.sqlite3"
export PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH="$PWD/data/langgraph/checkpoints.sqlite3"
export PAPERPILOT_TASK_MAX_RETRIES=3
export PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS=1
export PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS=30
export LANGGRAPH_STRICT_MSGPACK=true
```

The default worker limits are a 3-hour soft limit, a 3-hour 5-minute hard
limit, and a 4-hour Redis visibility timeout. The visibility timeout must
remain greater than the hard limit. Override the three values together with
`PAPERPILOT_TASK_SOFT_TIME_LIMIT_SECONDS`,
`PAPERPILOT_TASK_TIME_LIMIT_SECONDS`, and
`PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS`. Celery delivery is
**at-least-once**:
PaperPilot atomically claims pending work, and a redelivered message can reclaim
a running Task after worker loss. SQLite task creation and Redis publication
are not one transaction, so a transactional outbox/reconciler is still needed
before treating broker delivery as lossless. If publication is ambiguous, API
compensation may change only a still-`pending` Task to `failed`; it never
overwrites a Worker-owned `running`, `completed`, or `failed` Task.

Conversation infrastructure failures that escape the workflow have a bounded
retry policy shared by the Celery and thread executors. The default
`PAPERPILOT_TASK_MAX_RETRIES=3` means at most three retries after the initial
execution, for at most four total attempts. With
`PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS=1` and
`PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS=30`, the first three retry delays are
1, 2, and 4 seconds; larger retry counts continue exponentially and are capped
at 30 seconds. Celery schedules each retry as a new broker delivery, while the
thread executor waits and retries inside the same process and keeps its
admission slot reserved for the full lifecycle. A workflow failure already
converted into a deterministic business terminal state is not retried.

After the retry limit, an active Conversation Task is recorded as `failed`,
while the executor still reports the original exception. This prevents a Task
from remaining permanently `running` after a transient process-local failure,
but it cannot guarantee automatic recovery while the business SQLite database
or LangGraph checkpoint database remains unavailable. Long outages still need
operator recovery and, for lossless automated repair, a future reconciler.

The Research Agent has a separate, per-execution-attempt budget configured by
`PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT=6`,
`PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT=12`,
`PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS=4096`, and
`PAPERPILOT_RESEARCH_MODEL_RETRIES=1`. LangChain's model-call and tool-call
limit middleware enforces the logical-call bounds, while its model-retry
middleware permits at most one additional provider attempt for each failed
logical model call. The DeepSeek client itself has retries disabled so retry
ownership is explicit. The agent's two structured-response attempts share the
configured 6/12 totals rather than each receiving the full budget.

At the defaults, 6 logical model calls times 4,096 output tokens is a maximum
of 24,576 generated tokens within one execution attempt. Task execution may
make the initial attempt plus 3 infrastructure retries, so the extreme replay
bound is 98,304 generated tokens across 4 execution attempts. These are
theoretical output limits, not an exact billing cap based on usage metadata;
provider retries, input tokens, and provider billing semantics are separate.

Known deterministic failures are terminal: invalid task/checkpoint bindings,
unsupported graph or state schemas, malformed persisted or MCP payloads, and
agent budget exhaustion immediately create one safe `deep_reading_terminal`
failure event and are not retried. Database and checkpoint I/O failures, MCP
transport/timeouts, model provider/network failures (including exhausted model
retry middleware), and unknown infrastructure failures remain transient and
escape to the bounded Task retry policy. User-visible terminal events contain
only a stable error code/type and safe message; detailed exception data stays
in server logs.

## Web Runtime Protection

For the local, in-process thread executor, configure the runtime before
starting the API:

```bash
export PAPERPILOT_TASK_EXECUTOR=thread
export PAPERPILOT_THREAD_WORKERS=2
export PAPERPILOT_THREAD_QUEUE_CAPACITY=4
export PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS=1
export PAPERPILOT_TASK_MAX_RETRIES=3
export PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS=1
export PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS=30
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

`/health/live` probes no dependencies. `/health/ready` reports exactly the
`database`, `checkpoint`, and `executor` checks: both owned SQLite connections
must answer their health query and the executor must not have begun shutdown.
Redis, MCP, models, and temporary executor saturation are intentionally not
readiness probes. PaperPilot accepts an incoming `X-Request-ID` only when it
matches `[A-Za-z0-9._:-]` and is at most 128 characters; otherwise it generates
a new request ID.

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

The legacy task list APIs return bounded pagination envelopes. Task lists use
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

- **Conversation orchestration**: LangGraph with durable SQLite checkpoints
- **Conversation model/tools**: LangChain with Pydantic structured contracts
- **Legacy agent loop**: local synchronous ReAct loop
- **LLM**: DeepSeek (created lazily only for actual model work)
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

Run the Conversation-focused gate, then the complete automated suite with the
strict checkpoint serializer enabled:

```bash
LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests/deep_reading tests/web/test_checkpoint.py tests/web/test_conversation_store.py tests/web/test_conversation_worker.py tests/web/test_conversation_api.py tests/web/test_conversation_ui.py -q
LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests -q
```

The Task 15 local run produced `238 passed` for the focused gate and
`873 passed, 11 deselected` for the complete automated suite. The deselected
tests and any real model, arXiv, MCP-server, Redis/Celery-broker, or paid call
remain separate live-integration checks.

## License

MIT
