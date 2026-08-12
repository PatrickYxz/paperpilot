# PaperPilot

PaperPilot is a Conversation-based research service for searching, reading,
retrieving evidence from, and synthesizing academic papers. The supported
runtime is one durable Web architecture:

```text
FastAPI → Conversation → Executor → DeepReadingRunner → LangGraph → LangChain → MCP
```

FastAPI owns authentication and HTTP admission. A Conversation owns the stable
message head and active papers. The selected thread or Celery Executor runs a
Conversation-bound research Task. `DeepReadingRunner` binds that Task to a
durable LangGraph checkpoint, while LangChain provides the bounded research
agent, typed output, and model/tool middleware. MCP processes provide all paper
download, retrieval, citation-graph, and visual-reading extensions.

## Runtime capabilities

- Search arXiv and create an owned Conversation around one primary paper.
- Submit continuous follow-up questions with quick, standard, or deep research
  depth while preserving a stable published head.
- Search and prepare associated papers, then attach only validated paper and
  evidence identifiers to the Conversation.
- Build paper-isolated ColBERT indexes and return traceable evidence chunks.
- Preserve alternative branches and roll back to any compatible, complete
  assistant checkpoint without deleting later branches.
- Run locally with a bounded in-process thread executor or deploy an API and
  Celery Worker against the same databases.

## Data ownership and publication

PaperPilot deliberately uses two SQLite files with different owners:

- `PAPERPILOT_TASK_DB_PATH` is the business database. SQLAlchemy and Alembic
  own users, sessions, papers, Conversations, messages, research Tasks, events,
  artifacts, paper membership, and the authoritative Conversation head.
- `PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH` is the checkpoint database.
  LangGraph `SqliteSaver` owns graph state in its `checkpoints` and `writes`
  tables. It does not own authorization, user-visible messages, or the business
  head.

The API and every Worker must use the same two explicit paths. Each process
opens its own business and checkpoint connections; live SQLite connections are
never shared across processes. A result becomes visible only after its complete
checkpoint and assistant message are validated and the business head is moved
atomically to that pair.

## Conversation semantics

`POST /api/conversations/{conversation_id}/messages` requires
`expected_head_message_id`. Admission reserves executor capacity, rejects a
busy or stale Conversation with `409`, and creates an unstable user message plus
a Conversation-bound Task. The previous assistant remains the stable head while
the Task runs. Successful publication advances the message and checkpoint heads
together, so the next request is a continuous follow-up from an exact durable
state.

Alternatives are sibling user/assistant turns anchored to a complete assistant
message. Listing alternatives does not change the active branch. Rollback also
requires the expected current head, accepts only a complete assistant message
with a matching complete checkpoint, and atomically switches the stable head
and active-paper membership. It preserves the abandoned branch, so it remains
available as an alternative and can be selected again later.

## HTTP API

Authentication uses an HTTP-only session cookie. The supported API surface is:

- `POST /api/auth/register`
- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `GET /api/papers/search?q=...&limit=...`
- `POST /api/conversations`
- `GET /api/conversations`
- `GET /api/conversations/{conversation_id}`
- `PATCH /api/conversations/{conversation_id}` for title/archive updates
- `GET /api/conversations/{conversation_id}/messages`
- `POST /api/conversations/{conversation_id}/messages`
- `GET /api/conversations/{conversation_id}/messages/{message_id}/alternatives`
- `POST /api/conversations/{conversation_id}/rollback`
- `GET /api/conversations/{conversation_id}/tasks/{task_id}/updates`
- `GET /health/live`
- `GET /health/ready`

The conversation-scoped updates endpoint returns the Task plus bounded event and
artifact pages in one consistent business-database read. Clients advance
`after_event_id` and `after_artifact_id` watermarks while work is pending or
running.

## MCP extensions and local ColBERT behavior

`paperpilot/mcp_servers.json` starts four stdio MCP extensions:

1. `arxiv`: catalog search, PDF download, and text extraction.
2. `colbert`: per-paper index build, direct search, and planned retrieval.
3. `graph`: a NetworkX citation graph backed by Semantic Scholar metadata.
4. `vlm`: Qwen-VL page understanding through DashScope.

The ColBERT extension uses the local `lightonai/colbertv2.0` model and persists
an isolated PLAID index and `chunks.json` for each canonical paper ID under
`data/colbert_index/`. Its MCP child process sets `HF_HUB_OFFLINE=1`: production
startup does not download the model. Pre-cache that model for the operating
user before startup; a missing local model is an operator setup error. Existing
paper indexes are loaded lazily and IDs are converted to safe, hash-suffixed
directories.

## Installation and configuration

Use Python 3.12 and `uv` from the repository root:

```bash
python3.12 -m venv .venv
cp .env.example .env
# Replace credential placeholders and choose one executor before continuing.
set -a
source .env
set +a
uv pip sync requirements-lock.txt --python .venv/bin/python
```

The Web API and Celery Worker do not load `.env` before application import.
Every API and Worker shell must export the file first. `QwenClient` has a local
dotenv fallback, but operators should not rely on an MCP child process to load
production configuration. For Celery, set `PAPERPILOT_TASK_EXECUTOR=celery` in
`.env`; do not leave both executor choices active.

## Migration and startup order

Set both SQLite paths explicitly in `.env` so every process resolves the same
files:

```bash
PAPERPILOT_TASK_DB_PATH=/srv/paperpilot/data/web/tasks.sqlite3
PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH=/srv/paperpilot/data/langgraph/checkpoints.sqlite3
LANGGRAPH_STRICT_MSGPACK=true
```

For a Celery deployment, start Redis and run dependency sync, the business
migration, and checkpoint setup before any Worker or API process:

```bash
docker compose up -d redis
./.venv/bin/python -m alembic -c alembic.ini upgrade head
LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m paperpilot.web.checkpoint --setup
```

Alembic head is `20260807_0002`. Migration owns the business schema; checkpoint
setup owns only the LangGraph tables. Do not start multiple application
processes until both commands succeed.

After preparation, open two bash/zsh terminals in the repository root and load
the same `.env` in each. Start the Worker in terminal 1 and wait until it is
ready:

```bash
set -a
source .env
set +a
./.venv/bin/celery -A paperpilot.web.celery_app:celery_app worker --loglevel=INFO
```

Then start the API in terminal 2:

```bash
set -a
source .env
set +a
./.venv/bin/uvicorn paperpilot.web.app:app --host 127.0.0.1 --port 8000
```

For local thread execution, set `PAPERPILOT_TASK_EXECUTOR=thread`, export the
same two database paths, run the same migration and checkpoint setup, and start
only the API process.

## Capacity, health, and retry operations

The local thread executor defaults to 2 workers plus a queue capacity of 4,
which permits 6 unfinished tasks per Uvicorn process. With two Uvicorn workers,
the theoretical aggregate capacity across both processes is 12, although load
distribution is not guaranteed to be even. Tune
`PAPERPILOT_THREAD_WORKERS`, `PAPERPILOT_THREAD_QUEUE_CAPACITY`, and
`PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS` before startup. Saturated admission
returns `503` with `Retry-After` and creates no Task or event rows.

`/health/live` performs no dependency I/O. `/health/ready` evaluates exactly the
`database`, `checkpoint`, and `executor` checks. A failed readiness response
returns `503` and includes all three check details; a successful response is
only `{"status":"ready"}`. Redis, MCP, model providers, and temporary capacity
saturation are deliberately outside readiness. Request logs carry
`X-Request-ID`; configure
`PAPERPILOT_LOG_LEVEL`, `PAPERPILOT_LOG_FORMAT`,
`PAPERPILOT_SLOW_REQUEST_MS`, and `PAPERPILOT_ENV` as needed. Disable duplicate
Uvicorn access logs when PaperPilot logging is active:

```bash
./.venv/bin/uvicorn paperpilot.web.app:app \
  --host 127.0.0.1 --port 8000 --workers 2 --no-access-log
```

Conversation infrastructure failures use the same bounded policy in both
executors. With `PAPERPILOT_TASK_MAX_RETRIES=3`, one initial execution can be
followed by three retries. Backoff starts at
`PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS=1`, doubles, and is capped by
`PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS=30`. Contract, checkpoint-binding,
schema, malformed payload, deterministic MCP, and budget failures are terminal;
database/checkpoint I/O, MCP transport, model/network, and unknown
infrastructure failures are transient and may reach the Task retry policy.

Celery delivery is at-least-once. A redelivery may reclaim a running Task after
worker loss, but business SQLite creation and Redis publication are not one
transaction. A transactional outbox/reconciler is still required before broker
delivery can be treated as lossless. Ambiguous publication cleanup may fail
only a still-pending Task; it never overwrites Worker-owned `running`, `completed`, or `failed`
state.

Celery defaults are a 10,800-second soft limit, an 11,100-second hard limit,
and a 14,400-second Redis visibility timeout. Keep
`PAPERPILOT_TASK_SOFT_TIME_LIMIT_SECONDS` below
`PAPERPILOT_TASK_TIME_LIMIT_SECONDS`, and keep that below
`PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS`. Configure the broker with
`PAPERPILOT_CELERY_BROKER_URL`.

## Agent and nested-retrieval budgets

One execution attempt defaults to
`PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT=8`,
`PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT=12`,
`PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS=4096`,
`PAPERPILOT_RESEARCH_MODEL_RETRIES=1`, and a recursion limit of 24. Two
structured-response attempts share the configured total by receiving at most 4
logical model calls and 6 tool calls each. The DeepSeek client used by this
agent has provider retries disabled; LangChain middleware owns the one retry.

At the defaults, 8 logical model calls times 4,096 output tokens yields at most
32,768 generated tokens in one execution attempt. The initial execution plus
three infrastructure retries yields an extreme replay bound of 131,072
generated tokens. These are guardrail calculations, not billing guarantees:
input tokens, provider attempts, and provider usage metadata remain separate.

Every `retrieve_paper_evidence` tool call has an additional nested cost inside
the ColBERT MCP process. It performs at most one planner structured-output
invoke. Evidence verification defaults off; when explicitly enabled it is
capped at six requirement invokes, and only requirements with candidates are
sent. The nested retrieval model uses provider `max_retries=1`, which can add
one transport attempt after a transient failure for each planner or verifier
invoke. These nested invokes are outside the research agent middleware's 8-call
logical-model limit and must be included separately in cost approval.

## Backup and restore

Stop the API and all Workers so neither database can advance. Back up both files
as one timestamped pair and require both integrity checks to print `ok`:

```bash
BACKUP_DIR="data/backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
sqlite3 "$PAPERPILOT_TASK_DB_PATH" ".backup '$BACKUP_DIR/business.sqlite3'"
sqlite3 "$PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH" ".backup '$BACKUP_DIR/checkpoints.sqlite3'"
sqlite3 "$BACKUP_DIR/business.sqlite3" "PRAGMA integrity_check;"
sqlite3 "$BACKUP_DIR/checkpoints.sqlite3" "PRAGMA integrity_check;"
```

Restore both members of the same pair while the API and Workers remain stopped.
Restoring only one side can break business-message-to-checkpoint bindings:

```bash
RESTORE_DIR="data/backups/<selected-timestamp>"
sqlite3 "$RESTORE_DIR/business.sqlite3" ".backup '$PAPERPILOT_TASK_DB_PATH'"
sqlite3 "$RESTORE_DIR/checkpoints.sqlite3" ".backup '$PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH'"
./.venv/bin/python -m alembic -c alembic.ini upgrade head
LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m paperpilot.web.checkpoint --setup
sqlite3 "$PAPERPILOT_TASK_DB_PATH" "PRAGMA integrity_check;"
sqlite3 "$PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH" "PRAGMA integrity_check;"
```

Restart the Worker first and the API second only after migration, checkpoint
setup, and both SQLite integrity checks succeed.

## Verification boundary

Run deterministic automated validation with the project interpreter:

```bash
./.venv/bin/python -m pytest tests -q
uv pip check --python .venv/bin/python
git diff --check
```

The automated suite uses bounded fake models, fake MCP tools, controlled paper
search, temporary SQLite files, and broker doubles. It validates orchestration,
ownership, budgets, retry classification, and persistence without paid calls.
It does not prove live DeepSeek, arXiv, Hugging Face cache, MCP subprocess,
Semantic Scholar, DashScope, Redis, or Celery-worker connectivity.

A real-model smoke test is a separate manual integration gate. It requires
explicit approval before execution because it uses real credentials, network
services, and potentially paid model calls. Record the chosen model, budget,
database paths, and observed MCP/tool trace when approval is granted; never
infer production readiness from fake-model tests alone.

## License

MIT
