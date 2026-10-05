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
- Retrieve evidence via hybrid search: structure-aware chunks with contextual
  prefixes, ColBERT dense retrieval fused with BM25 by reciprocal rank fusion.
- Remember users across conversations: append-only long-term memories are
  extracted (with verbatim span verification) after each published turn,
  searched through the agent-owned `search_user_memory` tool, and distilled
  into a resident research profile injected as background data.
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

Alembic head is `20260901_0003`. Migration owns the business schema; checkpoint
setup owns only the LangGraph tables. Migration `20260901_0003` adds the
internal context-artifact, turn-archive, and compression-state tables and has
no downgrade. Do not start multiple application processes until both commands
succeed.

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

## Five-layer context management

The context-management runtime is provider-neutral and disabled by default.
The master flag enables the first four layers; full LLM compaction is a
separate flag and must be enabled only after the first four layers are stable.
The default model is the non-thinking `deepseek-v4-flash`, with an explicit
1,048,576-token model window.

The relevant configuration is:

```bash
PAPERPILOT_RESEARCH_MODEL_NAME=deepseek-v4-flash
PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED=false
PAPERPILOT_FULL_COMPACTION_ENABLED=false
PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS=1048576
```

| Variable | Default | Purpose |
|---|---:|---|
| `PAPERPILOT_RESEARCH_MODEL_NAME` | `deepseek-v4-flash` | Research model |
| `PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED` | `false` | Master flag |
| `PAPERPILOT_FULL_COMPACTION_ENABLED` | `false` | Stage B LLM compaction flag |
| `PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS` | `1048576` | Model context window |
| `PAPERPILOT_CONTEXT_ARTIFACT_ROOT` | `data/context-artifacts` | Artifact root |
| `PAPERPILOT_CONTEXT_TOOL_INLINE_MAX_TOKENS` | `2000` | Tool-result inline preview |
| `PAPERPILOT_CONTEXT_ARTIFACT_READ_MAX_TOKENS` | `2000` | One artifact-read bound |
| `PAPERPILOT_CONTEXT_MICRO_COMPACTION_TRIGGER_RATIO` | `0.70` | Micro-compaction trigger |
| `PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_TOKENS` | `8000` | Minimum micro reclaim |
| `PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_RATIO` | `0.10` | Minimum micro reclaim ratio |
| `PAPERPILOT_CONTEXT_MICRO_COMPACTION_KEEP_RECENT_TOOL_RESULTS` | `3` | Recent tool results kept |
| `PAPERPILOT_CONTEXT_ARCHIVE_BUDGET_TOKENS` | `4000` | Archive view budget |
| `PAPERPILOT_CONTEXT_ARCHIVE_MAX_RECORDS` | `5` | Maximum archive records |
| `PAPERPILOT_CONTEXT_ARCHIVE_RECENT_RECORDS` | `2` | Recent archive records |
| `PAPERPILOT_CONTEXT_FULL_COMPACTION_TRIGGER_RATIO` | `0.80` | Stage B trigger |
| `PAPERPILOT_CONTEXT_SESSION_MEMORY_TARGET_RATIO` | `0.65` | Stage A target |
| `PAPERPILOT_CONTEXT_FULL_COMPACTION_TARGET_RATIO` | `0.50` | Stage B target |
| `PAPERPILOT_CONTEXT_FULL_COMPACTION_RECENT_TURNS` | `2` | Protected recent turns |
| `PAPERPILOT_CONTEXT_COMPRESSION_FAILURE_THRESHOLD` | `3` | Breaker failure threshold |
| `PAPERPILOT_CONTEXT_COMPRESSION_TRANSIENT_RETRY_COUNT` | `1` | Transient compressor retries |
| `PAPERPILOT_CONTEXT_COMPRESSION_BREAKER_COOLDOWN_SECONDS` | `300` | Breaker cooldown |
| `PAPERPILOT_CONTEXT_SAFETY_MARGIN_RATIO` | `0.05` | Input-budget safety margin |

The model input budget is the 1,048,576-token window minus configured output
tokens and the safety margin. All candidate views are counted again before
adoption. When the master flag is false, the existing summary and graph path
remain active and no context-artifact root or context database write is
created.

### Artifact and database operations

Set `PAPERPILOT_CONTEXT_ARTIFACT_ROOT` to a directory owned by the API/Worker
operating user. For a deployment directory, create it before startup and use
permissions such as `chmod 750 data/context-artifacts`; published files are
immutable and should not be edited by hand. The first version does not
automatically delete published artifacts. A failed call may remove only its own
unpublished temporary/final files.

Back up the business SQLite file, the LangGraph checkpoint SQLite file, and the
entire artifact root as one timestamped set before enabling the flags:

```bash
BACKUP_DIR="data/backups/context-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
sqlite3 "$PAPERPILOT_TASK_DB_PATH" ".backup '$BACKUP_DIR/business.sqlite3'"
sqlite3 "$PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH" ".backup '$BACKUP_DIR/checkpoints.sqlite3'"
cp -a "$PAPERPILOT_CONTEXT_ARTIFACT_ROOT" "$BACKUP_DIR/context-artifacts"
sqlite3 "$BACKUP_DIR/business.sqlite3" "PRAGMA integrity_check;"
sqlite3 "$BACKUP_DIR/checkpoints.sqlite3" "PRAGMA integrity_check;"
```

Deploy the code with both flags false, run Alembic `20260901_0003`, and verify
the old Conversation and new Task baseline. Then enable
`PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED=true` for internal users and observe
the first four layers. Only after that baseline is stable, enable
`PAPERPILOT_FULL_COMPACTION_ENABLED=true` and observe candidate rejection,
compression, breaker, latency, and Task outcomes. 禁用时，关闭两个 flag（set
both flags to false）并重启 API/Worker。To recover to the pre-migration code,
stop all processes, restore the matching business/checkpoint/artifact backup
set, and deploy the old code; do not run old ORM code against the migrated
database and call that a completed rollback.

Context events are operational metadata only. They may contain bounded counts,
safe identifiers, digests, stages, reasons, validation types, and breaker
states, but must not contain Prompt text, user text, paper正文, full tool
output, hidden reasoning, or credentials. Provider cache hit/miss fields are
recorded only when the provider actually returns them; missing fields remain
`None`. Relevant event types include `artifact_externalized`,
`micro_compaction_completed`, `turn_archive_seeded`, `turn_archive_enriched`,
`turn_archive_failed`, `compression_*`, and `context_capacity_exhausted`.

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
`PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT=12`,
`PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT=12`,
`PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS=4096`,
`PAPERPILOT_RESEARCH_MODEL_RETRIES=1`, and a recursion limit of 24. Two
structured-response attempts share the configured total by receiving at most 6
logical model calls and 6 tool calls each. The DeepSeek client used by this
agent has provider retries disabled; LangChain middleware owns the one retry.

At the defaults, 12 logical model calls times 4,096 output tokens yields at most
49,152 generated tokens in one execution attempt. The initial execution plus
three infrastructure retries yields an extreme replay bound of 196,608
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

本地测试不等于生产收益。没有真实 DeepSeek 流量和线上观测，不能据此声称
输入 Token 降低 75%、缓存命中率提高、成本下降或线上成功率提升。上线验收还应
记录输入 Token 的平均值/p50/p95、Artifact 外置和读取、DROP/micro/两阶段压缩
事件、Archive 成败、Candidate rejection、breaker 三态、Provider cache
hit/miss、Task 成功率、p95 延迟和估算成本。

A real-model smoke test is a separate manual integration gate. It requires
explicit approval before execution because it uses real credentials, network
services, and potentially paid model calls. Record the chosen model, budget,
database paths, and observed MCP/tool trace when approval is granted; never
infer production readiness from fake-model tests alone.

## License

MIT
