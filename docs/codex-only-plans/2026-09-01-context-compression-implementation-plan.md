# PaperPilot Five-Layer Context Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不删除 TaskStore 权威业务历史、不改变 ResearchResult 与最终发布事务语义的前提下，为 PaperPilot 落地五层上下文管理：工具结果预算控制、确定性噪声立即删除、历史工具结果批量微压缩、逐业务轮次归档，以及带候选验证和熔断器的两阶段全量压缩。

**Architecture:** 新增 `paperpilot.deep_reading.context_management` 领域包，负责预算、生命周期决策、模型视图、归档和压缩算法；新增 Web 层 SQLite/本地文件适配器，负责 Artifact、TurnArchive 和 CompressionState 持久化。Runner 继续是 Task 到 LangGraph 的执行边界，通过协议端口注入上下文运行时；固定 System Prompt 保持静态，动态状态只进入每次请求的派生模型视图。

**Tech Stack:** Python 3.12、Pydantic 2、LangChain 1.3、LangGraph 1.2、SQLAlchemy 2、Alembic、SQLite、DeepSeek Chat Completions、pytest。

**Spec:** [2026-09-01-context-compression-design.md](/Users/patrick/Projects/paperpilot/docs/codex-only-plans/2026-09-01-context-compression-design.md)

## Global Constraints

- 本计划是实施说明，不构成生产代码修改授权。开始 Task 1 前，必须取得“实施前审批项”中各项的明确确认。
- 当前工作树已有未提交的 Prompt、Status Bar、Runner、配置和测试改动。执行时先运行 `git status --short`，逐文件保护既有修改；禁止 `reset`、`checkout`、`stash`、覆盖、删除或重排无关改动。
- 不改变公开 Web API、前端响应结构、`ResearchResult`、`AnswerDraft`、论文证据校验和现有 publication/finalization/failure/rollback 单事务语义。
- 不删除 TaskStore 消息树、已发布答案、业务 Event、Artifact、TurnArchive 或论文原文。模型视图可裁剪，权威数据不可因压缩被覆盖。
- 不引入向量数据库、Embedding 服务、服务端 DeepSeek context editing、跨 Conversation 记忆或新第三方依赖。
- 固定 System Prompt 不接收运行时变量。工具 schema、Prompt 版本或模型名发生变化时，只接受一次可解释的前缀缓存重建，之后保持稳定。
- 所有大文本写入、日志和 Event 必须排除完整 Prompt、用户原文、论文正文、完整工具输出、隐藏推理和凭证。Event 只保存 ID、计数、哈希、状态、原因和 Token 指标。
- 测试统一使用 `.venv/bin/python -m pytest`；不得使用已知会导致导入失败的 `.venv/bin/pytest`。
- 每个任务末尾的 Commit 步骤只是建议的审查边界。只有用户在实施时明确授权 Git 写入后才执行；否则跳过且不以此阻塞后续任务。
- 任何任务发现接口、数据结构、核心配置或行为需要超出本文时，停止该任务并回到设计审批，不得现场扩大范围。

## 实施前审批项

执行者必须在 Task 1 开始前逐项向用户复述并取得确认：

1. **数据库迁移：** 新增 `context_artifacts`、`turn_archives`、`compression_states` 三张内部表，迁移版本固定为 `20260901_0003`，不提供向下迁移；回滚依赖实施前数据库备份和旧版本代码。
2. **Artifact 文件策略：** 根目录默认 `data/context-artifacts`；发布后的文件不可变且第一版不自动删除。当前调用失败时只清理由该调用创建且尚未发布的临时/最终文件，不运行跨会话自动清理。
3. **Research 工具契约：** 在现有三个业务工具之外新增 `read_artifact_slice`、`search_artifact`；二者计入 Research tool budget 和 Status Bar。工具 schema 与 Research Prompt 版本变更会导致部署后一次前缀缓存重建。
4. **State/Graph 迁移：** `SCHEMA_VERSION` 从 1 升到 2，`GRAPH_VERSION` 从 `conversation-v1` 升到 `conversation-v2`；旧 v1 完成 checkpoint 可读取和迁移，但新写入只产生 v2。
5. **配置和灰度：** 新机制由 `PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED=false` 默认关闭；全量 LLM 压缩另由 `PAPERPILOT_FULL_COMPACTION_ENABLED=false` 默认关闭。先启用前四层，再单独启用第五层。
6. **模型兼容：** 当前代码固定使用已被 DeepSeek 官方停用的 `deepseek-chat`。推荐把默认模型改为非思考模式的 `deepseek-v4-flash`，并把上下文窗口显式配置为 `1_048_576`；若用户不批准模型迁移，则本计划只能完成本地逻辑，不能声称生产路径可运行。
7. **旧 Summary：** 首次启用新机制时，把 checkpoint 中的 `conversation_summary` 原始 JSON 写成 `legacy-summary-v1` 只读 Archive；它只作为不可信候选，不与 ActiveProjection 或 ContinuationCapsule 重复注入。
8. **失败语义：** Archive Seed/Narrative 失败不会反转已完成或已失败的业务 Task；压缩失败不会覆盖原 provider 异常。三次完整压缩流程失败后，按 Conversation 和 compressor version 打开熔断器。
9. **文档 allowlist：** 最终任务需要把本设计文档和本实施计划合并进当前已被用户修改的 `DOCS_KEEP_ALLOWLIST`，只能增量合并，不能覆盖现有 Status Bar 条目。

## 固定事件契约

实现必须注册并测试以下完整事件集合，不使用通配事件名替代具体类型：

```text
artifact_externalized
tool_result_dropped
micro_compaction_started
micro_compaction_completed
turn_archive_seeded
turn_archive_enriched
turn_archive_failed
session_memory_compaction_started
full_compaction_started
compression_candidate_rejected
compression_circuit_open
compression_circuit_half_open
compression_circuit_closed
compression_degraded
context_capacity_exhausted
```

每次压缩事件必须带 `stage`、`compressor_version`、`reason`、`before_tokens`、`after_tokens`、`reclaimed_tokens`、`protected_item_count`、Archive/Artifact 引用数量、`validation_failure_type` 和 breaker state；Provider 实际返回 cache token 时才写 cache hit/miss，缺失时保持 `None`。

---

### Task 1: 固化运行配置、模型标识和灰度边界

**Files:**

- Modify: `paperpilot/web/config.py`
- Modify: `paperpilot/deep_reading/runner.py`
- Modify: `paperpilot/deep_reading/nodes/context.py`
- Modify: `paperpilot/web/app.py`
- Modify: `paperpilot/web/worker_tasks.py`
- Modify: `.env.example`
- Modify: `tests/web/test_config.py`
- Modify: `tests/web/test_runtime_config.py`
- Modify: `tests/web/test_conversation_worker.py`
- Modify: `tests/deep_reading/test_runner.py`

**Interfaces:**

- `WebRuntimeConfig` 新增 `research_model_name: str`，默认 `deepseek-v4-flash`。
- `WebRuntimeConfig` 新增不可变的 `context_management: ContextManagementConfig`。
- `build_deep_reading_model(*, model_name: str, research_max_output_tokens: int)` 使用显式模型名。
- `DeepReadingRunner` 和 `DeepReadingContext` 只接收一个 `ContextManagementConfig`，避免增加十余个散落参数。

`ContextManagementConfig` 字段及默认值固定如下：

| 字段 | 默认值 |
|---|---:|
| `enabled` | `False` |
| `full_compaction_enabled` | `False` |
| `model_context_window_tokens` | `1_048_576` |
| `artifact_root` | `Path("data/context-artifacts")` |
| `tool_inline_max_tokens` | `2_000` |
| `artifact_read_max_tokens` | `2_000` |
| `micro_compaction_trigger_ratio` | `0.70` |
| `micro_compaction_min_reclaim_tokens` | `8_000` |
| `micro_compaction_min_reclaim_ratio` | `0.10` |
| `micro_compaction_keep_recent_tool_results` | `3` |
| `archive_context_budget_tokens` | `4_000` |
| `archive_context_max_records` | `5` |
| `archive_context_recent_records` | `2` |
| `full_compaction_trigger_ratio` | `0.80` |
| `session_memory_target_ratio` | `0.65` |
| `full_compaction_target_ratio` | `0.50` |
| `full_compaction_recent_turns` | `2` |
| `compression_failure_threshold` | `3` |
| `compression_transient_retry_count` | `1` |
| `compression_breaker_cooldown_seconds` | `300` |
| `context_safety_margin_ratio` | `0.05` |

环境变量映射固定如下，执行时不得自行缩写或增加别名：

```text
PAPERPILOT_RESEARCH_MODEL_NAME
PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED
PAPERPILOT_FULL_COMPACTION_ENABLED
PAPERPILOT_CONTEXT_MODEL_WINDOW_TOKENS
PAPERPILOT_CONTEXT_ARTIFACT_ROOT
PAPERPILOT_CONTEXT_TOOL_INLINE_MAX_TOKENS
PAPERPILOT_CONTEXT_ARTIFACT_READ_MAX_TOKENS
PAPERPILOT_CONTEXT_MICRO_COMPACTION_TRIGGER_RATIO
PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_TOKENS
PAPERPILOT_CONTEXT_MICRO_COMPACTION_MIN_RECLAIM_RATIO
PAPERPILOT_CONTEXT_MICRO_COMPACTION_KEEP_RECENT_TOOL_RESULTS
PAPERPILOT_CONTEXT_ARCHIVE_BUDGET_TOKENS
PAPERPILOT_CONTEXT_ARCHIVE_MAX_RECORDS
PAPERPILOT_CONTEXT_ARCHIVE_RECENT_RECORDS
PAPERPILOT_CONTEXT_FULL_COMPACTION_TRIGGER_RATIO
PAPERPILOT_CONTEXT_SESSION_MEMORY_TARGET_RATIO
PAPERPILOT_CONTEXT_FULL_COMPACTION_TARGET_RATIO
PAPERPILOT_CONTEXT_FULL_COMPACTION_RECENT_TURNS
PAPERPILOT_CONTEXT_COMPRESSION_FAILURE_THRESHOLD
PAPERPILOT_CONTEXT_COMPRESSION_TRANSIENT_RETRY_COUNT
PAPERPILOT_CONTEXT_COMPRESSION_BREAKER_COOLDOWN_SECONDS
PAPERPILOT_CONTEXT_SAFETY_MARGIN_RATIO
```

- [ ] 在 `tests/web/test_config.py` 先写失败测试，断言上述默认值、全部环境变量覆盖、布尔值解析、Path 解析及非法范围拒绝。
- [ ] 增加关系约束测试：`full_compaction_enabled=True` 时必须同时 `enabled=True`；比例满足 `0 < full_target < session_target < trigger < 1`；输出 Token 与安全余量后必须仍有正输入预算。
- [ ] 运行 `.venv/bin/python -m pytest tests/web/test_config.py -q`，确认因字段和解析器缺失而失败。
- [ ] 在 `paperpilot/web/config.py` 新增 `ContextManagementConfig`、`_read_bool`、`_read_ratio`，并从 `PAPERPILOT_CONTEXT_*` 环境变量一次性构造和校验配置；无效配置必须在应用启动时抛 `ValueError`。
- [ ] 在 `tests/deep_reading/test_runner.py` 写失败测试，断言模型工厂收到 `deepseek-v4-flash`，Runner 将同一个不可变配置对象传入 `DeepReadingContext`，恢复快路径不创建模型。
- [ ] 修改 `runner.py`、`nodes/context.py`、`app.py` 和 `worker_tasks.py` 的接线；保留注入 fake runner/model 的测试入口，不在模块 import 时读取环境变量。
- [ ] 在 `.env.example` 增加全部新变量，使用上表默认值；先不修改 README 的上线说明，留到 Task 12 与回滚步骤一起完成。
- [ ] 运行 `.venv/bin/python -m pytest tests/web/test_config.py tests/web/test_runtime_config.py tests/web/test_conversation_worker.py tests/deep_reading/test_runner.py -q`，确认配置、App、Thread/Celery Runner 接线全部通过。
- [ ] 运行 `git diff --check`，确认没有空白错误。
- [ ] 若已获得 Commit 授权，提交 `feat: define context management runtime bounds`；否则记录“未提交”并继续。

### Task 2: 建立上下文领域契约与统一 Token 预算

**Files:**

- Create: `paperpilot/deep_reading/context_management/__init__.py`
- Create: `paperpilot/deep_reading/context_management/models.py`
- Create: `paperpilot/deep_reading/context_management/ports.py`
- Create: `paperpilot/deep_reading/context_management/budget.py`
- Create: `tests/deep_reading/test_context_budget.py`
- Create: `tests/deep_reading/test_context_models.py`

**Interfaces:**

领域枚举固定为：

```python
class InitialAction(str, Enum):
    DROP_NOW = "DROP_NOW"
    EXTERNALIZE_NOW = "EXTERNALIZE_NOW"
    KEEP_INLINE = "KEEP_INLINE"


class FutureRetention(str, Enum):
    PROTECTED = "PROTECTED"
    CLEARABLE_AFTER_USE = "CLEARABLE_AFTER_USE"
```

`models.py` 的共享契约固定为：`ArtifactRef`、`ToolResultDisposition`、`ProtectedText`、`ArchiveSupersession`、`TurnArchiveSeed`、`ActiveProjection`、`RetrievedArchiveView`、`ContextView`、`SessionMemoryCandidate`、`ContinuationCapsule`、`CompressionAttempt` 和 `CompressionDecision`。所有模型使用 `ConfigDict(extra="forbid")`，所有 ID 和文本字段拒绝空白。

端口签名固定为：

```text
TokenCounter.count_messages(messages, *, tool_schemas=()) -> int
TokenCounter.count_text(text) -> int
TokenCounter.truncate_text(text, max_tokens) -> str
ContextArtifactPort.put(request) -> ContextArtifactRecord
ContextArtifactPort.get(artifact_id, *, conversation_id) -> ContextArtifactRecord | None
ContextArtifactPort.read_slice(artifact_id, *, conversation_id, cursor, max_tokens) -> ArtifactSlice
ContextArtifactPort.search(artifact_id, *, conversation_id, query, max_matches) -> ArtifactMatches
TurnArchivePort.seed(seed) -> TurnArchiveRecord
TurnArchivePort.list_for_conversation(conversation_id) -> list[TurnArchiveRecord]
CompressionStatePort.read(conversation_id, compressor_version) -> CompressionStateRecord
CompressionStatePort.record_outcome(outcome) -> CompressionStateRecord
```

预算公式固定为：

```text
usable_input_budget
  = model_context_window_tokens
  - configured_max_output_tokens
  - ceil(model_context_window_tokens * context_safety_margin_ratio)
```

Fallback 计数固定为 `max(ceil(UTF-8 bytes / 3), ceil(characters / 2))`，输入必须包含 role、name、tool call、tool result、结构化 wrapper 和 tool schema 的规范化 JSON。

- [ ] 在 `test_context_models.py` 写失败测试，覆盖所有枚举值、额外字段拒绝、空 ID 拒绝、重复引用拒绝和 `ToolResultDisposition` 构造后不可变。
- [ ] 在 `test_context_budget.py` 写失败测试，覆盖 ASCII、中文、混合文本、消息 role、tool_calls、tool schema、空输入和 safety margin 的精确边界。
- [ ] 增加测试：有模型级 `get_num_tokens_from_messages` 且成功时优先使用；模型计数不可用或抛错时使用保守 fallback；候选采用前必须重新计数。
- [ ] 运行 `.venv/bin/python -m pytest tests/deep_reading/test_context_models.py tests/deep_reading/test_context_budget.py -q`，确认模块缺失导致失败。
- [ ] 实现严格 Pydantic 模型和 Protocol；`ProtectedText` 的 `sha256` 必须由 UTF-8 exact text 计算并在 validator 中复核。
- [ ] 实现 `ModelAwareTokenCounter`：优先调用注入的模型计数器，失败时不向上泄漏计数器异常，改用保守 fallback；不得读取网络或下载 tokenizer。
- [ ] 实现 `usable_input_budget()`、`reclaim_threshold()` 和稳定的消息规范化函数，保证同一输入得到相同计数。
- [ ] 运行上述两个测试文件，确认通过。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: add context contracts and token budgeting`；否则跳过提交。

### Task 3: 新增三张内部表和显式 TaskStore 持久化接口

**Files:**

- Create: `migrations/versions/20260901_0003_context_memory.py`
- Modify: `paperpilot/web/db_models.py`
- Modify: `paperpilot/web/store/records.py`
- Modify: `paperpilot/web/store/helpers.py`
- Create: `paperpilot/web/store/context_memory.py`
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_db_models.py`
- Modify: `tests/web/test_db_migrations.py`
- Create: `tests/web/store/test_context_memory.py`
- Modify: `tests/architecture/test_web_module_boundaries.py`

**Database schema:**

`context_artifacts`：

| 列 | 约束 |
|---|---|
| `artifact_id` | `TEXT PRIMARY KEY` |
| `conversation_id` | `TEXT NOT NULL`, FK `conversations.id` |
| `task_id` | `TEXT NOT NULL`, FK `research_tasks.id` |
| `tool_call_id` | `TEXT NOT NULL` |
| `tool_name` | `TEXT NOT NULL` |
| `kind` | `TEXT NOT NULL` |
| `storage_key` | `TEXT NOT NULL UNIQUE` |
| `sha256` | `TEXT NOT NULL` |
| `byte_size` | `INTEGER NOT NULL` |
| `token_estimate` | `INTEGER NOT NULL` |
| `preview` | `TEXT NOT NULL` |
| `initial_action` | `TEXT NOT NULL` |
| `future_retention` | `TEXT NOT NULL` |
| `created_at` | `TEXT NOT NULL` |

唯一约束 `uq_context_artifacts_task_call_hash(task_id, tool_call_id, sha256)`；索引 `idx_context_artifacts_conversation_sha256(conversation_id, sha256)`。

`turn_archives`：使用设计文档 16.2 的列；`archive_id` 为主键，`narrative_summary` 可空，`narrative_status` 取 `pending|running|complete|failed`，唯一约束 `uq_turn_archives_conversation_message_version(conversation_id, user_message_id, archive_version)`，索引 `idx_turn_archives_conversation_created(conversation_id, created_at, archive_id)`。

`compression_states`：`conversation_id + compressor_version` 为联合主键，其余列使用设计文档 16.3；`state` 取 `CLOSED|OPEN|HALF_OPEN`。

**TaskStore methods:**

```text
create_context_artifact(*, record: NewContextArtifact) -> ContextArtifactRecord
get_context_artifact(artifact_id, *, conversation_id: str) -> ContextArtifactRecord | None
find_context_artifact_by_digest(*, conversation_id: str, tool_name: str, sha256: str) -> ContextArtifactRecord | None
seed_turn_archive(*, seed: TurnArchiveSeedRecord) -> TurnArchiveRecord
list_turn_archives(conversation_id: str) -> list[TurnArchiveRecord]
claim_turn_archive_narrative(archive_id: str) -> TurnArchiveRecord | None
complete_turn_archive_narrative(archive_id: str, *, narrative_summary: str) -> TurnArchiveRecord
fail_turn_archive_narrative(archive_id: str) -> TurnArchiveRecord
get_compression_state(conversation_id: str, compressor_version: str) -> CompressionStateRecord
record_compression_outcome(*, outcome: CompressionOutcomeRecord) -> CompressionStateRecord
```

- [ ] 先更新 `test_db_models.py` 和 `test_db_migrations.py` 的失败断言：12 张受管表、新列、三组唯一约束、外键和索引必须完全匹配。
- [ ] 增加迁移测试：空库直接升级、`20260807_0002` 库升级、已有业务行保留、重复 Archive Seed 被唯一约束拒绝、下迁移明确失败。
- [ ] 运行 `.venv/bin/python -m pytest tests/web/test_db_models.py tests/web/test_db_migrations.py -q`，确认新表尚不存在而失败。
- [ ] 实现 Alembic revision 和 SQLAlchemy Row；不得修改 `20260806_0001`、`20260807_0002` 历史迁移。
- [ ] 在 `records.py` 增加不可变 dataclass，在 `helpers.py` 增加 JSON 解码和 Row 转换；无效 JSON 必须抛确定性数据损坏错误，不返回空默认值。
- [ ] 在 `test_context_memory.py` 写失败测试，覆盖 Artifact 幂等查找、Archive Seed 幂等返回同一行、append-only seed、Narrative claim/complete/fail、Archive 顺序和 CompressionState 默认 CLOSED。
- [ ] 实现 `store/context_memory.py` 的短事务函数；Seed 已存在时必须逐字段比对，相同则返回，冲突则抛 `ValueError`。
- [ ] 实现 `record_compression_outcome` 在同一事务中更新 breaker row 并写脱敏 TaskEvent，保证状态和事件不会部分提交。
- [ ] 显式扩展 `TaskStore.__dict__` 方法及 `test_web_module_boundaries.py` 的方法集合、签名和委派测试；不使用 mixin、`__getattr__` 或动态注册。
- [ ] 运行 `.venv/bin/python -m pytest tests/web/test_db_models.py tests/web/test_db_migrations.py tests/web/store/test_context_memory.py tests/architecture/test_web_module_boundaries.py -q`。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: persist context artifacts archives and breaker state`；否则跳过提交。

### Task 4: 实现原子本地 ArtifactStore 与有界读取

**Files:**

- Create: `paperpilot/web/context_artifacts.py`
- Create: `tests/web/test_context_artifacts.py`
- Modify: `paperpilot/web/store/context_memory.py`
- Modify: `paperpilot/web/task_store.py`

**Interfaces and storage rules:**

- `LocalContextArtifactStore` 实现 Task 2 的 `ContextArtifactPort`，构造参数为 `root: Path`、`task_store: TaskStore`、`token_counter: TokenCounter`、`read_max_tokens: int`。
- `storage_key` 固定为 `<artifact_uuid>.json`，只允许 UUID 文件名；任何 `/`、`..`、绝对路径或软链接跳转均拒绝。
- 正文使用 UTF-8 canonical JSON；先写根目录内临时文件，flush + `os.fsync`，校验 SHA-256，再 `os.replace` 到最终名，最后写元数据。
- 元数据写失败时，只删除本调用刚创建且尚未被其他元数据引用的最终文件；已返回的 Artifact 文件永不覆盖。
- `read_slice` 的 `cursor` 是非负字符偏移，返回 `next_cursor: int | None`、`actual_tokens`、全文 `sha256` 和不超过配置上限的文本。
- `search` 使用 Unicode `casefold()` 子串匹配，`1 <= max_matches <= 10`，返回匹配字符范围和有界片段；总返回不超过 `artifact_read_max_tokens`。

- [ ] 在 `test_context_artifacts.py` 写失败测试，覆盖原子写入、canonical JSON、哈希、重复 digest 复用、会话隔离和不可变文件。
- [ ] 增加故障注入测试：临时写失败不落库；哈希失败不发布；DB 写失败清理本次文件；成功引用永不悬空。
- [ ] 增加安全测试：路径穿越、错误 conversation、未知 ID、哈希不匹配、负 cursor、越界 cursor、过大 max_tokens 和空 query 均返回确定性错误。
- [ ] 增加读取测试：连续 cursor 读取可无损重建 canonical JSON；每段 Token 不超限；search 结果顺序稳定且总预算受控。
- [ ] 运行 `.venv/bin/python -m pytest tests/web/test_context_artifacts.py -q`，确认实现缺失而失败。
- [ ] 实现 `LocalContextArtifactStore` 和内部 `_safe_artifact_path`；禁止向模型或 Event 暴露 `root`、临时路径和 `storage_key`。
- [ ] 在 TaskStore 增加只供 ArtifactStore 使用的“元数据创建失败清理判定”查询，不把文件 I/O 放进 SQLite transaction。
- [ ] 运行 `.venv/bin/python -m pytest tests/web/test_context_artifacts.py tests/web/store/test_context_memory.py -q`。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: add atomic local context artifact storage`；否则跳过提交。

### Task 5: 实现工具结果分类、冻结和工具专属 Preview

**Files:**

- Create: `paperpilot/deep_reading/context_management/artifacts.py`
- Create: `tests/deep_reading/test_context_artifact_policy.py`
- Modify: `paperpilot/deep_reading/context_management/models.py`
- Modify: `paperpilot/deep_reading/context_management/ports.py`

**Pipeline contract:**

```text
raw tool result
  -> deterministic noise classifier
  -> DROP_NOW, or budget/semantic-value policy
  -> KEEP_INLINE / EXTERNALIZE_NOW
  -> freeze ToolResultDisposition
  -> produce model-visible ToolMessage content
```

规则固定如下：

- 空/空白、相同工具指纹与相同哈希重复、协议回显、已被业务验证拒绝的记录、纯进度/调试元数据、State 已单独保存的重复状态，可 `DROP_NOW`。
- 无法确定语义价值时一律 `EXTERNALIZE_NOW`，不得 `DROP_NOW`。
- `token_estimate <= 2_000` 且当前步骤需要正文时 `KEEP_INLINE`；否则 `EXTERNALIZE_NOW`。
- `search_related_papers`、`retrieve_paper_evidence` 默认为 `CLEARABLE_AFTER_USE`；包含唯一失败验证、未完成 TODO 或受保护约束的结果为 `PROTECTED`。
- disposition 使用 `tool_call_id + content_sha256` 作为冻结键；同键重复调用必须得到完全相同的 action、preview 和 artifact ID，不能在 inline/ref 之间来回变化。

Preview 规则固定如下：

- Search：保留原顺序下全部 `external_id` 和 `title`；只给前 5 项保留最多 300 字符 abstract；作者和完整 abstract 留在 Artifact。
- Prepare：继续只返回 `external_id + status`，不保存或返回下载全文。
- Evidence：保留全部 `id`、`paper_external_id`、`paper_title`、`score`、`supports` 和全部 `summary_item_ids`；先保留 summary items 的完整 `chunk_text`，再按原顺序加入其他 chunk，直到 2,000 Token。
- DROP：返回固定 XML；属性集合严格为 `status`、`reason`、`result_id`、`tool_call_id`、`sha256`，其中 `status` 固定为 `dropped`，其余属性填入经过 XML 转义的实际 reason code、稳定 ID 和 64 位小写十六进制 digest；不生成 LLM 摘要。
- Externalize：返回固定 XML/JSON 组合，包含 `artifact_id`、`sha256`、`token_estimate`、preview 和读取工具名，不包含本地路径。

- [ ] 在 `test_context_artifact_policy.py` 写失败测试，逐条覆盖全部 DROP 规则，并断言分类过程中未调用模型。
- [ ] 写失败测试：语义不确定内容必定外置；2,000 Token 边界内联，2,001 Token 外置；Artifact 写失败且结果过大时工具调用失败而不是截断。
- [ ] 写 Search preview 属性测试：任意候选数都保留全部 ID/标题和顺序，最多 5 个 abstract，单 abstract 最多 300 字符，总体序列化稳定。
- [ ] 写 Evidence preview 测试：所有 Evidence ID/metadata 保留，summary item 优先，chunk 总预算不超过 2,000，完整 payload 可从 Artifact 无损恢复。
- [ ] 写 disposition 冻结测试：同一 tool call 重放返回相同引用；同一 fingerprint/digest 的后续重复结果变为 `DROP_NOW duplicate_result` 并指向既有权威 Artifact ID。
- [ ] 运行 `.venv/bin/python -m pytest tests/deep_reading/test_context_artifact_policy.py -q`，确认模块缺失而失败。
- [ ] 实现 `DeterministicNoiseClassifier`、`ToolResultPolicy`、三个 renderer 和 `ToolResultIngestor`；renderer 不得通过机械字符串切片破坏 JSON ID。
- [ ] 运行测试并检查每个 model-visible 返回都通过 TokenCounter 重计数。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: classify and externalize research tool results`；否则跳过提交。

### Task 6: 接入 Research Agent 工具、Status Bar 和批量微压缩

**Files:**

- Create: `paperpilot/deep_reading/context_management/editing.py`
- Modify: `paperpilot/deep_reading/research_agent.py`
- Modify: `paperpilot/deep_reading/research_status.py`
- Modify: `paperpilot/deep_reading/nodes/context.py`
- Modify: `tests/deep_reading/test_research_agent.py`
- Modify: `tests/deep_reading/test_research_status.py`
- Create: `tests/deep_reading/test_context_editing.py`

**Research wiring:**

- 三个现有工具使用 `response_format="content_and_artifact"` 返回模型内容和内部结构化 payload；`ResearchContextMiddleware.wrap_tool_call` 在结果第一次进入模型前调用 `ToolResultIngestor`，返回的新 ToolMessage 不保留大 `artifact` 字段。
- 新工具 schema 固定为：

```text
read_artifact_slice(artifact_id: str, cursor: int, max_tokens: int) -> dict
search_artifact(artifact_id: str, query: str, max_matches: int) -> dict
```

- `ContextEditingAdapter.edit(request, dispositions) -> EditedRequest` 定义在 `ports.py`；第一版只实现 `LocalContextEditingAdapter`。未来接 Provider 原生 editing 时只能替换 adapter，不得改变 DROP/EXTERNALIZE/retention 分类、Artifact 或归档语义。
- 两个读取工具必须校验 Artifact 属于当前 Conversation；`max_tokens <= artifact_read_max_tokens`，`max_matches <= 10`。
- `BUSINESS_TOOL_NAMES` 增加两个读取工具，使其计入同一 Research tool budget、重复调用 fingerprint、no-progress 和 Status Bar 观察。
- 中间件顺序固定为：ModelCallLimit → ResearchToolBudget → ResearchTodo → ResearchContextEditing → ResearchStatus → ModelRetry。必须用测试验证 Status Bar 仍是每次模型请求的最后一条 user-role 消息。
- `_RESEARCH_PROMPT_VERSION` 升为 `research-v7`；固定 System Prompt 只增加两项工具的可执行决策规则，不注入动态 Artifact ID。

**Micro-compaction eligibility:**

历史 ToolMessage 必须同时满足：已出现在至少一次成功 model call 的输入、`CLEARABLE_AFTER_USE`、下游已消费、可恢复、非 protected。触发必须同时满足：

```text
current_input_tokens >= 0.70 * usable_input_budget
reclaimable_tokens >= max(8_000, 0.10 * usable_input_budget)
```

选中后保留最近 3 个 ToolResult，其余在 request copy 中一次批量替换为稳定 `<artifact_ref>`；Agent state、LangGraph state、TaskStore 和 Artifact 文件不变。

- [ ] 在 `test_research_agent.py` 先写失败测试：大 Search/Evidence 结果只向第二次模型调用暴露 preview/ref，内部 candidate/evidence ledger 仍保留完整权威对象并能生成原 `ResearchResult`。
- [ ] 写读取工具测试：读取成功、跨 Conversation 拒绝、预算超限拒绝、读取调用进入 budget 和 tracker；Prepare 仍不返回论文全文。
- [ ] 在 `test_context_editing.py` 写失败测试，覆盖 eligibility 五条件、70% 边界、可回收双阈值、最近 3 条保留、一次批量替换和不修改原 request/state。
- [ ] 增加缓存稳定性测试：首次微压缩后，相同 disposition 的后续 model call 产生字节相同的历史替换序列；未达到阈值不编辑任何历史消息。
- [ ] 在 `test_research_status.py` 增加中间件组合测试，断言读取工具计数正确、Status XML 仍位于尾部、context middleware 失败不吞掉原工具异常。
- [ ] 运行 `.venv/bin/python -m pytest tests/deep_reading/test_context_editing.py tests/deep_reading/test_research_agent.py tests/deep_reading/test_research_status.py -q`，确认失败原因符合预期。
- [ ] 实现 `ResearchContextMiddleware` 的 `wrap_tool_call` 与 `wrap_model_call`；只复制 `request.messages`，禁止返回 LangGraph `RemoveMessage`。
- [ ] 修改三个工具的返回封装并新增两个读取工具；同步更新 Research Prompt 工具决策规则和 Prompt 结构测试。
- [ ] 发出脱敏事件 `artifact_externalized`、`tool_result_dropped`、`micro_compaction_started/completed`，payload 只含 stage、工具名、ID、before/after/reclaimed tokens 和原因。
- [ ] 运行上述 focused tests，随后运行 `.venv/bin/python -m pytest tests/deep_reading -q`。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: bound and edit research tool context`；否则跳过提交。

### Task 7: 生成逐业务轮次 Archive Seed 和后台 Narrative

**Files:**

- Create: `paperpilot/deep_reading/context_management/archives.py`
- Modify: `paperpilot/deep_reading/schemas.py`
- Modify: `paperpilot/deep_reading/state.py`
- Modify: `paperpilot/deep_reading/research_agent.py`
- Modify: `paperpilot/deep_reading/nodes/research_evidence.py`
- Modify: `paperpilot/deep_reading/runner.py`
- Create: `tests/deep_reading/test_context_archives.py`
- Modify: `tests/deep_reading/test_research_agent.py`
- Modify: `tests/deep_reading/test_runner.py`
- Modify: `tests/web/store/test_context_memory.py`

**Authoritative seed mapping:**

第一版不让归档 LLM 改写保护字段。`TurnArchiveSeedBuilder` 使用下列确定性来源：

| Seed 字段 | 权威来源 |
|---|---|
| `user_goal` | 当前 `MessageRecord.content` 原文 + message ID + SHA-256 |
| `constraints` | 经过 exact-span validator 的 `ResearchContextDelta.constraints`；完整当前用户请求仍由 `user_goal` 兜底保护 |
| `decisions` | 经过 exact-span validator 的用户决策、`ResearchResult.used_papers`、selected evidence IDs、`AnswerDraft.result_quality` |
| `paper_findings` | Evidence ID、Paper ID 和 citation label；不复制完整 chunk |
| `artifact_refs` | 本轮 ContextArtifact IDs |
| `failed_paths` | 结构化 research limitations、确定性工具错误 reason code |
| `verification` | citation validation、publication validation、terminal status 的 `pass|fail|not_run` |
| `unresolved_todos` | Research Agent 最终未完成 TODO；合法 structured response 前应为空 |
| `rollback_notes` | 当前 PaperPilot 无文件/数据回滚时为空列表 |
| `supersedes` | 经过 exact-span validator 的 `ResearchContextDelta.supersedes`；不得从 Narrative 推断 |

当前业务状态只有 `completed` 和 `failed` 终态：分别映射为 Archive 的 `success` 和 `failed`。Schema 预留 `cancelled`，但本任务不新增取消 API 或任务状态；未来只有权威 Task 状态真正支持取消后才能写 `cancelled`。

`AgentResearchDecision` 增加内部字段 `context_delta: AgentResearchContextDelta`，但对外 `ResearchResult` 不变。`context_delta` 固定包含：

```text
constraints: [{source_message_id, exact_text, kind}]
decisions: [{source_message_id, exact_text, kind}]
supersedes: [{target_protected_id, source_message_id, exact_text}]
```

validator 必须确认来源是同一 Conversation 的 user-role Message，`exact_text` 是该消息逐字子串，protected ID/hash 由服务端计算，supersedes target 已存在于本次 Authority Set。模型不能提交 hash、不能改写原文、不能引用未注入的旧约束。任何 context delta 校验失败只丢弃 delta，并发出 `turn_archive_failed`（stage=`context_delta`、reason=`invalid_exact_span`）；不得丢弃已经验证通过的 `ResearchResult`。完整当前 user goal 仍进入 Seed，因此不会静默丢失原请求。

`decisions` 中每个 Paper 选择记录 role 和 supporting Evidence IDs；这些显式引用就是可审计依据，不保存或臆造模型隐藏推理。用户写出的约束和理由以 exact text/hash 保护，后续压缩只能按 protected ID 原样携带。

`run_research_agent` 改为返回内部 `ResearchExecutionOutcome(result, trace, context_delta)`；`result` 是原 `ResearchResult`，字段和验证保持不变；`trace` 只含 TODO 状态、tool outcome reason、Artifact IDs 和验证状态，不含模型消息、用户原文或工具正文；只有 `context_delta` 可携带已经 exact-span 校验的用户原文片段。

Narrative Prompt 固定版本 `turn-archive-narrative-v1`，输出严格 `TurnArchiveNarrative(summary: str)`。它只解释 Seed，不返回也不更新 decisions、constraints、IDs、verification 或 supersedes。

- [ ] 在 `test_context_archives.py` 写失败测试，覆盖 success/failed 两种 Seed、精确 user goal hash、Paper/Evidence/Artifact 引用、TODO、verification 和空 file changes/rollback notes。
- [ ] 写测试证明旧 Summary 不会变成已确认事实；Narrative 无法修改 Seed；Seed JSON 在 Narrative 成功/失败前后字节相同。
- [ ] 在 `test_research_agent.py` 写失败测试，断言 `ResearchExecutionOutcome.result` 与旧返回完全等价，`trace` 不含 ToolMessage、Prompt、chunk text 或用户原文。
- [ ] 写 `ResearchContextDelta` 测试：合法 exact span 生成服务端 hash；改写一个字、引用 assistant message、跨 Conversation message、未知 protected target 均只丢弃 delta，不改变 ResearchResult。
- [ ] 写用户纠正 Seed 测试：新 delta 通过 `target_protected_id` 形成 scoped supersession，Archive 存储仍保留前后两条 Seed。
- [ ] 在 `test_runner.py` 写失败测试，覆盖顺序：先 `finalize_conversation_task`/`fail_conversation_task`，再独立 seed，再 Narrative；Seed/Narrative 失败都不改变终态。
- [ ] 增加 crash-window/redelivery 测试：终态已提交但 Seed 缺失时，下一次 no-claim/recovery 路径补写同一幂等 Seed；已存在 Seed 不重复。
- [ ] 运行 `.venv/bin/python -m pytest tests/deep_reading/test_context_archives.py tests/deep_reading/test_research_agent.py tests/deep_reading/test_runner.py tests/web/store/test_context_memory.py -q`。
- [ ] 实现 `ResearchExecutionOutcome`、`ResearchTrace` 和 Seed Builder；Runner 的 success、terminal failure、retry exhausted 三个终态入口统一调用 `_archive_terminal_best_effort`。
- [ ] 在 Seed 成功后，同一后台 worker 继续执行 Narrative enrichment；Task 的 completed/failed 事务已经提交，Narrative 只更新 `narrative_summary/status` 并发出脱敏事件。
- [ ] Narrative 使用独立 `DeepSeekUsageCallback`，以 `stage=archive_narrative` 和 `prompt_version=turn-archive-narrative-v1` 持久化用量；回调失败不掩盖 Narrative 或原 Task 结果。
- [ ] Narrative 临时 provider 错误只把状态置为 failed，允许后续幂等 claim 重试；不得走 Task execution retry，也不得覆盖原业务异常。
- [ ] 运行 focused tests 和 `.venv/bin/python -m pytest tests/deep_reading/test_runner.py tests/web/store/test_publications.py -q`。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: archive each terminal research turn`；否则跳过提交。

### Task 8: 构建 ActiveProjection、Archive 检索和唯一模型视图

**Files:**

- Create: `paperpilot/deep_reading/context_management/views.py`
- Create: `paperpilot/deep_reading/nodes/prepare_context.py`
- Modify: `paperpilot/deep_reading/research_agent.py`
- Modify: `paperpilot/deep_reading/nodes/write_answer.py`
- Modify: `paperpilot/deep_reading/nodes/summarize_history.py`
- Modify: `paperpilot/deep_reading/state.py`
- Modify: `paperpilot/deep_reading/nodes/__init__.py`
- Create: `tests/deep_reading/test_context_views.py`
- Modify: `tests/deep_reading/test_nodes.py`
- Modify: `tests/deep_reading/test_research_agent.py`

**View selection contract:**

未使用 Capsule：

```text
System Prompt
PaperPilot Runtime Context
Active Projection
Retrieved Turn Archives
Recent Conversation Messages
Research Status Bar（仅 Research Agent 尾部）
```

使用 Capsule：

```text
System Prompt
PaperPilot Runtime Context
Continuation Capsule
Recent Two Turns
Current User Message
Research Status Bar（仅 Research Agent 尾部）
```

`ContextViewBuilder` 每次从 TaskStore 稳定消息、TurnArchive Seed、当前业务状态和 supersession relation 重建，不以上一轮 ActiveProjection 为事实源。`current_goal` 始终取最新当前用户消息；旧 goal 仅留在 Archive。

Archive 检索算法固定为：

1. 提取当前请求中格式合法的 Archive/Paper/Evidence/Artifact ID，精确匹配；
2. 加入与当前 active paper ID 有交集的 Archive；
3. 加入最近 2 条 `turn-archive-v1`；
4. 若用户请求包含“历史/原因/变化/之前”等查询意图且精确命中 scoped supersession，加入关系两端；
5. 去重后按“精确 ID → active paper → recent → created_at/archive_id”排序；
6. 依次加入，最多 5 条且总计不超过 4,000 Token；单条超预算时保留 Seed IDs/verification，省略 Narrative。

第一版轻量文本匹配只对用户请求与 Narrative/goal 做 Unicode casefold 后的完整 ID/词项交集，不增加 Embedding。

- [ ] 在 `test_context_views.py` 写失败测试，覆盖 ActiveProjection 最新 goal、有效约束/决定、open TODO、failed verification、unresolved question 和 scoped supersession。
- [ ] 写用户纠正投影测试：新 Archive 的 scoped supersession 使 Active Projection 只采用新 exact text；历史查询仍能同时检索前后两条 Archive。
- [ ] 写 Archive 检索排序与预算测试：精确旧 ID 胜过最近记录、默认最多 5 条/4,000 Token、最近 2 条存在、superseded 内容仅在历史查询时加载。
- [ ] 写 legacy migration 测试：v1 `conversation_summary` 原 JSON 只写一次 `legacy-summary-v1`，视为 untrusted，成功后不再与 Projection/Capsule 同时注入。
- [ ] 在 `test_research_agent.py` 和 `test_nodes.py` 写失败测试，逐条断言两种消息顺序；Write Answer 使用同一 ContextView，而不是独立拼接旧 Summary 和六轮历史。
- [ ] 运行 `.venv/bin/python -m pytest tests/deep_reading/test_context_views.py tests/deep_reading/test_nodes.py tests/deep_reading/test_research_agent.py -q`。
- [ ] 实现 `ActiveProjectionBuilder`、`ArchiveRetriever`、`ContextViewBuilder` 和确定性 XML/JSON renderer；renderer 输出必须稳定排序，动态内容只使用 HumanMessage 技术槽位。
- [ ] 实现 `prepare_context` 节点的前四层外层视图准备；当 master flag 关闭时继续调用现有 `summarize_history` 兼容路径。
- [ ] 修改 `_research_messages` 和 `write_answer` 只消费 `context_view`；Research Runtime Context 仍把 primary paper 放首位，其余 active IDs 去重排序。
- [ ] `_ANSWER_PROMPT_VERSION` 升为 `answer-v3`，Research 保持 Task 6 的 `research-v7`；System 文本只有输入契约变化所必需的静态规则。
- [ ] 运行 focused tests，确认旧 `ConversationSummary` 不会重复注入，Status Bar 仍在 Research 尾部。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: assemble archive-backed context views`；否则跳过提交。

### Task 9: 实现两阶段压缩、候选验证和原子采用

**Files:**

- Create: `paperpilot/deep_reading/context_management/compaction.py`
- Modify: `paperpilot/deep_reading/context_management/models.py`
- Modify: `paperpilot/deep_reading/context_management/views.py`
- Modify: `paperpilot/deep_reading/nodes/prepare_context.py`
- Create: `tests/deep_reading/test_context_compaction.py`
- Modify: `tests/deep_reading/test_context_views.py`

**Coordinator contract:**

`CompressionCoordinator.prepare(view, *, task_id, conversation_id)` 先对前四层后的完整模型输入计数。低于 80% 直接返回原 view；达到 80% 且 full-compaction flag 开启时：

1. Stage A 只压缩 Projection 中已完成/已 supersede 的叙事、Retrieved Archive Narrative 和已归档旧结论副本。
2. Stage A Candidate 必须符合 `SessionMemoryCandidate`，经过引用和 protected set 校验，并使最终输入 `<= 65% usable` 才采用并停止。
3. Stage A 后仍 `>= 80% usable` 才运行 Stage B。
4. Stage B 生成 `ContinuationCapsule`，保留最近 2 个已完成业务轮次和当前用户消息。
5. Stage B Candidate 必须使最终输入 `<= 50% usable` 才采用。

当 master flag 开启但 full-compaction flag 关闭时，Coordinator 绝不调用 compressor：最终视图 `<= usable_input_budget` 时按前四层结果继续；超过 usable budget 时直接走 Task 10 的确定性最小安全视图，仍超限则返回 `context_capacity_exhausted`。不得回退到旧滚动 Summary，也不得把超预算请求直接发给 Provider。

`ContinuationValidator` 必须依次验证：

- Pydantic schema 严格通过且无 extra fields；
- Candidate 中所有 Archive/Artifact/Evidence/Paper ID 都存在于输入 authority set；
- `protected_ids_before == protected_ids_after`；
- 每个 `ProtectedText` 的 exact text SHA-256 一致；
- 当前用户目标、明确约束、未完成 TODO、fail verification、未解决问题、回滚说明和 active refs 覆盖率 100%；
- 对采用后的真实 model messages 重新 Token 计数并达到目标。

压缩 Prompt 固定版本：Stage A `session-memory-compressor-v1`，Stage B `continuation-compressor-v1`。模型只能返回 Candidate；不能直接改 State、Archive 或 Artifact。

- [ ] 在 `test_context_compaction.py` 写失败测试：79.99% 不调用模型，80% 调 Stage A，Stage A 达 65% 后不调 Stage B，Stage A 仍达 80% 才调 Stage B。
- [ ] 写 validator 参数化测试，分别拒绝 extra field、虚构 ID、缺失 protected ID、文本哈希变化、否定词丢失、dangling ref、65%/50% 未达标。
- [ ] 写原子性测试：Candidate 失败时返回原 view 的深拷贝等价值；成功时只替换派生 view，权威 Archive/Artifact/TaskStore mock 无写入。
- [ ] 写 transient retry 测试：timeout、429、5xx、连接中断每阶段最多额外重试 1 次；schema/ref/hash/insufficient reduction 同轮不重试。
- [ ] 写 `protected_context_oversized` 测试：受保护集合本身超过目标时不调用 Stage B 要求模型删除，直接返回确定性失败类型。
- [ ] 运行 `.venv/bin/python -m pytest tests/deep_reading/test_context_compaction.py -q`，确认模块缺失而失败。
- [ ] 实现两个结构化 compressor、`ContinuationValidator` 和 `CompressionCoordinator`；异常分类使用显式 enum，不用字符串包含判断。
- [ ] 在每次模型调用前后发出 `session_memory_compaction_started`、`full_compaction_started`、`compression_candidate_rejected` 及成功指标事件；不得记录 Candidate 文本。模型调用 metadata 分别使用 `stage=session_memory_compaction|full_compaction` 和对应 Prompt version，使现有 graph-level usage callback 自动聚合。
- [ ] 运行 `.venv/bin/python -m pytest tests/deep_reading/test_context_compaction.py tests/deep_reading/test_context_views.py -q`。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: validate two-stage context compaction`；否则跳过提交。

### Task 10: 实现持久化熔断器和最小安全降级路径

**Files:**

- Create: `paperpilot/deep_reading/context_management/breaker.py`
- Modify: `paperpilot/deep_reading/context_management/compaction.py`
- Modify: `paperpilot/deep_reading/context_management/views.py`
- Modify: `paperpilot/deep_reading/research_agent.py`
- Modify: `paperpilot/deep_reading/runner.py`
- Create: `tests/deep_reading/test_context_breaker.py`
- Modify: `tests/deep_reading/test_context_compaction.py`
- Modify: `tests/web/store/test_context_memory.py`
- Modify: `tests/deep_reading/test_runner.py`

**State machine:**

```text
CLOSED -- third consecutive full-pipeline failure --> OPEN
OPEN -- eligible recovery condition -------------> HALF_OPEN
HALF_OPEN -- one successful probe ----------------> CLOSED
HALF_OPEN -- one failed probe --------------------> OPEN
```

计数单位是一整条 Stage A/必要时 Stage B 流程；任一可采用 Candidate 成功即清零。Key 固定为 `(conversation_id, compressor_version)`，其中 compressor version 包含 model name、两个 Prompt version 和输入 schema version。

恢复规则固定为：

- transient failure：`opened_at + 300 seconds` 后允许一次 HALF_OPEN；
- deterministic schema/protected/ref failure：只有 compressor version 或 `last_input_digest` 改变才允许 HALF_OPEN；
- `protected_context_oversized`：等待不恢复，只有输入 digest 或 model context window 改变才允许探测。

OPEN 时禁止调用压缩模型。最小安全视图固定为 System + Runtime + current goal + 全部 protected constraints + open TODO + failed verification + authority refs + 最近 1 个完整业务轮次。若仍超 usable budget，写入一个内部 `continuation_capsule` Artifact 后抛 `ContextCapacityExhaustedError(error_code="context_capacity_exhausted")`。

- [ ] 在 `test_context_breaker.py` 写失败测试，覆盖前两次失败保持 CLOSED、第三次 OPEN、成功清零、进程重建后状态不丢失。
- [ ] 写 HALF_OPEN 测试：transient 冷却前拒绝、冷却后只允许一个 probe；deterministic 在版本/input digest 不变时拒绝；probe 成功/失败转移正确。
- [ ] 写测试证明 OPEN 和受保护内容过大时压缩 model mock 调用次数为 0。
- [ ] 写降级视图测试，逐项断言 protected 内容和引用完整；最小视图可容纳时继续并发 `compression_degraded`，不可容纳时抛安全终态错误并保存续接 Artifact。
- [ ] 写存储原子性测试：breaker state 和 `compression_circuit_*`/`compression_degraded` Event 同时提交或同时回滚。
- [ ] 运行 `.venv/bin/python -m pytest tests/deep_reading/test_context_breaker.py tests/web/store/test_context_memory.py -q`。
- [ ] 实现 `CompressionCircuitBreaker`、单 probe claim 和最小安全 `ContextViewBuilder`；并发 worker 通过 SQLite 条件 update 保证只有一个 HALF_OPEN probe。
- [ ] 将 breaker 接到 Coordinator 外层；provider 原异常保留为 `__cause__`，Archive/Narrative 错误不得增加压缩失败计数。
- [ ] 在 Runner 中把 `ContextCapacityExhaustedError` 纳入现有 `DeepReadingTaskError` 安全终态路径，public message 不泄漏上下文内容。
- [ ] 状态转换分别发出 `compression_circuit_open`、`compression_circuit_half_open`、`compression_circuit_closed`；降级和容量终止分别发出 `compression_degraded`、`context_capacity_exhausted`，不得只发一个泛化 `compression_*` 事件。
- [ ] 运行 focused tests 和 `.venv/bin/python -m pytest tests/deep_reading/test_runner.py -q`。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: circuit-break failed context compaction`；否则跳过提交。

### Task 11: 完成 State/Graph v2 迁移与 App/Worker 运行时接线

**Files:**

- Modify: `paperpilot/deep_reading/state.py`
- Modify: `paperpilot/deep_reading/graph.py`
- Modify: `paperpilot/deep_reading/nodes/initialize_turn.py`
- Modify: `paperpilot/deep_reading/nodes/context.py`
- Modify: `paperpilot/deep_reading/nodes/__init__.py`
- Modify: `paperpilot/deep_reading/runner.py`
- Create: `paperpilot/web/context_runtime.py`
- Modify: `paperpilot/web/app.py`
- Modify: `paperpilot/web/worker_tasks.py`
- Modify: `tests/deep_reading/test_graph.py`
- Modify: `tests/deep_reading/test_runner.py`
- Create: `tests/web/test_context_runtime.py`
- Modify: `tests/web/test_conversation_worker.py`
- Modify: `tests/web/test_celery_worker.py`

**State migration:**

新写入版本固定为 `SCHEMA_VERSION = 2`、`GRAPH_VERSION = "conversation-v2"`。`DeepReadingState` 新增 checkpoint-safe 字段：

```text
context_view: dict[str, object] | None
continuation_capsule: dict[str, object] | None
retrieved_archive_ids: list[str]
research_trace: dict[str, object] | None
context_input_tokens: int | None
```

数据库 handle、ArtifactStore、TokenCounter、模型和 Protocol 实例只放 `DeepReadingContext.context_management_runtime`，不得进入 checkpoint。

迁移规则固定为：

- v1 完成 base checkpoint：Runner 验证原绑定字段后允许作为只读 base，`initialize_turn` 在本轮第一条新 checkpoint 中补 v2 默认字段。
- v1 已发布但未 finalization 的 complete recovery checkpoint：仍按旧字段验证并完成原 finalization，随后 best-effort Seed；不重跑模型。
- 其他未知 graph/schema version：继续抛现有 unsupported errors。
- feature flag 关闭：走旧 `summarize_history` 路径并写 v2 基础字段；feature flag 开启：走 `prepare_context`，同一次调用不再运行旧 summary。

Graph 路由固定为：

```text
START -> initialize_turn
initialize_turn -> summarize_history（master flag off 且旧阈值触发）
initialize_turn -> prepare_context（master flag on）
initialize_turn -> prepare_primary_paper（master flag off 且无需旧 summary）
prepare_context/summarize_history -> prepare_primary_paper
prepare_primary_paper -> research_evidence -> write_answer -> publish_result -> END
```

- [ ] 在 `test_graph.py` 写失败测试，覆盖 flag off 的旧两条路径、flag on 的新路径和固定节点顺序；不增加 Research Agent 内部 recursion step。
- [ ] 在 `test_runner.py` 写 v1→v2 checkpoint 迁移测试、v1 crash recovery finalization 测试、未知版本拒绝测试和第二轮/rollback 分支测试。
- [ ] 在 `test_context_runtime.py` 写失败测试，断言 Web adapter 把同一 TaskStore、配置、LocalContextArtifactStore、TokenCounter、Archive/Compression ports 组装为一个不可变 runtime。
- [ ] 在 `test_conversation_worker.py` 和 `test_celery_worker.py` 写接线测试，覆盖 Thread/Celery 相同配置、资源关闭顺序、redelivery 不重复 seed、import 不读环境。
- [ ] 运行 `.venv/bin/python -m pytest tests/deep_reading/test_graph.py tests/deep_reading/test_runner.py tests/web/test_context_runtime.py tests/web/test_conversation_worker.py tests/web/test_celery_worker.py -q`。
- [ ] 实现 State 版本、Graph 路由和 Runner 双版本 validator；不得放宽 message/checkpoint head、task/message owner 和 published message 绑定。
- [ ] 实现 `build_context_management_runtime(store, config, model)`；master flag 关闭时返回 `None`，且不得创建 Artifact 根目录或做数据库外写。
- [ ] App 与 worker 只在实际执行新 Task 时构造模型和 context runtime；trusted recovery/no-model 路径仍不产生空 usage/compression Event。
- [ ] 运行 focused tests 后运行 `.venv/bin/python -m pytest tests/deep_reading tests/web -q`。
- [ ] 运行 `git diff --check`。
- [ ] 若已获得 Commit 授权，提交 `feat: wire context management into graph runtime`；否则跳过提交。

### Task 12: 完成回滚说明、架构门禁、全量验证和灰度验收

**Files:**

- Modify: `README.md`
- Modify: `.env.example`
- Modify: `tests/architecture/test_repository_allowlist.py`
- Modify: `tests/web/test_runtime_config.py`
- Modify: `tests/deep_reading/test_model_usage.py`
- Modify: `paperpilot/deep_reading/model_usage.py` only if provider fields require additive parsing
- Verify: all files changed by Tasks 1-11

**Rollout sequence:**

1. 备份业务 SQLite、LangGraph checkpoint SQLite 和 `data/context-artifacts` 根目录。
2. 部署代码并运行 Alembic 0003，保持两个 feature flag 为 false；验证旧 Conversation 和新 Task 基线。
3. 对内部测试用户开启 master flag，仅启用前四层；观察 Artifact 外置率、读取恢复、Archive seed 成功率和输入 Token。
4. 前四层稳定后再开启 full-compaction flag；观察 Candidate rejection、压缩降幅、breaker 和任务成功率。
5. 回滚时先关闭两个 flag，再部署旧代码并恢复实施前数据库/checkpoint/Artifact 备份；禁止对已迁移库运行旧 ORM 后声称完成回滚。

**Acceptance metrics:**

- 输入 Token 平均值和 p50/p95；
- Artifact 外置率、读取率、哈希/悬空引用错误数；
- DROP、micro、Stage A、Stage B 触发次数和 before/after/reclaimed tokens；
- Archive Seed/Narrative success/failure；
- Candidate rejection reason；
- breaker OPEN/HALF_OPEN/CLOSED；
- Provider 返回时的 cache hit/miss tokens；
- Task 成功率、p95 延迟和估算成本。

- [ ] 在 `test_runtime_config.py` 先更新失败断言，要求 `.env.example` 精确包含所有新变量；保留既有变量，不删除用户当前修改。
- [ ] 在 `test_repository_allowlist.py` 增量加入本设计文档和本实施计划；先读取当前集合，保留 Status Bar 文档及其他用户条目。
- [ ] 在 `test_model_usage.py` 增加可选 cache hit/miss 字段回归；Provider 未返回时保持 `None`，不得伪造 0。
- [ ] 更新 README：配置表、Artifact 目录权限/备份、迁移、灰度顺序、禁用/恢复步骤、Event 脱敏和“本地测试不等于生产收益”。
- [ ] 运行迁移专项：`.venv/bin/python -m pytest tests/web/test_db_migrations.py tests/web/test_db_models.py tests/web/store/test_context_memory.py -q`。
- [ ] 运行上下文专项：`.venv/bin/python -m pytest tests/deep_reading/test_context_budget.py tests/deep_reading/test_context_models.py tests/deep_reading/test_context_artifact_policy.py tests/deep_reading/test_context_editing.py tests/deep_reading/test_context_archives.py tests/deep_reading/test_context_views.py tests/deep_reading/test_context_compaction.py tests/deep_reading/test_context_breaker.py -q`。
- [ ] 运行 Research/Runner/Web 回归：`.venv/bin/python -m pytest tests/deep_reading tests/web tests/architecture -q`。
- [ ] 运行完整测试：`.venv/bin/python -m pytest -q`，记录 passed/failed/deselected/warnings 的实际数字，不沿用历史数字。
- [ ] 运行 `git diff --check`；再运行 `git status --short`，逐项核对未提交文件与实施前快照，确认未覆盖无关改动。
- [ ] 人工检查事件 payload：抽样 `artifact_externalized`、`micro_compaction_completed`、`turn_archive_*`、`compression_*`，确认无 Prompt、用户原文、论文正文、工具正文或凭证。
- [ ] 人工执行一个长 Search/Evidence 场景和一个连续压缩失败场景，保存本地 trace：必须看到完整 Artifact 可恢复、模型只见 preview、第三次失败后不再调用 compressor。
- [ ] 明确记录未验证项：没有真实生产流量时，不得声称 Token 降低 75%、缓存命中改善、成本下降或任务成功率提升。
- [ ] 若已获得 Commit 授权，提交 `docs: document context compression rollout`；否则保留可审查 diff，不 commit/push/merge。

---

## 任务依赖与停止条件

```text
Task 1 配置/模型审批
  -> Task 2 领域契约与预算
  -> Task 3 数据库持久化
  -> Task 4 Artifact 文件后端
  -> Task 5 工具结果策略
  -> Task 6 Research 接线与微压缩
  -> Task 7 TurnArchive
  -> Task 8 ActiveProjection/模型视图
  -> Task 9 两阶段压缩
  -> Task 10 熔断/降级
  -> Task 11 State/Graph/App 集成
  -> Task 12 灰度与全量验证
```

## 设计覆盖矩阵

| 已确认设计要求 | 实施任务 | 主要验证 |
|---|---|---|
| Token 预算和 5% safety margin | Task 1-2 | `test_context_budget.py` |
| 大结果外置、专属 Preview、有界读取 | Task 4-6 | Artifact 故障注入、Search/Evidence preview、读取工具测试 |
| 确定性噪声立即 DROP，不进 backlog | Task 5 | noise classifier 无模型调用、即时 XML replacement |
| 历史 CLEARABLE 结果在 70%/回收阈值批量编辑 | Task 6 | request-copy、不改 State、最近 3 条、稳定序列化 |
| 每个外层终态一条 append-only Archive | Task 3、7 | Seed 幂等、redelivery/crash window、Narrative 不改 Seed |
| Active Projection + 5 条/4,000 Token Archive | Task 8 | exact ID/recent/supersession 排序和预算测试 |
| 80% → 65% → 50% 两阶段压缩 | Task 9 | 边界、阶段短路、候选后重计数 |
| protected ID/hash 100% 和引用校验 | Task 2、9 | 否定词、缺 ID、虚构 ref、hash 变化拒绝 |
| 三次失败熔断和最小安全视图 | Task 3、10 | 持久化三态、并发 HALF_OPEN、无模型降级 |
| 旧 Summary 与 v1 checkpoint 迁移 | Task 8、11 | legacy archive 唯一注入、v1→v2、crash recovery |
| 固定 System + 动态尾部状态 | Task 6、8 | 两种消息顺序、Status Bar 尾部、稳定前缀 |
| 数据库/文件/配置/灰度/回滚审批 | Task 1、3-4、11-12 | migration、flag-off 回归、备份恢复说明 |
| 全量事件和生产指标 | Task 6-7、9-10、12 | exact event type、脱敏 payload、usage/cache 可选字段 |

以下任一情况出现时立即停止实施并回到用户审批：

- 迁移不能在已有 `20260807_0002` 数据上无损升级；
- 需要修改公开 API、`ResearchResult` 或 publication/finalization transaction；
- Artifact 引用可能先于正文/元数据持久化；
- 无法从 v1 checkpoint 安全验证并迁移绑定；
- 压缩 Candidate 无法做到 protected ID/hash 100% 等价；
- feature flag 关闭时旧行为或恢复路径发生变化；
- 需要新增依赖、自动删除已发布 Artifact 或保存用户/论文原文到日志；
- 当前工作树中的用户改动与计划修改无法可靠合并。

## 完成定义

只有同时满足以下条件，才能报告“五层上下文管理已实现”：

1. Tasks 1-12 的代码和测试步骤全部完成；
2. feature flag 关闭时旧行为回归通过；
3. 前四层和第五层分别有独立启用测试；
4. 完整测试与 `git diff --check` 通过；
5. 数据库迁移、v1 checkpoint 迁移、redelivery、Archive crash window 和 breaker 并发路径通过；
6. 无悬空 Artifact、无 Seed 重复、无 Candidate 未验证采用、无 TaskStore 历史删除；
7. 最终汇报列出实际测试数字、未验证的生产指标和工作树剩余改动；
8. 未经明确授权，不执行 commit、push、merge 或生产灰度操作。
