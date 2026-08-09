# PaperPilot Conversation + LangGraph 纵向切片实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留旧 `/api/tasks` 运行路径的前提下，交付支持论文选择、会话管理、完整回复、真实连续追问、持久化 checkpoint、非破坏性回滚和分支切换的 LangGraph 纵向切片。

**Architecture:** FastAPI 与 Celery 继续作为产品入口和后台执行器，SQLAlchemy/Alembic 业务 SQLite 保存用户可见的 Paper、Conversation、Message 和 Task；独立的 LangGraph `SqliteSaver` 保存每个 `conversation.id == thread_id` 的 Graph State。LangGraph 只负责编排固定宏观 SOP，LangChain 负责 DeepSeek 模型、structured output 和受限 Research Agent，Pydantic 负责节点及 API 数据契约。

**Tech Stack:** Python 3.12、FastAPI、SQLAlchemy 2、Alembic、SQLite WAL、Celery/Redis、LangGraph 1.2、LangChain 1.3、`langgraph-checkpoint-sqlite` 3.1、`langchain-deepseek` 1.1、Pydantic 2、原生 HTML/CSS/JavaScript、pytest。

## Global Constraints

- 设计依据固定为 `docs/codex-only-plans/2026-08-07-paperpilot-conversation-langgraph-vertical-slice-design.md`；实现不能暗中扩大到未来 Store、完整最终 SOP、token 流或旧入口删除。
- `conversation.id` 必须直接作为 LangGraph `thread_id`；不得再建立用户名到 thread 的重复映射表。
- 业务数据库继续使用 `PAPERPILOT_TASK_DB_PATH`；checkpoint 使用独立的 `PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH`，默认 `data/langgraph/checkpoints.sqlite3`。
- 依赖范围必须保持：`langgraph>=1.2.9,<1.3`、`langchain>=1.3.14,<1.4`、`langgraph-checkpoint-sqlite>=3.1,<3.2`、`langchain-deepseek>=1.1,<1.2`、`pydantic>=2.13,<3`。
- `LANGGRAPH_STRICT_MSGPACK` 缺省按安全值 `true` 处理；显式设置为 false 时启动失败。
- 新 Graph 的工具调用与 structured output 模型固定使用 `deepseek-chat`；不得使用 `deepseek-reasoner`。
- 长上下文配置固定为 `PAPERPILOT_SUMMARY_TOKEN_THRESHOLD=32000`、`PAPERPILOT_SUMMARY_RECENT_TURNS=6`、`PAPERPILOT_RESEARCH_RECURSION_LIMIT=12`，并在 `WebRuntimeConfig.from_env()` 中验证为正整数。
- `InMemorySaver` 只允许出现在单元测试；重启、回滚、fork 和并发承诺必须用真实 `SqliteSaver` 验证。
- 自动化测试全部使用 fake model/fake tools，不读取 `DEEPSEEK_API_KEY`；真实模型烟雾测试必须再次获得用户明确批准。
- 同一 Conversation 同时最多一个 `pending`/`running` Task；不同 Conversation 可以并行。
- 首版 Task 状态只允许 `pending/running/completed/failed`，不实现 interrupt、`waiting_input`、cancel 或暂停恢复 UI。
- rollback 只移动业务 head，不调用模型、不创建 checkpoint、不删除旧消息或旧 checkpoint。
- Agent 搜索到但未实际用于证据、比较或引用的论文不得写入 `conversation_papers`。
- 旧 `/api/tasks`、旧 `WorkflowRunner`、CLI、Eval 和现有测试必须保持兼容。
- 不引入 Ports、Adapters、Repository 或另一套自定义工作流框架。
- 执行计划前使用 `superpowers:using-git-worktrees` 建立隔离工作树；不得触碰当前主工作树中用户已有的未跟踪文件 `docs/codex-only-plans/2026-08-05-architecture-audit-plan.md`。

### 用户批准的 State 契约修正（2026-08-09）

Task 9 实施前确认到原计划只持久化 `evidence_items`，会丢失 Task 10 发布所需的
`ResearchResult.used_papers` 和 `limitations`。用户选择用一个完整业务对象替换分散字段：

- `DeepReadingState` 使用 `research_result: dict[str, object] | None`，不再使用顶层
  `evidence_items`。
- `research_evidence()` 只写入 `ResearchResult.model_dump(mode="json")` 的完整结果。
- `write_answer()` 从 `research_result.evidence_items` 校验引用；`publish_result()` 从
  `research_result.used_papers` 构造 `UsedPaperInput`。
- `initialize_turn()` 每轮把 `research_result` 清为 `None`。该切片尚未接入生产
  checkpoint，因此仍视为首版 schema，`SCHEMA_VERSION` 保持 `1`。

---

## 文件职责总览

### 新增运行时代码

- `paperpilot/papers.py`：arXiv 引用规范化、结构化搜索和精确解析；Web 与 arXiv MCP 共用。
- `paperpilot/deep_reading/state.py`：`DeepReadingState` 与消息 reducer。
- `paperpilot/deep_reading/schemas.py`：研究证据、论文使用、摘要和答案的 Pydantic 契约。
- `paperpilot/deep_reading/nodes.py`：固定 SOP 节点与可信 `DeepReadingContext`。
- `paperpilot/deep_reading/research_agent.py`：LangChain Agent 及受限论文搜索、准备和检索工具。
- `paperpilot/deep_reading/graph.py`：只定义 StateGraph 节点、固定边和摘要条件边。
- `paperpilot/deep_reading/runner.py`：Task 到 Graph 的唯一调用、恢复和 finalization 边界。
- `paperpilot/web/checkpoint.py`：`SqliteSaver` 连接、setup、health 和关闭生命周期。
- `paperpilot/web/conversation_routes.py`：Paper/Conversation/Message/rollback API 与 Pydantic API schema。

### 修改现有代码

- `requirements.txt`、`requirements-lock.txt`：依赖范围与可复现解析结果。
- `paperpilot/mcp_servers/arxiv.py`：复用 `paperpilot.papers`，保留原 MCP 文本输出兼容性。
- `paperpilot/web/db_models.py`、`migrations/versions/20260807_0002_conversations.py`：新表、Task 字段和索引。
- `paperpilot/web/task_store.py`：现有 SQLAlchemy 风格下增加 Conversation 事务方法，不新增 Repository。
- `paperpilot/web/config.py`、`.env.example`：checkpoint、摘要阈值和 Agent 预算配置。
- `paperpilot/web/workflow.py`、`paperpilot/web/worker_tasks.py`、`paperpilot/web/app.py`：按 `research_tasks.conversation_id` 路由新旧 Runner并管理进程级资源。
- `paperpilot/web/static/index.html`、`paperpilot/web/static/app.js`：增加会话入口并保留旧工作台。
- `paperpilot/web/static/conversations.js`、`paperpilot/web/static/conversations.css`：隔离新增会话交互，避免继续放大已有 619 行 `app.js` 和 676 行 `styles.css`。
- `README.md`：迁移、checkpoint setup、运行和恢复说明。

### 新增测试目录/文件

- `tests/papers/test_arxiv_catalog.py`
- `tests/deep_reading/test_dependency_contract.py`
- `tests/deep_reading/test_schemas.py`
- `tests/deep_reading/test_nodes.py`
- `tests/deep_reading/test_research_agent.py`
- `tests/deep_reading/test_graph.py`
- `tests/deep_reading/test_runner.py`
- `tests/web/test_checkpoint.py`
- `tests/web/test_conversation_store.py`
- `tests/web/test_conversation_api.py`
- `tests/web/test_conversation_worker.py`
- `tests/web/test_conversation_ui.py`

---

### Task 1: 锁定 LangGraph/LangChain 依赖与安全基线

**Files:**
- Modify: `requirements.txt:1-37`
- Create: `requirements-lock.txt`
- Create: `tests/deep_reading/__init__.py`
- Create: `tests/deep_reading/test_dependency_contract.py`

**Interfaces:**
- Consumes: 当前 Python 3.12 `.venv` 与 `requirements.txt`。
- Produces: 可导入的 `langgraph`、`langchain`、`langgraph-checkpoint-sqlite`、`langchain-deepseek`，以及锁定后的完整传递依赖集合。

- [ ] **Step 1: 写依赖契约测试**

```python
from importlib.metadata import version


def _major_minor(distribution: str) -> tuple[int, int]:
    major, minor, *_rest = version(distribution).split(".")
    return int(major), int(minor)


def test_langgraph_stack_uses_approved_minor_lines() -> None:
    assert _major_minor("langgraph") == (1, 2)
    assert _major_minor("langchain") == (1, 3)
    assert _major_minor("langgraph-checkpoint-sqlite") == (3, 1)
    assert _major_minor("langchain-deepseek") == (1, 1)
```

- [ ] **Step 2: 运行测试，确认当前环境因依赖缺失或版本不符而失败**

Run: `./.venv/bin/python -m pytest tests/deep_reading/test_dependency_contract.py -q`

Expected: FAIL，明确指出至少一个新 distribution 尚未安装或不在批准的小版本线上。

- [ ] **Step 3: 在 `requirements.txt` 增加已批准的五个范围**

```text
langgraph>=1.2.9,<1.3
langchain>=1.3.14,<1.4
langgraph-checkpoint-sqlite>=3.1,<3.2
langchain-deepseek>=1.1,<1.2
pydantic>=2.13,<3
```

- [ ] **Step 4: 生成跨平台锁文件并同步隔离工作树的虚拟环境**

Run: `uv pip compile requirements.txt --universal --python-version 3.12 --output-file requirements-lock.txt`

Run: `uv pip sync requirements-lock.txt --python .venv/bin/python`

Expected: 两条命令 exit 0；锁文件中的四个新 distribution 落在批准的小版本线上。

- [ ] **Step 5: 验证依赖契约与环境一致性**

Run: `./.venv/bin/python -m pytest tests/deep_reading/test_dependency_contract.py -q`

Run: `uv pip check --python .venv/bin/python`

Expected: 测试 PASS，`uv pip check` 报告所有依赖兼容。

- [ ] **Step 6: 提交依赖基线**

```bash
git add requirements.txt requirements-lock.txt tests/deep_reading
git commit -m "build: add LangGraph conversation dependencies"
```

---

### Task 2: 建立真实 SQLite Checkpointer 生命周期与 fork 契约

**Files:**
- Create: `paperpilot/web/checkpoint.py`
- Modify: `paperpilot/web/config.py:1-91`
- Modify: `.env.example:1-20`
- Create: `tests/web/test_checkpoint.py`
- Modify: `tests/web/test_config.py:1-91`

**Interfaces:**
- Consumes: `WebRuntimeConfig` 与 `langgraph.checkpoint.sqlite.SqliteSaver`。
- Produces: `WebRuntimeConfig.checkpoint_db_path: Path`；`resolve_checkpoint_db_path(path=None) -> Path`；`SqliteCheckpointRuntime.open(path) -> SqliteCheckpointRuntime`；属性 `saver`；方法 `check_health() -> None`、`close() -> None`；CLI `python -m paperpilot.web.checkpoint --setup`。

- [ ] **Step 1: 写路径、生命周期、重启和 health 失败测试**

```python
def test_checkpoint_runtime_persists_after_close_and_reopen(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    first = SqliteCheckpointRuntime.open(path)
    graph = _build_counter_graph(first.saver)
    graph.invoke({"value": 1}, {"configurable": {"thread_id": "conv-1"}})
    first.close()

    second = SqliteCheckpointRuntime.open(path)
    assert _build_counter_graph(second.saver).get_state(
        {"configurable": {"thread_id": "conv-1"}}
    ).values["value"] == 2
    second.close()
```

同时增加：显式 false 的 `LANGGRAPH_STRICT_MSGPACK` 被拒绝、重复 `close()` 安全、CLI setup 生成表、关闭后 health 失败、默认路径和环境覆盖测试。

- [ ] **Step 2: 写真实历史 fork 契约测试**

测试必须完成原分支两次 invoke，取得第一轮完整 checkpoint，再带该 `checkpoint_id` 和新输入 invoke；断言新结果的 `parent_config` 链能回到目标 checkpoint，旧分支最终 snapshot 仍可按原 ID 读取。rollback 设计只有这项契约通过才可继续。

- [ ] **Step 3: 运行测试，确认模块尚不存在**

Run: `./.venv/bin/python -m pytest tests/web/test_checkpoint.py tests/web/test_config.py -q`

Expected: FAIL with `ModuleNotFoundError: paperpilot.web.checkpoint` 或缺少配置字段。

- [ ] **Step 4: 实现连接与显式 setup**

```python
@dataclass
class SqliteCheckpointRuntime:
    path: Path
    connection: sqlite3.Connection
    saver: SqliteSaver
    _closed: bool = False

    @classmethod
    def open(cls, path: Path | str | None = None) -> "SqliteCheckpointRuntime":
        resolved = resolve_checkpoint_db_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            resolved,
            timeout=30,
            check_same_thread=False,
        )
        connection.execute("PRAGMA busy_timeout=30000")
        saver = SqliteSaver(connection)
        saver.setup()
        return cls(path=resolved, connection=connection, saver=saver)
```

`WebRuntimeConfig.from_env()` 从 `PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH` 读取路径，缺省为 `Path("data/langgraph/checkpoints.sqlite3")`。`open()` 在构造 Saver 前调用严格 msgpack 校验；`check_health()` 执行 `SELECT 1`；`close()` 只关闭一次。CLI 只调用 `open()`、`check_health()`、`close()`，不启动 Web 或模型。

- [ ] **Step 5: 增加多进程 SQLite 探针**

使用 `multiprocessing.get_context("spawn")` 启动 4 个进程，每个进程独立 `open()` 同一文件并向不同 `thread_id` 连续写 5 次；父进程断言全部 exit 0 且四个最终 State 都可读。如果出现持续 `database is locked`，停止后续实施并回到 Checkpointer 选型，不得通过删除探针掩盖失败。

- [ ] **Step 6: 运行 Checkpointer 测试**

Run: `LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests/web/test_checkpoint.py tests/web/test_config.py -q`

Expected: 全部 PASS；测试必须同时覆盖真实文件重启和 fork，而非只覆盖 `InMemorySaver`。

- [ ] **Step 7: 提交 Checkpointer 基础设施**

```bash
git add paperpilot/web/checkpoint.py paperpilot/web/config.py .env.example tests/web/test_checkpoint.py tests/web/test_config.py
git commit -m "feat(checkpoint): add SQLite LangGraph runtime"
```

---

### Task 3: 提取可复用的结构化 arXiv Paper Catalog

**Files:**
- Create: `paperpilot/papers.py`
- Modify: `paperpilot/mcp_servers/arxiv.py:1-125`
- Create: `tests/papers/__init__.py`
- Create: `tests/papers/test_arxiv_catalog.py`
- Modify: `tests/mcp_servers/test_arxiv_download.py:1-80`

**Interfaces:**
- Consumes: `arxiv.Client` 与现有 MCP 的搜索语义。
- Produces: `PaperCandidate`；`normalize_arxiv_id(value: str) -> str | None`；`search_arxiv_candidates(query: str, limit: int, client=None) -> list[PaperCandidate]`；`resolve_arxiv_candidate(external_id: str, client=None) -> PaperCandidate`。

- [ ] **Step 1: 写 URL/ID 规范化和结构化搜索测试**

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2401.12345", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345v2", "2401.12345v2"),
        ("https://arxiv.org/pdf/2401.12345v2.pdf", "2401.12345v2"),
        ("cs.AI/0501001v3", "cs.AI/0501001v3"),
        ("not an arxiv id", None),
    ],
)
def test_normalize_arxiv_id(raw: str, expected: str | None) -> None:
    assert normalize_arxiv_id(raw) == expected
```

Fake arXiv client 返回固定 Result；断言标题、作者列表、摘要、来源 URL 和显式 `v2` 被保留。搜索失败不得回退到 LLM。

- [ ] **Step 2: 运行新测试确认失败**

Run: `./.venv/bin/python -m pytest tests/papers/test_arxiv_catalog.py -q`

Expected: FAIL with `ModuleNotFoundError: paperpilot.papers`。

- [ ] **Step 3: 实现 Pydantic Paper 契约与直接 arXiv 查询**

```python
class PaperCandidate(BaseModel):
    source: Literal["arxiv"] = "arxiv"
    external_id: str
    title: str
    authors: list[str]
    abstract: str | None = None
    source_url: str
    pdf_url: str | None = None
    published: str | None = None
    primary_category: str | None = None
```

URL/ID 输入使用 `arxiv.Search(id_list=[normalized])` 精确解析；普通标题、作者和关键词使用 `arxiv.Search(query=query, max_results=limit)`。共享 Catalog 与旧 MCP 入口继续支持 1–50；Web Paper 搜索和 Research Agent 工具边界各自把用户输入限制为 1–20。空 query 抛 `ValueError`。Web PaperResponse 只暴露产品需要的字段；MCP 渲染继续使用 `pdf_url/published/primary_category`。

- [ ] **Step 4: 让 arXiv MCP 复用 catalog 并保持原文本格式**

`search_papers()` 调用 `search_arxiv_candidates()` 后仍渲染现有 `arxiv_id/title/authors/published/primary_category/pdf_url/abstract` 文本格式；download 路径保持不变，避免破坏旧 Agent 与 MCP 测试。

- [ ] **Step 5: 运行 catalog 与 MCP 回归**

Run: `./.venv/bin/python -m pytest tests/papers/test_arxiv_catalog.py tests/mcp_servers/test_arxiv_download.py tests/mcp_servers/test_mcp_manifest.py -q`

Expected: 全部 PASS，不发生网络请求。

- [ ] **Step 6: 提交 Paper Catalog**

```bash
git add paperpilot/papers.py paperpilot/mcp_servers/arxiv.py tests/papers tests/mcp_servers/test_arxiv_download.py
git commit -m "feat(papers): add structured arXiv catalog"
```

---

### Task 4: 增加 Conversation 业务表与 Task checkpoint 字段

**Files:**
- Modify: `paperpilot/web/db_models.py:1-92`
- Create: `migrations/versions/20260807_0002_conversations.py`
- Modify: `tests/web/test_db_models.py:1-100`
- Modify: `tests/web/test_db_migrations.py:1-260`

**Interfaces:**
- Consumes: 现有 `Base.metadata`、revision `20260806_0001` 和旧五表数据库。
- Produces: `PaperRow`、`ConversationRow`、`ConversationPaperRow`、`MessageRow`；扩展的 `ResearchTaskRow`。

- [ ] **Step 1: 先更新 ORM metadata 契约测试**

测试要求 metadata 包含九张业务表；`research_tasks` 新增 `conversation_id/base_checkpoint_id/final_checkpoint_id/result_quality`；新表字段与设计文档完全一致；Message 声明 `UNIQUE(task_id, role)`；Task 声明 SQLite 部分唯一索引 `uq_tasks_one_active_per_conversation`。

- [ ] **Step 2: 写 Alembic 真实文件升级测试**

从三种数据库升级：空库、当前五表库、带未知 `agent_runs` 表和旧 Task/Event 行的 legacy 库。断言：

```python
assert {
    "papers",
    "conversations",
    "conversation_papers",
    "messages",
} <= _table_names(db_path)
assert connection.execute(
    "SELECT question FROM research_tasks WHERE id = 'task-1'"
).fetchone() == ("question",)
assert "agent_runs" in _table_names(db_path)
```

另用两个并发事务创建同一 Conversation 的活动 Task，断言部分唯一索引拒绝第二条。

- [ ] **Step 3: 运行 metadata/migration 测试确认失败**

Run: `./.venv/bin/python -m pytest tests/web/test_db_models.py tests/web/test_db_migrations.py -q`

Expected: FAIL，报告缺少新表、字段或 migration head。

- [ ] **Step 4: 实现 ORM 表和稳定索引名**

固定使用这些约束/索引名：

```text
uq_papers_source_external_id
idx_conversations_user_updated_id
idx_conversation_papers_conversation_active
uq_messages_task_role
idx_messages_conversation_parent_created
uq_tasks_one_active_per_conversation
```

`conversation_papers` 使用 `(conversation_id, paper_id)` 复合主键。`conversations.head_message_id` 可空，和 `messages.conversation_id` 的循环关系使用具名 FK；migration 必须在真实 SQLite 上验证建表顺序。

- [ ] **Step 5: 编写非破坏性 revision `20260807_0002`**

revision 只新增表、Task 可空列和索引。为避免 SQLite 重建旧 `research_tasks` 丢失未知对象，`conversation_id` 采用与现有 legacy `user_id` 相同的兼容策略：ORM 声明 FK，migration 对既有表用 `op.add_column`，业务事务与部分唯一索引提供实际约束；新表的 FK 全部由 SQLite 实际执行。`downgrade()` 继续明确拒绝破坏性降级。

- [ ] **Step 6: 验证迁移、幂等和旧数据保留**

Run: `./.venv/bin/python -m pytest tests/web/test_db_models.py tests/web/test_db_migrations.py tests/web/test_database.py -q`

Run: `PAPERPILOT_TASK_DB_PATH=/tmp/paperpilot-plan-migration.sqlite3 ./.venv/bin/python -m alembic -c alembic.ini upgrade head`

Expected: 测试全 PASS，CLI upgrade exit 0；不得执行 downgrade。

- [ ] **Step 7: 提交业务 schema**

```bash
git add paperpilot/web/db_models.py migrations/versions/20260807_0002_conversations.py tests/web/test_db_models.py tests/web/test_db_migrations.py
git commit -m "feat(db): add conversation and message schema"
```

---

### Task 5: 实现 Paper 与 Conversation CRUD 事务

**Files:**
- Modify: `paperpilot/web/task_store.py:1-704`
- Create: `tests/web/test_conversation_store.py`

**Interfaces:**
- Consumes: Task 3 的 `PaperCandidate` 与 Task 4 的 ORM rows。
- Produces: `PaperRecord`、`ConversationRecord`、`ConversationPaperRecord`、`ConversationDetail`；`TaskStore.create_conversation()`、`list_conversations()`、`get_conversation_detail()`、`update_conversation()`。

- [ ] **Step 1: 写创建、复用、所有权和软归档测试**

```python
conversation = store.create_conversation(
    user_id=alice.id,
    paper=PaperCandidate(
        external_id="2401.12345v2",
        title="A Test Paper",
        authors=["Ada Lovelace"],
        abstract="abstract",
        source_url="https://arxiv.org/abs/2401.12345v2",
    ),
    title=None,
)
assert conversation.id.startswith("conv_")
assert conversation.user_id == alice.id
assert conversation.primary_paper_id is not None
assert conversation.head_message_id is None
assert conversation.head_checkpoint_id is None
```

再断言相同 source/external ID 只产生一个 Paper、Bob 无法读取 Alice Conversation、归档默认不出现在列表、主论文关联 `role=primary/added_by=user/is_active=True`。

- [ ] **Step 2: 运行 Store 测试确认方法缺失**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_store.py -q`

Expected: FAIL with missing dataclass or `TaskStore.create_conversation`。

- [ ] **Step 3: 增加直接映射业务 dataclass**

```python
@dataclass(frozen=True)
class ConversationRecord:
    id: str
    user_id: str
    primary_paper_id: str
    title: str
    head_message_id: str | None
    head_checkpoint_id: str | None
    created_at: str
    updated_at: str
    archived_at: str | None
```

`PaperRecord` 和 `ConversationPaperRecord` 同样一一映射数据库字段；`ConversationDetail` 组合 Conversation、主论文、当前 active papers 和活动 Task，不复制数据层。

- [ ] **Step 4: 在单个 SQLAlchemy transaction 中 upsert Paper 并创建 Conversation**

Paper upsert 先按 `(source, external_id)` 查询；已存在时刷新公开元数据但保留内部 ID。创建 Conversation 后立即插入 primary `ConversationPaperRow`。标题为空时使用规范化论文标题，标题只允许 1–200 字符。

- [ ] **Step 5: 实现所有权过滤的列表、详情和 patch**

`list_conversations(user_id, include_archived=False, limit=100)` 按 `updated_at DESC, id DESC`；`get_conversation_detail()` 对跨用户返回 `None`；`update_conversation()` 在活动 Task 存在时拒绝 archive，但允许标题更新。

- [ ] **Step 6: 运行 Store 新旧测试**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_store.py tests/web/test_task_store.py -q`

Expected: 全部 PASS，旧 TaskStore 行为不变。

- [ ] **Step 7: 提交 Conversation CRUD**

```bash
git add paperpilot/web/task_store.py tests/web/test_conversation_store.py
git commit -m "feat(web): persist papers and conversations"
```

---

### Task 6: 实现 Message 树、原子提交与单活动 Task 约束

**Files:**
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_conversation_store.py`
- Modify: `tests/web/test_task_store.py:1-980`

**Interfaces:**
- Consumes: Task 5 Conversation CRUD 与现有 `ResearchTask`/TaskEvent。
- Produces: `MessageRecord`、`ConversationTurn`、`ConversationAlternative`；`create_conversation_turn()`、`get_message()`、`get_task_message()`、`list_active_messages()`、`list_message_alternatives()`、`get_unstable_turn()`。

- [ ] **Step 1: 写 User Message + Task + queued event 原子性测试**

```python
turn = store.create_conversation_turn(
    user_id=alice.id,
    conversation_id=conversation.id,
    content="Explain the main contribution.",
    depth="standard",
    expected_head_message_id=None,
)
assert turn.user_message.parent_message_id is None
assert turn.task.conversation_id == conversation.id
assert turn.task.base_checkpoint_id is None
assert [event.type for event in _events(store, turn.task.id, alice.id)] == ["queued"]
```

用触发器拒绝 event insert，断言 Message 与 Task 一起回滚。

- [ ] **Step 2: 写冲突、树路径和 alternatives 测试**

覆盖：同一 Conversation 第二个活动 Task 被拒绝；不同 Conversation 可同时运行；旧 head 产生两个 User 子节点时 alternatives 返回各自完整 Assistant 子节点；active path 只沿 head 反向读取；循环 parent 数据被检测并抛出诊断错误。

- [ ] **Step 3: 运行目标测试确认失败**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_store.py tests/web/test_task_store.py -q`

Expected: FAIL，报告 turn/message 方法或 Task 字段缺失。

- [ ] **Step 4: 扩展 `ResearchTask` 但保持旧 API 字典**

```python
@dataclass(frozen=True)
class ResearchTask:
    id: str
    question: str
    depth: str
    status: str
    created_at: str
    updated_at: str
    user_id: str | None = None
    conversation_id: str | None = None
    base_checkpoint_id: str | None = None
    final_checkpoint_id: str | None = None
    result_quality: str | None = None
```

`to_dict()` 继续只返回旧 TaskResponse 的公开字段，避免旧 `/api/tasks` 响应扩张。

- [ ] **Step 5: 实现 `create_conversation_turn()` 的一个事务**

事务内按 owner 读取 Conversation、检查 archive、查询活动 Task、比较 `expected_head_message_id`、创建 User Message、ResearchTask 和 queued event。定义并稳定映射 `ConversationBusyError`、`StaleConversationHeadError`；捕获部分唯一索引 `IntegrityError` 并转换为 Busy，而不是泄漏 SQL。

- [ ] **Step 6: 实现 active path 与分叉查询**

`get_message()` 必须同时过滤 owner 与 Conversation，允许读取非 active 分支的 rollback 目标；`get_task_message(task_id, role)` 为 Runner 的幂等发布/恢复提供唯一查询。active path 从 `head_message_id` 使用 Message ID map 反向走 parent，检测不存在、跨 Conversation 和环；最后 reverse。alternatives 只接受 complete Assistant 分叉点，返回其直接 User 子消息及对应 complete Assistant 子消息，不推导全局树。

- [ ] **Step 7: 运行并发与旧 Task 回归**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_store.py tests/web/test_task_store.py -q`

Expected: 全部 PASS；并发测试必须实际使用同一 SQLite 文件的两个 TaskStore 实例。

- [ ] **Step 8: 提交 Message/Task 事务**

```bash
git add paperpilot/web/task_store.py tests/web/test_conversation_store.py tests/web/test_task_store.py
git commit -m "feat(web): add conversation turn transactions"
```

---

### Task 7: 实现幂等发布、finalization 与业务 head 切换

**Files:**
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_conversation_store.py`

**Interfaces:**
- Consumes: Task 6 Message/Task 数据。
- Produces: `UsedPaperInput`、`PublishedConversationResult`；`publish_conversation_result()`、`finalize_conversation_task()`、`fail_conversation_task()`、`switch_conversation_head()`。

- [ ] **Step 1: 写 Task-ID 幂等发布测试**

两次调用 `publish_conversation_result()`，输入相同 Task、答案、引用 metadata 和实际使用论文；断言只有一条 Assistant Message、一条 `kind=result` Artifact、每篇论文一个 ConversationPaper，返回相同 `message.id`。

- [ ] **Step 2: 写 finalization 与 rollback 原子性测试**

```python
final = store.finalize_conversation_task(
    task_id=turn.task.id,
    assistant_message_id=published.message.id,
    final_checkpoint_id="cp-final-1",
    result_quality="complete",
    active_paper_ids=published.active_paper_ids,
)
assert final.task.status == "completed"
assert final.conversation.head_message_id == published.message.id
assert final.conversation.head_checkpoint_id == "cp-final-1"
```

rollback 测试断言 expected head 不一致、活动 Task、跨用户和非 Assistant target 全部被拒绝；成功时 head 二元组和 active papers 同事务切换，旧消息仍存在。

- [ ] **Step 3: 注入两个崩溃窗口测试**

窗口 A 在发布后、finalization 前抛错；窗口 B 在 checkpoint ID 已知后让 finalization transaction 失败。重试后断言一条 User、一条 Assistant、一条结果 Artifact、一个 completed Task 和一致 head。

- [ ] **Step 4: 运行测试确认缺少实现**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_store.py -q`

Expected: FAIL on missing publish/finalize/switch methods。

- [ ] **Step 5: 实现幂等业务事务**

`publish_conversation_result()` 先按 `(task_id, role='assistant')` 查 Message；已存在时复用并验证 conversation/parent；Artifact 按 `(task_id, kind='result')` 查询复用；Agent 使用论文只从 `UsedPaperInput` upsert，并把首次采用它的 Task/Assistant 写入 `source_task_id/source_message_id`。`finalize_conversation_task()` 再次验证 Conversation 当前 `head_checkpoint_id == task.base_checkpoint_id`，以及 Assistant parent 链回到当前稳定 head，防止迟到 Worker 覆盖新路径。

- [ ] **Step 6: 实现失败和 head 切换**

`fail_conversation_task()` 只允许 pending/running 到 failed 并写一条幂等 failure event。`switch_conversation_head()` 在事务内复查 owner、archive、无活动 Task、expected head 和目标 Message，再更新 head 与 `is_active`；主论文无论目标 State 如何都保持 active。

- [ ] **Step 7: 运行 Store 故障恢复测试**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_store.py tests/web/test_task_store.py -q`

Expected: 全部 PASS。

- [ ] **Step 8: 提交发布和 head 事务**

```bash
git add paperpilot/web/task_store.py tests/web/test_conversation_store.py
git commit -m "feat(web): finalize conversation replies idempotently"
```

---

### Task 8: 定义 Graph State、Pydantic 契约与摘要策略

**Files:**
- Create: `paperpilot/deep_reading/__init__.py`
- Create: `paperpilot/deep_reading/state.py`
- Create: `paperpilot/deep_reading/schemas.py`
- Create: `paperpilot/deep_reading/nodes.py`
- Create: `tests/deep_reading/test_schemas.py`
- Create: `tests/deep_reading/test_nodes.py`
- Modify: `paperpilot/web/config.py`
- Modify: `tests/web/test_config.py`

**Interfaces:**
- Consumes: TaskStore records、LangChain messages 与 injected chat model。
- Produces: `DeepReadingState`、`DeepReadingContext`、`EvidenceItem`、`PaperUse`、`ResearchResult`、`ConversationSummary`、`AnswerCitation`、`AnswerDraft`、`initialize_turn()`、`needs_summary()`、`summarize_history()`。

- [ ] **Step 1: 写 Pydantic 拒绝非法研究结果和引用测试**

```python
def test_research_result_rejects_duplicate_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        ResearchResult(
            evidence_items=[_evidence("ev-1"), _evidence("ev-1")],
            used_papers=[],
            limitations=[],
        )
```

覆盖 role 枚举、空 chunk、score 边界、AnswerDraft `complete/partial`、ConversationSummary JSON dump/validate 往返。

契约字段固定为：

```python
class EvidenceItem(BaseModel):
    id: str
    paper_external_id: str
    paper_title: str
    chunk_text: str
    score: float
    supports: list[str]


class PaperUse(BaseModel):
    paper: PaperCandidate
    role: Literal["comparison", "citation", "background", "follow_up"]
    evidence_ids: list[str]


class ResearchResult(BaseModel):
    evidence_items: list[EvidenceItem]
    used_papers: list[PaperUse]
    limitations: list[str]


class AnswerCitation(BaseModel):
    evidence_id: str
    label: str


class AnswerDraft(BaseModel):
    content: str
    citations: list[AnswerCitation]
    result_quality: Literal["complete", "partial"]


class ConversationSummary(BaseModel):
    confirmed_facts: list[str]
    paper_findings: list[str]
    comparison_context: list[str]
    open_questions: list[str]
```

`used_papers` 只列关联论文；主论文由 `primary_paper_id` 和 primary ConversationPaper 管理，不在每轮结果中重复创建。

- [ ] **Step 2: 写每轮字段清零和摘要条件测试**

旧 State 预置上一轮 research result、draft、published ID 和 error；`initialize_turn()` 后这些字段必须回到空值，但历史 messages、summary 和 active paper IDs 延续。摘要阈值以下不调用 fake model；超过阈值只保留配置的最近 6 轮，并写结构化 summary。

- [ ] **Step 3: 运行测试确认模块缺失**

Run: `./.venv/bin/python -m pytest tests/deep_reading/test_schemas.py tests/deep_reading/test_nodes.py -q`

Expected: FAIL with missing deep_reading modules。

- [ ] **Step 4: 实现小型可序列化 State**

```python
class DeepReadingState(TypedDict, total=False):
    schema_version: int
    graph_version: str
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_summary: dict[str, object] | None
    current_task_id: str
    current_user_message_id: str
    primary_paper_id: str
    active_paper_ids: list[str]
    research_result: dict[str, object] | None
    answer_draft: dict[str, object] | None
    published_message_id: str | None
    error: dict[str, object] | None
```

固定 `SCHEMA_VERSION = 1` 与 `GRAPH_VERSION = "conversation-v1"`。除 LangChain 官方 Message 外，复杂对象全部 `model_dump(mode="json")` 后进入 State。

- [ ] **Step 5: 实现可信 Runtime Context**

`DeepReadingContext` 使用 frozen dataclass，包含 owner/conversation/task/user-message/base-checkpoint ID、TaskStore、chat model、MCP Tool map、paper search callable、event sink，以及三个有界配置：摘要阈值 32,000 estimated tokens、最近 6 轮、Research Agent recursion limit 12。数据库 Session 和 Saver 不进入 State。

`WebRuntimeConfig` 对应字段固定为 `summary_token_threshold`、`summary_recent_turns`、`research_recursion_limit`，分别读取 Global Constraints 中的三个环境变量；`create_app()`/Worker 构造 context 时显式传入，不允许节点自行读取环境变量。

- [ ] **Step 6: 实现摘要估算和 structured output**

token 估算使用明确的保守函数 `max(1, total_chars // 4)`；`summarize_history()` 通过 `model.with_structured_output(ConversationSummary)` 调用 fake/真实模型，将结果 JSON 写回。因为 `messages` 使用 `add_messages` reducer，裁剪必须从 `langchain.messages` 导入 `RemoveMessage`、从 `langgraph.graph.message` 导入 `REMOVE_ALL_MESSAGES`，返回 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 后按顺序重新加入最近 6 轮，不能只返回一个较短 list；业务数据库 Message 始终不删除。

- [ ] **Step 7: 运行 schema/node/config 测试**

Run: `./.venv/bin/python -m pytest tests/deep_reading/test_schemas.py tests/deep_reading/test_nodes.py tests/web/test_config.py -q`

Expected: 全部 PASS，测试不要求 API key。

- [ ] **Step 8: 提交 State 和契约**

```bash
git add paperpilot/deep_reading paperpilot/web/config.py tests/deep_reading tests/web/test_config.py
git commit -m "feat(graph): define deep-reading state contracts"
```

---

### Task 9: 实现受限的多论文 LangChain Research Agent

**Files:**
- Create: `paperpilot/deep_reading/research_agent.py`
- Modify: `paperpilot/deep_reading/nodes.py`
- Create: `tests/deep_reading/test_research_agent.py`
- Modify: `tests/deep_reading/test_nodes.py`

**Interfaces:**
- Consumes: `DeepReadingContext`、`ResearchResult`、结构化 Paper Catalog 与现有 MCP custom `Tool` map。
- Produces: `AgentPaperUseDecision`、`AgentResearchDecision`、`run_research_agent(state, context) -> ResearchResult`；Graph 节点 `research_evidence(state, runtime) -> dict`。

- [ ] **Step 1: 写三个受限工具的 fake 测试**

工具固定为：

```text
search_related_papers(query, limit)
prepare_paper(external_id)
retrieve_paper_evidence(question, external_id, top_k_each, summary_k)
```

断言 `prepare_paper` 只能接受主论文、当前 active paper 或本轮 search 返回的 external ID；`retrieve_paper_evidence` 只能检索已准备论文；MCP payload 非 JSON、缺少 evidence_pool 或 paper ID 不一致时抛 `ResearchContractError`。

- [ ] **Step 2: 写“搜索不等于持久化”契约测试**

Fake agent 搜索两篇论文，只准备/检索其中一篇并在 `structured_response` 中引用。断言 `ResearchResult.used_papers` 只包含实际产生被采用 evidence 的论文，未使用候选不出现在结果中。模型侧结构固定为：

```python
class AgentPaperUseDecision(BaseModel):
    external_id: str
    role: Literal["comparison", "citation", "background", "follow_up"]
    evidence_ids: list[str]


class AgentResearchDecision(BaseModel):
    selected_evidence_ids: list[str]
    paper_uses: list[AgentPaperUseDecision]
    limitations: list[str]
```

- [ ] **Step 3: 写 Agent 预算与结构化输出测试**

注入 fake `create_agent` factory，断言生产调用参数包含 `model=context.model`、三种工具、`response_format=ToolStrategy(AgentResearchDecision)`，invoke config 含 `recursion_limit=12`，结果从 `result["structured_response"]` 读取并再次 `model_validate`。

- [ ] **Step 4: 运行测试确认实现缺失**

Run: `./.venv/bin/python -m pytest tests/deep_reading/test_research_agent.py tests/deep_reading/test_nodes.py -q`

Expected: FAIL on missing research agent and node。

- [ ] **Step 5: 实现工具闭包与权威结果账本**

`run_research_agent()` 内维护三个局部 dict：search 返回的 `PaperCandidate`、已准备 external IDs、MCP 返回的 EvidenceItems。Agent 的 `AgentResearchDecision` 只选择 external/evidence IDs；函数用账本中的权威 metadata 重建最终 `ResearchResult`，拒绝模型凭空生成 Paper 或 Evidence。该账本只活在单次调用，不进入 checkpoint，也不形成新的持久化层。

`research_evidence()` 将完整 `ResearchResult.model_dump(mode="json")` 写入 State 的
`research_result` 字段；不得只保存 evidence 而丢失 `used_papers` 或 `limitations`。

- [ ] **Step 6: 包装现有 MCP 工具**

准备论文依次调用 `mcp__arxiv__download_paper` 与 `mcp__colbert__build_index`；检索调用 `mcp__colbert__planned_retrieval`。custom Tool handler 的文本结果用 `json.loads` 解码。工具调用事件通过 `context.event_sink` 写为 `prepare/research/tool_call/tool_result` 阶段。

- [ ] **Step 7: 实现 LangChain Agent**

```python
agent = create_agent(
    model=context.model,
    tools=[search_tool, prepare_tool, retrieval_tool],
    response_format=ToolStrategy(AgentResearchDecision),
)
result = agent.invoke(
    {"messages": research_messages},
    config={"recursion_limit": context.research_recursion_limit},
)
decision = AgentResearchDecision.model_validate(result["structured_response"])
structured = _validate_and_materialize_result(decision, authoritative_ledger)
```

结构化输出重试最多 2 次；预算耗尽转为明确的 `DeepReadingTaskError`，不得无限循环。

- [ ] **Step 8: 运行 Research Agent 测试**

Run: `./.venv/bin/python -m pytest tests/deep_reading/test_research_agent.py tests/deep_reading/test_nodes.py -q`

Expected: 全部 PASS，无网络和模型调用。

- [ ] **Step 9: 提交 Research Agent**

```bash
git add paperpilot/deep_reading/research_agent.py paperpilot/deep_reading/nodes.py tests/deep_reading/test_research_agent.py tests/deep_reading/test_nodes.py
git commit -m "feat(graph): add bounded multi-paper research agent"
```

---

### Task 10: 完成论文准备、答案、发布节点与 StateGraph

**Files:**
- Create: `paperpilot/deep_reading/graph.py`
- Modify: `paperpilot/deep_reading/nodes.py`
- Modify: `paperpilot/deep_reading/__init__.py`
- Create: `tests/deep_reading/test_graph.py`
- Modify: `tests/deep_reading/test_nodes.py`

**Interfaces:**
- Consumes: Task 7 publish transaction、Task 8 State/Context、Task 9 Research Agent。
- Produces: `prepare_primary_paper()`、`write_answer()`、`publish_result()`、`build_deep_reading_graph(checkpointer)`。

- [ ] **Step 1: 写固定路径和摘要条件边测试**

Fake nodes 记录执行顺序，断言短上下文为：

```python
[
    "initialize_turn",
    "prepare_primary_paper",
    "research_evidence",
    "write_answer",
    "publish_result",
]
```

长上下文只在 initialize 与 prepare 之间多一个 `summarize_history`。`graph.py` 不包含 SQL、Prompt 或 MCP handler。

- [ ] **Step 2: 写真实 `InMemorySaver` 两轮 State 测试**

同一 `thread_id="conv-two-turn"` 调用两轮不同 `DeepReadingContext`；第二轮 fake writer 必须看到第一轮 Human/AI 消息和 summary/active papers，但 research result、draft、published ID 已清零后重新生成。该测试只验证 Graph 语义，不声称进程重启能力。

- [ ] **Step 3: 写节点边界测试**

`prepare_primary_paper` 必须调用 download/build，且重复准备不改变业务数据；`write_answer` 必须拒绝引用不存在于 `research_result.evidence_items` 的 evidence ID；`publish_result` 必须把 `research_result.used_papers` 中 Agent 实际使用的 Paper 转成 `UsedPaperInput` 并把 Store 返回的内部 Paper IDs 写回 `active_paper_ids`。

- [ ] **Step 4: 运行 Graph 测试确认失败**

Run: `./.venv/bin/python -m pytest tests/deep_reading/test_graph.py tests/deep_reading/test_nodes.py -q`

Expected: FAIL on missing graph/nodes。

- [ ] **Step 5: 实现确定性节点与 structured writer**

`prepare_primary_paper()` 通过 context tool map 调用准确的主论文 ID；`write_answer()` 使用 `model.with_structured_output(AnswerDraft)`，Prompt 只包含 summary、最近消息、EvidenceItem 和论文 metadata；`publish_result()` 是唯一写 Assistant/Artifact 的节点，并返回 `published_message_id`、active IDs，以及 `messages=[{"role": "assistant", "content": answer.content, "id": published.message.id}]`，保证下一轮 checkpoint 同时保留上一轮问答。

- [ ] **Step 6: 只在 `graph.py` 声明图结构**

```python
builder = StateGraph(DeepReadingState, context_schema=DeepReadingContext)
builder.add_node("initialize_turn", initialize_turn)
builder.add_node("summarize_history", summarize_history)
builder.add_node("prepare_primary_paper", prepare_primary_paper)
builder.add_node("research_evidence", research_evidence)
builder.add_node("write_answer", write_answer)
builder.add_node("publish_result", publish_result)
builder.add_edge(START, "initialize_turn")
builder.add_conditional_edges("initialize_turn", route_after_initialize)
builder.add_edge("summarize_history", "prepare_primary_paper")
builder.add_edge("prepare_primary_paper", "research_evidence")
builder.add_edge("research_evidence", "write_answer")
builder.add_edge("write_answer", "publish_result")
builder.add_edge("publish_result", END)
return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 7: 运行 Graph 单元测试**

Run: `LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests/deep_reading/test_graph.py tests/deep_reading/test_nodes.py tests/deep_reading/test_schemas.py -q`

Expected: 全部 PASS，只有此处及 Checkpointer 通用单测允许 `InMemorySaver`。

- [ ] **Step 8: 提交最小 Graph**

```bash
git add paperpilot/deep_reading tests/deep_reading
git commit -m "feat(graph): build conversation deep-reading workflow"
```

---

### Task 11: 实现 DeepReadingRunner、重启恢复和两个崩溃窗口

**Files:**
- Create: `paperpilot/deep_reading/runner.py`
- Modify: `paperpilot/deep_reading/__init__.py`
- Create: `tests/deep_reading/test_runner.py`

**Interfaces:**
- Consumes: TaskStore、`SqliteCheckpointRuntime.saver`、MCPRuntime、ChatDeepSeek factory 与 compiled graph。
- Produces: `DeepReadingCheckpoint`、`DeepReadingRunner.run(task_id: str) -> None`；`read_checkpoint(conversation_id, checkpoint_id) -> DeepReadingCheckpoint | None`；`DeepReadingTaskError`。

- [ ] **Step 1: 写成功运行与最终 checkpoint 关联测试**

创建真实业务 Conversation/turn 和真实文件 Saver，注入 fake model/MCP tools。运行后断言 Task completed、Assistant/Artifact 各一条、`final_checkpoint_id` 非空、Conversation head 二元组等于 Assistant/final checkpoint，snapshot `next == ()` 且 `current_task_id` 正确。

- [ ] **Step 2: 写进程重启和连续两轮测试**

关闭第一组 TaskStore/Saver/Runner，再重新打开相同两个 SQLite 文件；创建第二轮 turn 并运行。fake writer 断言收到第一轮消息；第二轮 Task `base_checkpoint_id == 第一轮 final_checkpoint_id`。

- [ ] **Step 3: 写业务 rollback + 下一轮 fork 集成测试**

完成两轮，API 外层等价地把业务 head 切回第一轮 snapshot，然后创建第三轮并运行。断言第三轮 checkpoint parent 链从第一轮分出，第二轮原 checkpoint 仍能 `get_state`，rollback 阶段 fake model call count 不变。

- [ ] **Step 4: 写两个崩溃窗口与 redelivery 测试**

窗口 A 在 `publish_result` 后让 checkpoint write 抛错；窗口 B 在 Graph 完成后让 `finalize_conversation_task` 第一次抛 `OperationalError`。第二次 `run(task_id)` 必须通过 `current_task_id/published_message_id/graph_version/snapshot.next` 验证恢复点并完成，不重复 Message/Artifact。

- [ ] **Step 5: 运行 Runner 测试确认模块缺失**

Run: `./.venv/bin/python -m pytest tests/deep_reading/test_runner.py -q`

Expected: FAIL with missing runner。

- [ ] **Step 6: 实现可信加载、调用和 finalization**

Runner 从 TaskStore 加载 Task、owner、Conversation、User Message 和主论文；构造 context；在一个 `MCPRuntime.lease_tools()` 内编译/调用 Graph。Graph 输入固定包含带业务 Message ID 的新 HumanMessage 和当前轮标识：

```python
graph_input = {
    "messages": [
        HumanMessage(content=user_message.content, id=user_message.id)
    ],
    "current_task_id": task.id,
    "current_user_message_id": user_message.id,
    "primary_paper_id": conversation.primary_paper_id,
}
```

同一个 User Message 在 redelivery 中复用相同 ID，让 `add_messages` 覆盖而非重复追加。首轮 config 只含 `thread_id`；有 base 时才加入 `checkpoint_id`。Graph 完成后从 `snapshot.config["configurable"]["checkpoint_id"]` 取得最终 ID并调用 TaskStore finalization。

对 API 暴露的只读 checkpoint 视图固定为：

```python
@dataclass(frozen=True)
class DeepReadingCheckpoint:
    checkpoint_id: str
    state: DeepReadingState
    is_complete: bool
```

`read_checkpoint()` 内部调用 compiled graph `get_state({"configurable": {"thread_id": conversation_id, "checkpoint_id": checkpoint_id}})`，不接受 metadata filter，也不向 API 返回原始 Checkpointer。

- [ ] **Step 7: 实现恢复与异常边界**

运行前若已存在 published Assistant，则查该 thread 最新完整 snapshot；只有 State task/message/version 与当前 Task 完全匹配才执行 finalization。Runner 只捕获 `DeepReadingTaskError` 并标记业务 failed；SQLAlchemy、SQLite、Checkpointer、进程和未知异常必须重新抛给 Celery redelivery。

- [ ] **Step 8: 实现 DeepSeek 默认 factory**

```python
def build_deep_reading_model() -> ChatDeepSeek:
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=2,
    )
```

factory 只在真正执行新 Conversation Task 时调用；import Web app、搜索论文和 rollback 不读取 API key。

- [ ] **Step 9: 运行 Runner/SQLite 集成测试**

Run: `LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests/deep_reading/test_runner.py tests/web/test_checkpoint.py -q`

Expected: 全部 PASS；测试使用真实 `SqliteSaver` 文件且模型为 fake。

- [ ] **Step 10: 提交 Runner**

```bash
git add paperpilot/deep_reading/runner.py paperpilot/deep_reading/__init__.py tests/deep_reading/test_runner.py
git commit -m "feat(graph): run and recover conversation tasks"
```

---

### Task 12: 把新 Runner 接入 Thread Executor 与 Celery 子进程

**Files:**
- Modify: `paperpilot/web/workflow.py:1-218`
- Modify: `paperpilot/web/worker_tasks.py:1-90`
- Modify: `paperpilot/web/app.py:137-174`
- Modify: `paperpilot/web/task_executor.py:1-273`
- Create: `tests/web/test_conversation_worker.py`
- Modify: `tests/web/test_celery_worker.py:1-210`
- Modify: `tests/web/test_task_executor.py:1-530`
- Modify: `tests/web/test_workflow.py:1-160`

**Interfaces:**
- Consumes: `ResearchTask.conversation_id` 与 `DeepReadingRunner.run()`。
- Produces: 新旧 Task 的明确分派；每个 Celery 子进程一个 MCPRuntime 和一个 SqliteCheckpointRuntime；Web 进程拥有并关闭自己的资源。

- [ ] **Step 1: 写新旧 Task 路由测试**

旧 Task `conversation_id is None` 仍调用旧 `WorkflowRunner.run_real()`；Conversation Task 调用 fake `DeepReadingRunner.run(task_id)`，旧 `conversation.run()` fake call count 为 0。simulated 旧 Task 行为不变。

- [ ] **Step 2: 写 Celery 进程资源复用/关闭测试**

两个 Conversation Task 复用同一 `_runtime` 和 `_checkpoint_runtime`；`_close_worker_runtime()` 各关闭一次并清空全局；redelivered running Conversation Task 可恢复，completed Task 直接跳过。

- [ ] **Step 3: 运行 worker/executor 回归确认失败**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_worker.py tests/web/test_celery_worker.py tests/web/test_task_executor.py tests/web/test_workflow.py -q`

Expected: FAIL on missing DeepReading routing。

- [ ] **Step 4: 实现最小 Runner 分派**

`WorkflowRunner` 增加可选 `deep_reading_runner`；`run_real()` 读取 Task 后若 `conversation_id` 非空则委托新 Runner并立即返回，否则执行原代码。`WorkflowRunnerLike` 无需新增方法，TaskExecutor/Celery payload 仍只传 `(task_id, "real")`，避免公开队列协议变化。

- [ ] **Step 5: 实现 Celery 子进程惰性资源**

增加 `_get_checkpoint_runtime()` 与 `_build_deep_reading_runner(store, mcp_runtime, checkpoint_runtime)`；只在 claimed Task 有 `conversation_id` 时创建 ChatDeepSeek/Graph相关对象。shutdown signal 按 checkpoint、MCP 顺序关闭；不能继承父进程 SQLite connection。

- [ ] **Step 6: 更新 Web app 默认生命周期**

默认 `create_app()` 为 thread executor 创建 application-owned CheckpointRuntime、MCPRuntime 和 DeepReadingRunner，并在 shutdown 中关闭；测试注入 TaskStore 时 checkpoint 默认放在 `store.db_path.parent / "checkpoints.sqlite3"`，避免写项目 `data/`。新增可选 injection 参数供 API/worker 测试，不改变现有调用方。

- [ ] **Step 7: 运行分派与旧运行时测试**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_worker.py tests/web/test_celery_worker.py tests/web/test_task_executor.py tests/web/test_workflow.py -q`

Expected: 全部 PASS；旧 WorkflowRunner failure/status 测试不修改预期。

- [ ] **Step 8: 提交执行器接线**

```bash
git add paperpilot/web/workflow.py paperpilot/web/worker_tasks.py paperpilot/web/app.py paperpilot/web/task_executor.py tests/web/test_conversation_worker.py tests/web/test_celery_worker.py tests/web/test_task_executor.py tests/web/test_workflow.py
git commit -m "feat(web): route conversation tasks to LangGraph"
```

---

### Task 13: 实现 Paper、Conversation、Message 与 rollback API

**Files:**
- Create: `paperpilot/web/conversation_routes.py`
- Modify: `paperpilot/web/app.py:1-509`
- Create: `tests/web/test_conversation_api.py`
- Modify: `tests/web/test_web_app.py:1-1020`

**Interfaces:**
- Consumes: TaskStore Conversation 方法、Paper Catalog、TaskExecutor reservation、DeepReadingRunner checkpoint reader。
- Produces: 设计文档第 7 节全部 API；跨用户统一 404；Busy/Stale 统一 409。

- [ ] **Step 1: 写论文搜索和会话 CRUD API 测试**

固定响应：

```json
{"items":[{"source":"arxiv","external_id":"2401.12345v2","title":"A Test Paper","authors":["Ada Lovelace"],"abstract":"abstract","source_url":"https://arxiv.org/abs/2401.12345v2"}]}
```

覆盖自然语言搜索、URL 精确搜索、创建 Conversation、列表、详情、改标题、软归档、未登录 401、Bob 访问 Alice 资源 404。Fake paper search 证明无 LLM 调用。

- [ ] **Step 2: 写消息提交、进度与冲突 API 测试**

`POST /api/conversations/{id}/messages` 返回 202，body 含 `user_message/task/stable_head_message_id`；相同 expected head 的第二个活动请求返回 409；stale head 返回 409；executor capacity 503 时数据库零写入；queue submit 失败时 Task failed 且 Conversation head 不移动。

`GET /api/conversations/{id}/messages` 使用 `ConversationMessagesResponse(items: list[MessageResponse], unstable_turn: UnstableTurnResponse | None)`，其中 `UnstableTurnResponse` 固定包含 `user_message: MessageResponse` 与 `task: TaskResponse`，使 pending/running/failed User Message 能在不移动稳定 head 的情况下恢复展示。

- [ ] **Step 3: 写 messages、alternatives 和 rollback API 测试**

active path 只返回当前分支；alternatives 返回直接分支 User/Assistant pair；rollback 注入会在调用时抛错的 fake model，断言响应 200 且 fake model call count 为 0；目标 checkpoint 不存在、schema 不支持、活动 Task、跨用户分别映射 409/404 且 head 不变。

- [ ] **Step 4: 运行 API 测试确认路由缺失**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_api.py tests/web/test_web_app.py -q`

Expected: FAIL with 404 on new routes。

- [ ] **Step 5: 定义精确 API schema 与 router factory**

```python
def create_conversation_router(
    *,
    store: TaskStore,
    executor: TaskExecutorLike,
    deep_reading_runner: DeepReadingRunner,
    require_user: Callable[[Request, str | None], WebUser],
    paper_search: Callable[[str, int], list[PaperCandidate]],
) -> APIRouter:
    router = APIRouter()
    return router
```

请求模型固定包含 `CreateConversationRequest.paper`、`CreateMessageRequest.content/depth/expected_head_message_id`、`RollbackRequest.message_id/expected_head_message_id`。`expected_head_message_id` 字段必传但允许 JSON null。

- [ ] **Step 6: 实现 reservation + 业务事务 + submit 顺序**

新消息 route 复用旧 `/api/tasks` 的 admission 模式：先 `executor.reserve()`，再 `create_conversation_turn()`，最后 `reservation.submit(task.id, "real")`。持久化后 submit 异常调用 `fail_pending_task()` 和 failure event；如果 Worker 已 claim，则返回已接受结果，不误标 failed。

- [ ] **Step 7: 实现 checkpoint 校验后再切 head**

rollback route 先读取业务 target，再调用 `deep_reading_runner.read_checkpoint()`；验证 `schema_version==1`、`graph_version==conversation-v1`、`is_complete is True`、`published_message_id==target`，随后把 `active_paper_ids` 交给 `switch_conversation_head()` 在业务事务中二次检查并更新。

- [ ] **Step 8: 在 `create_app()` 挂载 Router 并扩展 readiness**

`app.include_router(conversation_router)` 使用现有 nested `require_user`；readiness checks 精确为 `database/checkpoint/executor`，checkpoint 失败时 503，liveness 仍始终不探测 DB/模型。

- [ ] **Step 9: 运行 API 与 auth/health 回归**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_api.py tests/web/test_web_app.py tests/web/test_auth.py -q`

Expected: 全部 PASS，新旧接口都要求原有 cookie auth。

- [ ] **Step 10: 提交 Conversation API**

```bash
git add paperpilot/web/conversation_routes.py paperpilot/web/app.py tests/web/test_conversation_api.py tests/web/test_web_app.py
git commit -m "feat(api): expose conversation and rollback endpoints"
```

---

### Task 14: 增加最小 Web 会话界面并保留旧工作台

**Files:**
- Modify: `paperpilot/web/static/index.html:1-127`
- Modify: `paperpilot/web/static/app.js:1-619`
- Create: `paperpilot/web/static/conversations.js`
- Create: `paperpilot/web/static/conversations.css`
- Create: `tests/web/test_conversation_ui.py`
- Modify: `tests/web/test_web_app.py:676-721`

**Interfaces:**
- Consumes: Task 13 HTTP API 与现有 `requestJson()`/auth 生命周期。
- Produces: 搜索选择论文、会话列表、active path、完整回复轮询、rollback 与“其他版本”选择器。

- [ ] **Step 1: 写静态 UI 契约测试**

断言 HTML 包含 `conversationView/conversationList/paperSearch/paperCandidates/messageList/messageComposer/branchSelector`；加载两个新静态文件；JS 只请求新 Conversation API 和现有 `/api/tasks/{id}/updates`，不出现 WebSocket/EventSource；旧 `taskForm/taskList/evalSnapshot` 仍存在。

- [ ] **Step 2: 运行 UI 测试确认失败**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_ui.py tests/web/test_web_app.py::test_event_ui_static_assets_are_served -q`

Expected: FAIL，报告缺少会话 DOM/静态文件。

- [ ] **Step 3: 增加不破坏旧页面的导航与布局**

登录后默认显示 Conversations 标签，Legacy Workbench 标签保留现有 Task/Eval/Detail DOM。`app.js` 在认证成功/退出时分别 dispatch `paperpilot:authenticated` 与 `paperpilot:unauthenticated`，不搬迁旧 Task 逻辑。

- [ ] **Step 4: 在独立 `conversations.js` 实现状态机**

客户端状态固定为：

```javascript
const conversationState = {
  selectedConversationId: null,
  selectedConversation: null,
  messages: [],
  activeTaskId: null,
  eventAfterId: 0,
  artifactAfterId: 0,
  pollTimer: null,
  requestVersion: 0,
};
```

实现 `searchPapers/createConversation/loadConversations/loadConversation/submitMessage/pollConversationTask/loadAlternatives/rollbackToMessage`。提交与 rollback 的 `expected_head_message_id` 始终取当前已加载 Conversation 的稳定 head。所有 ID 用 `encodeURIComponent`；切换会话递增 requestVersion 并取消旧 timer，防止旧轮询覆盖新页面。

- [ ] **Step 5: 实现产品交互约束**

活动 Task 时禁用发送、rollback 和版本切换；轮询只显示阶段事件，Task completed 后重新拉 messages 并一次渲染 Assistant 完整正文；409 时显示“会话已更新，请重试”并 reload；失败 Task 显示在稳定 head 后且允许再次提问。

- [ ] **Step 6: 实现线性路径和局部分支选择器**

默认只 render messages response 的 active path。每个有 alternatives 的 Assistant 分叉点显示“其他版本”按钮；选择项调用 rollback API，把该 Assistant ID 作为 target，成功后重新加载 Conversation，不绘制完整树。

- [ ] **Step 7: 运行 UI 与完整 Web API 测试**

Run: `./.venv/bin/python -m pytest tests/web/test_conversation_ui.py tests/web/test_web_app.py tests/web/test_conversation_api.py -q`

Expected: 全部 PASS。

- [ ] **Step 8: 提交 Web 会话 UI**

```bash
git add paperpilot/web/static tests/web/test_conversation_ui.py tests/web/test_web_app.py
git commit -m "feat(ui): add conversation reading workspace"
```

---

### Task 15: 完成安全、文档、端到端和全量回归门禁

**Files:**
- Modify: `README.md:68-332`
- Modify: `.env.example`
- Modify: `tests/web/test_runtime_config.py:1-120`
- Modify: `tests/deep_reading/test_runner.py`
- Modify: `tests/web/test_conversation_api.py`

**Interfaces:**
- Consumes: 前 14 个 Task 的完整纵向切片。
- Produces: 可执行的迁移/setup/启动手册与最终验收证据。

- [ ] **Step 1: 增加严格序列化拒绝测试**

子进程设置 `LANGGRAPH_STRICT_MSGPACK=true`，构造 Graph 节点返回 State schema 未声明的自定义类并尝试写/读 checkpoint；断言该对象不能从 Saver 恢复。另断言 API 没有 metadata filter 或原始 checkpoint 枚举参数，用户给出的 `checkpoint_id` 不能绕过 Message ownership。

- [ ] **Step 2: 增加端到端 fake 产品测试**

在同一测试中完成：注册 Alice → 搜索并选择论文 → 创建 Conversation → 第一问完成 → Web/Worker资源重开 → 第二问看到第一轮 → rollback 到第一轮 → 第三问产生分支 → alternatives 能切回原第二轮。断言过程中 fake model/tool 有界调用且无 API key。

- [ ] **Step 3: 更新运行与恢复文档**

README 必须给出以下顺序和准确命令：

```bash
uv pip sync requirements-lock.txt --python .venv/bin/python
./.venv/bin/python -m alembic -c alembic.ini upgrade head
LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m paperpilot.web.checkpoint --setup
./.venv/bin/celery -A paperpilot.web.celery_app:celery_app worker --loglevel=INFO
./.venv/bin/uvicorn paperpilot.web.app:app --host 127.0.0.1 --port 8000
```

同时说明两个 SQLite 文件用途、backup/restore 必须成对处理、旧 `/api/tasks` 仍是 legacy、Store 尚未实现、真实模型烟雾测试不在自动测试中。

- [ ] **Step 4: 运行 migration 和 checkpoint setup 门禁**

Run: `PAPERPILOT_TASK_DB_PATH=/tmp/paperpilot-final-business.sqlite3 ./.venv/bin/python -m alembic -c alembic.ini upgrade head`

Run: `PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH=/tmp/paperpilot-final-checkpoints.sqlite3 LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m paperpilot.web.checkpoint --setup`

Expected: 两条命令 exit 0；业务库包含 Alembic head，新 checkpoint 库包含 checkpoints/writes 表。

- [ ] **Step 5: 运行分层回归**

Run: `LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests/deep_reading tests/web/test_checkpoint.py tests/web/test_conversation_store.py tests/web/test_conversation_worker.py tests/web/test_conversation_api.py tests/web/test_conversation_ui.py -q`

Expected: 全部 PASS，零 deselected（这些目标文件不带 slow marker）。

- [ ] **Step 6: 运行完整测试套件与静态差异检查**

Run: `LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests -q`

Run: `git diff --check`

Expected: 完整测试 0 failed；仅保留项目已有的 slow deselection 和已知第三方 warning；diff check exit 0。

- [ ] **Step 7: 检查未授权范围没有进入 diff**

Run: `git diff --name-status e80038b..HEAD`

确认没有 Store 数据库、SSE/WebSocket、旧入口删除、PostgreSQL、全局 Repository 层或真实 API key。确认用户原有未跟踪 plan 文件未被 stage。

- [ ] **Step 8: 提交文档和最终门禁测试**

```bash
git add README.md .env.example tests/web/test_runtime_config.py tests/deep_reading/test_runner.py tests/web/test_conversation_api.py
git commit -m "docs: document conversation runtime operations"
```

- [ ] **Step 9: 请求代码审查并处理结果**

使用 `superpowers:requesting-code-review` 对照设计文档、这份计划、最终 diff 和完整测试输出做审查。任何修复必须重新运行对应目标测试和完整 `pytest tests -q`，再进入分支收尾。

---

### Task 16: 为 Conversation 执行补齐有限重试与失败终态

**背景：**
- 最终分支审查确认 Celery 5.6 的 `task_acks_on_failure_or_timeout=True` 会 ACK 普通 Python/SQLite/checkpoint 异常；现有 `task_acks_late` 和 `task_reject_on_worker_lost` 只覆盖 Worker 丢失，不会重投普通异常。
- `WorkflowRunner.run_real()` 有意让 Conversation 基础设施异常逃出，以便执行边界重试；但当前 Celery Task 没有调用 `self.retry()`，线程 Future 的回调也只释放容量，导致业务 Task 可永久停在 `running`。
- 用户选择方案 1：首次失败后最多重试 3 次，指数退避从 1 秒开始且上限 30 秒；Celery 显式重投，线程进程内重试；耗尽后业务 Task 必须进入 `failed`。

**Files:**
- Modify: `paperpilot/web/config.py`
- Modify: `paperpilot/web/task_executor.py`
- Modify: `paperpilot/web/worker_tasks.py`
- Modify: `paperpilot/web/workflow.py`
- Modify: `tests/web/test_config.py`
- Modify: `tests/web/test_task_executor.py`
- Modify: `tests/web/test_celery_worker.py`
- Modify: `tests/web/test_conversation_worker.py`（仅在真实业务终态覆盖需要时）
- Modify: `tests/web/test_runtime_config.py`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces and exact contract:**
- `WebRuntimeConfig` 新增 `task_max_retries=3`、`task_retry_backoff_seconds=1`、`task_retry_backoff_max_seconds=30`。
- 对应环境变量固定为 `PAPERPILOT_TASK_MAX_RETRIES`、`PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS`、`PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS`；最大重试次数允许 0，两个退避值必须为正整数，初始值不得大于上限。
- “3 次重试”表示首次执行之外最多再执行 3 次，总尝试次数最多 4 次。第 1、2、3 次重试的默认延迟依次为 1、2、4 秒；配置更大的重试次数时使用 `min(initial * 2**retry_index, max)` 封顶。
- 只重试从 `WorkflowRunner` 逃出的异常；Graph 内部已经转成确定业务终态的失败不得重复执行。不得把 `task_acks_on_failure_or_timeout=False` 当作修复，因为 Celery 对普通失败会 reject 且 `requeue=False`。
- Celery bound task 在 `request.retries < task_max_retries` 时调用 `self.retry(exc=..., countdown=..., max_retries=...)`。显式 retry 消息未必带 broker `redelivered`，因此 `_execute_research_task(..., redelivered=...)` 的恢复条件必须是 `delivery_info.redelivered` 或 `request.retries > 0`。
- 线程执行器必须使用同一退避公式在进程内重试，并允许测试注入无等待 sleeper；容量 reservation 覆盖完整重试生命周期，完成回调必须观察并记录最终 Future 异常后再释放容量。
- 重试耗尽时，通过 `WorkflowRunner` 的小型显式方法和现有 `TaskStore.fail_conversation_task()` 幂等终结活动 Conversation Task；不得新增 Repository 层或改变数据库结构。
- 终态事件固定为 `type="failed"`、`stage="execution_retry_exhausted"`、`message="Conversation execution failed after retry limit."`，payload 至少包含 `backend`、`attempts`、`max_retries`、`error_type`。用户可见事件不得保存原始异常消息、traceback、密钥或论文内容；完整异常只写服务端日志。
- 若并发恢复已经把 Task 置为 `completed`，晚到的耗尽处理必须保持 completed 且不新增 failed 事件；若 Task 不存在或不是 Conversation Task，保持原异常失败语义，不伪造 Conversation 终态。
- Celery 耗尽并完成业务失败落库后仍重新抛出原异常，使 Celery 任务自身成为 FAILURE；线程 Future 同样保留原异常。若失败落库本身不可用，记录服务端异常并保留原执行异常，不能吞掉。
- `SynchronousTaskExecutor` 是测试用确定性执行器，不引入睡眠或生产重试；生产 `build_task_executor()` 必须把同一份已验证的 runtime config 传给 `TaskExecutor`。

- [ ] **Step 1: 先写配置、Celery、线程和终态回归测试并验证 RED**

至少覆盖：默认与环境变量/非法关系；线程第二或第三次成功；线程默认 4 次均失败后只有一个 failed 事件；Celery 普通异常进入显式 retry；`request.retries > 0` 能重新 claim `running` Task；Celery 耗尽后真实 SQLite 业务 Task 为 failed；并发 completed 不被晚到失败覆盖。先运行目标测试，记录它们因缺少新行为而失败，而不是因 fixture/导入错误失败。

- [ ] **Step 2: 实现共享退避计算和配置校验**

退避计算保持为本模块可测试的小函数或等价的单一实现；Celery 与线程不得各复制一套可能漂移的公式。配置由 `WebRuntimeConfig.from_env()` 一次验证，执行器和 Worker 只消费已验证值。

- [ ] **Step 3: 实现线程有限重试和 Future 异常观测**

`TaskExecutor` 对逃出的异常最多重试配置次数；每次 retry 前记录结构化服务端 warning；耗尽时调用 runner 的 Conversation 失败终结方法，再让 Future 保留原异常。回调先观察/记录异常再可靠释放 reservation 容量。

- [ ] **Step 4: 实现 Celery 显式 retry 和 retry-aware claim**

普通异常未耗尽时通过 `self.retry()` 重投；retry 次数和 broker redelivery 任一成立都允许恢复 `running` Task。耗尽后以新 `TaskStore` 连接终结业务 Task、关闭连接并重新抛出原异常。不得在 Web 进程启动 MCP。

- [ ] **Step 5: 实现幂等且不泄密的 Conversation 失败终结方法**

方法只处理活动或已 failed 的 Conversation Task；completed 是安全 no-op；首次耗尽写一个固定 failed 事件，重复耗尽不追加。错误类型只写类名，原异常正文仅进入 logger。

- [ ] **Step 6: 更新运行配置文档**

在 `.env.example` 与 README 说明三个参数、总尝试次数语义、默认退避序列、Celery/线程差异，以及 SQLite/checkpoint 长时间不可用时仍需要人工恢复或未来 reconciler 的限制。

- [ ] **Step 7: 运行目标与完整回归门禁**

Run: `LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests/web/test_config.py tests/web/test_task_executor.py tests/web/test_celery_worker.py tests/web/test_conversation_worker.py tests/web/test_runtime_config.py -q`

Run: `LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests -q`

Run: `git diff --check`

Expected: 所有目标测试与完整测试 0 failed；仅保留项目已有的 slow deselection 和已知第三方 warning；diff check exit 0。

- [ ] **Step 8: 提交并请求任务级、整分支级复审**

提交只包含 Task 16 与 Task 3 计划文字校正。任务复审必须检查真实 Celery ACK/retry 语义、总尝试次数、completed race、失败事件泄密边界和线程容量释放；通过后重新生成 `bf003b2..HEAD` 整分支 review package，处理最终结论。

---

## 计划自检映射

| 设计要求 | 实施 Task |
| --- | --- |
| 三类存储边界、真实 SqliteSaver、重启/fork | 2、11、15 |
| Paper 搜索与用户明确选择 | 3、5、13、14 |
| 新业务表、旧数据兼容、单活动 Task | 4、6 |
| Message 树、active path、alternatives | 6、7、13、14 |
| 固定 SOP + LangChain Agent + Pydantic | 8、9、10 |
| 主论文锚点 + 实际使用的关联论文 | 5、7、9、10 |
| 幂等发布、两个崩溃窗口、Celery redelivery 与普通异常有限重试 | 7、11、12、16 |
| 连续追问、rollback 零模型调用、后续 fork | 11、13、15 |
| 进度轮询 + 完整 Assistant Message | 13、14 |
| 旧 `/api/tasks` 兼容 | 12、13、15 |
| 不实现 Store/token streaming/cancel/Postgres | Global Constraints、15 |
| 无付费模型自动测试与完整回归 | 1–16，重点 15、16 |

## 执行停止条件

- 真实 `SqliteSaver` 历史 fork 契约与设计不一致时，停止并重新评审 rollback 语义。
- 四进程 SQLite 探针在 30 秒 busy timeout 下仍稳定失败时，停止并重新评审 Checkpointer，不继续堆补丁。
- Alembic migration 无法在保留旧 rows 和未知表的情况下升级时，停止并先修订 migration 方案。
- DeepSeek 当前批准版本无法同时完成工具调用和 structured output fake/契约测试时，停止并重新选择 Agent 输出策略。
- 任一故障窗口会产生重复 Assistant/Artifact 或错移 Conversation head 时，不得进入 UI Task。
- 完整测试出现旧功能回归时，不得以“与新功能无关”为由跳过。
