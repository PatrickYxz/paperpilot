# PaperPilot Web 业务模块整理设计

## 背景与问题

当前 `paperpilot/web/` 已经只服务 Conversation + LangGraph 正式架构，但业务持久化代码仍集中在单个 `task_store.py` 中。该文件约 1943 行，同时承载用户、登录 Session、论文、Conversation、消息树、ResearchTask、Event、Artifact、回答发布、最终确认和回滚等职责。`routes/conversations.py` 也同时包含 Conversation CRUD、消息提交、分支读取和回滚逻辑。

这种组织方式没有直接造成功能错误，但显著提高了阅读和修改成本：开发者很难只围绕一个业务概念阅读代码，也很难在修改某项行为时快速确定事务边界和测试范围。

## 目标

1. 按业务职责整理 `paperpilot/web/`，使开发者可以根据“用户”“会话”“消息”“发布”“任务”“进度产物”等概念定位代码。
2. 将 `task_store.py` 缩减为清晰、显式的兼容门面，同时保留 `TaskStore` 的现有公开调用方式。
3. 拆分 `routes/conversations.py`，使 Conversation CRUD、消息 API 和回滚 API 分别位于明确模块。
4. 将对应的大型测试文件按相同业务边界拆分，保持并发、事务、权限和回滚测试覆盖。
5. 为关键模块、公开业务函数和复杂事务补充解释“业务规则与原因”的注释，提高代码可阅读性。

## 非目标

本次只整理结构，不改变业务设计。明确不做以下事项：

- 不修改公开 HTTP API 路径、请求或响应结构；
- 不修改 SQLAlchemy 表、约束、索引或 Alembic revision；
- 不迁移、删除或清洗现有业务数据；
- 不修改 LangGraph checkpoint 表或 checkpoint 语义；
- 不改变 Task 状态转换、Conversation head、消息分支或回滚行为；
- 不引入 Repository、Service、Ports、Adapters、Mixin 多继承或动态代理；
- 不重构 LangGraph、LangChain、MCP、Executor、Celery 或前端业务行为；
- 不因追求统一目录外观而拆分当前职责已经集中的模块。

## 已确认决策

1. 生产代码和对应测试一起整理。
2. 优先按业务概念组织代码，不采用仅按行数机械切割的方式。
3. 采用“业务函数模块 + `TaskStore` 兼容门面”方案。
4. `TaskStore` 保留显式方法；不使用 `__getattr__`、动态注册或 Mixin 多继承。
5. 现有从 `paperpilot.web.task_store` 导入类型、异常和 `TaskStore` 的方式继续有效。
6. 注释重点解释业务不变量、事务边界、并发原因和失败保证，不逐行翻译代码。

## 目标生产目录

```text
paperpilot/web/
├── app.py
├── auth.py
├── config.py
├── schemas.py
├── observability.py
│
├── store/
│   ├── __init__.py
│   ├── records.py
│   ├── users.py
│   ├── conversations.py
│   ├── messages.py
│   ├── publications.py
│   ├── tasks.py
│   ├── updates.py
│   └── helpers.py
│
├── task_store.py
│
├── routes/
│   ├── auth.py
│   ├── papers.py
│   ├── task_updates.py
│   └── conversations/
│       ├── __init__.py
│       ├── router.py
│       ├── crud.py
│       ├── messages.py
│       ├── rollback.py
│       └── presenters.py
│
├── database.py
├── db_models.py
├── db_migrations.py
├── checkpoint.py
├── task_executor.py
├── worker_tasks.py
├── celery_app.py
└── static/
```

## Store 模块职责

### `store/records.py`

保存业务代码使用的不可变数据对象和业务异常，不包含 SQL 查询：

- `ResearchTask`
- `WebUser`
- `TaskEvent`
- `TaskArtifact`
- `TaskEventBatch`
- `TaskArtifactBatch`
- `TaskUpdates`
- `PaperRecord`
- `ConversationRecord`
- `ConversationPaperRecord`
- `ConversationDetail`
- `MessageRecord`
- `ConversationTurn`
- `ConversationAlternative`
- `UsedPaperInput`
- `PublishedConversationResult`
- `FinalizedConversationTask`
- `DuplicateUsernameError`
- `ConversationBusyError`
- `StaleConversationHeadError`

`paperpilot.web.task_store` 必须重新导出这些现有公开符号，避免修改全部调用方。

### `store/users.py`

负责用户与登录 Session：

- `create_user`
- `get_user_by_username`
- `get_user_by_id`
- `create_session`
- `get_user_for_session`
- `delete_session`

### `store/conversations.py`

负责 Conversation 本身、主论文和 ConversationPaper 关联：

- `create_conversation`
- `list_conversations`
- `get_conversation_detail`
- `update_conversation`

该模块负责会话标题、归档状态、主论文 upsert、活跃论文读取和 Conversation 所有权边界，但不负责消息树和回答发布。

### `store/messages.py`

负责消息树与一次追问的创建：

- `create_conversation_turn`
- `get_message`
- `get_task_message`
- `list_active_messages`
- `list_message_alternatives`
- `get_unstable_turn`

该模块集中表达 `parent_message_id`、当前活跃消息路径、分支 alternatives，以及尚未成为稳定 head 的用户回合。

### `store/publications.py`

负责回答发布、Task 终态、稳定 head 和回滚：

- `publish_conversation_result`
- `finalize_conversation_task`
- `fail_conversation_task`
- `switch_conversation_head`

这些函数共同维护 Assistant Message、结果 Artifact、Task 终态、最终 checkpoint、Conversation 两个 head 和活跃论文集合。由于它们包含关键并发控制和原子事务，不再继续拆成更细的 Repository 或步骤函数。

### `store/tasks.py`

负责 Task 自身的查询、领取和队列投递失败：

- `get_task`
- `claim_task`
- `fail_pending_task`
- `check_health`

Task 成功或业务执行失败的终态与 Conversation 发布一致性高度相关，因此仍由 `store/publications.py` 负责；本模块只处理不需要同时推进 Conversation head 的状态操作。

### `store/updates.py`

负责 Task Event、Artifact 和增量更新快照：

- `add_event`
- `list_events_page`
- `add_artifact`
- `list_artifacts_page`
- `get_conversation_task_updates`

该模块保留 `(task_id, id)` 水位游标语义，并确保同一次 updates 请求中的 Task、Event 和 Artifact 来自同一个业务数据库快照。

### `store/helpers.py`

只保存两个以上业务模块确实共享的底层辅助函数：

- SQLAlchemy Row 到业务 Record 的转换；
- JSON 字段解码；
- UTC 时间生成；
- 共享的所有权查询；
- 活跃论文 ID 的读取、校验和更新；
- Event/Artifact 增量分页读取；
- 共享的数据库一致性诊断。

模块私有的校验和事务辅助函数继续放在所属业务模块。`helpers.py` 不得成为无明确归属代码的收纳目录。

## `TaskStore` 兼容门面

`paperpilot/web/task_store.py` 继续定义 `TaskStore`，并负责：

1. 解析业务数据库路径；
2. 确保 Alembic schema 为当前版本；
3. 创建和关闭 SQLAlchemy Engine 与 SessionFactory；
4. 使用显式同名方法转发到对应 `store/*` 业务函数；
5. 重新导出现有公开 Record 和业务异常。

示意：

```python
class TaskStore:
    def create_conversation(self, **kwargs) -> ConversationRecord:
        return conversations.create_conversation(
            self._session_factory,
            **kwargs,
        )
```

所有原公开方法必须显式存在于 `TaskStore` 类上。这能够保留 IDE 跳转、子类重写、类型检查和现有测试行为。门面不得依赖 `__getattr__` 或运行时方法注册。

## Conversation 路由拆分

现有 `paperpilot/web/routes/conversations.py` 调整为包 `paperpilot/web/routes/conversations/`：

### `__init__.py`

只重新导出 `build_conversation_router`，保留现有导入路径：

```python
from paperpilot.web.routes.conversations import build_conversation_router
```

### `router.py`

创建统一前缀为 `/api/conversations` 的 `APIRouter`，组装 CRUD、消息和回滚子路由。该文件不实现具体业务处理。

### `crud.py`

负责：

- 创建 Conversation；
- 列出 Conversation；
- 读取 Conversation 详情；
- 修改标题和归档状态。

### `messages.py`

负责：

- 读取当前活跃消息路径；
- 查询消息 alternatives；
- 提交用户追问；
- Executor 容量预留、提交和提交失败清理；
- busy/stale/capacity 异常到 HTTP 状态的映射。

### `rollback.py`

负责：

- 回滚入场校验；
- 目标 Assistant Message 和 Task 校验；
- 从 `DeepReadingRunner` 读取目标 checkpoint；
- checkpoint 与业务对象绑定校验；
- 调用 `switch_conversation_head`。

### `presenters.py`

负责将 Store Record 转换成 API 响应字典，不执行数据库查询或业务决策。

## 依赖方向

允许的依赖方向为：

```text
database.py + db_models.py
        ↓
store/records.py + store/helpers.py
        ↓
store/users.py / conversations.py / messages.py
store/publications.py / tasks.py / updates.py
        ↓
task_store.py
        ↓
routes/ + auth.py + DeepReadingRunner + worker_tasks.py
        ↓
app.py
```

禁止以下反向依赖：

- `store/*` 依赖 FastAPI Router；
- `store/*` 依赖 `app.py`；
- `store/*` 回头依赖 `TaskStore`；
- HTTP presenter 调用数据库；
- Store 业务模块读取 FastAPI Request 或生成 HTTPException。

业务模块之间应尽量不互相调用；真正共享且不拥有事务的能力进入 `helpers.py`。一个完整的业务事务必须由一个明确业务函数持有。

## 必须保持的事务边界

### 创建追问

`create_conversation_turn` 在同一事务中完成：

1. 创建 pending `ResearchTaskRow`；
2. 创建 User `MessageRow`；
3. 创建 queued `TaskEventRow`。

失败时三者全部回滚，并且不推进 Conversation 的稳定 head。

### 发布回答

`publish_conversation_result` 在同一事务中完成：

1. 创建或读取该 Task 唯一的 Assistant Message；
2. 创建或读取该 Task 唯一的 result Artifact；
3. upsert 本次实际使用的论文；
4. 更新 ConversationPaper 关联并返回候选 active paper IDs。

该步骤不推进 Conversation 稳定 head，并继续保持幂等和并发串行化语义。

### 最终确认

`finalize_conversation_task` 在同一事务中完成：

1. 验证 Task、User Message、Assistant Message 和 Conversation 绑定；
2. 验证业务 message head 与 checkpoint head 仍基于 Task 的原始 head；
3. 将 Task 标记为 completed 并保存 `final_checkpoint_id`；
4. 更新 active papers；
5. 同时推进 `head_message_id` 与 `head_checkpoint_id`；
6. 创建 completed Event。

### 失败终态

`fail_conversation_task` 保留现有幂等、并发和“不覆盖 completed Task”的行为，并保证失败 Event 与 Task 失败状态一致。

### 回滚

`switch_conversation_head` 在同一事务中完成：

1. 拒绝归档或存在活跃 Task 的 Conversation；
2. 校验调用者提供的 expected head；
3. 校验目标完整 Assistant Message；
4. 更新 active papers；
5. 同时切换 `head_message_id` 与 `head_checkpoint_id`。

回滚不删除 Message、Task、Artifact、Event 或 LangGraph checkpoint。

## 注释与可阅读性标准

### 模块文档字符串

每个新增业务模块必须在文件顶部说明：

- 本模块负责什么；
- 本模块明确不负责什么；
- 它使用哪些主要数据库对象；
- 关键事务或并发边界是什么。

### 公开业务函数文档字符串

公开业务函数需要说明：

- 前置条件；
- 会读取或写入哪些核心对象；
- 是否改变稳定 head；
- 失败时的原子性或幂等保证。

### 行内注释

行内注释只解释代码无法直接表达的原因，例如：

- SQLite 无行级 `SELECT FOR UPDATE` 时为何使用 no-op UPDATE 获取写入权；
- 为什么发布 Assistant Message 后暂时不推进稳定 head；
- 为什么 `head_message_id` 与 `head_checkpoint_id` 必须同步切换；
- 为什么 active papers 必须随回滚恢复；
- 为什么提交失败不能覆盖已被 Worker claim 的 Task。

禁止为明显的赋值、查询或循环逐行添加翻译式注释。

## 测试目录整理

Store 测试按业务职责拆分：

```text
tests/web/store/
├── conftest.py
├── test_lifecycle.py
├── test_users.py
├── test_conversations.py
├── test_messages.py
├── test_publications.py
├── test_tasks.py
└── test_updates.py
```

Conversation API 测试按路由职责拆分：

```text
tests/web/routes/conversations/
├── conftest.py
├── test_crud.py
├── test_message_submission.py
├── test_message_reads.py
├── test_rollback.py
└── test_task_updates.py
```

测试拆分遵循以下规则：

- 只重新归类，不删除断言或降低覆盖；
- 公共 fixture 和小型构造器放入相邻 `conftest.py`；
- 不建立新的复杂测试框架；
- 并发创建追问测试归入 messages；
- 并发发布、finalize/fail 竞争和幂等测试归入 publications；
- 回滚 API 与并发 head 切换测试归入 rollback；
- Event/Artifact 水位、索引和单快照测试归入 updates；
- 生命周期、迁移、WAL、健康检查测试归入 lifecycle。

现有测试对 `paperpilot.web.task_store.uuid`、`_utc_now` 等私有实现的替换，改为替换真正拥有该逻辑的业务模块。私有模块成员不作为公开兼容 API 保留。

## 保持不拆的模块

以下模块当前职责集中，本轮不拆：

- `app.py`：FastAPI 应用组装和资源生命周期；
- `task_executor.py`：统一的 Task 投递、容量与重试策略；
- `worker_tasks.py`：Celery Worker 入口和 Worker 资源生命周期；
- `observability.py`：请求日志和观测；
- `schemas.py`：HTTP Pydantic 契约；
- `db_models.py`：SQLAlchemy 表定义；
- `checkpoint.py`：LangGraph checkpoint SQLite 生命周期；
- `auth.py`、`routes/auth.py` 和 `routes/papers.py`：当前职责已经明确。

## 兼容性契约

重构后以下导入继续有效：

```python
from paperpilot.web.task_store import (
    TaskStore,
    WebUser,
    ResearchTask,
    ConversationRecord,
    MessageRecord,
    UsedPaperInput,
    ConversationBusyError,
    StaleConversationHeadError,
)
```

以下行为不得改变：

- API 路径、请求与响应 schema；
- 错误类型、HTTP 状态码和 `Retry-After`；
- SQLAlchemy schema、Alembic revision 和数据库路径；
- User 和 Login Session 生命周期；
- Conversation 所有权隔离；
- Message 父子树、active path 和 alternatives；
- 一个 Conversation 同时最多一个 pending/running Task；
- Task claim、redelivery、retry 和失败语义；
- Assistant Message 与 result Artifact 的幂等发布；
- Conversation message/checkpoint 双 head；
- continuous follow-up、checkpoint 恢复和 rollback；
- Event/Artifact 增量水位和单快照读取；
- Thread 和 Celery 两种 Executor 行为。

## 实施顺序原则

详细实施步骤将在单独计划中给出，但必须遵循以下顺序：

1. 先建立兼容性和依赖方向测试；
2. 提取 Record 与共享 helper；
3. 按业务域逐组迁移 Store 方法，每组完成后运行对应测试；
4. 保持 `TaskStore` 显式转发和可子类重写行为；
5. 拆分 Conversation Router，并保持导入路径兼容；
6. 最后移动和分组测试文件；
7. 运行聚焦测试、完整 Web 测试、Deep Reading 测试和全量测试。

## 验证要求

至少运行：

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/web/store -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/web/routes/conversations -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/deep_reading -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/web -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests -q
uv pip check
git diff --check
```

并进行以下结构检查：

- `TaskStore` 的全部原公开方法仍显式存在；
- 原公开 Record 和异常仍可从 `paperpilot.web.task_store` 导入；
- `paperpilot.web.routes.conversations` 继续导出 `build_conversation_router`；
- Store 层不存在对 Router、FastAPI 或 `app.py` 的反向依赖；
- Alembic schema 元数据与 revision 无变化；
- 自动测试不调用真实模型或产生模型费用；
- 现有业务和 checkpoint 数据库不被测试修改。

## 风险与控制

### 事务被无意拆散

控制方式：每个关键事务整体迁入一个业务函数；不允许门面逐步调用多个数据库写函数完成同一事务。

### 私有 helper 移动造成 monkeypatch 失效

控制方式：生产兼容只承诺公开 API；测试改为 patch 真正拥有逻辑的模块，并为时间和 ID 相关并发场景保留确定性验证。

### 显式门面产生重复样板

这是有意接受的成本。少量转发方法换取稳定公开接口、清晰 IDE 跳转、可重写性和低迁移风险。

### 模块之间重新形成循环依赖

控制方式：Record 不依赖业务模块；helper 不依赖 `TaskStore`；业务模块不依赖 Router；依赖测试扫描禁止反向引用。

### 测试移动时遗漏覆盖

控制方式：移动前后核对测试函数清单和总收集数量；聚焦测试通过后再运行完整测试集。

## 完成标准

当且仅当以下条件全部满足时，本次整理完成：

1. `task_store.py` 成为短小、显式、可阅读的兼容门面；
2. Store 业务逻辑按用户、会话、消息、发布、任务和更新分组；
3. Conversation Router 按 CRUD、消息和回滚分组；
4. 对应测试按业务边界整理且无覆盖减少；
5. 关键模块和复杂事务具备解释业务原因的注释；
6. 全部公开兼容契约保持不变；
7. 所有验证命令通过；
8. 未修改数据库 schema、现有业务数据或 checkpoint 数据。
