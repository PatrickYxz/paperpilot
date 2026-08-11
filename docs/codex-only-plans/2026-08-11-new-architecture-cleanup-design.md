# PaperPilot 新架构白名单清理设计

## 背景

PaperPilot 已交付 Conversation + LangGraph 纵向切片，但当前仓库仍同时保留旧 CLI、同步 ReAct Agent Loop、旧 `/api/tasks` Web Workbench、历史评测和 Day 脚本。新架构虽然可以运行，却仍通过 `WorkflowRunner`、旧 Task API 和共享 `core.adapter.Tool` 与 legacy 代码耦合，导致项目目录无法直接表达当前唯一产品架构。

本次清理采用“原地白名单式清理”：保留已验证的新架构及其共享 MCP/检索基础设施，先解除 legacy 兼容依赖，再删除当前产品不再支持的代码、测试、脚本和文档。不得以复制一套 `paperpilot_v2` 或重新搭建骨架的方式重写项目。

## 已确认的用户决策

1. 保留全部用户和运行数据：业务数据库、Conversation、Message、Task、checkpoint、论文缓存和 ColBERT 索引。
2. 删除旧 CLI、旧 Agent Loop、旧 `/api/tasks`、Legacy Workbench 和旧 WorkflowRunner 路径。
3. 从当前项目移除历史评测模块、Day 脚本、评测测试和评测产物；已跟踪内容继续保留在 Git 历史中。
4. 只保留新架构文档和测试；未跟踪的旧架构审计计划先移到项目外备份目录。
5. arXiv、ColBERT、citation graph 和 VLM 全部保留，并明确定义为新架构的 MCP 扩展模块。
6. 使用当前包原地清理，不创建 `paperpilot_v2`，不重建仓库。
7. ColBERT planned retrieval 内部的 query planner 和可选 evidence verifier 从旧 `core.adapter.LLMClient` 迁移到 LangChain `ChatDeepSeek` + Pydantic structured output；planner 每次 retrieval 最多调用模型一次，verifier 继续默认关闭。

## 目标

- 当前工作树只呈现一个受支持的 PaperPilot 产品架构。
- FastAPI 是唯一产品入口，Conversation 是唯一用户交互模型。
- LangGraph 负责固定论文精读 SOP，LangChain 负责 Research Agent 与 structured output，Pydantic 负责状态和 API 契约。
- Thread/Celery 只执行绑定 Conversation 的 Deep Reading Task。
- 前端只保留 Conversation 工作台。
- 四类 MCP 能力继续作为新架构基础设施存在。
- 保持已有用户数据、checkpoint、历史消息和索引可用。

## 非目标

- 不实现 LangGraph Store 用户画像或跨 Conversation 记忆。
- 不增加 token streaming、cancel、interrupt 或 Postgres。
- 不引入 Repository、Ports、Adapters 或另一套自写框架。
- 不重写已经验证的 LangGraph、LangChain、checkpoint、Task claim、重试和幂等发布语义。
- 不迁移或清洗旧数据库 rows；只停止通过产品入口创建、执行和展示 legacy task。
- 自动验证不调用真实 DeepSeek，不产生模型费用。

## 最终运行架构

唯一支持的调用链：

```text
FastAPI Web
  -> Conversation API
  -> Task Executor (thread or Celery)
  -> DeepReadingRunner
  -> LangGraph fixed SOP
  -> bounded LangChain Research Agent
  -> MCP Runtime
  -> arXiv / ColBERT / Citation Graph / VLM
  -> business SQLite + LangGraph checkpoint SQLite
```

职责边界：

- FastAPI：认证、论文搜索、Conversation/Message/rollback API、任务进度查询、进程生命周期。
- SQLAlchemy/Alembic：用户可见的 Paper、Conversation、Message、Task、Event、Artifact 和 Conversation head。
- Task Executor：容量控制和 thread/Celery 投递，不包含研究工作流策略。
- DeepReadingRunner：Task claim、可信上下文构造、Graph 调用、恢复和最终发布。
- LangGraph：固定节点、边、摘要条件、checkpoint state。
- LangChain：DeepSeek 模型、Research Agent、工具循环、structured output 和官方预算 middleware。
- Pydantic：Graph、ResearchResult 和 API schema 校验。
- MCP：arXiv 下载/搜索、ColBERT 索引/检索、citation graph、VLM 页面理解。

## 目标目录结构

```text
paperpilot/
├── __init__.py
├── papers.py
├── deep_reading/
│   ├── graph.py
│   ├── nodes.py
│   ├── research_agent.py
│   ├── runner.py
│   ├── schemas.py
│   └── state.py
├── web/
│   ├── app.py
│   ├── auth.py
│   ├── celery_app.py
│   ├── checkpoint.py
│   ├── config.py
│   ├── database.py
│   ├── db_migrations.py
│   ├── db_models.py
│   ├── observability.py
│   ├── schemas.py
│   ├── task_executor.py
│   ├── task_store.py
│   ├── worker_tasks.py
│   ├── routes/
│   │   ├── auth.py
│   │   ├── papers.py
│   │   ├── conversations.py
│   │   └── task_updates.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── styles.css
├── tools/
│   ├── mcp_client.py
│   ├── mcp_runtime.py
│   └── types.py
├── mcp_servers/
│   ├── arxiv.py
│   ├── colbert/
│   ├── graph/
│   └── vlm/
└── retrieval/
    ├── __init__.py
    ├── evidence_pool.py
    ├── evidence_verifier.py
    ├── llm_query_planner.py
    ├── planned_retrieval.py
    ├── query_plan.py
    └── query_plan_validator.py
```

`paperpilot/tools/types.py` 只承载新架构共享的简单 `Tool` 数据类型。旧 `core.adapter` 中的 Anthropic-compatible `LLMClient`、`ToolCall`、`ToolResult` 和 `ParsedResponse` 不迁移；新架构的模型调用继续完全交给 LangChain。

## API 设计

保留：

```text
POST /api/auth/register
POST /api/auth/login
POST /api/auth/logout
GET  /api/auth/me

GET  /api/papers/search

POST  /api/conversations
GET   /api/conversations
GET   /api/conversations/{conversation_id}
PATCH /api/conversations/{conversation_id}
GET   /api/conversations/{conversation_id}/messages
POST  /api/conversations/{conversation_id}/messages
GET   /api/conversations/{conversation_id}/messages/{message_id}/alternatives
POST  /api/conversations/{conversation_id}/rollback
GET   /api/conversations/{conversation_id}/tasks/{task_id}/updates

GET /health/live
GET /health/ready
```

删除：

```text
/api/tasks/*
/api/eval/*
```

新 task updates 路由必须同时校验 `user_id`、`conversation_id` 和 `task_id`，不得退化为仅按 task ID 查询。响应继续包含 Task 状态、增量 Event 和增量 Artifact，以保持现有 Conversation UI 的完整回复和进度展示能力。

## 唯一数据流

1. 用户搜索并显式选择论文。
2. API 创建 Conversation；`conversation.id` 直接作为 LangGraph `thread_id`。
3. 用户发送消息；业务事务创建 User Message 和绑定该 Conversation 的 Task。
4. Executor 只提交 `task_id`；公开 API 不再接受 `execution_mode`。
5. Worker 通过 Task 反查 Conversation、User Message、主论文和可信 checkpoint。
6. DeepReadingRunner claim Task 后调用 LangGraph。
7. LangGraph 使用固定 SOP，并在 Research Agent 节点调用有界 LangChain Agent 和 MCP 工具。
8. checkpoint 写入独立 SQLite；业务发布事务写入 Assistant Message、引用论文、Task 终态和最终 checkpoint ID。
9. UI 通过 conversation-scoped updates 路由轮询进度；完成后重新读取 Message active path。
10. 连续追问复用同一 `thread_id`；rollback 只切换业务 head，不调用模型、不删除历史或 checkpoint。

## Task 与数据库兼容策略

- 新创建 Task 必须同时拥有 `conversation_id` 和 `user_message_id`。
- 删除 simulated task 和无 Conversation one-shot task 的产品创建/执行路径。
- Executor 和 Celery task 不再传递 legacy execution mode。
- 数据库现有 nullable 字段、legacy rows 和 Alembic 历史不做破坏性清理。
- 旧 rows 保留在 SQLite 中，但不再由 API 列出、执行或展示。
- `task_events` 与 `task_artifacts` 是新架构的运行记录，继续保留。
- TaskStore 删除只服务 legacy API/runner/eval 的方法，保留 Conversation、Message、Task claim、重试、发布、rollback、Event/Artifact 和 scoped updates 所需方法。
- 现有 migration 必须继续支持从空数据库升级到 head，并保留未知表/旧 rows 的既有安全边界。

## Web 模块整理

`paperpilot/web/app.py` 只负责：

- 创建 FastAPI app；
- 组装 Router；
- 初始化/关闭 TaskStore、Checkpoint Runtime、MCP Runtime 和 Executor；
- 安装认证、日志和 request-id 中间件；
- 暴露 health endpoints。

FastAPI 原生 `APIRouter` 按 auth、papers、conversations 和 task updates 分文件。Pydantic 请求/响应模型统一放在 `paperpilot/web/schemas.py`，避免继续把 schema、helper 和路由全部堆在 `app.py` 或单个 `conversation_routes.py` 中。

`paperpilot/web/task_executor.py` 继续只负责容量和投递；其协议改为 Conversation Task 语义，不再依赖 `WorkflowRunnerLike` 或 mode 字符串。`paperpilot/web/worker_tasks.py` 只构造并调用 DeepReadingRunner，删除 `paperpilot.conversation.run`、WorkflowRunner 和 legacy failure 分支。

## 前端整理

静态资源最终只保留：

```text
paperpilot/web/static/index.html
paperpilot/web/static/app.js
paperpilot/web/static/styles.css
```

- 删除 Legacy Tab、旧 task form、旧 task list 和 eval snapshot。
- 将当前 `app.js` 中仍需保留的登录、注册、登出和 session 恢复逻辑，与 `conversations.js` 合并为一个新 `app.js`。
- 将 Conversation 样式与仍需保留的全局/auth 样式合并为一个 `styles.css`。
- updates polling 改用 conversation-scoped task route。
- 保留论文搜索/选择、Conversation 列表、消息 active path、完整 Assistant Message、连续追问、alternatives、rollback、错误提示和登录状态保护。

## MCP 与检索保留策略

四类 MCP Server 全部保留：

- arXiv：搜索、规范化 ID、PDF 下载和正文缓存。
- ColBERT：按论文隔离的索引、search 和 planned retrieval。
- Citation Graph：邻居、共同引用和最短路径。
- VLM：论文页面图片理解。

`mcp_servers.json` 继续注册四个 server。MCP Client/Runtime 保持进程内复用和 transport invalidation 语义。仅将共享 `Tool` import 从 `paperpilot.core.adapter` 改到 `paperpilot.tools.types`。

保留 `paperpilot/retrieval/` 中 ColBERT server 实际依赖的 planned retrieval、query plan、evidence pool/verifier 等模块；删除只服务旧 eval/CLI 且无新架构调用者的模块前，必须用 import/dependency scan 证明无引用。

### Retrieval 内部模型迁移

当前 `llm_query_planner.py` 和 `evidence_verifier.py` 仍调用旧 `core.adapter.LLMClient`。这是新 ColBERT MCP 链路中的隐藏 legacy 依赖，必须在删除 `core/` 前迁移：

- 使用 `langchain_deepseek.ChatDeepSeek`，模型固定为 `deepseek-chat`。
- 使用 Pydantic structured-output schema 与 `include_raw=True`，不再手写从任意模型文本中提取 JSON 作为主路径。
- Planner 每次 `planned_retrieval` 恰好最多执行一次模型 invoke；解析失败或 provider 调用失败时继续返回安全的 literal-query fallback，并在 `query_plan_meta.fallback_reason` 记录分类。
- Planner 复用 `PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS` 和 `PAPERPILOT_RESEARCH_MODEL_RETRIES`，不新增另一套隐式无限重试。
- `MAX_QUERIES=6` 保持不变；新增 `MAX_EVIDENCE_REQUIREMENTS=6`，避免可选 verifier 的调用数随模型输出无限增长。
- Verifier 继续缺省 `verify_evidence=False`。显式启用时，每个 required requirement 最多一次 structured-output invoke，总数最多 6；默认 Deep Reading Research Agent 不启用 verifier，因此默认链路没有 verifier 模型费用。
- Research Agent 的 `ModelCallLimitMiddleware` 只统计 Agent 自身调用，不统计 MCP 子进程内 planner。运行说明必须明确：每次 `retrieve_paper_evidence` 还包含最多一次 planner 模型调用；它的总次数由 Agent 的 tool-call limit 间接封顶。
- 完成迁移后，requirements 中若无其他调用者，删除旧 Anthropic SDK 依赖。

该迁移保持 planned retrieval 的默认调用形状和 literal fallback，不在本次清理中重写查询规划算法或 evidence ranking。

## 删除清单

计划删除的运行代码：

```text
paperpilot/agent/
paperpilot/builtin_tools/
paperpilot/core/
paperpilot/eval/
paperpilot/skills/
paperpilot/main.py
paperpilot/conversation.py
paperpilot/bulk_input.py
paperpilot/document_store.py
paperpilot/message_codec.py
paperpilot/session_store.py
paperpilot/web/workflow.py
paperpilot/web/eval_summary.py
paperpilot/web/event_mapper.py
paperpilot/web/pagination.py
```

计划删除的研发资产：

- `scripts/day*.py`；
- 旧 eval/semantic audit/calibration/case-study 脚本；
- `tests/agent/`、`tests/builtin_tools/`、`tests/eval/`；
- 旧 CLI、ConversationSession、Agent Loop、DocumentStore、SessionStore、MessageCodec 测试；
- 仅覆盖 legacy `/api/tasks`、WorkflowRunner、eval endpoint 和 Legacy Workbench 的 Web 测试；
- 已跟踪的旧 Day 文档、旧架构计划和历史评测文档；
- `data/eval/` 与历史 trace 产物。

保留或重写：

- `tests/deep_reading/`；
- `tests/papers/`；
- 四类 `tests/mcp_servers/`；
- MCP Runtime/Client 测试（改用新 Tool import）；
- `tests/retrieval/` 中仍覆盖新 MCP 能力的测试；
- Web auth、database、migration、config、checkpoint、Conversation Store/API/UI、Executor、Worker、Celery、observability 测试；
- 与新 Web admission、数据库和 checkpoint 运维直接相关的 benchmark/smoke。

## 文档与未跟踪文件

- README 重写为新架构、唯一 API、启动顺序、数据所有权、恢复和预算说明。
- 当前新架构设计、清理设计和实施计划可以保留。
- 已跟踪的旧文档删除后由 Git 历史继续保存。
- 未跟踪文件 `docs/codex-only-plans/2026-08-05-architecture-audit-plan.md` 在任何删除前移动到：

```text
/Users/patrick/Documents/PaperPilot-archive/2026-08-05-architecture-audit-plan.md
```

移动后必须校验源路径不存在、目标文件存在且内容字节一致。该外部备份不提交到 PaperPilot Git。

## 数据保护

以下目录和文件不得删除、重建或迁移：

```text
data/web/
data/langgraph/
data/papers/
data/colbert_index/
```

清理前记录现有业务数据库与 checkpoint 数据库路径、文件大小和只读 health/integrity 结果；清理后再次只读验证。实现和自动测试一律使用临时数据库，不得把现有用户数据库作为测试目标。

`data/eval/` 和 trace 是已确认移除的研发产物，不属于保留的用户/运行数据。若其中存在未跟踪文件，实施前必须列出精确路径并单独确认删除范围，不能用宽泛递归命令处理整个 `data/`。

## 错误处理与运行保护

- Pydantic validation、Task/Message/checkpoint binding、schema/graph version、Research contract 和 MCP deterministic tool failure：terminal，立即失败。
- 数据库/checkpoint I/O、MCP transport/timeout 和 provider/network：按现有有界策略重试。
- Thread/Celery 都必须先原子 claim Conversation Task；重复投递不得产生重复 Assistant Message、Artifact 或 head 移动。
- LangChain model/tool call、max output tokens、model retry 和 recursion limit 继续使用当前已验证配置。
- 删除 legacy 分支不得扩大 broad exception catch，也不得改变失败事件中的公开安全消息。

## 实施顺序

1. 在隔离分支和工作树中记录基线测试、Git 状态和数据只读健康结果。
2. 将未跟踪旧架构审计计划移动到项目外备份并校验。
3. 新增架构约束测试：禁止 legacy imports、legacy routes、Legacy UI 标识和无 Conversation Task 创建。
4. 迁移 `Tool` 到 `paperpilot.tools.types`，保持 MCP/Deep Reading tests green。
5. 将 retrieval planner/verifier 迁移到 LangChain structured output，并用调用次数和 fallback 测试锁定成本及失败语义。
6. 新增 conversation-scoped task updates route，并迁移前端 polling。
7. 用 FastAPI Router/Pydantic schema 重组 Web composition，不改变新 API 行为。
8. 将 Executor/Worker 收敛为 DeepReadingRunner-only，删除 execution mode 和 WorkflowRunner 分支。
9. 合并前端静态资源并完成真实浏览器 smoke。
10. 通过依赖扫描后删除 legacy 运行代码、测试、脚本、文档和明确批准的 eval 产物。
11. 清理 requirements 中仅由已删除模块使用的依赖并重新生成/同步 lock。
12. 重写 README 和运维说明。
13. 运行完整验证与独立 review，确认无 Critical/Important 后才交付分支。

## 验证门禁

### 静态和架构门禁

- `rg`/AST import scan 无 legacy module 引用。
- OpenAPI 不再包含 `/api/tasks` 或 `/api/eval`。
- 静态 HTML/JS/CSS 不含 Legacy Workbench、legacy tab 或旧 API 请求。
- 新 Task 创建路径强制绑定 Conversation 和 User Message。
- `git diff --check` 通过，工作树只包含计划内变化。

### 自动测试

- Deep Reading Graph、nodes、Research Agent、Runner 和 schema。
- Paper catalog、arXiv cache、ColBERT、Citation Graph、VLM 和 MCP runtime。
- Retrieval 实际保留模块。
- Retrieval planner/verifier 使用真实 LangChain structured-output 边界的 fake-model 测试，覆盖一次调用上限、Pydantic parser failure、literal fallback、最多 6 个 requirement 和 verifier 默认关闭。
- Web auth、Conversation API/Store/UI、checkpoint、database、migration、Executor、Thread/Celery Worker、observability。
- 新 conversation-scoped updates 的 owner/conversation/task 三重隔离。
- 连续追问、rollback 零模型调用、alternatives、checkpoint restart/fork、幂等发布和有界重试。

### 持久化和运行 smoke

- Alembic 从临时空库升级到 head。
- 临时 checkpoint DB setup 后存在 `checkpoints` 和 `writes`。
- 现有业务 DB 和 checkpoint DB 清理前后只读 health/integrity 一致。
- `uv pip check` 通过。
- `/health/live`、`/health/ready` 返回 200。
- 四个 MCP Server 启动并列出预期工具。
- 浏览器验证注册、登录、论文搜索、Conversation 创建、会话列表、消息页面和登出；不调用真实模型。

真实 DeepSeek + arXiv 精读 smoke 仍需用户单独批准，因为会产生模型费用和外部网络访问。

## 验收标准

- 项目代码中只有一个产品入口、一个 Conversation API 和一个 LangGraph/LangChain 执行链。
- 目标目录结构成立，不存在 legacy package、旧 API、旧 UI、eval 模块或 Day 脚本。
- 四类 MCP 能力均保留并可启动。
- 当前用户数据库、checkpoint、论文缓存和 ColBERT 索引未被修改或删除。
- 旧数据库 rows 仍存在，但产品不再创建、执行或展示 legacy task。
- 新架构所有保留测试通过，fresh migration/checkpoint、dependency、health、MCP 和 browser gates 通过。
- README 只描述新架构和唯一启动方式。
- 删除范围经过独立审查，无未解释的 import、API、schema、数据或部署回归。

## 主要风险与控制

1. **新 UI 仍依赖 `/api/tasks/{id}/updates`。** 先新增并切换 conversation-scoped route，后删除旧 API。
2. **新 Deep Reading/MCP 仍依赖 `core.adapter.Tool`。** 先迁移最小 Tool 类型，后删除 core。
3. **Worker/Executor 仍通过 WorkflowRunner 做新旧分流。** 先收敛为 DeepReadingRunner-only 并覆盖 Thread/Celery，再删除 WorkflowRunner。
4. **TaskStore 混有 legacy 和 Conversation 方法。** 通过真实调用扫描和测试覆盖逐项删除，不在一轮机械重写中替换整个 Store。
5. **删除旧 tests 可能掩盖共享模块回归。** MCP、retrieval、database、auth、observability 等共享能力必须保留对应测试后才能删除旧测试组。
6. **数据保留与代码清理冲突。** 不做破坏性 schema migration，不删除 legacy rows；清理只作用于代码入口和已明确批准的非用户评测产物。
7. **历史资料丢失。** 已跟踪内容由 Git 历史保存；唯一未跟踪架构审计计划在任何删除前移到项目外并校验。
8. **ColBERT planned retrieval 隐藏调用旧 LLMClient。** 先迁移 planner/verifier 到有显式上限的 LangChain structured output，验证默认调用次数和 fallback 后才删除 `core/` 与 Anthropic SDK。
