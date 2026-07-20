# PaperPilot Agent Runtime 控制平面设计

## 1. 背景

PaperPilot 已经具备完整的研究型 Agent 原型能力：

- `paperpilot/core/loop.py` 提供同步 ReAct 循环；
- `paperpilot/conversation.py` 负责系统提示词、内置工具和 MCP 工具装配；
- `paperpilot/builtin_tools/subagent.py` 可以并行深读多篇论文；
- `paperpilot/tools/mcp_runtime.py` 可以在 Celery worker 进程内复用 MCP 子进程；
- `paperpilot/web/*` 已有认证、用户隔离、任务、事件、产物、线程执行器和可选 Celery 执行器；
- SQLite 已启用 WAL、分页读取和基础运行保护。

当前缺口不再是“缺少一个工具”，而是缺少统一的 Agent 运行控制平面。现有 Web
任务只保存粗粒度状态、事件和最终产物，真实 Agent 的消息、步骤、工具调用、预算和
恢复位置仍主要存在于执行进程内。一旦进程退出、worker 被回收或任务需要取消，系统
无法从稳定边界继续执行，也无法精确判断一次工具调用是否可以安全重试。

本设计是“较成熟 Agent 体系”的第一阶段。后续阶段将在该控制平面之上加入结构化
Planner、通用子 Agent 编排、Verifier/Synthesizer 契约、长期记忆和在线质量闭环。

## 2. 目标

第一阶段交付一个统一、持久、可恢复的 Agent Runtime，使线程执行器和 Celery worker
共享相同的运行语义。

必须实现：

1. 每个真实研究任务对应一个持久化 `AgentRun`。
2. 每次 LLM、工具、子 Agent、上下文压缩和最终合成都形成有序 `AgentStep`。
3. 每个安全边界写入可恢复的 `RunCheckpoint`，保存协议安全的消息和运行预算。
4. 用户可以请求取消 `pending`、`running`、`waiting_retry` 状态的运行。
5. 可重试故障按照显式策略进入 `waiting_retry`，不得无限重试。
6. worker 重启后可以从最近完整检查点恢复，不重复提交已经完成的步骤。
7. 工具调用拥有稳定的执行 ID、错误分类、耗时和结果摘要。
8. 所有状态转换均通过条件更新完成，防止重复投递并发推进同一运行。
9. 现有 Web task API、认证和用户隔离保持兼容；新增运行详情和取消能力。
10. 运行事件仍映射到现有增量更新接口，前端无需理解底层消息协议。

## 3. 非目标

第一阶段明确不做以下工作：

- 不替换现有 ReAct 循环和 MCP 协议层；
- 不引入新的 Agent 编排框架；
- 不把 SQLite 替换为 PostgreSQL；
- 不实现任意位置的指令级恢复，只在已提交的安全边界恢复；
- 不实现通用 DAG Planner、动态角色创建或跨任务长期记忆；
- 不重做检索、证据验证和 QASPER 评测逻辑；
- 不承诺对外部有副作用工具的 exactly-once 语义；
- 不在本阶段修复 SQLite 与 Redis 发布之间的事务双写窗口。

## 4. 总体架构

```text
FastAPI Task API
  -> ResearchTask                      用户级任务入口
  -> AgentRun                          一次可恢复执行
  -> AgentRuntime                      统一推进器
       -> ExecutionPolicy              预算、重试、取消、权限
       -> AgentLoopDriver              按一步推进现有 ReAct 逻辑
       -> ToolExecutor                 工具调用记录与错误分类
       -> MCPRuntime                   常驻 MCP 资源
       -> RunStore                     run/step/checkpoint 持久化
  -> task_events / task_artifacts      Web 增量展示
```

核心原则是将“决定下一步”和“执行完整任务”拆开。`AgentLoopDriver` 每次只推进一个
可提交步骤，`AgentRuntime` 在步骤前后检查所有权、取消标记和预算，并负责写入状态。
线程和 Celery 只负责调度 `run_id`，不再各自决定 Agent 状态语义。

## 5. 模块边界

### 5.1 `paperpilot/agent/models.py`

定义不依赖 SQLite、FastAPI 和具体模型 SDK 的领域对象：

- `AgentRun`
- `AgentStep`
- `RunCheckpoint`
- `ToolExecution`
- `RunStatus`
- `StepKind`
- `StepStatus`
- `FailureClass`

这些对象只表达状态，不包含数据库连接或 MCP handler。

### 5.2 `paperpilot/agent/store.py`

定义 `RunStore` 协议及其 SQLite 实现。第一阶段允许 SQLite 实现复用
`TaskStore` 的连接配置，但 Agent Runtime 只能依赖 `RunStore` 接口，不能直接拼接
Web API SQL。

关键接口：

```python
class RunStore(Protocol):
    def create_run(self, *, task_id: str, policy: RunPolicy) -> AgentRun: ...
    def get_run(self, run_id: str) -> AgentRun | None: ...
    def claim_run(self, run_id: str, *, owner_id: str, lease_seconds: int) -> AgentRun | None: ...
    def renew_lease(self, run_id: str, *, owner_id: str, lease_seconds: int) -> bool: ...
    def append_step(self, step: AgentStep) -> AgentStep: ...
    def complete_step_and_checkpoint(self, *, step: AgentStep, checkpoint: RunCheckpoint) -> None: ...
    def request_cancel(self, run_id: str) -> AgentRun | None: ...
    def transition_run(self, run_id: str, *, expected: set[str], target: str, fields: dict) -> AgentRun | None: ...
```

`complete_step_and_checkpoint` 必须在同一个 SQLite 事务中完成，避免出现“步骤完成但
恢复点仍旧”的状态。

### 5.3 `paperpilot/agent/policy.py`

`ExecutionPolicy` 统一解释现有分散配置：

- 最大 Agent 迭代次数；
- 总 Token 预算；
- 单次 LLM 与工具超时；
- 最大可重试次数；
- 指数退避下限与上限；
- 允许的工具集合；
- 是否允许子 Agent；
- 租约时长与续租间隔。

策略在 run 创建时生成不可变快照并持久化。运行恢复时使用原策略，不受进程环境变量
后来变化影响。

### 5.4 `paperpilot/agent/tool_executor.py`

负责真实工具调用，但不负责让 LLM 选择工具。职责包括：

- 为调用生成稳定 `tool_execution_id`；
- 在执行前记录工具名、规范化参数、步骤和 attempt；
- 检查工具权限；
- 使用 `run_id`、`step_id` 和当前 `owner_id` 校验未过期租约与最新 step attempt；
- 调用现有 `Tool.handler`；
- 记录成功结果、错误类型、耗时和截断后的展示摘要；
- 将原始结果写入检查点消息；
- 根据工具类别和错误类别判断是否可重试。

第一阶段工具分为两类：

- `read_only`：搜索、读取、检索、图查询等，可以在传输类故障后重试；
- `idempotent_write`：例如按确定 `paper_id` 建立/覆盖索引，只能使用相同
  `tool_execution_id` 和相同参数重试。

未声明类别的工具默认 `non_retryable`。这比默认重试更保守。

`owner_id` 必须由 `AgentRuntime` 经 `AgentLoopDriver` 传到 `ToolExecutor`，再传给
`RunStore` 的工具开始与失败写入。工具写入不得只凭稳定 execution ID 修改记录；条件中
必须同时包含当前 step 和有效租约，防止旧 attempt 的迟到回调污染新 owner 的记录。

### 5.5 `paperpilot/agent/driver.py`

`AgentLoopDriver` 从当前 checkpoint 恢复消息，并一次推进一个边界：

1. 执行上下文预检，必要时产生 `compact` 步骤；
2. 调用一次 LLM，产生 `llm` 步骤；
3. 若无工具调用，产生最终结果并结束；
4. 若有工具调用，依次产生 `tool` 步骤；
5. 工具结果写回消息后返回控制权给 Runtime。

现有 `agent_loop` 在迁移期间保留给 CLI 和已有测试。共享的消息解析、参数修复和上下文
压缩逻辑应提取为小函数，由旧循环和新 Driver 共用，避免同时维护两份行为。

### 5.6 `paperpilot/agent/runtime.py`

`AgentRuntime` 是唯一能够推进持久化 run 的入口：

```python
class AgentRuntime:
    def execute(self, run_id: str, *, owner_id: str) -> RunOutcome: ...
```

执行流程：

1. 条件 claim run 并获取带过期时间的执行租约；
2. 读取最新完整 checkpoint；
3. 在每个步骤前检查取消标记、预算和租约；
4. 调用 Driver 推进一步；
5. 原子提交 step 与 checkpoint；
6. 续租并继续，或转换到终态/等待重试状态；
7. 将领域事件投影到 `task_events`；
8. 完成时将最终回答投影到 `task_artifacts`。

### 5.7 Web 和执行器适配

`WorkflowRunner.run_real(task_id)` 改为：

1. 获取或创建该任务的 active run；
2. 调用 `AgentRuntime.execute(run_id, owner_id=...)`；
3. 不再自己捕获所有异常并直接把 task 标记为 failed。

线程执行器和 Celery worker 继续接收 `task_id` 与 execution mode，以保持现有提交协议
兼容；进入 `WorkflowRunner` 后统一解析到 `run_id`。未来可以单独迁移为直接投递
`run_id`，但不属于第一阶段的必要接口变更。

## 6. 数据模型

### 6.1 `agent_runs`

| 字段 | 含义 |
|---|---|
| `id` | `run_<uuid>` |
| `task_id` | 关联 `research_tasks.id`，第一阶段一项任务最多一个 active run |
| `status` | 运行状态 |
| `attempt` | 当前 run 级尝试次数，从 1 开始 |
| `current_step` | 最近成功提交的步骤序号 |
| `cancel_requested_at` | 用户请求取消的时间 |
| `retry_at` | 下一次允许调度时间 |
| `failure_class` | 规范化失败分类 |
| `failure_message` | 供诊断的截断错误信息 |
| `policy_json` | 创建时固定的运行策略 |
| `owner_id` | 当前执行者身份 |
| `lease_expires_at` | 执行租约过期时间 |
| `schema_version` | Agent Runtime schema 版本，第一阶段固定为 1 |
| `created_at/updated_at/started_at/finished_at` | 生命周期时间 |

约束：同一 `task_id` 只能有一个非终态 run。SQLite 使用部分唯一索引实现。

### 6.2 `agent_steps`

| 字段 | 含义 |
|---|---|
| `id` | `step_<uuid>` |
| `run_id` | 所属 run |
| `sequence` | run 内从 1 递增 |
| `kind` | `llm/tool/subagent/compact/finalize` |
| `status` | `started/completed/failed/cancelled` |
| `attempt` | 此步骤尝试次数 |
| `input_json` | 结构化输入摘要 |
| `output_json` | 结构化输出摘要 |
| `error_json` | 规范化错误 |
| `schema_version` | Agent Runtime schema 版本，第一阶段固定为 1 |
| `started_at/finished_at` | 步骤时间 |

约束：`(run_id, sequence, attempt)` 唯一。

### 6.3 `run_checkpoints`

| 字段 | 含义 |
|---|---|
| `id` | `checkpoint_<uuid>` |
| `run_id` | 所属 run |
| `step_sequence` | 对应已完成步骤 |
| `messages_json` | 经 `message_codec` 编码的消息 |
| `runtime_state_json` | Guardrail、预算、下载文档和 Driver 状态 |
| `schema_version` | Agent Runtime schema 版本，第一阶段固定为 1 |
| `created_at` | 创建时间 |

约束：`(run_id, step_sequence)` 唯一。只允许读取已完成步骤对应的 checkpoint。

### 6.4 `tool_executions`

| 字段 | 含义 |
|---|---|
| `id` | 稳定工具执行 ID |
| `run_id/step_id` | 所属 run 和 step |
| `tool_name` | 完整工具名 |
| `arguments_json` | 规范化参数 |
| `classification` | `read_only/idempotent_write/non_retryable` |
| `status` | `started/completed/failed` |
| `result_preview` | 有界展示文本 |
| `failure_class/failure_message` | 失败分类与信息 |
| `duration_ms` | 调用耗时 |
| `schema_version` | Agent Runtime schema 版本，第一阶段固定为 1 |
| `started_at/finished_at` | 调用时间 |

原始工具结果不重复写入该表，而是通过 checkpoint 消息保存，避免两份结果产生漂移。

## 7. 状态机

运行状态：

```text
pending -> running
running -> completed
running -> waiting_retry -> pending
running -> failed
pending/running/waiting_retry -> cancelling -> cancelled
running -- lease expired --> pending
```

规则：

- 只有持有未过期租约的 owner 可以提交步骤和修改 `running` 状态；
- `completed/failed/cancelled` 是不可逆终态；
- `cancel_requested_at` 一经设置不可清除；
- 取消优先于重试，检查到取消后不得再创建新步骤；
- 租约过期只释放执行所有权，不回滚已经提交的步骤；
- `waiting_retry` 到期后由线程定时器或 Celery 重投递转换为 `pending`；
- 重复消息无法 claim 已被有效租约持有的 run。

任务状态保持现有四态以维持 API 兼容：

- run `pending/waiting_retry` 投影为 task `pending`；
- run `running/cancelling` 投影为 task `running`；
- run `completed` 投影为 task `completed`；
- run `failed/cancelled` 暂时投影为 task `failed`，详细原因由 run API 和事件区分。

后续若产品需要在任务列表直接区分 cancelled，再单独扩展 task 公共状态，不在第一阶段
静默改变现有 API 枚举。

## 8. 检查点和恢复语义

### 8.1 安全边界

允许创建检查点的位置：

- LLM 响应已经完整收到并编码；
- 单个工具结果已经完整收到并追加到消息；
- 子 Agent 已经全部结束并形成聚合结果；
- 上下文压缩结果已经替换消息；
- 最终回答已经形成但尚未/已经投影到 artifact。

正在进行中的网络调用不是检查点。进程在调用中退出时，恢复逻辑回到上一个完整检查点，
再根据步骤和工具分类决定是否重试。

### 8.2 工具调用恢复

- 若 `tool_execution` 已是 `completed`，恢复时直接使用 checkpoint 中的结果，不再调用；
- 若状态是 `started` 且工具为 `read_only`，标记前一次 attempt 为传输中断并重试；
- 若状态是 `started` 且工具为 `idempotent_write`，只允许携带相同 ID 和参数重试；
- 若状态是 `started` 且工具为 `non_retryable`，run 进入 `failed`，要求人工重新发起任务。

内部 `tool_execution_id` 用于识别同一次逻辑调用，但不会自动传递给不支持幂等键的
外部服务。`idempotent_write` 只有在工具本身存在稳定自然键并验证重复执行等价时才可
声明，例如 ColBERT `build_index` 以相同 `paper_id` 和相同文档内容覆盖同一索引。
没有这类业务保证时，即使内部 ID 相同也必须归类为 `non_retryable`。

### 8.3 最终产物幂等

最终 artifact 的 payload 保存 `run_id` 和 `step_id`，并通过唯一索引确保同一 run 只创建
一个 result artifact。若进程在创建 artifact 后、标记 run completed 前退出，恢复时复用
已有 artifact，再完成状态转换。

## 9. 错误分类与重试

`FailureClass` 第一阶段固定为：

- `validation`：参数、schema 或状态非法，不重试；
- `authentication`：API key 或权限问题，不重试；
- `rate_limit`：可重试，使用服务端提示或指数退避；
- `timeout`：只对允许重试的步骤重试；
- `transport`：MCP/HTTP 连接中断，可按工具类别重试；
- `capacity`：执行资源不足，回到队列等待；
- `cancelled`：用户取消，不重试；
- `internal`：未知代码错误，默认不自动重试并保留诊断事件。

默认最大自动重试 2 次。退避为 2 秒、8 秒，并加入不超过 20% 的抖动。策略快照允许
部署配置调整这些值，但测试使用固定随机源保证确定性。

MCP transport 故障继续触发现有 `MCPRuntime` 失效逻辑。下一次 attempt 重新创建
MCPClient，不在损坏的 session 上重试。

## 10. 取消语义

新增 `POST /api/tasks/{task_id}/cancel`：

- 仍使用现有登录和任务所有权校验；
- 对终态任务返回当前状态，不重复产生取消事件；
- 对可取消任务写入 `cancel_requested_at` 并返回 `202`；
- Runtime 在每个步骤前和步骤完成后检查取消；
- 正在执行的同步 LLM/MCP 调用不会被强制终止；
- 调用返回后丢弃尚未提交的新工作，写入取消事件并进入 `cancelled`；
- Celery revoke 不作为正确性机制，只能作为可选的资源优化。

## 11. 并发、租约与崩溃恢复

每次执行生成进程内唯一 `owner_id`。`claim_run` 使用条件 SQL：

- run 必须处于 `pending`，或 `running` 但租约已经过期；
- 写入 owner 和新租约必须在同一条更新中完成；
- 提交步骤时再次校验 owner 和租约；
- 长步骤由 Runtime 独立心跳线程续租，不能依赖步骤返回后才续租；
- 失去租约的执行者不得再写 checkpoint 或终态。

线程执行器崩溃后，由 API 启动时的轻量 reconciler 扫描过期 run 并重新提交。Celery
模式由 redelivery 和同一 reconciler 共同触发，但最终仍依赖 `claim_run` 保证单一推进者。

reconciler 只处理数据库中已经存在的 run，不能解决“数据库已提交但 Redis 发布尚未
发生”的任务双写窗口。该窗口继续作为已知风险，后续用 transactional outbox 单独解决。

## 12. 事件与可观测性

所有 Runtime 领域事件先保存到 Agent 表，再投影到现有 `task_events`。第一阶段至少
产生：

- `run_created/run_claimed/run_resumed`
- `step_started/step_completed/step_failed`
- `tool_started/tool_completed/tool_failed`
- `retry_scheduled`
- `cancel_requested/run_cancelled`
- `checkpoint_saved`
- `run_completed/run_failed`

事件 payload 必须包含 `run_id`、`step_id`、`sequence`、`attempt`、`owner_id` 和
`request_id` 中可获得的字段。工具参数和结果继续使用有界 preview，认证信息、cookie、
API key 和完整论文正文不得写入 Web 事件。

需要新增的聚合指标先通过结构化日志提供：

- run 数量和终态分布；
- run/step/tool 延迟；
- 重试次数与 failure class；
- 取消延迟；
- lease 过期与恢复次数；
- checkpoint 大小。

Prometheus/OpenTelemetry 适配仍保持为后续部署工作。

## 13. API 兼容与新增接口

保留现有接口和响应字段。新增：

- `GET /api/tasks/{task_id}/run`：返回当前 run 概览；
- `GET /api/tasks/{task_id}/run/steps`：按 sequence 增量分页；
- `POST /api/tasks/{task_id}/cancel`：请求取消；
- `/api/tasks/{task_id}/updates` 的事件 payload 增加 run/step 标识，但不删除旧字段。

创建 task 的请求结构第一阶段不新增必填字段。`depth` 映射为默认 policy：

- `quick`：较小迭代与 Token 预算，不启用通用子 Agent；
- `standard`：当前默认能力；
- `deep`：更高预算并允许论文深读子 Agent。

`execution_mode=simulated` 继续仅用于演示和测试。本阶段不静默修改其默认值；将产品默认
切换到 real 需要独立确认和部署前密钥检查。

## 14. 数据迁移策略

项目当前没有迁移框架，`TaskStore` 通过幂等 DDL 和列检查维护 SQLite schema。第一阶段
延续这一模式，不新增依赖：

1. 使用 `CREATE TABLE IF NOT EXISTS` 创建四张 Agent 表；
2. 使用 `CREATE INDEX IF NOT EXISTS` 创建约束与查询索引；
3. 旧任务不回填 run，只有新建真实任务或首次执行旧 pending 任务时惰性创建；
4. schema 初始化失败时 readiness 返回失败，API 不接受新任务；
5. 每张表记录 `schema_version=1` 所需字段，未来引入正式迁移工具时可识别。

## 15. 测试策略

所有实现遵循测试先行。测试层次：

### 15.1 领域与存储测试

- 状态转换合法性与终态不可逆；
- active run 唯一约束；
- claim/renew/lease expiry 的并发条件；
- step 与 checkpoint 原子提交；
- cancel request 幂等；
- 工具执行 ID 与结果复用；
- run、step 分页和用户隔离。

### 15.2 Runtime 测试

- 从空 run 执行到完成；
- 每个步骤后都能恢复且不重复已完成步骤；
- 在 LLM、工具、artifact 投影前后模拟崩溃；
- transport/rate-limit/timeout 的重试与上限；
- validation/auth/internal 错误不重试；
- 取消在步骤边界生效；
- owner 失去租约后不能继续写入；
- MCP transport 故障使 runtime 在下一 attempt 重建客户端。

### 15.3 Web 与执行器测试

- run 和 steps API 的认证、所有权、分页；
- cancel API 的 `202`、幂等和终态行为；
- 线程与 Celery 都调用同一个 AgentRuntime；
- 重复 Celery 消息只允许一个 owner 推进；
- updates 保持兼容并带 run/step 标识；
- readiness 覆盖 Agent schema 初始化失败。

### 15.4 回归与故障注入

- 运行现有完整 pytest 套件；
- 使用 fake LLM、fake tools 和可控时钟做确定性故障注入；
- 增加一个无需网络的端到端恢复场景；
- 运行 `compileall`、`git diff --check`；
- Celery 使用 `memory://` 验证任务注册和重复投递保护。

## 16. 分阶段实施顺序

### 阶段 1A：领域模型和持久化

实现 run/step/checkpoint/tool execution 表、状态机、租约和存储测试。此阶段不改真实
Agent 执行路径。

### 阶段 1B：单步 Driver 与 Runtime

提取现有 loop 的共用逻辑，加入单步推进、策略快照、错误分类和 checkpoint。CLI 旧路径
继续工作。

### 阶段 1C：Web、线程和 Celery 接入

将真实 WorkflowRunner 接到 AgentRuntime，增加 run/steps/cancel API、事件投影和恢复
reconciler。

### 阶段 1D：故障注入与运行验证

验证进程退出、租约过期、重复消息、工具 transport failure、取消和最终 artifact 幂等。

完成阶段 1 后再启动第二份独立设计：结构化 Planner、专业 Worker、Verifier 和
Synthesizer 的通用编排契约。

## 17. 验收标准

第一阶段只有在以下证据全部成立时才算完成：

1. 一个真实任务在每个已提交 step 后都可从 checkpoint 恢复；
2. 恢复不会再次执行已完成工具调用；
3. 两个并发 owner 不能同时推进同一 run；
4. 用户取消最终稳定进入 cancelled，且不会产生后续步骤；
5. 可重试故障遵循策略上限，不可重试故障直接失败；
6. 最终 artifact 在崩溃恢复后仍只有一份；
7. 线程和 Celery 路径使用同一个 Runtime 语义；
8. 新 API 保持认证和用户隔离；
9. 现有完整测试套件继续通过；
10. README 记录运行模型、配置、恢复边界和剩余风险。

## 18. 风险与后续工作

- SQLite 仍是单写者，checkpoint 频率提高后必须测量写入压力和数据库大小；
- 消息 checkpoint 可能包含大工具结果，需要在后续阶段评估压缩或外部 blob 存储；
- 同步 SDK 无法可靠强制取消正在进行的网络调用，取消延迟上限取决于当前调用超时；
- 外部有副作用工具只能提供幂等约束，不能由本地数据库单方面保证 exactly-once；
- Redis 发布双写窗口仍需 transactional outbox/reconciler 专项设计；
- 多 Agent 编排、长期记忆和在线质量治理建立在本控制平面之上，不应绕过 run/step
  模型另建一套状态。

## 19. 已确认决策

- 采用渐进式控制平面，不替换现有 Agent/MCP 栈；
- 第一阶段优先可靠性，再做自主编排和质量治理；
- SQLite 继续作为本地默认存储，并通过接口保留 PostgreSQL 迁移空间；
- 线程和 Celery 共享同一 AgentRuntime；
- 恢复粒度为安全步骤边界，不承诺调用中恢复；
- 设计确认日期：2026-07-19。
