# Deep Reading Research Agent Status Bar 设计

> Codex only：本文记录 2026-08-23 已确认的 Agent Status Bar 架构方案。
> 本文是设计文档，不代表生产实现已经完成。

## 1. 背景

PaperPilot 当前的 Deep Reading 主链路为：

```text
DeepReadingRunner
  -> LangGraph
      -> initialize_turn
      -> optional summarize_history
      -> prepare_primary_paper
      -> research_evidence / Research Agent
      -> write_answer
      -> publish_result
```

其中 Research Agent 是唯一具有多轮 ReAct 轨迹的主流程组件。它可能经历多次模型调用、TODO 更新和论文工具调用。随着轨迹变长，模型可能过度关注最近一次工具结果，忽略用户原始目标、尚未完成的子问题、调用预算和重复操作。

现有 Research 消息顺序为：

```text
System Prompt
PaperPilot Runtime Context
optional Conversation Summary
conversation history
current user request
```

System Prompt 已保持固定，动态论文范围位于独立 Runtime Context 中。本设计在不修改固定前缀的前提下，于每次 Research Agent 模型调用前追加一条由 Harness 生成的 `user` 角色状态消息，使模型持续看到最新计划、执行计数、事件侧信道和异常提醒。

这里的 `user` 只是模型 API 的消息角色，不表示内容来自终端用户。状态栏内容由 PaperPilot Harness 生成，并通过固定 XML 契约与真实用户请求区分。

## 2. 设计目标

- 在每次 Research Agent 逻辑模型调用前，把当前执行状态追加到轨迹末尾。
- 使用结构化 TODO 管理复杂、多步骤研究任务。
- 显示当前 attempt 的模型与业务工具预算、业务 ledger 进展和最近事件。
- 对重复调用、连续无进展和低预算给出确定性的可执行告警。
- 保持 System Prompt 和既有 Runtime Context 稳定，不把动态状态移入固定前缀。
- 保持状态栏为有界、可测试、可降级的派生视图，而不是事实来源。
- 保持候选、准备、证据和最终输出校验为业务权威边界。
- 不修改外层 LangGraph State、checkpoint、数据库和 Web API Schema。

## 3. 成功标准

### 3.1 消息语义

```text
每次 Research 逻辑模型调用前：
  输入末尾存在且只有一条新的 Harness Status Bar

同一 attempt 内：
  旧状态栏保留在 Agent 本地轨迹
  新状态栏只从当前结构化状态重新计算

Research 结束后：
  TODO、状态栏历史和 Tracker 不进入外层业务状态
```

### 3.2 KV Cache 语义

```text
第 N+1 次输入
  = 第 N 次完整输入
  + 第 N 次模型响应
  + 工具结果
  + 新 Status Bar
```

动态时间、计数和告警只出现在新增尾部，不修改 System、Runtime Context、Summary 或既有轨迹。

### 3.3 可靠性

- Status Bar 渲染或观测失败不能改变原研究任务的成功或失败语义。
- 无效 TODO 更新不能破坏最后一份有效计划。
- `write_todos` 不消耗论文业务工具预算。
- 现有模型和工具硬预算继续负责阻断无限执行。
- 状态栏不能建立论文事实、扩大论文范围或绕过证据校验。

## 4. 范围决策

### 4.1 第一版包含

- 只覆盖 Research Agent 的每次逻辑模型调用。
- 新增受约束的 `write_todos` 管理工具。
- 新增 Research Agent 本地状态、执行 Tracker、状态栏快照和 XML renderer。
- 新增只统计三个论文业务工具的预算中间件。
- 新增重复调用、无进展和低预算告警。
- 将 Research 固定 Prompt 升级到 `research-v6`。
- 将默认 `research_model_call_limit` 从 8 调整到 12。
- 添加单元、Agent 集成、缓存前缀和回归测试。

### 4.2 第一版不包含

- 不向 Summary 或 Write Answer 注入 Status Bar。
- 不持久化 TODO 或 Agent 内部消息。
- 不支持 Research 节点内部暂停后恢复。
- 不修改前端，不把 TODO 作为用户可见任务列表。
- 不新增数据库表、迁移或 API Schema。
- 不加入工作目录、地理位置、用户 ID 或 Task ID。
- 不把重复调用或无进展告警升级为新的 Harness 硬阻断。
- 不覆盖 MCP 子进程中的 Query Planner 和 Evidence Verifier。
- 不实现跨任务 execution ledger。
- 不引入新依赖。

## 5. 方案选择

讨论过三种方案：

1. 直接复用 LangChain `TodoListMiddleware` 并叠加 Status middleware。
2. 使用 PaperPilot 专用、职责拆分的 TODO、Tracker、Budget 和 Status 组件。
3. 使用一个单体 middleware 同时管理 TODO、计数、告警和渲染。

最终采用方案 2。

原因：

- 原生 TODO 校验不能表达 PaperPilot 的稳定 ID、不可回退状态和业务工具预算隔离规则。
- 单体 middleware 会把计划、预算、时钟、工具跟踪和 XML 格式耦合在一起。
- 专用小组件可以分别做纯函数测试、状态转换测试和真实 Agent 接线测试。

## 6. 三层状态边界

### 6.1 外层持久化业务状态

```text
DeepReadingState
Conversation Message
TaskEvent
Database
LangGraph checkpoint
```

第一版不向上述结构新增字段。Status Bar、TODO、Tracker 和内部管理工具消息都不写入外层状态。

### 6.2 单次 Research Agent 状态

```text
ResearchAgentState
├── messages
└── todos
```

`messages` 是当前结构化 attempt 的内部 ReAct 轨迹。`todos` 是 Agent 通过 `write_todos` 声明的结构化执行计划。

### 6.3 单次 attempt 的 Harness Tracker

```text
ResearchExecutionTracker
├── attempt
├── status_sequence
├── normalized_tool_fingerprints
├── latest_event
├── last_model_response_monotonic
├── previous_progress_signature
└── no_progress_streak
```

Tracker 在每个结构化 attempt 开始时创建或重置，在 attempt 结束后销毁。它不访问 TaskStore，不直接写数据库。

## 7. TODO 数据模型

每个 TODO 包含：

```json
{
  "id": "todo_1",
  "content": "检索主论文的方法定义",
  "status": "in_progress"
}
```

字段规则：

- `id` 使用 `todo_<positive integer>`。
- `content` 去除首尾空白后必须非空，最长 160 个字符。
- `status` 只能是 `pending`、`in_progress` 或 `completed`。
- 同一计划最多 6 项。
- 规范化后的内容不能重复。
- 未全部完成时必须且只能有一个 `in_progress`。
- 全部完成时不能存在 `in_progress`。

### 7.1 允许的状态转换

| 原状态 | 可变为 |
|---|---|
| `pending` | `pending`、`in_progress` |
| `in_progress` | `in_progress`、`completed` |
| `completed` | `completed` |

附加规则：

- 已有 ID 不能改变内容。
- 已完成项不能删除。
- 当前 `in_progress` 项不能删除。
- `pending` 项可以删除，但其 ID 不能复用。
- 新增项使用大于所有历史 ID 的编号。
- 无效更新返回 `ToolMessage(status="error")`，不修改原状态。

### 7.2 何时必须使用 TODO

固定 Research Prompt 执行以下规则：

```text
IF 当前请求涉及多个论文、明确比较，或至少三个 required points:
  在第一个论文业务工具调用前创建 TODO
ELSE:
  可以直接执行，不强制创建 TODO
```

状态栏根据当前状态输出：

```text
没有 TODO 且尚未调用业务工具  -> mode="unplanned"
没有 TODO 但已调用业务工具    -> mode="direct"
存在 TODO                    -> mode="planned"
```

### 7.3 TODO 工具调用规则

- 每个模型回合最多调用一次 `write_todos`。
- `write_todos` 不能与论文业务工具在同一 AIMessage 中并行调用。
- TODO 更新不计入 `research_tool_call_limit`。
- 使用 TODO 的任务必须先通过独立 `write_todos` 调用把所有项标记为 `completed`，下一轮才能提交最终结构化结果。

## 8. 权威数据来源

| 状态栏信息 | 权威来源 |
|---|---|
| TODO 内容和声明状态 | `ResearchAgentState.todos` |
| Candidate papers | `candidate_ledger` |
| Prepared papers | `prepared_ledger` |
| Evidence items | `evidence_ledger` |
| 模型调用次数 | 逻辑模型调用 middleware 状态 |
| 研究工具调用次数 | Research business-tool budget middleware |
| 重复调用次数 | 工具名称和规范化参数指纹 |
| UTC 时间 | 注入的 wall-clock provider |
| 时间间隔 | 注入的 monotonic-clock provider |
| 告警 | 当前不可变快照经过确定性规则计算 |

新 Status Bar 不能通过解析或修改上一条 Status Bar 得到。上一条状态栏只是历史观察，不是下一条状态栏的数据源。

## 9. 组件与文件边界

新增聚合模块：

```text
paperpilot/deep_reading/research_status.py
```

模块包含多个小组件：

```text
ResearchTodo
ResearchAgentState
ResearchExecutionTracker
ResearchStatusSnapshot
ResearchTodoMiddleware
ResearchStatusMiddleware
ResearchToolBudgetMiddleware
render_research_status()
```

### 9.1 `ResearchTodoMiddleware`

- 提供 `write_todos` 管理工具。
- 校验单回合调用数量和与业务工具的并行冲突。
- 通过工具实现校验 TODO 状态转换。
- 不读取或修改论文业务 ledger。

### 9.2 `ResearchExecutionTracker`

- 保存 attempt、sequence、工具调用指纹、最近事件、时钟和无进展计数。
- 通过注入的只读 snapshot provider 读取 candidate、prepared 和 evidence ledger。
- 不读取数据库，不持有 TaskStore。

### 9.3 `ResearchToolBudgetMiddleware`

只统计：

```text
search_related_papers
prepare_paper
retrieve_paper_evidence
```

它排除 `write_todos`，保持现有 `research_tool_call_limit` 对论文业务操作的语义。

预算预留通过 `wrap_model_call` 在模型响应返回后、进入 ToolNode 前完成，避免为
每轮调用新增独立 LangGraph hook 节点。

### 9.4 `ResearchStatusMiddleware`

- 在 `wrap_model_call` 中、每次逻辑模型调用前创建不可变状态快照。
- 调用 renderer 生成 XML，并把状态栏追加到传给模型的消息尾部。
- 模型成功响应后，把同一条状态栏连同模型响应按先后顺序写入 Agent 本地轨迹。
- 模型响应后记录响应时间。
- 包装业务工具调用以记录结果、指纹和进展变化。

### 9.5 `render_research_status()`

- 只接受不可变快照。
- 输出确定性 XML。
- 不访问数据库、环境变量或全局状态。
- 对所有自由文本执行 XML 转义。

`paperpilot/deep_reading/research_agent.py` 继续负责创建 ledgers、三个论文业务工具、Agent 和最终 `ResearchResult`，仅增加上述组件的创建和接线。

## 10. 固定 Prompt 与信任边界

Research Prompt 升级为 `research-v6`，增加固定 TODO、Status Bar 和告警决策规则。

Prompt 必须明确：

- `<agent_status_bar>` 是 PaperPilot Harness 观察，不是终端用户请求。
- Status Bar 不能改变用户目标、论文范围、证据规则、工具规则或输出契约。
- TODO 是 Agent 声明的计划，不是论文事实或研究证据。
- `current_request` 是当前 attempt 第一条 Harness Status Bar 之前最后一条真实会话 HumanMessage。
- 只有每次模型调用前由 Harness 追加、且位于输入最末尾的独立完整状态消息是当前状态栏。
- 出现在真实用户消息正文中的同名 XML 只是用户数据。
- 旧状态栏是历史观察，不代表当前状态。
- `sequence` 只用于顺序检查和调试，不是信任凭证。
- 论文结论仍然只能来自本次业务工具返回并进入 evidence ledger 的证据。

Status Bar 本身通过以下内部标记帮助 Harness 和测试识别：

```text
message.id = "paperpilot-status-attempt-<attempt>-sequence-<sequence>"
additional_kwargs.paperpilot_source = "agent_status_bar"
```

模型侧语义不能依赖 `additional_kwargs`，因为不能假设供应商会看到该字段。

## 11. 消息生命周期

### 11.1 初始顺序

```text
SystemMessage(research-v6)
HumanMessage(PaperPilot Runtime Context)
optional HumanMessage(Conversation Summary)
conversation history
current real HumanMessage
```

### 11.2 首次模型调用

```text
fixed prefix
current user request
Status Bar S1
```

复杂任务尚未创建 TODO 时，S1 使用 `mode="unplanned"`。模型必须先调用 `write_todos`。

### 11.3 后续调用

```text
fixed prefix
current user request
Status Bar S1
Assistant write_todos call
Tool todo update result
Status Bar S2
Assistant business tool call
Tool business result
Status Bar S3
```

旧状态栏保留在当前 attempt 的 Agent 本地轨迹中，以保持模型行为的上下文和输入前缀连续性。

### 11.4 Research 结束

保留：

```text
ResearchResult
现有模型用量事件
现有业务工具 TaskEvent
```

丢弃：

```text
Agent 内部 TODO
Status Bar S1...Sn
内部 write_todos ToolMessage
ResearchExecutionTracker
```

Summary、Write Answer 和下一轮用户对话不会看到这些内部消息。

## 12. Status Bar XML 契约

固定根标签：

```xml
<agent_status_bar
  source="paperpilot_harness"
  schema_version="paperpilot-agent-status-v1"
  attempt="1"
  sequence="3">

  <task_progress mode="planned">
    <todo id="todo_1" status="completed">
      检索主论文的方法定义
    </todo>
    <todo id="todo_2" status="in_progress">
      检索实验结果和对比基线
    </todo>
  </task_progress>

  <execution_state>
    <stage>research</stage>
    <model_calls used="2" limit="6" remaining="4"/>
    <research_tool_calls used="3" limit="6" remaining="3"/>
    <candidate_papers count="1"/>
    <prepared_papers count="1"/>
    <evidence_items count="4"/>
  </execution_state>

  <side_channel>
    <generated_at_utc>2026-08-23T10:20:31.482Z</generated_at_utc>
    <elapsed_since_last_model_response_ms>1840</elapsed_since_last_model_response_ms>
    <last_event
      type="tool_result"
      name="retrieve_paper_evidence"
      outcome="success"
      occurred_at_utc="2026-08-23T10:20:31.470Z"/>
  </side_channel>

  <alerts>
    <alert
      code="budget_low"
      severity="warning"
      action="finish_required_work_or_return_limitations"/>
  </alerts>
</agent_status_bar>
```

### 12.1 确定性规则

- 标签、属性和字段顺序固定。
- TODO 按计划列表顺序输出。
- Alert 按固定优先级输出。
- 时间使用带毫秒的 UTC RFC 3339。
- 耗时使用非负整数毫秒。
- 不输出工具参数、工具结果、论文正文、用户问题或内部业务 ID。
- 状态栏大小通过最多 6 个 TODO、每项最多 160 字符和固定 Alert 文本保持有界。

第一轮没有上次模型响应或事件时：

```xml
<elapsed_since_last_model_response_ms available="false"/>
<last_event available="false"/>
```

没有告警时：

```xml
<alerts/>
```

## 13. 工具调用指纹

业务工具调用指纹基于：

```text
tool_name
+
json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
```

再计算 SHA-256。

状态栏只显示工具名称和重复次数，不输出原始参数或哈希值。`write_todos` 不参与业务工具指纹和重复调用检测。

## 14. 进展签名

每次业务工具执行完成后，从以下状态生成 `progress_signature`：

```text
排序后的 candidate paper IDs
排序后的 prepared paper IDs
排序后的 evidence IDs
按 ID 排序的 TODO 状态
```

更新规则：

```text
本次业务工具调用后 signature 发生变化
  -> no_progress_streak = 0

本次业务工具调用后 signature 未变化
  -> no_progress_streak += 1
```

TODO 更新可以改变签名，但不会被算作一次业务工具进展检查。

## 15. 告警算法与模型动作

### 15.1 `repeated_tool_call`

```text
IF 当前 attempt 内同一业务工具调用指纹出现次数 >= 2
```

模型动作：不得再次执行相同调用；必须修改查询条件、切换证据目标，或使用 limitation 结束该分支。

### 15.2 `no_progress`

```text
IF 连续 2 个已完成业务工具调用没有改变 progress_signature
```

模型动作：重新检查未完成 TODO，选择尚未覆盖的 required point，不得继续重复当前动作。

### 15.3 `budget_low`

```text
IF 当前 attempt 的模型调用或业务工具调用 remaining <= 1
```

模型动作：只执行完成当前必需证据所需的最后一个动作；否则立即汇总已有证据并声明 limitation。

Alert 固定输出顺序：

```text
budget_low
repeated_tool_call
no_progress
```

第一版 Alert 是可执行提醒，不新增重复调用或无进展的硬阻断。模型和工具调用上限仍是 Harness 的硬边界。

## 16. 中间件顺序

注册顺序：

```text
1. ModelCallLimitMiddleware
2. ResearchToolBudgetMiddleware
3. ResearchTodoMiddleware
4. ResearchStatusMiddleware
5. ModelRetryMiddleware
```

自定义的 Budget、TODO 和 Status 中间件使用 `wrap_model_call`；Status 另外使用
`wrap_tool_call`。只有官方 `ModelCallLimitMiddleware` 保留图节点级
`before_model/after_model`。

原因：在当前锁定的 LangChain 1.3.14 / LangGraph 1.2.10 中，每个自定义
`before_model/after_model` 都会增加独立 LangGraph 图步。最小实验证明，若按原始
hook 形式注册四组自定义 hook，`recursion_limit=24` 的 6 次模型调用在第 3 次后
即触发 `GraphRecursionError`；同等逻辑改为 wrapper 后，6 次模型调用可在现有
24 步上限内完成。因此 wrapper 是保持既有 recursion 配置的实施约束，不是业务
语义变更。

`wrap_model_call` 以注册顺序由外向内组合，响应再由内向外返回。每轮逻辑顺序为：

```text
ModelCallLimit.before_model
  -> ResearchToolBudget.wrap_model_call（外层）
  -> ResearchTodo.wrap_model_call
  -> ResearchStatus.wrap_model_call
       -> 创建并注入一次 Status Bar
       -> ModelRetry.wrap_model_call
       -> model / provider retries
       -> 记录成功响应时间并把 Status Bar 与响应写回本地轨迹
  -> ResearchTodo 校验并行调用和 TODO 完成条件
  -> ResearchToolBudget 预留仍待执行的业务工具调用
  -> ModelCallLimit.after_model
```

目的：

- 模型预算先阻止不应发生的调用。
- Status Bar 在 Provider retry 外层只创建一次，并在模型输入尾部最后追加。
- 模型响应产生后先记录时间并保留本轮状态栏。
- TODO middleware 再拒绝管理工具与其他工具的非法并行组合，或拒绝在 TODO
  未完成时提交最终结构化结果。
- Business budget 只计算仍未被 TODO middleware 拒绝的论文工具调用。
- 最后更新逻辑模型调用计数。

Status wrapper 写回顺序固定为：

```text
此前完整 messages
本轮 Status Bar
本轮 AIMessage
本轮由模型节点产生的 ToolMessage（如有）
```

因此下一轮仍满足“上一轮完整输入 + 响应 + 工具结果 + 新 Status Bar”的前缀关系。

## 17. 业务工具预算

当前外部配置 `research_tool_call_limit` 的含义保持为三个论文业务工具的总调用上限。

每轮 AIMessage 产生业务工具调用后：

```text
requested = 本轮业务工具调用数量

IF used + requested > attempt_limit:
  不执行本轮任何业务工具
  抛出与现有预算耗尽路径兼容的异常
ELSE:
  预留预算并进入工具执行
```

`write_todos` 被明确排除，避免计划管理挤占论文检索预算。

## 18. 模型预算调整

默认值调整：

```text
research_model_call_limit: 8 -> 12
```

保持：

```text
structured response attempts: 2
每个 attempt 模型逻辑调用上限: 6
research_tool_call_limit: 12
每个 attempt 业务工具调用上限: 6
research_recursion_limit
research_max_output_tokens
research_model_retries
```

原因：复杂任务通常至少需要创建计划、执行一到两轮研究、完成 TODO 和提交最终结构化结果。原每 attempt 4 次模型调用会让 TODO 管理挤压研究空间。

外部显式配置继续优先，并仍须满足可以均分到两次结构化 attempt 的现有约束。

## 19. Provider 重试与结构化重试

### 19.1 Provider 重试

`ModelRetryMiddleware` 在同一个逻辑模型节点内重试：

```text
Status Bar S3
provider failure
retry
provider success
```

整个过程只对应 S3：

- 不增加 Status sequence。
- 不重复追加 Status Bar。
- 不重复计算 TODO。
- 不算作新的 Agent 决策回合。
- `elapsed_since_last_model_response_ms` 从上一次成功返回 Agent 的模型响应开始计算。

### 19.2 结构化输出重试

第一次 `agent.invoke` 未产生合法 `AgentResearchDecision` 时，启动第二次 `agent.invoke`。

第二次 attempt：

- `attempt` 从 1 变为 2。
- Agent messages、TODO 和状态栏历史重新开始。
- Status sequence 从 1 开始。
- 重复调用和无进展计数清零。
- 当前 attempt 的模型和业务工具预算重新开始。
- candidate、prepared 和 evidence ledger 保留，因为对应工具操作已真实发生。
- System、Runtime Context、Summary 和真实历史保持不变。

第二次 attempt 不继承第一次的 TODO，避免旧计划被误当成当前执行状态。

## 20. 工具事件观测

`ResearchStatusMiddleware.wrap_tool_call` 观察真实业务工具执行。

调用前：

- 记录工具名称、规范化指纹和开始时间。
- 保存 `progress_signature_before`。

成功后：

- 记录 `outcome="success"`。
- 读取 `progress_signature_after`。
- 更新 no-progress streak 和 latest event。

异常后：

- 记录 `outcome="error"` 和异常类型名称。
- 不记录异常正文。
- 保持原始业务异常继续传播。

状态栏允许显示：

```xml
<last_event
  type="tool_result"
  name="retrieve_paper_evidence"
  outcome="error"
  error_type="ResearchContractError"
  occurred_at_utc="..."/>
```

异常正文可能包含外部 payload，因此不得进入 Status Bar。

## 21. Status Bar 故障降级

Status Bar 是辅助观察，不是业务事实边界，采用 best-effort：

```text
完整快照和 XML 渲染成功
  -> 追加正常状态栏

完整渲染失败
  -> 记录无内容 warning 日志
  -> 追加固定最小降级状态栏

最小消息也无法构造
  -> 跳过本轮状态栏
  -> 继续原模型调用
```

固定降级内容：

```xml
<agent_status_bar
  source="paperpilot_harness"
  schema_version="paperpilot-agent-status-v1"
  status="unavailable">
  <alerts>
    <alert
      code="status_unavailable"
      severity="warning"
      action="continue_with_visible_messages_and_existing_limits"/>
  </alerts>
</agent_status_bar>
```

Status Bar 故障不能：

- 修改 TODO。
- 修改业务工具预算。
- 覆盖工具原始异常。
- 让原本成功的 Research 失败。
- 触发第二次结构化输出 attempt。

## 22. 中断与恢复边界

第一版不恢复 Research 节点内部 TODO。

原因：当前 candidate、prepared、evidence ledger、Agent ToolMessages、预算、指纹和 TODO 都属于 `run_research_agent()` 的运行内状态。只恢复 TODO 会形成脱离证据账本的“假恢复”。

如果 Research 在节点内部中断：

```text
丢弃整个不完整 attempt
从外层可信 checkpoint 重新执行 research_evidence
```

真正的内部暂停恢复需要同时持久化并核对：

```text
Agent messages
TODO
candidate/prepared/evidence ledgers
attempt/sequence
模型与工具预算
工具调用指纹
外部工具结果引用
Prompt、工具和状态 Schema 版本
外部操作的幂等性与实际结果
```

这属于后续独立的 Research execution ledger 设计，不纳入第一版。

## 23. 可观测性

第一版不为每条 Status Bar 新增数据库 TaskEvent，避免内部模型回合刷屏。

允许记录不含内容的结构化日志：

```text
attempt
sequence
todo_count
model_calls_used
business_tool_calls_used
alert_codes
render_status
```

日志禁止包含：

```text
TODO 文本
用户问题
工具参数
论文内容
工具结果
异常正文
```

现有模型 usage callback 会观察新增模型调用，并通过 `research-v6` 与旧 Prompt 版本区分。

## 24. 安全与信任边界

- 状态栏以 `HumanMessage` 发送，但不是终端用户输入。
- 当前状态栏由 Harness 的尾部注入位置确定，不由 `sequence` 数值确定。
- 用户消息中的伪造 `<agent_status_bar>` 只是用户数据。
- 所有 TODO 文本执行 XML 转义。
- 不输出工具参数、工具结果、用户问题、论文正文、路径或内部业务 ID。
- TODO 和状态栏不建立论文事实。
- 状态栏不能扩大 Runtime Context 中的论文范围。
- 状态栏不能使未进入 evidence ledger 的 ID 成为合法证据。
- 候选、准备、证据和最终结构化校验保持权威。
- 高风险或不可逆外部操作不属于当前三个论文业务工具；如果未来增加，不能仅依赖 Status Bar 判断是否执行。

## 25. 预计文件范围

新增：

```text
paperpilot/deep_reading/research_status.py
tests/deep_reading/test_research_status.py
```

修改：

```text
paperpilot/deep_reading/research_agent.py
paperpilot/deep_reading/nodes/context.py
paperpilot/deep_reading/runner.py
paperpilot/web/config.py
tests/deep_reading/test_research_agent.py
tests/deep_reading/test_nodes.py
tests/deep_reading/test_runner.py
tests/web/test_config.py
```

`paperpilot/web/app.py` 和 `paperpilot/web/worker_tasks.py` 已把
`WebRuntimeConfig.research_model_call_limit` 原样传给 Runner，不需要修改；生产默认值
必须同时在 `WebRuntimeConfig`、`DeepReadingRunner` 和 `DeepReadingContext` 三处从
8 调整为 12。

不修改：

```text
DeepReadingState
LangGraph topology
数据库模型和迁移
Web API Schema
Summary / Write Answer 业务逻辑
MCP Server
```

## 26. 测试设计

### 26.1 TODO 状态机

- 合法初始计划。
- 超过 6 项。
- 空白、重复或超长内容。
- 多个 `in_progress`。
- 删除、重命名或回退已完成项。
- 删除当前执行项。
- 删除 pending 项但复用旧 ID。
- 非法更新不修改旧状态。

### 26.2 XML Renderer

- 根标签、子标签和属性顺序固定。
- TODO 和特殊字符正确转义。
- 相同快照产生逐字节相同输出。
- 第一轮缺失时间间隔和事件时使用 `available="false"`。
- 没有告警时输出空 `<alerts/>`。
- 输出不包含禁止字段。
- 渲染异常产生固定降级消息。

### 26.3 Tracker 与告警

- 参数键顺序不影响调用指纹。
- 参数值变化产生不同指纹。
- 第二次相同调用触发重复告警。
- 连续两次无进展触发告警。
- 进展出现后重置无进展计数。
- 剩余调用数小于等于 1 时触发低预算告警。
- 新 attempt 重置临时计数但保留业务 ledger。

### 26.4 Agent 集成

- 每次逻辑模型调用前追加一条新状态栏。
- Status Bar 是位于输入末尾的 `HumanMessage`。
- 旧状态栏保留在当前 attempt 轨迹。
- 第二次模型输入以前一次完整输入、响应和工具结果为前缀。
- `write_todos` 不计入研究工具预算。
- TODO 与业务工具并行调用被拒绝。
- Provider retry 不重复状态栏。
- Status Bar 故障不改变 Research 结果。
- 两个结构化 attempt 的内部状态相互隔离。
- 内部消息不写入 `DeepReadingState`。
- Research metadata 使用 `research-v6`。

### 26.5 配置与 Runner

- 默认模型调用总上限为 12。
- 每个结构化 attempt 上限为 6。
- 显式配置继续覆盖默认值。
- 业务工具总上限仍为 12。
- 现有模型 usage 聚合继续工作。

## 27. 验证命令

实施后至少运行：

```bash
.venv/bin/python -m pytest \
  tests/deep_reading/test_research_status.py \
  tests/deep_reading/test_research_agent.py \
  tests/deep_reading/test_nodes.py \
  tests/deep_reading/test_runner.py \
  tests/deep_reading/test_model_usage.py -q

.venv/bin/python -m pytest tests/deep_reading -q
.venv/bin/python -m pytest tests/architecture/test_repository_allowlist.py -q
git diff --check
```

自动化测试不访问真实 DeepSeek。

它们可以证明：

- 消息顺序和前缀满足缓存复用条件。
- TODO、预算、告警和降级行为符合本地契约。
- 外层状态和业务校验未被绕过。

它们不能证明：

- DeepSeek 服务端一定命中 KV Cache。
- 模型一定按告警改变行为。
- 新增模型回合在真实流量中的成本收益为正。

这些结论必须在部署后通过真实 usage、日志和任务质量样本评估。

## 28. 风险与控制

### 28.1 TODO 增加模型调用

风险：创建和完成 TODO 会占用逻辑模型回合。

控制：默认模型调用总上限从 8 提高到 12；业务工具预算不增加；简单任务允许 direct 模式。

### 28.2 Status Bar 增加 token

风险：每轮状态栏都会增加输入 token。

控制：最多 6 个 TODO、单项 160 字符、固定 Alert 文本、不包含原始工具数据；每个 attempt 最多 6 条状态栏。

### 28.3 user 角色混淆

风险：模型把 Harness 状态当成终端用户新请求。

控制：固定 XML 根标签、固定 System 信任规则、始终尾部单独注入、外层消息不持久化。

### 28.4 用户伪造状态标签

风险：用户在问题正文中写入同名 XML。

控制：当前状态由尾部独立消息位置确定，不使用最大 sequence 作为信任规则。

### 28.5 Alert 误报

风险：合法重复操作被判定为异常。

控制：第一版 Alert 只引导模型，不新增硬阻断；跨 attempt 重置重复和无进展计数。

### 28.6 状态栏故障干扰业务

风险：时钟、快照或 XML 渲染错误导致 Research 失败。

控制：完整状态栏、固定降级状态栏、最终跳过的三级 best-effort；原业务异常始终优先。

### 28.7 假恢复

风险：只恢复 TODO 而丢失证据和工具结果。

控制：第一版不支持内部暂停恢复；中断时从外层可信 checkpoint 重跑完整 Research 节点。

## 29. 实施前审批边界

本文确认的是架构设计。下一阶段应单独形成可执行实施计划，并明确：

- TDD 顺序和每个 RED/GREEN 检查点。
- 现有脏工作区的文件归属和保护方式。
- 具体类、TypedDict、Tool Schema 和 middleware hook 签名。
- Prompt `research-v6` 的精确固定文本修改。
- 分批验证与回滚点。

在实施计划获得确认前，不修改生产代码。
