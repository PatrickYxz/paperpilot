# PaperPilot 五层上下文压缩架构设计

> Codex only：本文记录 2026-09-01 已逐层讨论并确认的上下文压缩方案。
> 本文是设计文档，不代表生产实现已经完成。涉及内部持久化结构、配置和运行行为的修改，必须在本文审阅通过并形成实施计划后才能执行。

## 1. 背景

PaperPilot 当前 Deep Reading 主链路为：

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

当前上下文压缩只覆盖外层会话历史：

- 当估算 Token 严格大于 `summary_token_threshold=32_000` 时触发；
- Token 使用 `(消息字符数 + 旧摘要序列化字符数) // 4` 粗略估算；
- 将旧 `ConversationSummary` 和当前消息重新生成一个四字段摘要；
- 用新摘要覆盖旧摘要；
- 从 LangGraph 活动消息状态中移除旧消息，只保留最近六个 HumanMessage 锚定的轮次；
- TaskStore 中的完整业务消息历史不受 `RemoveMessage` 影响。

该实现属于“滚动合并摘要”，相当于不断执行 `git squash`，尚未覆盖以下能力：

1. 大型工具结果的统一外置、预览和按需读取；
2. 确定性噪声的立即删除；
3. 对已失去即时价值的历史工具结果进行批量微压缩；
4. 像 `git log` 一样保留每个业务轮次的独立归档；
5. 前四层仍无法满足预算时的全量压缩、结果验证和连续失败熔断。

PaperPilot 当前生产模型由 `ChatDeepSeek(model="deepseek-chat")` 构造。DeepSeek 当前 Responses API 不支持 `context_management`，因此本设计不能依赖服务端原生上下文编辑或压缩，必须提供本地、Provider-neutral 的实现边界。

## 2. 目标与成功标准

### 2.1 目标

- 根据不同信息的生命周期采用不同压缩策略。
- 在内容第一次进入模型之前优先完成预算控制，减少不必要的 KV Cache 失效。
- 让确定性噪声立即退出模型工作集，不对噪声生成摘要。
- 将大型或语义价值不确定的内容完整外置，并保留可恢复引用。
- 保留每个外层业务轮次的决策、约束、验证、失败路径和未完成事项。
- 将归档存储与模型本次可见上下文分离，只按需注入少量相关记录。
- 把全量压缩限制为最后手段，并禁止未经验证的压缩结果覆盖有效状态。
- 防止会话陷入反复压缩失败和重复付费的循环。
- 保持固定 System Prompt 不包含动态压缩状态，继续满足 KV Cache 友好原则。

### 2.2 成功标准

```text
新工具结果：
  在第一次模型可见前完成 DROP_NOW / EXTERNALIZE_NOW / KEEP_INLINE 决策

确定性噪声：
  立即删除，不等待累计，不生成摘要

历史工具结果：
  只批量回收 CLEARABLE_AFTER_USE 且非 PROTECTED 的内容

业务轮次结束：
  追加一条独立 TurnArchive，不覆盖既有归档

模型调用：
  只注入当前投影、最多五条相关归档和最近消息

全量压缩：
  候选结果通过结构、受保护信息、引用和 Token 目标验证后才原子采用

连续压缩失败：
  三次后打开熔断器，不再重复调用压缩模型
```

## 3. 范围与非目标

### 3.1 设计范围

- Deep Reading 外层会话和 Research Agent 工具结果的上下文预算管理。
- 统一内部 ArtifactStore 抽象和本地文件后端。
- 工具结果的初始动作和未来保留策略。
- 本地 Context Editing 视图构造。
- 逐轮 TurnArchive、Active Projection 和相关归档检索。
- 两阶段全量压缩、Continuation Capsule 验证和熔断器。
- 必要的内部持久化、配置、事件和指标边界。

### 3.2 第一版非目标

- 不删除或重写 TaskStore 中的完整业务消息树。
- 不把所有归档放回每次模型调用。
- 不做跨用户或跨 Conversation 的长期记忆共享。
- 不在第一版引入向量数据库或新 Embedding 依赖。
- 不将 TurnArchive 直接暴露为新的公开 Web API 或前端页面。
- 不保存隐藏推理、供应商私有 Chain-of-Thought 或 API 凭证。
- 不依赖 DeepSeek 当前不支持的 `context_management`、`conversation` 或服务端 compaction。
- 不改变论文证据校验、ResearchResult 契约和最终发布事务语义。

## 4. 核心边界

### 4.1 权威数据与模型视图分离

```text
权威层
  TaskStore              完整业务消息、Task、Event、发布结果
  ArtifactStore          大型工具原文和可按需读取内容
  TurnArchiveStore       逐轮追加的结构化业务归档
  Evidence Ledger        经过业务校验的论文证据

派生层
  ToolResultDisposition  工具结果生命周期决策
  ActiveProjection       当前仍有效的目标、约束、TODO 和验证状态
  CompressionState       压缩失败计数和熔断状态

模型视图
  固定 System Prompt
  Runtime Context
  ActiveProjection 或 ContinuationCapsule
  Retrieved Archives
  Recent Messages
  Research Status Bar（仅 Research Agent 尾部）
```

模型视图可以被裁剪或重建，但不能成为业务事实来源。发生冲突时，以 TaskStore、ArtifactStore、TurnArchiveStore 和 Evidence Ledger 为准。

### 4.2 “语义处置”与“未来生命周期”是两个维度

单个 `KEEP / DELETE / EXTERNALIZE` 标签无法同时表达“现在怎么处理”和“以后能不能回收”。本设计使用两个字段：

```text
initial_action:
  DROP_NOW
  EXTERNALIZE_NOW
  KEEP_INLINE

future_retention:
  PROTECTED
  CLEARABLE_AFTER_USE
```

规则如下：

| `initial_action` | 当前动作 |
|---|---|
| `DROP_NOW` | 立即从模型可见结果中删除，不生成摘要 |
| `EXTERNALIZE_NOW` | 完整内容写入 ArtifactStore，模型只接收预览和引用 |
| `KEEP_INLINE` | 当前模型调用保留完整内容 |

| `future_retention` | 后续动作 |
|---|---|
| `PROTECTED` | 微压缩不得清除 |
| `CLEARABLE_AFTER_USE` | 当前使用完成后，可由第三层批量回收 |

`DROP_NOW` 是立即执行的动作，不是等待第三层处理的标签。第三层只处理已经进入过模型、当时有用、现在已经 `CLEARABLE_AFTER_USE` 的历史结果。

## 5. 总体数据流

```text
工具返回原始结果
    |
    v
统一入口先执行第二层的确定性噪声规则
    |- DROP_NOW -> 立即移除，不做无意义 Artifact 写入
    `- 不能证明是噪声 -> 进入第一层预算控制
    |
    v
第一层：工具结果预算控制
    |- 小且当前需要 -> KEEP_INLINE
    `- 大或价值不确定 -> ArtifactStore + Preview + ArtifactRef
    |
    v
第三层：历史工具结果微压缩
    |- 上下文未达阈值 -> 不改历史
    `- 达阈值且可回收量足够 -> 一次批量替换 CLEARABLE_AFTER_USE
    |
    v
第四层：业务轮次归档
    |- 同步写 Archive Seed
    `- 后台补充 Narrative Summary
    |
    v
模型下一次调用
    System + Runtime + ActiveProjection + RetrievedArchives + RecentMessages
    |
    v
第五层：最后兜底
    |- 先压缩会话记忆
    `- 仍超预算 -> 全量压缩 + 验证 + 熔断
```

五层表示信息生命周期和责任边界，不要求运行函数严格按编号串行。工具结果第一次进入模型前，第一层预算控制和第二层噪声规则由同一个 ingestion pipeline 完成；为避免给确定性噪声做无意义的 Artifact 写入，运行时先执行廉价的 `DROP_NOW` 规则，再对其余结果执行内联或外置判断。

## 6. 上下文预算模型

### 6.1 可用输入预算

```text
usable_input_budget
  = model_context_window
  - configured_max_output_tokens
  - safety_margin
```

- `configured_max_output_tokens` 使用当前 Runner 配置，默认 4,096。
- `safety_margin` 默认取模型上下文窗口的 5%。
- 所有比例阈值都基于 `usable_input_budget`，而不是供应商标称总窗口。

### 6.2 Token 计数

定义统一 `TokenCounter` 端口：

1. 优先使用模型或 Provider 对应 tokenizer 的精确消息计数；
2. 无精确 tokenizer 时使用经过线上 `input_tokens` 校准的保守估算器；
3. 无校准数据时使用 `max(ceil(UTF-8 bytes / 3), ceil(characters / 2))`；
4. 估算器必须把消息角色、工具 schema、工具参数和结构化包装计入预算；
5. 触发压缩后必须对候选模型输入重新计数，不能只相信压缩模型声称的长度。

当前 `characters // 4` 只保留为迁移前兼容行为，不能作为新压缩协调器的最终计数方式。

## 7. 第一层：工具结果预算控制

### 7.1 决策时机

工具返回后、结果第一次进入模型之前执行。运行时先应用第八节的确定性噪声规则；只有未被 `DROP_NOW` 的结果才进入本节预算判断。最终动作写入 `ToolResultDisposition` 后即冻结，后续调用不得在完整正文与预览之间反复切换。

默认规则：

```text
result_tokens <= 2,000
  且当前步骤需要原文
    -> KEEP_INLINE

result_tokens > 2,000
  或语义价值不确定
    -> EXTERNALIZE_NOW
```

`2,000` 是第一版可配置默认值，不替代工具专属 renderer。工具 renderer 必须优先保留业务标识和恢复入口，而不是机械截取前 2,000 Token。

### 7.2 PaperPilot 三个业务工具的预览规则

#### `search_related_papers`

- ArtifactStore 保存完整结构化候选列表。
- 预览保留全部候选的 `external_id`、`title` 和原始顺序。
- 仅前五个候选保留截断后的 abstract，单条最多 300 字符。
- 预览附带 `artifact_ref`，可读取完整作者和 abstract。

#### `prepare_paper`

- 当前实现已将论文正文留在下载/索引链路中，只向 Research Agent 返回 `external_id + status`。
- 保持该行为；不得把 PDF 原文或全文重新放入 ToolMessage。

#### `retrieve_paper_evidence`

- ArtifactStore 保存完整 `evidence_items` 和 `summary_item_ids`。
- 预览保留所有 Evidence ID、paper ID、来源位置和分数等选择所需字段。
- 完整正文优先保留 `summary_item_ids` 对应项，再按原始排序保留直到达到 2,000 Token。
- 未内联的 Evidence 只能通过 `artifact_ref` 有界读取，不能静默丢失。

### 7.3 有界按需读取

内部提供稳定工具契约：

```text
read_artifact_slice(artifact_id, cursor, max_tokens)
search_artifact(artifact_id, query, max_matches)
```

规则：

- 单次读取 `max_tokens <= 2,000`；
- 返回下一游标、实际 Token 数和内容哈希；
- 读取次数计入 Research 工具预算和 Status Bar；
- 路径由 ArtifactStore 解析，模型不能提交文件系统路径；
- 找不到、哈希不一致或越界时返回确定性错误。

### 7.4 写入原子性

只有完整 Artifact 成功持久化并通过哈希校验后，才能向模型返回 `artifact_ref`。写入失败时不得返回悬空引用；若正文又超过硬内联预算，则工具调用明确失败，不进行无提示截断。

## 8. 第二层：确定性噪声直接删除

### 8.1 可直接证明的噪声

以下规则不调用 LLM：

- 空结果、空白字段和固定协议包装；
- 完全重复且已有相同内容哈希的结果；
- 同一工具指纹的重复结果，且权威 Artifact 已存在；
- 已被业务校验明确拒绝、后续节点绝不会消费的无效记录；
- 心跳、进度回显、调试日志和无业务语义的传输元数据；
- 已由结构化 State 单独保存的重复状态文本。

### 8.2 语义不确定时的规则

系统只能证明 `DROP_NOW` 或明确的 `KEEP_INLINE`。无法证明是噪声、但又不应长期内联的内容必须选择 `EXTERNALIZE_NOW`，不得猜测性删除。

### 8.3 执行动作

```xml
<tool_result status="dropped" reason="duplicate_result" />
```

- 新结果在第一次模型调用前立即替换；
- 不生成 LLM 摘要；
- 只保留最小审计字段：result ID、tool call ID、reason、content hash；
- 不形成等待第三层处理的 DELETE backlog。

## 9. 第三层：历史工具结果微压缩

### 9.1 处理对象

第三层只处理同时满足以下条件的历史 ToolResult：

1. 已经进入过至少一次模型调用；
2. `future_retention == CLEARABLE_AFTER_USE`；
3. 当前下游节点已消费完成；
4. 完整内容已存在于 ArtifactStore，或有其他权威来源可恢复；
5. 不包含受保护约束、未完成 TODO 或唯一验证证据。

`DROP_NOW` 已在第二层执行，不属于第三层目标。

### 9.2 触发规则

同时满足以下条件才执行一次批量微压缩：

```text
current_input_tokens >= 70% * usable_input_budget

reclaimable_tokens >= max(8,000, 10% * usable_input_budget)
```

微压缩后保留最近三个 ToolResult。一次操作中处理所有已选目标，禁止每发现一条就重写一次历史。

### 9.3 本地等价实现

PaperPilot 使用本地 Context Editing middleware：

```text
LangGraph State 完整消息
    -> 复制本次 request.messages
    -> 批量替换目标 ToolMessage
    -> 调用 DeepSeek
```

替换规则：

```xml
<artifact_ref
  id="artifact_123"
  summary="本结果已完成消费，原文可按需读取"
  sha256="..."
/>
```

本地 LangGraph State、TaskStore 历史和 ArtifactStore 原文保持不变。未来如 Provider 支持原生 context editing，可在同一策略接口下增加 Provider adapter，但不得改变业务分类语义。

### 9.4 KV Cache 语义

- 新工具结果在第一次发送前被外置或删除，不存在对该结果的历史缓存重建。
- 已被模型看过的旧结果被替换时，从最早修改点开始的后续 KV Cache 失效；修改点之前的相同前缀仍可复用。
- 因此第三层只在可一次回收足够 Token 时批量触发。
- 编辑后的表示和 disposition 必须冻结，后续调用复用同一序列化结果。

## 10. 第四层：逐轮归档式摘要

### 10.1 归档单位

一条归档对应一个外层用户业务轮次到达终态：

```text
success / failed / cancelled
```

Research Agent 内部模型调用、工具调用和 TODO 更新不是独立 TurnArchive；它们由 Event、Status Tracker、Artifact 和 Evidence Ledger 记录，最终汇总进该业务轮次。

### 10.2 两阶段形成

```text
业务轮次计算出终态结果
    -> 原有发布/失败终结事务先按既有语义完成
    -> 立即在独立、幂等、best-effort 写入中保存确定性 Archive Seed
    -> 后台 LLM 补充 Narrative Summary
```

Archive Seed 直接从权威状态复制：

- 用户目标和明确约束；
- 采用的 Paper ID、Evidence ID 和 Artifact ID；
- TODO 完成状态；
- 验证结果；
- 工具失败和未解决问题；
- 最终任务状态。

后台 LLM 只压缩自然语言叙事：本轮发生了什么、为什么采取该方案、获得什么结论、后续仍缺什么。Archive Seed 或 Narrative 补充失败都不回滚已经完成的原有发布/失败事务，但必须记录失败事件并支持幂等重试。后续上下文只能使用已经成功持久化的 Archive，不得使用仅存在于进程内存中的 Seed。

### 10.3 `TurnArchive` 契约

```json
{
  "archive_id": "archive_003",
  "conversation_id": "conversation_001",
  "task_id": "task_003",
  "user_message_id": "message_003",
  "terminal_status": "success",
  "user_goal": "...",
  "decisions": [],
  "constraints": [],
  "paper_findings": [],
  "evidence_refs": [],
  "artifact_refs": [],
  "failed_paths": [],
  "verification": [],
  "unresolved_todos": [],
  "rollback_notes": [],
  "supersedes": [],
  "narrative_summary": null,
  "archive_version": "turn-archive-v1",
  "created_at": "..."
}
```

幂等键固定为：

```text
(conversation_id, user_message_id, archive_version)
```

归档只允许追加。用户修正旧决策时创建新 Archive，并通过 `supersedes` 或 `invalidates` 指向旧记录，不直接改写旧记录。

### 10.4 明确保留优先级

以下字段不得由 LLM 自由改写：

- 架构决策、关键约束及其理由：保留精确结构化值、来源消息 ID 和必要原文；
- 已修改文件列表和关键变更记录：完整保留；PaperPilot 当前不修改文件时字段为空；
- 验证状态：明确保存 `pass / fail / not_run`；
- 未解决 TODO 和回滚笔记：完整保留；
- Evidence、Artifact、Paper：保留稳定 ID；
- 工具输出正文：归档只保存 pass/fail 结论和 Artifact 引用。

问候、确认语、重复内容、工具 chatter 和隐藏推理不进入 TurnArchive。

### 10.5 Active Projection

Active Projection 是从权威状态和 TurnArchive 派生的当前有效视图，不是归档本身：

```json
{
  "current_goal": "...",
  "active_constraints": [],
  "active_decisions": [],
  "active_paper_ids": [],
  "open_todos": [],
  "failed_verifications": [],
  "unresolved_questions": []
}
```

旧投影不得作为下一次投影的唯一事实来源；每次从当前业务状态、Archive Seed 和 supersession 关系重新计算。

### 10.6 归档检索与注入

默认规则：

```text
archive_context_budget = 4,000 tokens
max_archive_records = 5
recent_archive_records = 2
```

检索优先级：

1. 当前有效约束、未完成 TODO 和失败验证始终进入 Active Projection；
2. 用户明确提到的 Archive ID、Paper ID、Evidence ID 和 Artifact ID 精确命中；
3. 与当前目标和论文范围相同的归档；
4. 最近两条业务归档；
5. 同优先级按更新时间和直接相关性排序。

被 supersede 的旧记录默认不注入；只有用户询问历史、原因或变化过程时才同时加载前后记录。第一版使用结构化过滤、ID 匹配和轻量文本匹配，不引入向量检索。

## 11. 第五层：两阶段全量压缩

### 11.1 触发条件

前四层全部执行后：

```text
current_input_tokens >= 80% * usable_input_budget
```

才允许进入第五层。全量压缩是最后兜底，不是每轮固定步骤。

### 11.2 阶段 A：压缩会话记忆

只压缩派生状态：

- Active Projection 中已完成或已 supersede 的普通叙事；
- 当前检索出来的 Archive 解释文本；
- 已完成 TODO 的展开描述；
- 已归档的旧研究结论副本。

不处理 System Prompt、当前用户请求、关键约束原文、未完成 TODO、失败验证、最近消息和不可变 Archive。

成功目标：

```text
post_compaction_tokens <= 65% * usable_input_budget
```

达到目标后立即停止，不进入阶段 B。

### 11.3 阶段 B：全量工作集压缩

阶段 A 后仍满足：

```text
post_stage_a_tokens >= 80% * usable_input_budget
```

才生成 `ContinuationCapsule`：

```json
{
  "current_goal": "...",
  "exact_constraints": [],
  "decisions_and_rationales": [],
  "active_papers": [],
  "evidence_refs": [],
  "artifact_refs": [],
  "completed_work": [],
  "verification": [],
  "failed_paths": [],
  "unresolved_todos": [],
  "rollback_notes": [],
  "source_archive_ids": [],
  "capsule_version": "continuation-v1"
}
```

采用后模型工作集为：

```text
System Prompt
Runtime Context
Continuation Capsule
最近两轮完整消息
当前用户请求
```

成功目标：

```text
post_full_compaction_tokens <= 50% * usable_input_budget
```

原始消息、Artifact、TurnArchive 和业务 Event 不被删除。

## 12. 压缩候选验证

压缩模型输出只能形成 Candidate。验证全部通过后才在一个状态更新中原子采用：

```text
original working set
    -> generate candidate
    -> validate candidate
        |- pass -> atomic replace
        `- fail -> discard candidate, keep original
```

### 12.1 结构验证

- 严格符合 `ContinuationCapsule` Schema；
- 必填字段齐全且禁止额外字段；
- Archive、Artifact、Evidence 和 Paper ID 必须存在；
- 不允许生成输入中不存在的新 ID。

### 12.2 受保护信息验证

每个受保护项具有稳定 ID 和必要原文哈希。压缩前后的 protected ID 集合必须完全相等：

```text
protected_ids_before == protected_ids_after
```

以下信息覆盖率必须为 100%：

- 当前用户目标；
- 用户明确约束；
- 架构决策及理由；
- 未完成 TODO；
- `fail` 验证状态；
- 未解决问题和回滚信息；
- 当前有效 Paper、Evidence 和 Artifact 引用。

### 12.3 压缩效果验证

- 阶段 A 未达到 65% 目标时，不把 Candidate 作为成功结果；
- 阶段 B 未达到 50% 目标时，判定 `insufficient_reduction`；
- 受保护字段本身已超过目标时，判定 `protected_context_oversized`，禁止继续要求 LLM 删除关键内容。

## 13. 连续失败熔断器

### 13.1 计数语义

一次完整压缩流程最多执行：

- 阶段 A 一次；
- 必要时阶段 B 一次；
- 只有 timeout、429、5xx 和连接中断允许每阶段额外重试一次；
- Schema、保护信息丢失、引用无效和压缩不足不在同一业务轮次语义重试。

只有整条压缩流程没有产生可采用 Candidate，才把 `consecutive_failures` 增加一。任一合法压缩成功后计数归零。

### 13.2 状态机

```text
CLOSED
  -- consecutive_failures == 3 --> OPEN

OPEN
  -- recovery condition --> HALF_OPEN

HALF_OPEN
  -- one probe succeeds --> CLOSED
  -- one probe fails ----> OPEN
```

熔断器按 Conversation 和 compressor version 持久化。Worker 重启、Task retry 和 broker redelivery 不得清零。

### 13.3 OPEN 降级路径

OPEN 状态下禁止调用压缩模型，确定性构造最小安全工作集：

```text
System Prompt
+ 当前用户目标
+ 全部受保护约束
+ 未完成 TODO
+ 失败验证
+ Evidence / Artifact / Archive 引用
+ 最近一轮消息
```

- 最小安全工作集能够容纳：记录 `compression_degraded`，任务以降级模式继续；
- 最小安全工作集仍无法容纳：终止当前执行，返回 `context_capacity_exhausted`，并生成可用于新 Conversation 的续接包；
- 任何情况下都不得静默删除受保护字段。

### 13.4 恢复条件

- timeout、429、5xx 等临时失败：冷却 300 秒后允许一次 HALF_OPEN 探测；
- Schema、保护字段丢失等确定性失败：只有 compressor model、Prompt version 或输入结构变化后允许探测；
- `protected_context_oversized`：必须先外部减少受保护内容或切换更大上下文模型，单纯等待不构成恢复。

## 14. 消息装配顺序

未启用 Continuation Capsule 时：

```text
System Prompt
PaperPilot Runtime Context
Active Projection
Retrieved Turn Archives
Recent Conversation Messages
Research Status Bar（仅 Research Agent，每次调用尾部追加）
```

启用 Continuation Capsule 时：

```text
System Prompt
PaperPilot Runtime Context
Continuation Capsule
Recent Two Turns
Current User Message
Research Status Bar（仅 Research Agent，每次调用尾部追加）
```

同一次模型调用不得同时注入旧 `ConversationSummary`、Active Projection 和 Continuation Capsule 的重复内容。迁移完成后，由 Context View Builder 根据状态选择唯一主记忆表示。

所有动态压缩内容位于固定 System 和规范化 Runtime Context 之后，不能修改固定前缀。

## 15. 组件职责

| 组件 | 职责 |
|---|---|
| `TokenCounter` | 统一计算模型可见输入 Token |
| `ArtifactStore` | 原子保存大结果、校验哈希、提供预览和有界读取 |
| `ToolResultPolicy` | 产生并冻结 initial action / future retention |
| `ContextViewBuilder` | 从完整 State 构造本次模型可见消息 |
| `ContextEditingMiddleware` | 批量替换历史 CLEARABLE ToolResult |
| `TurnArchiveBuilder` | 从权威状态生成 Archive Seed 和后台叙事摘要 |
| `TurnArchiveStore` | 追加、幂等保存和查询独立归档 |
| `ActiveProjectionBuilder` | 计算当前有效目标、约束、TODO 和验证状态 |
| `ArchiveRetriever` | 在 4,000 Token / 5 条预算内选择相关归档 |
| `CompressionCoordinator` | 执行 80% -> 65% -> 50% 两阶段策略 |
| `ContinuationValidator` | 校验 Schema、protected IDs、引用和 Token 目标 |
| `CompressionCircuitBreaker` | 持久化连续失败和 CLOSED/OPEN/HALF_OPEN 状态 |

组件不得反向依赖 Web Router 或前端。模型视图构造失败不能修改权威存储；权威持久化失败不能返回悬空引用。

## 16. 内部持久化设计

本设计不复用现有面向任务结果展示的 `TaskArtifact.content` 存储大型原始工具输出，避免把内部上下文 Artifact 混入用户可见 Artifact 流。第一版新增三个内部持久化结构；这部分涉及数据库迁移，必须在实施计划开始前得到明确确认。

### 16.1 `context_artifacts`

保存 Artifact 元数据，正文存入受控本地文件后端：

```text
artifact_id
conversation_id
task_id
tool_call_id
kind
storage_key
sha256
byte_size
token_estimate
preview
initial_action
future_retention
created_at
```

文件后端使用应用数据目录下的受控根路径，以 UUID storage key 定位；采用临时文件写入、fsync、原子 rename。模型和 API 不接触真实路径。

### 16.2 `turn_archives`

```text
archive_id
conversation_id
task_id
user_message_id
terminal_status
archive_version
seed_json
narrative_summary
narrative_status
supersedes_json
created_at
updated_at
```

唯一约束：

```text
(conversation_id, user_message_id, archive_version)
```

除后台补充 `narrative_summary / narrative_status` 外，不允许原地改写 Archive Seed。

### 16.3 `compression_states`

```text
conversation_id
compressor_version
state
consecutive_failures
last_failure_type
last_input_digest
opened_at
updated_at
```

唯一约束：

```text
(conversation_id, compressor_version)
```

## 17. 配置默认值

| 配置 | 默认值 |
|---|---:|
| `tool_inline_max_tokens` | 2,000 |
| `artifact_read_max_tokens` | 2,000 |
| `micro_compaction_trigger_ratio` | 0.70 |
| `micro_compaction_min_reclaim_tokens` | `max(8,000, usable * 0.10)` |
| `micro_compaction_keep_recent_tool_results` | 3 |
| `archive_context_budget_tokens` | 4,000 |
| `archive_context_max_records` | 5 |
| `archive_context_recent_records` | 2 |
| `full_compaction_trigger_ratio` | 0.80 |
| `session_memory_target_ratio` | 0.65 |
| `full_compaction_target_ratio` | 0.50 |
| `full_compaction_recent_turns` | 2 |
| `compression_failure_threshold` | 3 |
| `compression_transient_retry_count` | 1 |
| `compression_breaker_cooldown_seconds` | 300 |
| `context_safety_margin_ratio` | 0.05 |

所有配置启动时验证范围；无效值应阻止应用启动，不得在运行中静默修正。阈值变更属于行为和成本变更，需要版本化配置及回归测试。

## 18. 失败处理与一致性

### 18.1 Artifact

- 原文写入和哈希校验成功后才发布引用；
- 元数据落库失败时清理尚未发布的临时文件；
- 引用已发布后不得原地覆盖正文；新内容创建新 Artifact ID。

### 18.2 TurnArchive

- Archive Seed 使用幂等键追加；
- Narrative Summary 为 best-effort，可独立重试；
- 归档失败不把已成功发布的用户答案改为失败，但必须写可观测错误；
- 后续模型不能把失败的 Narrative 当成 Archive Seed 缺失的替代品。

### 18.3 压缩

- Candidate 验证前不得更新活动消息；
- 验证失败保留最后一份有效工作集；
- 熔断状态和失败事件在同一持久化边界内更新；
- 压缩失败不得覆盖原始 provider 或业务异常。

## 19. 可观测性

新增结构化事件：

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

每次压缩记录：

- stage、compressor version 和原因；
- before/after tokens；
- reclaimed tokens；
- protected item count；
- Archive/Artifact 引用数量；
- cache hit/miss（Provider 返回时）；
- validation failure type；
- breaker state。

日志和事件不得包含 Prompt、论文正文、完整工具输出、用户原文或凭证。

## 20. 迁移策略

### 20.1 兼容现有 ConversationSummary

首次在旧 Conversation 上启用新机制时：

1. 保留原 `conversation_summary` 原始 JSON；
2. 以 `legacy-summary-v1` 类型写入一个只读迁移 Archive；
3. 将其内容视为未经确认的 carry-forward candidate；
4. 从 TaskStore 当前稳定消息、业务状态和后续 TurnArchive 构建 Active Projection；
5. 新机制稳定后，Context View Builder 不再同时注入旧 Summary。

不批量回放全部历史生成 Archive，避免一次性成本和错误放大。旧历史仍可从 TaskStore 恢复。

### 20.2 Provider Adapter

第一版只实现本地 adapter。未来 Provider 原生支持时：

- `ToolResultPolicy`、Artifact、Archive 和保护规则保持不变；
- 只替换 Context Editing / Full Compaction 的执行 adapter；
- 原生返回的 opaque compaction item 只能作为模型续接表示，不能替代可审计 TurnArchive。

## 21. 验证策略

### 21.1 单元测试

- 工具专属 preview 的确定性、Token 上限和完整 ID 保留；
- DROP_NOW 规则不调用 LLM；
- 语义不确定内容落入 EXTERNALIZE_NOW；
- disposition 冻结后不可回退；
- micro compaction eligibility 和批量阈值；
- Archive 幂等键、supersedes 和检索排序；
- protected ID/hash 全覆盖校验；
- breaker 三态和恢复条件。

### 21.2 集成测试

- 大结果原文成功存盘后，模型只看到 preview/ref；
- Artifact 写失败时不产生悬空引用；
- 新结果预处理不破坏已有消息前缀；
- 历史批量编辑只改变本次 request copy；
- TaskStore 完整业务历史保持不变；
- 每个终态业务轮次只产生一个 Archive Seed；
- 全量压缩 Candidate 缺少约束时被拒绝且原状态不变；
- 连续三次失败后不再发起模型压缩调用；
- 最小安全上下文超限时返回 `context_capacity_exhausted`。

### 21.3 场景回归

- 用户纠正旧约束：新 Archive supersedes 旧 Archive，当前投影只采用新约束；
- 用户追问历史原因：同时检索 superseding 和 superseded Archive；
- Research 搜索返回大量候选：保留全部 ID 和标题，完整 metadata 可按需读取；
- Evidence 结果过大：保留 Evidence ID 和 summary items，引用可以恢复其余正文；
- Provider 429：只重试一次，最终失败才增加 breaker 计数；
- 压缩模型连续遗漏否定词：哈希校验拒绝 Candidate。

### 21.4 生产验证

实现后的本地测试只能证明逻辑和接口，不能证明生产收益。上线后至少观察：

- 平均/分位输入 Token；
- Artifact 外置率和恢复读取率；
- micro/full compaction 触发率；
- before/after Token 降幅；
- DeepSeek cache hit ratio；
- Candidate 拒绝率和原因；
- breaker 打开次数；
- 任务成功率、延迟和总成本。

## 22. 风险与控制

| 风险 | 控制 |
|---|---|
| Preview 丢失模型当前需要的细节 | 保留稳定 ID、工具专属 renderer、Artifact 有界读取 |
| 噪声规则误删有价值内容 | 只允许确定性 DROP；不确定时 EXTERNALIZE |
| 微压缩频繁破坏缓存 | 70% + 最低可回收量双阈值，一次批量编辑 |
| 归档不断增长 | Archive 不全部进 Prompt；按 4,000 Token/5 条检索 |
| LLM 摘要改写约束 | protected ID + exact text hash 验证 |
| 全量压缩反复失败 | 每轮语义不重试，连续三次熔断 |
| Artifact 引用失效 | 原子写入、哈希、不可变 ID、引用前校验 |
| 新持久化结构扩大范围 | 实施前单独审批迁移和回滚方案 |
| 估算 Token 偏差 | Provider tokenizer 优先，保守 fallback，候选后重新计数 |
| Provider 能力变化 | adapter 隔离，业务策略不依赖原生 API |

## 23. 已确认决策

1. 采用五层策略，不把全部问题交给单个滚动摘要器。
2. ArtifactStore 使用统一抽象、工具专属预览和有界读取。
3. 确定性噪声立即删除，不等待积累。
4. `DROP_NOW` 与第三层历史微压缩严格分离。
5. 第三层只批量回收 `CLEARABLE_AFTER_USE` 的历史 ToolResult。
6. 归档按外层业务轮次追加，不覆盖旧归档。
7. Archive Seed 同步形成，LLM Narrative 后台补充。
8. 归档注入采用精确关联优先，最多五条、4,000 Token。
9. 达到 80% 后先压缩会话记忆，目标 65%；仍超限才全量压缩，目标 50%。
10. 压缩结果必须先验证，连续失败三次后熔断。

## 24. 实施前审批边界

本文审阅通过后，下一步只编写实施计划，不直接修改生产代码。实施计划必须单独列出并请求确认：

- 三个内部持久化结构及数据库迁移；
- Artifact 文件根目录、生命周期和清理策略；
- 新增内部读取工具对 Research Agent 工具 schema 和缓存的影响；
- DeepReadingState / Graph version 的迁移方式；
- 新配置项及 `.env.example` 变化；
- 失败回滚、旧 Conversation 兼容和灰度开关；
- 分阶段测试与生产指标验收。

## 25. 参考资料

- [LangChain Short-term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [LangChain Context Editing Middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [LangChain Long-term Memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [LangMem Core Concepts](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [OpenAI Responses Compaction](https://developers.openai.com/api/reference/java/resources/responses/methods/compact)
- [DeepSeek Responses API Compatibility](https://api-docs.deepseek.com/guides/responses_api/)
- [Microsoft Circuit Breaker Pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)
