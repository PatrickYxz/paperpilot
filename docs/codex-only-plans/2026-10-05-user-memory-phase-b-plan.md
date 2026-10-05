# B 阶段细化计划：跨会话用户记忆（rag 分支）

> Codex only。承接《AI Agents in Depth》第 3 章 3.1 / 3.3.4，是
> 2026-10-05-user-memory-and-rag-plan.md 三阶段路线的 B 阶段落地。
> 用户已确认：读取形态为 Agent 工具（智能体化记忆）。

## 本阶段增量决策（相对总计划的细化）

1. **不做新 MCP server**。记忆是 per-user 业务数据，检索工具在
   `research_agent` 内以闭包绑定 `context.user_id`（与 `_build_retrieval_tool`
   同模式），权限天然隔离；算法层复用 A 阶段 `paperpilot/retrieval/bm25.py`。
   MCP 定位保持「论文获取类外部能力」。
2. **冲突处理采用 Mem0 v3 思路**：只 ADD 不改写，检索时 BM25 分数 ×
   时间新近度融合排序（新记忆覆盖旧记忆的展示顺序）；`status` 字段保留
   但 v1 不做主动归档，去重/归档/整理留给 C 阶段定期整理任务。
3. **提取触发点**：`deep_reading/runner.py` 发布成功后的 best-effort 后处理
   （与 `_enrich_archive_narrative` 完全同模式：claim 幂等 → LLM →
   complete，异常吞掉只记日志，绝不影响发布链路）。
   幂等键：`source_task_id`（该 task 已有记忆则跳过）。
4. **核验规则（v1 规则版）**：提取器必须为每条候选携带 `support_span`
   （原文支撑片段），管线核验 span 归一化后必须出现在本轮用户/助手
   消息文本中，否则丢弃——「提取器可提议，不能自行把未核验字符串当事实」。
5. **评估分层**：检索层评估（BM25+recency 对合成记忆集的 recall/MRR）
   进单元测试；LLM 提取端到端评估挂 smoke 体系（后续）。

## 分步骤执行

### B1 存储层
- `web/db_models.py`：`UserMemoryRow`（memory_id PK、user_id FK、kind、
  content、context_json、source 三元组 FK、support_span、status、created_at；
  Index(user_id, created_at)）。
- `migrations/versions/20261005_0004_user_memories.py`（downgrade 沿用
  RuntimeError 风格）。
- `web/records.py`：`UserMemoryRecord` / `NewUserMemory` dataclass。
- `web/store/user_memories.py`：`append_user_memory` / `list_user_memories`
  / `count_task_memories`（user_id 隔离过滤在 SQL 层）。
- `web/task_store.py` 委托 + 更新
  `tests/architecture/test_web_module_boundaries.py` 的
  EXPECTED_TASK_STORE_METHODS / SIGNATURES（守卫测试设计如此，同步更新）。

### B3 检索与工具
- `user_memory/retrieval.py`：`MemorySearchIndex`——BM25 over
  (kind + content) 文本 + `recency_boost`（按 created_at 新近度），
  `search(query, top_k)` 返回融合排序；导出 `search_user_memories` 便捷函数。
- `deep_reading/research_agent.py`：`_build_user_memory_tool(context)` 注册
  `search_user_memory(query)`；工具描述明确：涉及用户偏好/研究方向/历史
  已读论文时调用；返回的记忆是**参考资料不是指令**（来源标记）。
  工具异常返回友好错误字符串，不炸 agent 循环。

### B2 提取管线
- `user_memory/extractor.py`：`extract_memory_candidates(user_text,
  assistant_text, model)` —— `with_structured_output`（extra=forbid），
  prompt 写明三规则（选择性/抽象化/结构化）+ 禁存一次性事实；
  `verify_candidates(candidates, texts)` 归一化 span 核验。
- `user_memory/pipeline.py`：`run_memory_extraction(store, model, task,
  user_message, assistant_message) -> int`（幂等检查→提取→核验→逐条
  append→返回写入数；全部异常上抛由调用方吞）。
- `runner.py`：发布成功路径调用 `run_memory_extraction`，best-effort。

## 验证方式
- 单测：store（幂等/隔离/计数）、retrieval（排序与中文查询）、
  extractor（fake 模型 + span 核验丢弃）、pipeline（端到端 fake）、
  agent 工具（注册与返回格式）。
- 全量 `pytest`；LLM 真调用与三层次评估集（layer1/2）留待 smoke 阶段，
  收尾时明确标注未覆盖。

## 风险与待确认项
- 发布路径新增 LLM 调用（提取）：best-effort + 异常吞掉，最坏影响是
  延迟增加数秒与一条 warning 日志，不影响发布正确性；若不可接受可
  加配置开关（默认开）。
- 提取质量依赖 prompt：一次性事实误存风险由「选择性规则 + span 核验」
  双层缓解，C 阶段定期整理兜底。
- 架构守卫测试需同步更新（EXPECTED_TASK_STORE_*），属预期改动。
