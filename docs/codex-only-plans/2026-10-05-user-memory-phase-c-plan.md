# C 阶段细化计划：双层记忆架构（rag 分支）

> Codex only。三阶段路线收官（书 3.3.5 双层架构 + 3.3.3 定期整理的 v1）。

## 增量决策

1. **概览层 = 物化 profile**（而非每轮动态拼 top-N 记忆）：LLM 从全部
   active 记忆归纳一份 ≤120 词的用户研究画像，去重消冲突后常驻 agent
   消息（SystemMessage，标注 background data）。触发：提取管线每写入
   PROFILE_REFRESH_DELTA(=5) 条新记忆后 best-effort 刷新。
2. **profile 存储**：新表 `user_memory_profiles`（user_id PK、
   profile_text、source_memory_count、updated_at），migration 0005。
3. **细节层深化（C2）**：`search_user_memory` 升级混合检索——BM25 over
   记忆条目（已有）+ BM25 over 跨会话轮次摘要（turn_archives 的
   narrative_summary，缺失时退回 user question），后者带
   `[conversation X; date; question: ...]` 上下文前缀（复用 A2 思路），
   RRF 融合。新增 store 函数 `list_user_turn_summaries(user_id, limit)`。
4. **范围外**（留后续）：完整 Proposer-Reviewer PR 审核整理、smoke
   三层次评估集（提取质量先靠真实使用观察）。

## 步骤

1. `user_memory/profile.py`：`generate_profile(model, memories) -> str|None`
   （结构化输出，失败返 None）+ `_PROFILE_WORD_BUDGET` 截断。
2. 表 + migration 0005 + store（get/upsert）+ TaskStore 委托 +
   守卫测试登记（表/列/索引/FK、方法签名、迁移头）。
3. pipeline 尾部触发 profile 刷新（阈值比较，best-effort）。
4. research_agent：`_research_messages` 增 `user_profile` 参数，
   run_research_agent 经 `_load_user_profile(context)` 传入。
5. C2：store `list_user_turn_summaries` + retrieval 增
   `search_turn_summaries` + 工具改 RRF 混合 + presentation 渲染两路。
6. 测试：profile 生成/截断/触发阈值、注入位置、混合检索融合、
   store 层；全量 pytest。

## 验证与风险

- 全量 pytest；LLM 真归纳质量未端到端验证（同 B 阶段，fake 模型测逻辑）。
- profile 注入会占 ~200 token 预算：截断 + 仅在有 profile 时注入。
- 混合检索默认改变工具返回形态（多一路会话命中）：字段向后兼容。
