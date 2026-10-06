# 第四章落地：工具描述审计 + 记忆 Reviewer 审批（rag 分支）

> Codex only。用户已确认 A+C 路线（书 4.2 工具描述艺术/保真 + 4.5
> Proposer-Reviewer 事前审批，与第三章记忆管线跨章组合）。

## 任务 A：工具描述与保真性审计（书 4.2.4 / 4.2.5）

书中原则：描述核心是"什么时候用"；**边界与反例比能力更重要**；参数
用具体示例；返回值结构说明；静默转换必须声明（保真性）。

审计对象与改法：
1. `retrieve_paper_evidence`：前置条件显式化（"必须先 prepare_paper，
   否则返回引导错误"——把上轮代码兜底的行为写进描述）；top_k 参数
   带示例。
2. `prepare_paper`：与 retrieve 的关系写清（"为后续检索准备索引"）；
   只接受本会话 active 论文 ID 的边界。
3. `search_related_papers`：何时用/何时不用（找新论文 vs 检索已准备
   论文内容的反例）；limit 示例。
4. `search_user_memory`：已有较完整描述，补"返回卡片格式说明"。
5. `read_artifact_slice`/`search_artifact`：offset/limit 截断语义
   （显式截断声明，书 4.4）。
6. 保真性声明：`_canonical_arxiv_id` 的规范化行为（v1 后缀补全等）
   写进 search/prepare 的参数描述。
7. MCP server（arxiv/colbert）工具 docstring 同步审计。

验证：描述断言单测（关键边界词存在，防静默退化）；prompt_version
research-v7 → v8；smoke 关键场景（bert-adversarial、memory-layer1）
回归。

## 任务 C：记忆写入的 Proposer-Reviewer 事前审批（书 4.5）

异源互审落地：**DeepSeek 提取（Proposer）+ Qwen 审批（Reviewer）**——
书中建议的不同家族能力相近配对；DashScope SDK 已在依赖内（vlm 在用），
零新依赖。

1. `user_memory/dashscope_model.py`：`BaseChatModel` 最小实现
   （dashscope Generation 兼容模式，支持 with_structured_output）。
2. `user_memory/reviewer.py`：`review_memory_cards(cards, source_texts,
   model) -> ReviewVerdict`（结构化输出：每卡 approve/reject + 理由；
   审查维度：span 是否真支撑、是否一次性事实、是否过度泛化）。
   Reviewer prompt 与 Proposer 规则一致但视角不同（风险控制导向）。
3. pipeline 集成：span 核验（规则）→ Reviewer 审批（LLM）→ 只写
   批准卡；拒绝理由进 event（反馈闭环的书内实践）。
4. 配置：`PAPERPILOT_MEMORY_REVIEWER_MODEL`（空=禁用审批，行为不变；
   默认 qwen-plus）；allowlist + .env.example 登记。
5. 测试：fake reviewer（批准/拒绝/异常降级）；全量 pytest；
   真实 Qwen 验证审批质量（用上轮卡片场景对照）。

## 验证与风险

- 全量 pytest + smoke 回归；A 的行为变化靠 bert-adversarial ×3 与
  记忆场景稳定性监控。
- C 风险：Reviewer 延迟（审批加一次 LLM 调用，发布后 best-effort
  路径可接受）；Qwen 不可用时降级为跳过审批（配置空）。
- 审批过严会丢记忆：Reviewer 拒绝率纳入真实验证观察项。
