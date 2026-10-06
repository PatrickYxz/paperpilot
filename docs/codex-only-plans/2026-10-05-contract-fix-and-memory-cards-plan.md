# 修 ResearchContractError + 记忆 Advanced JSON Cards（rag 分支）

> Codex only。收尾验证发现的既有问题修复 + 记忆存储按书 3.1.3 卡片化。

## 任务一：ResearchContractError 概率性修复

症状：混合意图问题（背景句+论文问题）与对抗问题在 quick depth 下
偶发 `research_contract_invalid`（1 次模型调用即终止，重试仍失败）。
main 上同样存在（既有问题）。

步骤：
1. 读 `_validate_and_materialize_result` 全部 raise 点与 agent 终止路径
2. 合约失败时把原始 decision dump 进 task_events（插桩，本身也是有用的
   可观测性改进）
3. 重跑 bert-adversarial 复现，定位具体失败字段
4. 按根因修复（候选：repair message 引导、prompt 规则、合约校验放宽）
5. 回归：bert-adversarial × 3 + 记忆种子措辞 × 3 稳定通过

## 任务二：记忆卡片化（Advanced JSON Cards，书 3.1.3）

书中设计：卡片除事实外带主体（person）、与用户关系（relationship）、
获取背景（backstory）——解决"张医生是谁的医生"式消歧；关键少量数据
用卡片、大量事实用简条，混合模式。

落地（不动表结构，`context_json` 本就是预留字段）：
1. `MemoryCandidate` 增 `subject`（默认 "user"）/`relationship`/
   `backstory`（≤120 字）/`topic`（细主题），提取 prompt 同步升级；
   support_span 核验规则不变
2. 写入：结构化字段进 `context_json`；kind 保留
3. 检索：BM25 索引文本扩为 kind+topic+subject+content+backstory
   （消歧信息参与词法匹配）
4. 呈现：`[kind|topic|subject]` 标签前缀 + backstory 摘要
5. 兼容：旧记忆（context 空）检索与呈现不回归

## 验证

- 单测：卡片提取/核验/检索/呈现；全量 pytest
- 真实：bert-adversarial × 3 稳过；记忆组件级验证（卡片字段质量）；
  合约修复后激活 memory-layer1/2 场景进回归

## 风险

- 合约修复可能改变拒答行为：bert-adversarial 的 expect_refusal/
  trap_terms 检查必须仍通过
- 卡片字段增加提取成本（输出 token 变多）：可控
