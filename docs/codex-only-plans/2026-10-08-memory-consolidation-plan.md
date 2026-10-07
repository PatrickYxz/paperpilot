# 第八章落地：记忆睡眠学习（定期整理）（rag 分支）

> Codex only。第三章 C 阶段遗留项 + 第八章 8.3.5 方法论。用户未及
> 确认时按推荐方案 A 执行（append-only 回滚 + Reviewer 审批 +
> 保守阈值，风险受控）。

## 设计（五步周期 → paperpilot）

1. **触发**：提取管线尾部检查——active 卡数 ≥ 15（保守阈值）且
   进程内该用户冷却 ≥ 1h（防整理后仍超阈值时每轮重跑；进程重启
   重跑一次幂等无害，注释说明）。
2. **整理提案（LLM，不直接改库）**：`consolidation.py` 读全部
   active 卡 → 结构化提案：
   - `merge`：同事实多表述/碎片卡合并为一张（merged_content 可
     补适用条件——第三章 qualification 原则）；新卡 context 记
     `merged_from` 溯源
   - `archive`：过期、被新卡取代、无长期价值
   - `keep`：保留
   完整性约束：每卡至多被一个提案引用；引用未知 id 的提案丢弃。
3. **审批**：复用第四章 Reviewer（deepseek-flash，JSON mode）逐
   提案批准；拒绝即不执行该提案（fail-open 仅当 reviewer 不可用，
   同提取审批语义）。
4. **执行与回滚**：archive = status 置 archived + context 记
   reason（原始卡永不删除）；merge = 新卡 append（support_span
   取源卡最长 span，context 标 consolidated）+ 源卡归档。回滚 =
   状态翻转，审计靠来源三元组与 merged_from。
5. **验证**：整理后 profile 重归纳（复用）；检索回归——整理前后
   对固定查询集的命中对比（组件级验证）；smoke memory-layer1
   场景回归。

## 明确不做

参数更新（第七章已定）、Prompt 经验沉淀（无生产流量）、
自我修改代码、新表（水位用进程内冷却替代）。

## 验证与风险

- 单测：提案解析（未知 id 丢弃/单提案引用约束）、审批通过/拒绝
  执行路径、merge 溯源字段、archive 可回滚（状态翻转检索可见性）。
- 真实：构造重复卡用户跑整理（deepseek 提案+审批），验证合并后
  检索不丢关键事实。
- 风险：LLM 误合并 → Reviewer 审批 + append-only 可回滚 +
  support_span 保留；阈值触发风暴 → 冷却。

---

## 附：第九章落点（2026-10-08 追加）

第九章（语音/CUA/机器人）与垂直论文助手不适用，不强行落地。
唯一衔接点：第四章 4.4.1「工具化多模态分析」——`analyze_paper_page`
工具把 VLM server（understand_paper_page）接入 agent 工具面，论文
图表/布局问题走视觉路径；描述写明「纯文本问题优先文本检索」的
边界（4.4 输出形态原则：布局敏感内容保留图像）。flag：
PAPERPILOT_PAGE_VISION_TOOL_ENABLED（默认开；DashScope 配额恢复前
工具运行时优雅降级返回提示文本）。真实验证待 DashScope 恢复。
