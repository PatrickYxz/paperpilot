# Codex only: context-compression review fixes

## 背景与目标

修复 2026-09-02 审查确认的上下文管理集成缺陷，使五层策略按既有设计运行，同时保持公开 Web API、ResearchResult、AnswerDraft、业务终态事务和 feature-flag-off 行为不变。

## 约束

- 保留当前未提交工作树，不执行 reset、checkout、stash、commit、push 或 merge。
- 不引入新依赖，不删除权威 TaskStore 消息、TurnArchive 或 Artifact。
- 每项修复先增加能复现缺陷的测试并确认失败，再做最小生产修改。
- 继续使用 `.venv/bin/python -m pytest` 验证。

## 执行步骤

1. 修正统一 Token 预算：按真实模型消息和工具 schema 计数，并在候选采用后写回重计数值。
2. 收紧压缩候选验证：所有 Paper、Evidence、Artifact、Archive 引用必须来自 authority set，active refs 必须 100% 保留。
3. 修正 ContextView 生命周期：按配置保留最近业务轮次，持久化 ContinuationCapsule，避免下一轮清空。
4. 接通微压缩状态机：保存 ToolMessage 索引，标记已进入成功调用及被下游消费，只批量替换正确历史消息。
5. 修正 TurnArchive 完整性：解析并保存 ArtifactRef，只保存未完成 TODO，并补齐终态后的 crash-window/redelivery 修复。
6. 修正 legacy-summary 过滤、配置参数接线、熔断阈值/重试次数和容量终止续接 Artifact。
7. 运行上下文专项、迁移/Runner 集成测试、完整回归和 `git diff --check`。

## 验证方式

- 每个缺陷对应的新增测试先 RED、修复后 GREEN。
- 上下文管理专项测试全部通过。
- 完整 `.venv/bin/python -m pytest -q` 通过。
- `git diff --check` 通过。

## 风险

- Token 计数必须避免把 Provider 不支持的 tool schema 参数直接传给模型计数器。
- Capsule 是派生状态，不能反向修改 TaskStore 权威历史。
- Archive 修复必须幂等，不能重复生成 Narrative 或改变已完成 Task。
- 容量终止 Artifact 只能保存受控续接数据，事件中不得暴露正文或文件路径。
