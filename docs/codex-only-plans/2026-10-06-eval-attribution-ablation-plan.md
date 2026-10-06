# 第六章落地：失败归因 + Pass^k + 消融开关（rag 分支）

> Codex only。用户已确认 A→B 连做；评估 API 成本优先走用户的 ZAI
> 订阅（glm-5.3-flash）。

## GLM 接入（零代码改动，配置就绪）

paperpilot 的模型层已支持 `DEEPSEEK_BASE_URL`（ChatDeepSeek 兼容端点）
与 `PAPERPILOT_RESEARCH_MODEL_NAME`。接入 GLM 只需在 .env 换：
`DEEPSEEK_BASE_URL=https://api.z.ai/api/paas/v4`（或 bigmodel 端点）、
`PAPERPILOT_RESEARCH_MODEL_NAME=glm-5.3-flash`、`DEEPSEEK_API_KEY=<ZAI key>`。
待用户提供 key 后验证一次 smoke；开发期用 DeepSeek。

## A1：失败归因器（书 6.5.2）

`paperpilot/smoke/attribution.py`：从实例业务库 task_events 流生成
结构化归因——首个错误定位（事件序号+类型）+ 错误类别 + 证据引用。

类别（首版）：contract_failure（合约，含插桩 decision dump）/
retrieval_empty（检索空或未引用证据）/ tool_error（工具异常，含
repetition warning 关联）/ budget_exhausted（预算/重试耗尽）/
check_failure（任务完成但 check 未过——归因到答案内容层）。
原则：归因首个错误，后续重试记为后果不记为根因。

集成：runner 失败路径调归因器，报告行加 `attribution` 块；
CLI 失败输出打印归因摘要。纯事件流分析，零 API 成本。

## A2：--repeat k 可靠性协议（书 6.2.2/6.7）

smoke CLI 加 `--repeat k`（默认 1）：每场景独立跑 k 次，报告
Pass^k（连过）/ Pass@k（至少一过）+ 单次成功率 Wilson 95% 置信区间。
报告行增 repeat 聚合块。用于激活决策与回归阈值判断。

## B：消融开关 + 矩阵（书 6.10.1 / 6.6.3）

补齐四大特性开关（env，默认开=行为不变；allowlist+.env.example 登记）：
- `PAPERPILOT_MEMORY_TOOL_ENABLED`（search_user_memory 注册与注入）
- `PAPERPILOT_PROFILE_INJECTION_ENABLED`（画像常驻注入）
- `PAPERPILOT_COMPUTATION_TOOL_ENABLED`（run_computation）
- `PAPERPILOT_RETRIEVAL_MODE`（hybrid/dense，复用既有 search mode）
布线原则（书中 6.10.1）：启动路径早期读取，避免模块级常量捕获。

`paperpilot/smoke/ablation.py` + CLI `--ablate`：跑「基线（全开）vs
逐项关闭」矩阵 × 场景子集，输出每特性的通过率差与 token 成本差
（model_usage 事件聚合）表。矩阵跑分在 GLM key 就绪后执行。

## 验证与风险

- 单测：归因类别判定（合成事件流）、repeat 聚合数学、开关布线。
- 真实：已知失败场景（构造合约失败）归因正确；--repeat 3 on
  记忆场景输出合理区间；GLM key 后 1 场景验证 + 消融矩阵。
- 风险：repeat 的种子独立性（API 温度非零，天然随机）；消融矩阵
  API 成本由 repeat × 场景数控制，默认小矩阵。
