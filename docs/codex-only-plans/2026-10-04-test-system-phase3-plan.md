# Test System Phase 3 Plan（多轮追问场景）

## 背景与目标

Phase 1/2（见 `2026-10-03-real-business-test-system-plan.md`、
`2026-10-03-test-system-phase2-plan.md`）交付了 CI、真实 E2E harness、题库、
要点评分与两报告对比，但所有场景都是**单轮提问**——`codex/context-management`
分支的核心能力（跨轮次的分层上下文管理、微压缩、TurnArchive）完全没有被
真实场景覆盖。Phase 3 补上多轮追问：

- 场景可带 `follow_ups`（追问列表）：同一会话内，主问题完成后逐条追问。
- 第 N+1 轮的 `expected_head_message_id` 链式取第 N 轮助手消息 id
  （与 `_validate_turn_admission` 的语义一致；首轮为 `None`）。
- 逐轮观测与断言：每轮任务终态、答案非空；主问题的引用/要点/judge 判定
  绑定第一轮答案；retry/泄漏/artifact 检查聚合所有轮。

本计划对应方案 C 中"多轮场景"一项；第二供应商裁判、定期回归与多版本
趋势沉淀仍留待 Phase 4+。

## 约束条件

- 报告行保持向后兼容：`citations_count` 仍指第一轮（主问题）引用数，
  compare 工具无需改动；新增 `turns` 摘要数组随行携带。
- 对外行为兼容：无 `follow_ups` 的场景执行路径与 Phase 2 完全一致。
- 不新增环境变量；不改 web 公开 API。

## 分步骤执行计划

### Step 1：场景模型
- `scenarios.py`：`Scenario.follow_ups: tuple[str, ...] = ()`；
  `load_scenarios` 校验（非空字符串列表）。

### Step 2：适配层
- `adapter.py`：`ask(content, depth, head_message_id=None)` 改为任意提问；
  `assistant_message()` 改为返回**最后一条**助手消息（多轮累加）。

### Step 3：观测与断言
- `checks.py`：新增 `TurnObservation`（单轮的 task/assistant/citations/
  events/artifacts/error）；`ScenarioOutcome.turns: list[TurnObservation]`，
  保留 `task/assistant_message/citations/events/artifacts` 作为"最后一轮"
  的兼容属性；会话级错误仍在 `ScenarioOutcome.error`。
- `evaluate_checks` 调整：
  - `task_completed`/`assistant_answer_present` 改为逐轮判定（命名不变，
    语义为 every turn）；
  - `citations_gte`/`expected_points_covered`/judge 绑定第一轮答案；
  - `no_retry_events`/`no_secret_leak`/`context_artifacts_present` 聚合全部轮；
  - `turns_completed`：完成轮数 == len(questions)（防追问提前中断）。

### Step 4：执行与报告
- `runner.py`：按 `[question, *follow_ups]` 逐轮执行，轮间链式传 head id；
  某轮异常记入会话级 error 并停止后续追问。
- `build_report_row`：新增 `turns: [{status, answer_chars, citations_count}]`。
- 题库补 1 条 inactive 多轮草稿场景（追问链：实验设置→数值结果→局限）。

### Step 5：测试与验证
- conftest fake client 升级多轮（按轮脚本化 task id / 消息累加 / head 校验），
  兼容单轮旧用法。
- 新增/更新测试：follow_ups 加载校验、多轮调用序列与 head 链、逐轮断言、
  追问中断、报告 turns 摘要、单轮场景行为不变。
- 全量 `pytest -q` + `python -m paperpilot.smoke --dry-run`。

## 验证方式

- 单元测试全绿（fake 驱动，无网络无凭证）；单轮场景旧测试不改语义通过
  即证明向后兼容。
- 真实多轮效果留待首次真实运行校准（多轮场景的引用/要点期望值届时录入）。

## 风险与待确认项

- 多轮场景 token 消耗数倍于单轮（每轮一次完整研究）；真实运行预算需注意。
- 逐轮 assistant_message 取"最后一条"，依赖消息列表按时间有序返回；
  若 API 排序语义变化，适配层单点修复。
- `turns_completed` 在追问导致 409（head 变化）等场景下会失败并停止——
  这正是要暴露的问题，不是误报。
