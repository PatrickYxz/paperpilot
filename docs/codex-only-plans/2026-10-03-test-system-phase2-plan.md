# Test System Phase 2 Plan（要点评分与趋势对比）

## 背景与目标

Phase 1（见 `2026-10-03-real-business-test-system-plan.md`）交付了 CI、真实 E2E harness、
场景题库与 harness 自测，但只能回答"跑通了、有引用"，不能回答"**答得对不对**"，
且两次运行的报告只能人眼对比。Phase 2 补齐这两块：

1. **要点覆盖**：场景可带 `expected_points`（标准答案要点），两层判定——
   - 程序化严格层：要点子串（大小写不敏感）必须出现在答案里，适合精确标识符/数字；
   - LLM-as-judge 语义层（`--judge`）：判定改写/同义覆盖，数字必须精确匹配。
2. **趋势对比**：`scripts/smoke_compare.py` 对比两份 `smoke_report.jsonl`，
   自动标出回归/改善/新增/移除场景与失败项变化，退出码可直接当门禁用。

## 约束条件

- 不新增环境变量；judge 复用 `DEEPSEEK_API_KEY` 与
  `PAPERPILOT_RESEARCH_MODEL_NAME`（`--judge-model` 为 CLI 参数而非新变量）。
- judge 单次调用、零重试（复用 `build_deep_reading_model` 的 temperature=0 /
  thinking 禁用 / max_retries=0），输出解析失败按失败处理，不阻塞其他场景。
- 报告行新增字段向后兼容（旧报告没有 `citations_count` 等字段时 compare 容错）。
- 新增计划文档登记 `DOCS_KEEP_ALLOWLIST`。
- 题库本批不填 `expected_points`（论文事实要点需首次真实运行时校准录入），
  字段与判定逻辑先用单元测试覆盖。

## 分步骤执行计划

### Step 1：场景要点字段 + 程序化覆盖检查
- `scripts/smoke_real.py`：`Scenario` 增加 `expected_points: tuple[str, ...]`；
  `load_scenarios` 校验（非空字符串列表）；`evaluate_checks` 在有要点时追加
  `expected_points_covered` 检查（大小写不敏感子串）。
- 测试：要点全中/部分缺失/答案为空三种情形；非法要点行被拒。

### Step 2：LLM-as-judge
- `scripts/smoke_real.py`：`judge_answer(question, expected_points, answer, *, model_name, invoke=None)`；
  `invoke` 可注入（测试传 fake，生产经 `build_deep_reading_model` 懒加载构造）。
  固定评分 prompt，输出 JSON `{"covered": [...], "missing": [...], "reason": str}`；
  解析失败抛 `JudgeError`。
- CLI 增加 `--judge` 与 `--judge-model`：主循环在场景有要点且 `--judge` 时追加
  `judge_points_covered` 检查；无要点场景打印跳过说明；judge 失败不中断运行。
- 测试：fake invoke 返回合法 JSON → 通过；返回无 JSON 文本 → 检查失败；
  要点部分 missing → 失败详情列出缺失项。

### Step 3：报告行补充可比字段
- `build_report_row` 接收 outcome，新增 `citations_count`、`answer_chars`。
- 测试更新：字段存在性与数值正确。

### Step 4：趋势对比脚本 `scripts/smoke_compare.py`
- 输入两份报告 JSONL；按 `scenario_id` 对齐；逐场景标注
  SAME / REGRESSION / IMPROVED / NEW / REMOVED，回归与改善附失败项差异明细；
  汇总段落；`--json` 输出机器可读结果。
- 退出码：当前报告有失败或存在回归 → 1，否则 0；文件缺失/为空 → 2。
- 测试：全过→0；单场景回归→1 且明细含新失败项名；fail→pass 标 IMPROVED；
  新增/移除场景；旧报告缺新字段时容错；`--json` 可解析。

### Step 5：全量验证
- `pytest tests/smoke tests/architecture -q` 与全量 `pytest -q`。
- `scripts/smoke_real.py --dry-run`、`scripts/smoke_compare.py` 冒烟自检。

## 验证方式

- 全部新逻辑由单元测试覆盖（fake invoke / 临时报告文件），无网络无凭证。
- judge 与 expected_points 的真实效果留待首次真实运行（用户批准 + 凭证）后校准。

## 风险与待确认项

- **judge 同源偏差**：业务与 judge 同用 DeepSeek，可能同源偏好；如需更强可信度，
  后续可接第二供应商模型当裁判（需新增凭证变量，另行确认）。
- **程序化子串层对英文词形变化不敏感**（如 "2-point improvement" 匹配不上
  "improves by 2 points"）——这正是设语义层的原因；要点录入时应优先精确标识符/数值。
- 评分标准（多少缺失算失败）当前为"全要点必须覆盖"；若过严，真实运行后再调。

## 结构调整附记（2026-10-04）

应用户要求，把单文件 harness 拆分为独立测试系统包 `paperpilot/smoke/`，
一职责一模块：

- `scenarios.py` 题库模型与 JSONL 加载；`cases.jsonl` 随包自包含。
- `checks.py` 断言清单与观测模型（含引用提取、密钥泄漏扫描）。
- `adapter.py` HTTP 适配层（唯一入口知识）。
- `judge.py` LLM-as-judge（可注入、零重试）。
- `runtime.py` 隔离实例构建（仅复用已声明环境变量）。
- `runner.py` 场景执行与报告行；`cli.py`/`__main__.py` 命令行入口。
- `compare.py` 趋势对比（可独立 `-m` 执行）。

入口变更：`python -m paperpilot.smoke`、`python -m paperpilot.smoke.compare`。
原 `scripts/smoke_*.py` 移除（从未提交）。测试镜像拆分到 `tests/smoke/`
（按模块一一对应），共享 fake 收敛进 `conftest.py`，importlib 动态加载
hack 随之删除，改为常规包导入。行为与 Phase 1/2 已验证逻辑一致。
