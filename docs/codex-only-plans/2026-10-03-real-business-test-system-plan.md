# Real Business Test System Plan（真实业务测试系统搭建计划）

## 背景与目标

- 2026-09-17 的真实冒烟是手工一次性作业（隔离实例 + 人眼验收），无法复现、无法对比、无法扩展场景。
- 项目 README 已明确：fake 模型的 832 个自动化测试不能推断生产就绪；真实模型测试是需人工批准的独立门禁，但该门禁目前没有工具支撑。
- 目标：把"真实业务测试"从手工作业变成可重复执行、可落盘对比的最小系统，分两期：
  - **本期（Phase 1）**：CI（守住 fake 快速套件）+ 真实 E2E harness（固化冒烟：场景表、隔离实例、程序化断言、报告落盘）。
  - **下期（Phase 2，不在本计划实施范围）**：在 harness 上叠加质量评分（要点覆盖、LLM-as-judge）、趋势对比。
- 本计划回应的核心约束："一边升级项目一边跑测试，很多入口都会变"。设计原则是**业务语义与接口调用分离**：
  - 场景数据（题目、论文、期望）是纯业务语言，不含任何 API 细节——入口怎么变都不需要动。
  - 所有 HTTP 调用收敛在 harness 的单一适配层 `ConversationApi`（约 50 行）——入口变了只改这一个类。
  - 底层 fake 测试随入口变红是设计使然（保护契约），修的是测试本身。

## 约束条件

- 不新增环境变量：`.env.example` 变量集合被 `tests/architecture/test_repository_allowlist.py` 精确锁定，harness 只复用已声明的变量。
- 不触碰架构门禁禁止的路径：不得创建 `paperpilot/eval`、`tests/eval`、`data/eval`、`data/traces`、`scripts/day*`。
- 新增 docs 文件必须同步登记 `DOCS_KEEP_ALLOWLIST`（本计划文档自身也要登记）。
- 真实模型调用维持 README 门禁：需要真实凭证、花真钱，harness 无凭证时拒绝执行（`--dry-run` 除外）。
- 不引入新依赖：harness 只用标准库 + 已有依赖（fastapi TestClient 已在 requirements）。
- 默认执行环境与生产默认一致：context management 默认关闭（`--context-management` 显式开启，对齐 9-17 冒烟配置：开启 context management、关闭 full compaction）。

## 分步骤执行计划

### Step 1：计划文档 + 架构白名单登记
- 新建 `docs/codex-only-plans/2026-10-03-real-business-test-system-plan.md`（本文件）。
- 修改 `tests/architecture/test_repository_allowlist.py`：`DOCS_KEEP_ALLOWLIST` 增加本文件路径。
- 目的：文档先行，且不让架构测试变红。
- 验证：`pytest tests/architecture -q` 通过。

### Step 2：CI workflow
- 新建 `.github/workflows/ci.yml`：push/PR 触发，ubuntu + Python 3.12，`pip install -r requirements-lock.txt`，跑 `python -m pytest -q`（默认排除 slow，纯 fake，无需 secrets）。
- 真实 LLM 层永不进 CI。
- 验证：YAML 可解析；本地等价命令已通过（现有 832 套件）。

### Step 3：真实 E2E harness（`scripts/smoke_real.py`）
单一脚本，模块级只依赖标准库（重量级 import 延迟到构建实例时），包含：
- `Scenario` 场景模型 + `load_scenarios(jsonl)`：从数据文件加载校验。
- `ConversationApi` 适配层：**唯一知道 HTTP 入口形状的地方**（注册登录、建会话、提问、轮询任务更新、取消息）。
- `evaluate_checks`：程序化断言清单，每项独立命名——task 终态为 completed、助手答案非空、引用数达标、（开启 context management 时）artifact 存在、无 retry 事件、事件流无凭证泄漏（`sk-` 扫描）。
- `build_isolated_app`：环境变量注入隔离路径（复用 `PAPERPILOT_TASK_DB_PATH` 等已声明变量指向临时目录），进程内 `TestClient` 拉起真实 web 实例；场景数据带论文元数据时注入 stub `PaperSearch`（对齐 9-17 冒烟"仅注入会话创建元数据，Research/MCP/LLM 全真实"的模式）。
- 报告：JSONL 落盘（含 git commit、模型名、耗时、逐项 check 结果），退出码反映整体 pass/fail。
- CLI：`--dry-run`（无凭证可跑，打印执行计划）、`--only <id>`、`--context-management`、`--timeout`、`--base-dir`、`--model`。
- 凭证门禁：非 dry-run 且缺 `DEEPSEEK_API_KEY` 时直接拒绝执行。
- 目的：把 9-17 手工冒烟固化，场景可扩展、结果可对比。
- 验证：单元测试（Step 5）+ `--dry-run` 实跑。

### Step 4：场景数据（`scripts/smoke_cases.jsonl`）
- 1 个 active 场景：复用 9-17 已验证的 arXiv 2001.09899v1 + quick 深度（期望值保守设置：引用 ≥1、答案非空、终态 completed）。
- 2 个 inactive 草稿场景（standard 数字题 / deep 对比题），期望值留待首次真实运行校准后激活。
- 数据纯业务语言：论文 ID、问题、深度、期望，零 API 细节。
- 验证：`load_scenarios` 单元测试通过。

### Step 5：harness 单元测试（`tests/smoke/test_smoke_real.py`）
- fake client（无网络、无凭证）驱动适配层：注册→建会话→提问→轮询的调用序列与载荷形状。
- 断言清单逐项 RED/GREEN：每种失败模式（终态 failed、答案空、引用不足、retry 事件、凭证泄漏）对应一条测试。
- 报告行结构：含 git_commit、逐项 check、整体 rollup。
- 凭证门禁与 `--dry-run` 的 CLI 行为（子进程级验证，显式清空凭证环境变量）。
- 验证：新测试全绿。

### Step 6：全量验证
- `.venv/bin/python -m pytest -q` 全量（原 832 + 新增）。
- `scripts/smoke_real.py --dry-run` 实跑输出检查。
- CI YAML 解析检查（`python -c "import yaml..."`）。
- 真实运行（需凭证与人工批准）不在本次自动执行范围，留作首次人工触发。

## 验证方式汇总

- 架构测试、全量套件、dry-run、YAML 解析——本次全部执行并在结果中报告。
- 真实 LLM 冒烟运行——待用户在有凭证环境显式触发（维持 README 门禁），首次运行同时校准 active 场景的期望值。

## 风险与待确认项

- **active 场景期望值未校准**：9-17 冒烟未记录确切题目与引用数下限，本计划保守设为引用 ≥1；首次真实运行后应调紧（如 ≥3）并激活草稿场景。
- **引用字段的 HTTP 形状**：`AnswerDraft.citations` 经由助手消息 `metadata` 暴露的具体键名以首次真实运行为准；适配层 `_extract_citations` 是唯一改动点，形状确认前采用防御性递归查找。
- **事件 retry 判定是启发式**：按事件字段中 `retry` 子串判定；若真实事件词汇不同，首次运行后按实际词汇收紧。
- **CI 首跑可能暴露安装问题**：requirements-lock.txt 在 ubuntu 上的可安装性未验证过（本地为 macOS），首次 CI 运行可能需要修依赖。
- Phase 2（评分、趋势）范围另立计划。
