# Day 10 Research Todo 执行计划

> Codex only. 本文件记录 Codex 对 Claude Code 生成方案的确认和执行拆解；主要方案仍以 `docs/superpowers/plans/2026-05-03-day10-research-todo.md` 为准。

## 背景与目标

Day 10 要新增 PaperPilot 第二个 L2 内嵌 tool `research_todo`，与 `load_skill` 同层注册，让 LLM 在多步研究任务中先列计划并逐步推进状态。

确认后的落地目标：

1. 新增 `paperpilot/builtin_tools/research_todo.py`，提供 `TodoStore`、`research_todo_tool(store)`、`render(items)` 和 `RESEARCH_TODO_NUDGE`。
2. 只修改 `find-classics` skill 的步骤段，加入 step 0 作为演示路径的硬触发。
3. 在 `paperpilot/main.py` 启动流程中实例化 `TodoStore`，把 `research_todo` 合并进工具列表，并把 nudge 拼入 system prompt。
4. 增加单元测试、main 集成测试和 Day 10 端到端 smoke。

## 约束条件

- 不修改 `paperpilot/core/loop.py`、`paperpilot/tools/mcp_client.py` 或 MCP server。
- 不引入新依赖。
- 不做 todo 持久化、id、priority、due、tag、多 store 或自动同步到 system prompt。
- `research_todo` 输出必须是字符串，避免触碰 Day 7 FastMCP list 输出监控边界。
- 仅校验“至多 1 个 in_progress”，其它字段依赖工具 schema。

## 分步骤执行计划

1. Task 1: 新增 `research_todo` 单元测试，确认失败后实现内嵌 tool，再运行目标单测。
2. Task 2: 精确替换 `paperpilot/skills/find-classics.md` 的 `## 步骤` 段，随后验证 skill loader 能正常读取。
3. Task 3: 修改 `paperpilot/main.py` 和 `tests/test_main_integration.py`，验证 system prompt nudge 和工具注册。
4. Task 4: 新增 `scripts/day10_smoke.py`，验证真实 LLM 会经由 `load_skill(find-classics)` 触发 `research_todo` 并调用 arxiv/graph 工具链。
5. 收尾检查：运行默认测试集、必要的 slow/smoke、TODO/FIXME 扫描和 git 状态检查。

## 验证方式

- `.venv\Scripts\python.exe -m pytest tests\builtin_tools\test_research_todo.py -v`
- `.venv\Scripts\python.exe -m pytest tests\builtin_tools\test_skill_loader.py -v`
- `.venv\Scripts\python.exe -m pytest tests\test_main_integration.py -v`
- `.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py`
- `.venv\Scripts\python.exe scripts\day10_smoke.py`
- 可选：`.venv\Scripts\python.exe scripts\day9_smoke.py`

## 风险与待确认项

- Day 10 smoke 依赖真实 LLM 和外部 MCP 数据链路，可能因模型不主动用 skill、arxiv/graph 抖动或限流失败；若实现正确但 smoke 抖动，需要记录具体原因。
- Claude 计划要求分 4 个 commit；除非用户明确要求提交，我先完成代码和验证，不主动提交。
