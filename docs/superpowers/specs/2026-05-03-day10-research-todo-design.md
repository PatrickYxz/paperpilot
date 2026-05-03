# PaperPilot Day 10 设计:`research_todo` 内嵌 tool + find-classics 集成

| 项 | 值 |
|---|---|
| 日期 | 2026-05-03 (Day 9 skill loading 完工后, 设计 Day 10 research_todo) |
| 范围 | (1) 新增 L2 内嵌 tool `research_todo`, 与 `load_skill` 同层; (2) 新增 `paperpilot/builtin_tools/research_todo.py` 实现 `TodoStore`; (3) base system prompt 末尾拼"## 多步任务规划" nudge; (4) 改 `find-classics.md` 加 step 0 显式触发 todo 流程; (5) 单测 + day10 smoke |
| 不在范围 | todo 持久化 (跨 CLI invocation); 优先级 / due / tag / id 字段; 自动同步 todo 到 system prompt; 多 todo store / 嵌套; 改 deep-read-paper / explore-citations skill |
| 状态 | Draft, 待用户 review |

---

## 1. 目标

让 LLM 在多步研究任务里自规划并跟踪进度, 真正闭环 PaperPilot 第二个 L2 内嵌 tool:

1. base system prompt 提示 LLM "多步任务先 research_todo 列计划"
2. `find-classics` skill 显式硬触发: step 0 给 LLM 完整 4 项 todo 模板
3. LLM 每完成一步, 调 `research_todo` 把对应项 status 改 completed, 推进下一项 in_progress
4. tool_result 返新 list 渲染的 markdown checkbox, 自然进 conversation history

这是 PaperPilot 第二个 L2 内嵌 tool, 与 `load_skill` 共同构成"agent 自规划自决策"的核心机制。

同时严守方案 C / Day 5-9 锁定的红线:

- **L1 基座层 + Day 5 mcp_client 零改动** —— 内嵌 tool 与 MCP tool 在 agent_loop 里同等对待
- **server 之间不互通信** —— `research_todo` 不知道任何 mcp server / skill 存在
- **决策由 LLM 做** —— nudge 只说"建议", `find-classics` step 0 只是 prose 模板, LLM 仍可不按模板填写
- **不做推测性抽象** —— 不留持久化 / 优先级 / id / 多 store 等口子
- **复用 Day 7 trip wire** —— `research_todo` tool_result 是 str, 不撞 FastMCP list[dict] 监控

---

## 2. 关键设计决策 (Q1-Q5)

| ID | 决策 | 选项 | 主要理由 |
|---|---|---|---|
| Q1 | tool 接口形态 | **单 tool 整表覆写** (vs 多细粒度 tool / 单 tool + action 字段) | 与 Claude Code TodoWrite 完全对齐; 实现最简; 一次完成"加 + 改"; tool 数不膨胀, 不稀释 attention |
| Q2 | item 字段 | **`{content, status}`, status ∈ {pending, in_progress, completed}** (vs CC 原汤含 activeForm / 砍 in_progress 的简化版) | 砍 activeForm (CC 给 UI 用, CLI 无意义); 留 in_progress (LLM 多轮里能定位"现在卡哪步"); 与 CC 数据契约偏离最小, 简历可解释为"针对 CLI 做减法保留核心 3 状态" |
| Q3 | 状态可见性 | **唯一通路: tool_result** (vs system prompt 每轮重注入 / 加 `read_todos` tool) | 守 "loop.py 一行不改" 红线; DeepSeek 在 Day 9 已证能可靠串 3-4 轮 + 引用; tool_result 渲染只 5-10 行, attention 抓得住; 真挂再升级 |
| Q4 | 触发机制 | **base nudge + find-classics 硬触发组合** (vs 零提示 / 单 nudge / 单 skill 集成) | 单 nudge LLM 可能不听, smoke 抓不到; 单 skill 集成又局限到一个场景; 组合方案 demo 可控且作通用 L2 tool 仍灵活 |
| Q5 | 验证严格度 | **只校验 "至多 1 个 in_progress"** (vs 零校验 / 全校验 content+status 单调) | 单 in_progress 是 CC TodoWrite 设计哲学核心 ("一次只专注一件事"); 其它规则不破坏 demo, 严校验易触发 LLM retry 循环; A 完全不校验丢掉最有讲解价值的规则 |

### 隐含决策 (已锁)

| 项 | 值 | 备注 |
|---|---|---|
| 内嵌 tool 模块路径 | `paperpilot/builtin_tools/research_todo.py` | 与 `skill_loader.py` 平级 |
| 状态作用域 | 单 CLI 进程内存 | 一次 `python -m paperpilot.main --query "..."` = 一个 `TodoStore` 实例; 进程退出销毁 |
| state holder | `TodoStore` 类 | 与 `SkillRegistry` 同模式: 由 main 实例化, 传给 handler factory |
| 整表覆写语义 | 漏传 = 删除 | 不维护 id / 不去重 / 不 merge |
| 空 list 处理 | 不报错, render 返 "empty" | 允许 "全完成清空" / "放弃清空" |
| 改哪些 skill | 只改 `find-classics` | `deep-read-paper` / `explore-citations` 都是 1 step skill, 加 todo 噪声化 |
| `additionalProperties` | False (item 层) | 阻止 LLM 误传 `activeForm` / `priority` / `id` |
| status 校验 | JSON Schema enum (LLM 端) + handler 单 in_progress (后端) | 不重复校验 status 拼写 |

---

## 3. 架构

```
┌──────────────────────────────────────────────────────────────────┐
│  main.py 启动                                                    │
│    1. SkillRegistry(...)         # Day 9                         │
│    2. TodoStore()                 # Day 10 新, 持有 list[dict]   │
│    3. mcp = MCPClient(...).start()                              │
│    4. tools = [load_skill_tool(reg),                            │
│                research_todo_tool(store),    # Day 10 新        │
│                *mcp.list_tools()]                                │
│    5. system = SYSTEM_PROMPT_BASE                                │
│                + render_skill_section(reg.list_metadata())       │
│                + RESEARCH_TODO_NUDGE      # Day 10 新, ~1 段     │
│    6. agent_loop(messages, system, tools, ...)                  │
└──────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  agent_loop 运行时 (loop.py 一行不改)                            │
│    LLM 看 system 知道有 research_todo                           │
│      → 调 research_todo(todos=[{content,status},...])           │
│      → handler 校验 "至多 1 个 in_progress"                      │
│      → store.replace(todos)                                      │
│      → tool_result = render(items)   # markdown checkbox list    │
│    后续轮次 LLM 看 conversation history 知道当前进度,            │
│    再调 research_todo 标 completed / 推进 in_progress / 加新项   │
└──────────────────────────────────────────────────────────────────┘
```

**关键边界:**

- `TodoStore` 与 `SkillRegistry` 平级, `paperpilot/builtin_tools/research_todo.py` 单文件实现 store + handler factory + render
- `loop.py` / `adapter.py` / `mcp_client.py` 一行不改 — 守 Day 5-9 红线
- `research_todo` 是 L2 内嵌 tool (Python 函数 handler), **不走 MCP 协议** —— 与 `load_skill` 同层
- 内嵌 tool 与 MCP tool 在 agent_loop 里**同等对待** (统一走 `Tool` dataclass + handler dispatch)
- 一次 `python -m paperpilot.main --query "..."` = 一个 `TodoStore` 实例; CLI 进程退出销毁
- `find-classics` skill 改一行 (step 0 加 research_todo 列计划) — 唯一一个 skill 改动

---

## 4. 文件布局

### 新增 / 改动

| 路径 | 状态 | 责任 | 预估行数 |
|---|---|---|---|
| `paperpilot/builtin_tools/research_todo.py` | 新 | `TodoStore` + `research_todo_tool(store)` 工厂 + `render(items)` | ~70 |
| `paperpilot/builtin_tools/__init__.py` | 改 (可选) | re-export | +2 |
| `paperpilot/skills/find-classics.md` | 改 | step 0 加 research_todo 列计划; 收尾说明 "每完成一步推进 status" | +6 |
| `paperpilot/main.py` | 改 | 实例化 `TodoStore`, 合并 tool, system prompt 末尾拼 `RESEARCH_TODO_NUDGE` | +12 |
| `tests/builtin_tools/test_research_todo.py` | 新 | TodoStore / handler / 校验 / render 单测 | ~120 |
| `tests/test_main_integration.py` | 改 | 加 1 个 fast 测: system prompt 含 nudge + tools 含 `research_todo` | +8 |
| `scripts/day10_smoke.py` | 新 | 端到端真 LLM: find-classics 路径, smoke 抓 `research_todo` ≥ 2 次 | ~70 |

### 不动

- `paperpilot/core/loop.py` / `adapter.py` / `guardrail.py` — 一行不改
- `paperpilot/tools/mcp_client.py` — 一行不改 (Day 9 codex 加的 `_resolve_command` 已稳)
- `paperpilot/builtin_tools/skill_loader.py` — 一行不改
- 现有 mcp_servers (arxiv/colbert/graph) — 一行不改
- 现有 `deep-read-paper.md` / `explore-citations.md` — 一行不改

---

## 5. 数据契约

### todo item shape

```python
{
    "content": str,    # 必填, 非空, 描述任务 (例: "搜索 RAG 领域代表性论文")
    "status": str,     # 必填, enum: "pending" | "in_progress" | "completed"
}
```

### `research_todo` tool schema

```python
Tool(
    name="research_todo",
    description=(
        "维护多步研究任务清单。整表覆写语义: 每次调用必须传完整新 list, "
        "后端会替换之前的 list。同一时刻至多 1 个 in_progress。"
        "适用: 用户问题需要 3+ 步骤 (找论文 → 构图 → 求共引 → 综合) 时, "
        "先用此 tool 列出计划再执行; 每完成一步把对应项 status 改 completed。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "minLength": 1},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content", "status"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["todos"],
    },
    handler=...,
)
```

### handler 行为

```python
def _handler(args: dict) -> str:
    todos = args["todos"]
    in_progress_count = sum(1 for t in todos if t["status"] == "in_progress")
    if in_progress_count > 1:
        raise ValueError(
            f"only one in_progress allowed, got {in_progress_count}; "
            "complete or revert the others first"
        )
    store.replace(todos)
    return render(store.items())
```

### `TodoStore` 接口

```python
class TodoStore:
    def __init__(self):
        self._items: list[dict] = []

    def replace(self, todos: list[dict]) -> None:
        self._items = list(todos)

    def items(self) -> list[dict]:
        return list(self._items)
```

### `render` 函数 (tool_result 字符串)

```python
def render(items: list[dict]) -> str:
    if not items:
        return "## Research Todos (empty)"
    lines = [f"## Research Todos ({len(items)} items)", ""]
    for it in items:
        if it["status"] == "completed":
            mark = "[x]"
            suffix = ""
        elif it["status"] == "in_progress":
            mark = "[→]"
            suffix = " (in progress)"
        else:
            mark = "[ ]"
            suffix = ""
        lines.append(f"- {mark} {it['content']}{suffix}")
    return "\n".join(lines)
```

例:

```
## Research Todos (3 items)

- [x] 找出 diffusion 领域 5 篇代表性论文
- [→] 构建引用图并跑共引 (in progress)
- [ ] 把 top-10 共引按 cited_by_count 排序综合
```

### `RESEARCH_TODO_NUDGE` (拼到 system prompt)

```
## 多步任务规划
涉及多步研究 (找论文 → 检索 → 综合 / 比较多篇 paper / 跨 server 协同) 时, 先调
research_todo 列计划, 每完成一步把对应项 status 标 completed, 推进 in_progress
到下一项。任务简单 (1-2 步) 时不必用。
```

---

## 6. find-classics.md 改动

**只改 "## 步骤" 段, 其它段落不动。**

改后:

```
## 步骤
0. 先列计划: research_todo(todos=[
     {"content":"找该领域 5 篇代表性当代 paper","status":"in_progress"},
     {"content":"把这批 paper 拉进引用图","status":"pending"},
     {"content":"求共引找经典","status":"pending"},
     {"content":"综合按 cited_by_count + year 排序","status":"pending"},
   ])
1. 找当代 paper: mcp__arxiv__search_papers(query="该领域关键词", max_results=5)
2. 拉进引用图: mcp__graph__build_graph(arxiv_ids=[...])
3. 求共引: mcp__graph__get_common_citations(arxiv_ids=[...], top_k=10)
4. 综合回答按 cited_by_count + year 排序

每完成一步, 调 research_todo 把该项 status 改 completed, 把下一项 status 改 in_progress。
```

---

## 7. 错误处理

按 Day 5/9 红线分两类:

### A. 启动期 hard-fail

无。`TodoStore()` 是纯内存对象, 无 IO / 无解析。

### B. 运行时 soft-fail (handler raise → agent_loop 转 is_error=true → LLM 决策)

| 失败 | 触发 | LLM 看到 |
|---|---|---|
| 多个 in_progress | LLM 一次标 2 个 in_progress | `ValueError("only one in_progress allowed, got 2; complete or revert the others first")` → "我标错了, 改成只 1 个" |
| 缺 `todos` / `content` / `status` | LLM 漏字段 | JSON Schema 不通过, anthropic SDK / DeepSeek 端报错 → agent_loop 转 is_error → LLM 补字段 |
| `status` 不在 enum | LLM 写 "doing" | 同上 (JSON Schema enum 校验) |
| 空 `content` | minLength=1 | 同上 |
| 空 `todos: []` | LLM 显式清空 | **不报错**, render 返 "empty" 字符串 |

**全部不做:**
- 自动 fuzzy 修复 status 拼写
- 自动 fallback / retry
- 自动加 `id` 字段 (无去重需求)

---

## 8. 测试策略

### 层 1: 单元测试 `tests/builtin_tools/test_research_todo.py` (秒级, CI 跑)

| 测试 | 测什么 |
|---|---|
| `test_store_starts_empty` | `TodoStore().items() == []` |
| `test_store_replace_overwrites` | 连续两次 `replace`, 第 2 次完全覆盖 |
| `test_store_items_returns_copy` | mutate 返回值不影响内部状态 |
| `test_handler_replaces_state` | handler 调用后 `store.items()` 反映新 list |
| `test_handler_returns_render_string` | tool_result 是 str, 含 `## Research Todos`, 含每项 content |
| `test_handler_marks_in_progress_with_arrow` | render 里 in_progress 项含 `[→]` |
| `test_handler_marks_completed_with_x` | completed 项含 `[x]`, pending 含 `[ ]` |
| `test_handler_empty_list_returns_empty_render` | 空 list → `## Research Todos (empty)` |
| `test_handler_rejects_multiple_in_progress` | 2 个 in_progress 抛 `ValueError`, message 含 "only one in_progress" + "got 2" |
| `test_handler_allows_zero_in_progress` | 全 pending / 全 completed 都 OK |
| `test_handler_allows_one_in_progress` | 恰好 1 个 in_progress, 正常返回 |
| `test_tool_metadata` | `name == "research_todo"`; input_schema item 层含 `additionalProperties: False` |
| `test_handler_raises_keyerror_when_todos_missing` | `args = {}` 时 `args["todos"]` 抛 KeyError (由 agent_loop 转 is_error) |

### 层 2: main.py 集成 `tests/test_main_integration.py` 加 1 个 fast 测

```python
def test_build_system_prompt_includes_research_todo_nudge():
    prompt = _build_system_prompt()
    assert "## 多步任务规划" in prompt
    assert "research_todo" in prompt
    # Day 9 已有断言不动: ## 可用 skill / deep-read-paper 等

# slow 测扩展: tools 里多 1 个 name
def test_build_tools_contains_load_skill_and_research_todo_and_mcp_tools():
    tools, mcp = _build_tools()
    try:
        names = [t.name for t in tools]
        assert "load_skill" in names
        assert "research_todo" in names    # 新加
        assert any(n.startswith("mcp__") for n in names)
    finally:
        mcp.close()
```

### 层 3: Day 10 smoke `scripts/day10_smoke.py` (端到端真 LLM + 真 MCP)

```
prompt: "我想入门 retrieval-augmented generation 领域, 帮我找出该领域被反复
        引用的几篇必读经典。"

期望 LLM 行为:
  1. 自己判断是多步任务 → load_skill(name="find-classics")
  2. 看到 skill step 0 → research_todo(todos=[4 项, 第 1 项 in_progress])
  3. 调 mcp__arxiv__search_papers
  4. 调 research_todo 把第 1 项标 completed, 第 2 项 in_progress
  5. 调 mcp__graph__build_graph
  6. 调 research_todo 推进
  7. 调 mcp__graph__get_common_citations
  8. 调 research_todo 推进
  9. 综合回答, 末轮可选: research_todo 全部 completed

assertions:
  EXPECT_TOOLS = {
    "load_skill", "research_todo",
    "mcp__arxiv__search_papers",
    "mcp__graph__build_graph",
    "mcp__graph__get_common_citations",
  }
  assert all of EXPECT_TOOLS in saw
  assert research_todo_call_count >= 2     # 至少列计划 + 推进 1 次
  assert load_skill called with name="find-classics" 至少 1 次
```

smoke prompt **不**显式说 "先 research_todo / 先 load_skill" — 让 LLM 自决, 这是 Day 10 机制存在意义的验证。

**编码** (沿用 Day 5/6/8/9 模式): smoke 顶部 `sys.stdout.reconfigure(encoding="utf-8")`。

---

## 9. 工作量预估 + 完工标志

| 阶段 | 预估 |
|---|---|
| 写 plan | ~25 min |
| Task 1: TodoStore + handler + render + 13 单测 | ~50 min |
| Task 2: 改 find-classics skill | ~10 min |
| Task 3: main.py 集成 (实例化 store + 拼 nudge + 集成测扩展) | ~30 min |
| Task 4: scripts/day10_smoke.py + 真 LLM 联调 | ~45 min |
| **合计** | **~2.5 小时** (预算 6h, 留 ~3.5h buffer 给 LLM 不主动用 todo 时调 nudge / find-classics 文案) |

**完工标志 (Definition of Done)**:

1. `pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py` 全绿 (原 57 + Task 1 新 13 + Task 3 新 1 fast = 71 个; slow 5 个 deselected, Task 3 只是给已有 slow 测加 1 个 assertion 不增加 slow 测数)
2. `python scripts/day9_smoke.py` 无回归 (deep-read 链路仍通)
3. `python scripts/day10_smoke.py` 退出 0 + stdout 末尾 `Day 10 smoke PASSED`
4. day10 smoke tracer 必须抓到 `research_todo` 至少 2 次 + `load_skill(name="find-classics")` 至少 1 次
5. `git grep -E "TODO|FIXME" paperpilot/builtin_tools/research_todo.py paperpilot/skills/find-classics.md` 空
6. 4 个 commit: Task 1 / Task 2 / Task 3 / Task 4

---

## 10. trip wire (Day 7 监控复用)

`research_todo` tool_result 是 **str (markdown 文本)**, 不是 list[dict] —— 不撞 Day 7 锁的 FastMCP list[dict] 监控。

新增以下监控条件:

| 触发条件 | 行动 |
|---|---|
| 同 session 内 LLM 反复传同一 list (无变化) > 3 次 | LLM 没在推进, 可能 nudge 太弱 / find-classics step 0 太死板 |
| 单次传入 todos 项数 > 10 | LLM 把任务列得太细, 检查 find-classics step 0 是否需要更宽口径 |
| 多个 in_progress 错误重试 > 3 次 | LLM 没消化 ValueError; 调 description 文案更明确 |

第一版**不写**监控脚本 —— 上述条件作为"未来某天数据量级跳变"的提醒钉子。

---

## 11. 不在范围 / 推迟到未来的事

| 项 | 推迟到何时触发 |
|---|---|
| todo 持久化 (跨 CLI invocation 续跑) | 真出现 "多 session 接力" 需求时 |
| todo 优先级 / due date / tag | 真演示场景需要时 (YAGNI) |
| 自动同步 todo 到 system prompt (Q3 的 B 方案) | demo 显示 LLM 长 session 漂移忘标 completed 时 |
| 多 todo store (嵌套 / parent-child) | 永远不做 — TodoWrite 哲学就是扁平 |
| 加 `id` 字段去重 | 永远不做 — content 直接当 key, 去重无业务价值 |
| 改 deep-read-paper / explore-citations skill 加 todo | 这两个都是 1 step skill, 加 todo 噪声化; 真出现 demo 需要再说 |
| `pdf-parse-mcp` (第 4 个 MCP server) | Day 11+ |
| `paper_deep_read` subagent | Week 3 |
| `compact_context` 内嵌 tool | 长 PDF 真撞 token 上限时 |
| `vlm-mcp` + Qwen-VL | Week 3 |

---

## 附: 与已有 spec / 架构的衔接点

- **复用 Day 4 agent_loop** —— 内嵌 tool 与 mcp tool 走同一 `Tool` dataclass + handler dispatch, agent_loop 一行不改
- **复用 Day 5 mcp_client 启动** —— `research_todo` 注册不影响 mcp_client; main.py 启动顺序: SkillRegistry.scan() → TodoStore() → mcp_client.start() → tools 合并 → agent_loop
- **复用 Day 6 build_index 参数 repair** —— 不冲突; find-classics 不走 colbert
- **复用 Day 7 trip wire** —— `research_todo` 不是 list[dict] 输出, 不撞 FastMCP 监控
- **复用 Day 8 graph-mcp** —— find-classics step 1-3 直接 prose 调 graph 3 个 tool
- **复用 Day 9 skill loading** —— `research_todo` 与 `load_skill` 同层 L2, 共同构成 "agent 自规划 + 自决策" 核心机制; find-classics skill 改动验证两个内嵌 tool 协同
