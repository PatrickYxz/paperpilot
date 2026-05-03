# Day 10 Research Todo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 PaperPilot 第 2 个 L2 内嵌 tool `research_todo`,与 `load_skill` 同层,让 LLM 多步任务里自规划 / 跟踪进度。Claude Code TodoWrite 的 PaperPilot 适配版(单 tool 整表覆写,3 状态,只校验单 in_progress)。

**Architecture:** 新建 `paperpilot/builtin_tools/research_todo.py` 实现 `TodoStore` (单进程内存) + `research_todo_tool(store)` 工厂 + `render(items)`。`main.py` 启动时实例化 `TodoStore`,把 `research_todo` 与 `load_skill` 一起塞进 tools,system prompt 末尾追加 `RESEARCH_TODO_NUDGE` 段。改 `find-classics.md` 加 step 0(研究 todo 模板)做硬触发演示。

**Tech Stack:** Python stdlib + 现有 `paperpilot.core.adapter.Tool` dataclass + pytest。

**Spec:** `docs/superpowers/specs/2026-05-03-day10-research-todo-design.md`

---

## File Structure

| 路径 | 状态 | 责任 |
|---|---|---|
| `paperpilot/builtin_tools/research_todo.py` | 新 | `TodoStore` + `research_todo_tool(store)` + `render(items)` + `RESEARCH_TODO_NUDGE` 常量 |
| `paperpilot/skills/find-classics.md` | 改 | "## 步骤" 段改写,加 step 0 调 research_todo 列模板 |
| `paperpilot/main.py` | 改 | `_build_tools` 接 `TodoStore` 实例化, `_build_system_prompt` 末尾拼 `RESEARCH_TODO_NUDGE` |
| `tests/builtin_tools/test_research_todo.py` | 新 | TodoStore / handler / 校验 / render 单测 (13 个) |
| `tests/test_main_integration.py` | 改 | 加 1 个 fast 测 (system prompt 含 nudge); slow 测扩展 1 个 assertion |
| `scripts/day10_smoke.py` | 新 | 端到端真 LLM, find-classics 路径,smoke 抓 `research_todo` ≥ 2 次 |

---

## Task 1: TodoStore + research_todo_tool + render + 单测

**Files:**
- Create: `paperpilot/builtin_tools/research_todo.py`
- Create: `tests/builtin_tools/test_research_todo.py`

- [ ] **Step 1.1: Write failing tests**

Write `tests/builtin_tools/test_research_todo.py`:

```python
"""TodoStore / research_todo_tool / render 单测。"""
from __future__ import annotations

import pytest

from paperpilot.builtin_tools.research_todo import (
    TodoStore,
    render,
    research_todo_tool,
)


def test_store_starts_empty():
    assert TodoStore().items() == []


def test_store_replace_overwrites():
    s = TodoStore()
    s.replace([{"content": "a", "status": "pending"}])
    s.replace([
        {"content": "b", "status": "in_progress"},
        {"content": "c", "status": "pending"},
    ])
    items = s.items()
    assert len(items) == 2
    assert items[0]["content"] == "b"
    assert items[1]["content"] == "c"


def test_store_items_returns_copy():
    s = TodoStore()
    s.replace([{"content": "a", "status": "pending"}])
    items = s.items()
    items.append({"content": "x", "status": "pending"})
    assert len(s.items()) == 1


def test_handler_replaces_state():
    s = TodoStore()
    tool = research_todo_tool(s)
    tool.handler({"todos": [
        {"content": "step 1", "status": "in_progress"},
        {"content": "step 2", "status": "pending"},
    ]})
    items = s.items()
    assert len(items) == 2
    assert items[0]["status"] == "in_progress"


def test_handler_returns_render_string():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "step 1", "status": "pending"},
    ]})
    assert isinstance(out, str)
    assert "## Research Todos" in out
    assert "step 1" in out


def test_handler_marks_in_progress_with_arrow():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "doing", "status": "in_progress"},
    ]})
    assert "[→]" in out
    assert "(in progress)" in out


def test_handler_marks_completed_with_x():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "done", "status": "completed"},
        {"content": "todo", "status": "pending"},
    ]})
    assert "[x] done" in out
    assert "[ ] todo" in out


def test_handler_empty_list_returns_empty_render():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": []})
    assert out == "## Research Todos (empty)"
    assert s.items() == []


def test_handler_rejects_multiple_in_progress():
    s = TodoStore()
    tool = research_todo_tool(s)
    with pytest.raises(ValueError) as exc:
        tool.handler({"todos": [
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ]})
    msg = str(exc.value)
    assert "only one in_progress" in msg
    assert "got 2" in msg


def test_handler_allows_zero_in_progress():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "a", "status": "pending"},
        {"content": "b", "status": "completed"},
    ]})
    assert "## Research Todos (2 items)" in out


def test_handler_allows_one_in_progress():
    s = TodoStore()
    tool = research_todo_tool(s)
    out = tool.handler({"todos": [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "in_progress"},
        {"content": "c", "status": "pending"},
    ]})
    assert "[x] a" in out
    assert "[→] b" in out
    assert "[ ] c" in out


def test_tool_metadata():
    s = TodoStore()
    tool = research_todo_tool(s)
    assert tool.name == "research_todo"
    assert "todos" in tool.input_schema["properties"]
    item_schema = tool.input_schema["properties"]["todos"]["items"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"]["status"]["enum"] == [
        "pending", "in_progress", "completed",
    ]


def test_handler_raises_keyerror_when_todos_missing():
    s = TodoStore()
    tool = research_todo_tool(s)
    with pytest.raises(KeyError):
        tool.handler({})
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `python -m pytest tests/builtin_tools/test_research_todo.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.builtin_tools.research_todo'`.

- [ ] **Step 1.3: Implement research_todo.py**

Write `paperpilot/builtin_tools/research_todo.py`:

```python
"""Research todo built-in tool.

L2 内嵌 tool, 与 load_skill 同层。单进程内存 state, 整表覆写语义,
3 状态 (pending / in_progress / completed), 只校验"至多 1 个 in_progress"。
"""
from __future__ import annotations

from typing import Any

from paperpilot.core.adapter import Tool


RESEARCH_TODO_NUDGE = """
## 多步任务规划
涉及多步研究 (找论文 → 检索 → 综合 / 比较多篇 paper / 跨 server 协同) 时, 先调
research_todo 列计划, 每完成一步把对应项 status 标 completed, 推进 in_progress
到下一项。任务简单 (1-2 步) 时不必用。
""".rstrip()


class TodoStore:
    def __init__(self) -> None:
        self._items: list[dict] = []

    def replace(self, todos: list[dict]) -> None:
        self._items = list(todos)

    def items(self) -> list[dict]:
        return list(self._items)


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


def research_todo_tool(store: TodoStore) -> Tool:
    def _handler(args: dict[str, Any]) -> str:
        todos = args["todos"]
        in_progress_count = sum(
            1 for t in todos if t["status"] == "in_progress"
        )
        if in_progress_count > 1:
            raise ValueError(
                f"only one in_progress allowed, got {in_progress_count}; "
                "complete or revert the others first"
            )
        store.replace(todos)
        return render(store.items())

    return Tool(
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
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                ],
                            },
                        },
                        "required": ["content", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["todos"],
        },
        handler=_handler,
    )
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `python -m pytest tests/builtin_tools/test_research_todo.py -v`

Expected: PASS, 13 tests passed.

- [ ] **Step 1.5: Commit**

```bash
git add paperpilot/builtin_tools/research_todo.py tests/builtin_tools/test_research_todo.py
git commit -m "Day 10 Task 1: TodoStore + research_todo_tool + render + 13 单测"
```

---

## Task 2: 改 find-classics.md 加 step 0

**Files:**
- Modify: `paperpilot/skills/find-classics.md`

- [ ] **Step 2.1: Replace "## 步骤" 段**

旧内容(行 14-21):
```markdown
## 步骤
1. 找当代 paper: 调 `mcp__arxiv__search_papers(query="领域关键词", max_results=5)`。
2. 从搜索结果文本中提取 3-5 个有代表性的 `arxiv_id`。
3. 构建引用图: 调 `mcp__graph__build_graph(arxiv_ids=[提取出的 arxiv_id])`。
   - 如果返回 `missing`,用剩下的 id 继续,不要中断。
4. 求共引: 调 `mcp__graph__get_common_citations(arxiv_ids=[同一批可用 id], top_k=10)`。
   - 返回 `cited_by_count >= 2` 的共同引用文献。
5. 综合回答: 按 `cited_by_count`、年份和主题相关性列出 top 文献,解释为什么它们可能是该 cluster 的基础工作。
```

替换为:
```markdown
## 步骤
0. 先列计划: 调 `research_todo(todos=[
     {"content":"找该领域 5 篇代表性当代 paper","status":"in_progress"},
     {"content":"提取 arxiv_id 并构建引用图","status":"pending"},
     {"content":"求共引找经典文献","status":"pending"},
     {"content":"综合按 cited_by_count + year 排序","status":"pending"},
   ])`。
1. 找当代 paper: 调 `mcp__arxiv__search_papers(query="领域关键词", max_results=5)`。
2. 从搜索结果文本中提取 3-5 个有代表性的 `arxiv_id`。
3. 构建引用图: 调 `mcp__graph__build_graph(arxiv_ids=[提取出的 arxiv_id])`。
   - 如果返回 `missing`,用剩下的 id 继续,不要中断。
4. 求共引: 调 `mcp__graph__get_common_citations(arxiv_ids=[同一批可用 id], top_k=10)`。
   - 返回 `cited_by_count >= 2` 的共同引用文献。
5. 综合回答: 按 `cited_by_count`、年份和主题相关性列出 top 文献,解释为什么它们可能是该 cluster 的基础工作。

每完成一步, 调 `research_todo` 把该项 `status` 改 `completed`, 把下一项 `status` 改 `in_progress`。
```

用 Edit tool 精确替换"## 步骤"那一段(包括所有 5 个原步骤),不要触碰 `## 适用场景` / `## 注意` 段以及 frontmatter。

- [ ] **Step 2.2: Verify skill still parses**

Run:
```bash
python -c "from pathlib import Path; from paperpilot.builtin_tools.skill_loader import SkillRegistry; r = SkillRegistry(Path('paperpilot/skills')); body = r.load('find-classics'); assert 'research_todo' in body; assert 'step 0' not in body.lower() or '0. 先列计划' in body; print('OK')"
```

Expected stdout: `OK`(skill 仍能正常 load,且 body 含 `research_todo` 字符串)。

- [ ] **Step 2.3: Run skill_loader tests to verify no regression**

Run: `python -m pytest tests/builtin_tools/test_skill_loader.py -v`

Expected: PASS, 13 tests passed (Day 9 全绿,与 find-classics 内容无关)。

- [ ] **Step 2.4: Commit**

```bash
git add paperpilot/skills/find-classics.md
git commit -m "Day 10 Task 2: find-classics 加 step 0 (research_todo 列计划)"
```

---

## Task 3: main.py 集成 + 集成测扩展

**Files:**
- Modify: `paperpilot/main.py`
- Modify: `tests/test_main_integration.py`

- [ ] **Step 3.1: Write failing fast integration test**

在 `tests/test_main_integration.py` 末尾追加(保留原 2 个测不动):

```python
def test_build_system_prompt_includes_research_todo_nudge():
    prompt = _build_system_prompt()
    assert "## 多步任务规划" in prompt
    assert "research_todo" in prompt
    # Day 9 已有断言保持成立 (这里只看新增段, 不重复测)
```

同时把已有 slow 测从 `test_build_tools_contains_load_skill_and_mcp_tools` 改名加 assertion(保持 `@pytest.mark.slow`):

```python
@pytest.mark.slow
def test_build_tools_contains_load_skill_research_todo_and_mcp_tools():
    tools, mcp = _build_tools()
    try:
        names = [tool.name for tool in tools]
        assert "load_skill" in names
        assert "research_todo" in names
        assert any(name.startswith("mcp__") for name in names)
    finally:
        mcp.close()
```

(用 Edit tool 把旧 slow 测整段替换。原 fast 测 `test_build_system_prompt_includes_skills` 一个字不改。)

- [ ] **Step 3.2: Run new fast test to verify it fails**

Run: `python -m pytest tests/test_main_integration.py::test_build_system_prompt_includes_research_todo_nudge -v`

Expected: FAIL with `AssertionError: assert '## 多步任务规划' in prompt`(因为 main 还没拼 `RESEARCH_TODO_NUDGE`)。

- [ ] **Step 3.3: Modify main.py**

读现状: `paperpilot/main.py` 顶部 import 和现有结构如下(Day 9 已固化):

```python
from paperpilot.builtin_tools.skill_loader import (
    SkillRegistry,
    load_skill_tool,
    render_skill_section,
)
```

把 import 段扩展为:

```python
from paperpilot.builtin_tools.research_todo import (
    RESEARCH_TODO_NUDGE,
    TodoStore,
    research_todo_tool,
)
from paperpilot.builtin_tools.skill_loader import (
    SkillRegistry,
    load_skill_tool,
    render_skill_section,
)
```

把 `_build_system_prompt` 从:

```python
def _build_system_prompt(registry: SkillRegistry | None = None) -> str:
    registry = registry or SkillRegistry(SKILLS_DIR)
    return SYSTEM_PROMPT_BASE + render_skill_section(registry.list_metadata())
```

改成:

```python
def _build_system_prompt(registry: SkillRegistry | None = None) -> str:
    registry = registry or SkillRegistry(SKILLS_DIR)
    return (
        SYSTEM_PROMPT_BASE
        + render_skill_section(registry.list_metadata())
        + "\n\n"
        + RESEARCH_TODO_NUDGE
    )
```

把 `_build_tools` 从:

```python
def _build_tools(
    registry: SkillRegistry | None = None,
) -> tuple[list[Tool], MCPClient]:
    """Return (tools, mcp_client); caller is responsible for close()."""
    registry = registry or SkillRegistry(SKILLS_DIR)

    mcp = MCPClient(MANIFEST_PATH)
    try:
        mcp.start()
        tools: list[Tool] = [load_skill_tool(registry), *mcp.list_tools()]
        return tools, mcp
    except Exception:
        mcp.close()
        raise
```

改成:

```python
def _build_tools(
    registry: SkillRegistry | None = None,
    todo_store: TodoStore | None = None,
) -> tuple[list[Tool], MCPClient]:
    """Return (tools, mcp_client); caller is responsible for close()."""
    registry = registry or SkillRegistry(SKILLS_DIR)
    todo_store = todo_store or TodoStore()

    mcp = MCPClient(MANIFEST_PATH)
    try:
        mcp.start()
        tools: list[Tool] = [
            load_skill_tool(registry),
            research_todo_tool(todo_store),
            *mcp.list_tools(),
        ]
        return tools, mcp
    except Exception:
        mcp.close()
        raise
```

把 `run` 从:

```python
def run(query: str, *, max_iter: int = 8, on_event=None) -> list[dict]:
    """Run one complete agent conversation and return final messages."""
    load_dotenv()

    registry = SkillRegistry(SKILLS_DIR)
    tools, mcp = _build_tools(registry)
    try:
        messages = [{"role": "user", "content": query}]
        return agent_loop(
            messages,
            system=_build_system_prompt(registry),
            ...
```

改成(把 `TodoStore()` 实例化拉到顶,统一传给 `_build_tools`):

```python
def run(query: str, *, max_iter: int = 8, on_event=None) -> list[dict]:
    """Run one complete agent conversation and return final messages."""
    load_dotenv()

    registry = SkillRegistry(SKILLS_DIR)
    todo_store = TodoStore()
    tools, mcp = _build_tools(registry, todo_store)
    try:
        messages = [{"role": "user", "content": query}]
        return agent_loop(
            messages,
            system=_build_system_prompt(registry),
            ...
```

(`run` 函数其它部分一字不改。)

- [ ] **Step 3.4: Run new fast test to verify it passes**

Run: `python -m pytest tests/test_main_integration.py::test_build_system_prompt_includes_research_todo_nudge -v`

Expected: PASS.

- [ ] **Step 3.5: Run all fast tests in test_main_integration.py**

Run: `python -m pytest tests/test_main_integration.py -v`

Expected: 2 passed, 1 deselected(原 fast 测 `test_build_system_prompt_includes_skills` + 新加 fast 测都 PASS;1 个 slow 测改名后 deselected)。

- [ ] **Step 3.6: Run renamed slow test to verify**

Run: `python -m pytest tests/test_main_integration.py::test_build_tools_contains_load_skill_research_todo_and_mcp_tools -v -m slow`

Expected: PASS(真启 mcp servers, ~5s)。

- [ ] **Step 3.7: Run full default test suite to check for regressions**

Run: `python -m pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py`

Expected: 71 passed, 5 deselected
(原 57 fast + Task 1 新 13 fast + Task 3 新 1 fast = 71 fast;slow 5 个不变。)

- [ ] **Step 3.8: Commit**

```bash
git add paperpilot/main.py tests/test_main_integration.py
git commit -m "Day 10 Task 3: main 集成 research_todo + nudge 注入 system prompt"
```

---

## Task 4: scripts/day10_smoke.py 端到端真 LLM

**Files:**
- Create: `scripts/day10_smoke.py`

- [ ] **Step 4.1: Write day10_smoke.py**

Write `scripts/day10_smoke.py`:

```python
"""Day 10 smoke: main loop -> load_skill(find-classics) -> research_todo + arxiv + graph -> answer.

prompt 不显式说"先 load_skill / 先 research_todo", 让 LLM 自决:
  - 自己识别这是入门类多步问题 -> load_skill("find-classics")
  - 看到 step 0 -> research_todo 列 4 项
  - 每完成一步 -> 再调 research_todo 推进 status
若 LLM 未触发 research_todo 至少 2 次, smoke fail (机制没起作用)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

EXPECT_TOOLS = {
    "load_skill",
    "research_todo",
    "mcp__arxiv__search_papers",
    "mcp__graph__build_graph",
    "mcp__graph__get_common_citations",
}


def main() -> None:
    saw: set[str] = set()
    research_todo_calls = 0
    load_skill_targets: list[str] = []

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        nonlocal research_todo_calls
        if kind == "tool_call":
            name = payload["name"]
            saw.add(name)
            if name == "research_todo":
                research_todo_calls += 1
            elif name == "load_skill":
                load_skill_targets.append(
                    payload.get("arguments", {}).get("name", "")
                )
            print(f"  -> {name}({_preview(name, payload.get('arguments', {}))})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            preview = (
                content[:160] if isinstance(content, str) else str(content)[:160]
            )
            print(f"  <- {payload['name']}: {preview}...")
        elif kind == "guardrail_stop":
            print(f"  !! guardrail: {payload['reason']}")
        elif kind == "tool_arg_repair":
            print(f"  ~ repaired {payload['name']}: {payload['repaired']}")

    messages = run(
        "我想入门 retrieval-augmented generation 领域, 帮我找出该领域被反复"
        "引用的几篇必读经典。",
        max_iter=12,
        on_event=tracer,
    )

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for block in last:
            if hasattr(block, "text"):
                print(block.text)
    else:
        print(last)

    missing = EXPECT_TOOLS - saw
    assert not missing, f"FAIL: expected tools missing: {missing}; saw {saw}"
    assert research_todo_calls >= 2, (
        f"FAIL: research_todo only called {research_todo_calls} times; "
        "expected >= 2 (initial plan + at least 1 progress update)"
    )
    assert "find-classics" in load_skill_targets, (
        f"FAIL: load_skill never called with name='find-classics'; "
        f"saw load_skill targets: {load_skill_targets}"
    )
    print("\nDay 10 smoke PASSED")


def _preview(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "research_todo":
        todos = args.get("todos", [])
        return {
            "todos_count": len(todos),
            "in_progress": [
                t.get("content", "")[:30]
                for t in todos
                if t.get("status") == "in_progress"
            ],
            "completed_count": sum(
                1 for t in todos if t.get("status") == "completed"
            ),
        }
    if "documents" in args:
        docs = args.get("documents") or []
        preview_docs = []
        for doc in docs:
            if isinstance(doc, dict):
                preview_docs.append({
                    "paper_id": doc.get("paper_id"),
                    "text_len": len(doc.get("text", "")),
                })
        return {**args, "documents": preview_docs}
    return args


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run smoke (real LLM + real MCP servers)**

Run: `python scripts/day10_smoke.py`

Expected:
- tracer 抓到 5 个 tool name(`load_skill` / `research_todo` / `mcp__arxiv__search_papers` / `mcp__graph__build_graph` / `mcp__graph__get_common_citations`)
- `research_todo` 被调 ≥ 2 次(初始列计划 + 至少 1 次推进 status)
- `load_skill` 调用至少含 `name="find-classics"`
- 退出码 0,stdout 末尾 `Day 10 smoke PASSED`
- 最终回答含按 cited_by_count + year 排序的经典文献列表

如果失败可能原因 + 调法:
- LLM 没主动 `load_skill("find-classics")`:
  - 检查 `_build_system_prompt()` 输出含 `## 可用 skill` 段含 find-classics 行
  - 调 prompt: 把 "我想入门..." 加 "用你工具箱里能找到的经典文献流程"
- LLM 没主动 `research_todo`:
  - 检查 system prompt 末尾真拼了 `## 多步任务规划`
  - 检查 find-classics body 含 "0. 先列计划: research_todo"
  - 必要时把 nudge 文案改强(从 "建议" → "应该")
- `research_todo_calls < 2`(只列了一次没推进):
  - find-classics 收尾说明 "每完成一步, 调 research_todo 把该项 status 改 completed" 是否够明显
  - max_iter 拉到 14
- arxiv / graph 限流: 等几分钟重跑

- [ ] **Step 4.3: Verify regression on prior smokes (optional)**

```bash
python scripts/day9_smoke.py
```

Expected: `Day 9 smoke PASSED`(deep-read-paper 链路不动)。

(若 arxiv 429 / SS 429 等外部抖动,可重跑或记录但不阻塞 commit;day10 通过即视为完工。)

- [ ] **Step 4.4: Commit**

```bash
git add scripts/day10_smoke.py
git commit -m "Day 10 Task 4: day10_smoke 端到端 (load_skill find-classics + research_todo + graph)"
```

---

## Definition of Done

1. `python -m pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py` 全绿(原 57 + Task 1 新 13 + Task 3 新 1 fast = 71 个;slow 5 个 deselected,Task 3 只是给已有 slow 测加 1 个 assertion 不增加 slow 测数)
2. `python scripts/day9_smoke.py` 无回归(可选,arxiv 抖动允许跳过)
3. `python scripts/day10_smoke.py` 退出 0 + 打印 `Day 10 smoke PASSED`
4. day10 smoke tracer 必须满足:
   - 5 个 expected tool name 全抓到
   - `research_todo` 被调 ≥ 2 次
   - `load_skill` 调用至少含 `name="find-classics"`
5. `git grep -E "TODO|FIXME" paperpilot/builtin_tools/research_todo.py paperpilot/skills/find-classics.md` 空
6. 4 个 commit:Task 1 / Task 2 / Task 3 / Task 4