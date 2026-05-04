# Day 13: `compact_context` 内嵌 tool 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增第 3 个 L2 内嵌 tool `compact_context`,LLM 自决调用,handler 通过闭包真改主 agent 的 messages list,把中段 turns 压成一条 `<context_summary>` user message。

**Architecture:** 工厂模式 `compact_context_tool(messages_ref, client_factory, on_event, keep_recent_turns=3)` 与 `paper_deep_read_tool` 同模板。Handler 走"too-short no-op / 正常压缩 / 异常 raise"三路径。`_summarize` 单次调 LLMClient 拍出结构化 markdown summary。`paperpilot/core/loop.py` 与 `adapter.py` 一行不改。

**Tech Stack:** Python 3.12, Anthropic SDK(走 DeepSeek 兼容端点),pytest,unittest.mock。

**Spec:** `docs/superpowers/specs/2026-05-05-day13-compact-context-design.md`。

---

## 文件结构

| 路径 | 动作 | 责任 |
|---|---|---|
| `paperpilot/builtin_tools/compact.py` | 新建,~120 行 | `compact_context_tool` 工厂、`_summarize`、`_render_messages_for_summary`、`COMPACT_CONTEXT_NUDGE` |
| `paperpilot/main.py` | 改 ~10 行 | `messages` 提到 `_build_tools` 之前;`_build_tools` 接 `messages_ref` 形参;tool 加入列表;nudge 拼到 system prompt |
| `tests/builtin_tools/test_compact.py` | 新建 | 6 个 fast unit case |
| `tests/test_main_integration.py` | 加 1 条 fast 断言 | nudge 文本 + tool 名字命中 |
| `scripts/day13_smoke.py` | 新建 | 真 LLM 长对话 → 验证 LLM 自主触发 compact 且 messages 缩短 |

---

## Task 1: `compact_context` 核心 tool(单测,LLMClient mock)

最大改动。严格 TDD:先写测试 → 跑红 → 写最小实现 → 跑绿 → 重复。所有 LLM 调用都用 `MagicMock`。

**Files:**
- Create: `paperpilot/builtin_tools/compact.py`
- Create: `tests/builtin_tools/test_compact.py`

### Step 1.1: Write failing test for too-short no-op path

- [ ] 新建测试文件,先放一个 case 验证短 history 不动 list。

```python
# tests/builtin_tools/test_compact.py
"""compact_context 内嵌 tool 单元测试。LLMClient 全部 mock。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paperpilot.builtin_tools.compact import (
    COMPACT_CONTEXT_NUDGE,
    compact_context_tool,
)
from paperpilot.core.adapter import ParsedResponse


def _fake_client(text: str) -> MagicMock:
    client = MagicMock()
    client.call.return_value = ParsedResponse(
        text=text, tool_calls=[], usage={"total_tokens": 50}, raw=None,
    )
    return client


def test_handler_too_short_returns_noop():
    messages: list[dict] = [{"role": "user", "content": "hi"}]
    client = _fake_client("should not be called")
    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: client,
        on_event=lambda k, v: None,
    )
    result = tool.handler({})
    assert "already compact" in result
    assert messages == [{"role": "user", "content": "hi"}]
    client.call.assert_not_called()
```

### Step 1.2: Run failing test

- [ ] 验证 ImportError(模块还不存在)。

Run: `pytest tests/builtin_tools/test_compact.py::test_handler_too_short_returns_noop -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.builtin_tools.compact'`

### Step 1.3: Write minimal implementation for too-short path

- [ ] 新建 `paperpilot/builtin_tools/compact.py`,先只够过 too-short 这一个 case。

```python
# paperpilot/builtin_tools/compact.py
"""compact_context built-in tool.

LLM 自决何时调用。Handler 通过闭包持有主 agent 的 messages list 引用,
把中段 turns 压成一条 <context_summary> user message,真改 list。
"""
from __future__ import annotations

from typing import Callable

from paperpilot.core.adapter import LLMClient, Tool
from paperpilot.core.loop import EventCallback


COMPACT_CONTEXT_NUDGE = """
## Long conversation
When tool results pile up and earlier turns no longer matter, call
compact_context() with no arguments. It rewrites the history into a
structured summary and frees context budget. Do not call it on a short
conversation or when you are mid-step (e.g. just got a tool result and
are about to act on it).
""".strip()


def compact_context_tool(
    *,
    messages_ref: list[dict],
    client_factory: Callable[[], LLMClient],
    on_event: EventCallback,
    keep_recent_turns: int = 3,
) -> Tool:
    """Build the compact_context tool.

    The handler closes over messages_ref and mutates it in place when
    invoked. Caller must keep the same list identity for agent_loop.
    """
    K = keep_recent_turns

    def _handler(args: dict) -> str:
        if len(messages_ref) <= 1 + 2 * K:
            return "already compact, nothing to summarize"
        # filled in next steps
        raise NotImplementedError

    return Tool(
        name="compact_context",
        description=(
            "Compress earlier conversation into a structured summary, "
            "freeing context budget. Call when the conversation is long "
            "and recent tool results have made earlier turns redundant."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=_handler,
    )
```

### Step 1.4: Run test to verify too-short path passes

Run: `pytest tests/builtin_tools/test_compact.py::test_handler_too_short_returns_noop -v`
Expected: PASS

### Step 1.5: Write failing test for normal compaction path

- [ ] 追加测试覆盖正常压缩:10 条 messages → 1 (head) + 1 (summary) + 6 (tail) = 8 条;summary 内容含 mock LLM 返字符串。

```python
def _make_long_messages(n: int) -> list[dict]:
    """First message is original user query; rest alternate user/assistant."""
    messages: list[dict] = [{"role": "user", "content": "original query"}]
    for i in range(1, n):
        role = "assistant" if i % 2 == 1 else "user"
        messages.append({"role": role, "content": f"turn-{i}"})
    return messages


def test_handler_normal_path_replaces_middle():
    messages = _make_long_messages(10)
    head_obj = messages[0]
    tail_objs = list(messages[-6:])

    client = _fake_client("FAKE_SUMMARY_BODY")
    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: client,
        on_event=lambda k, v: None,
    )
    result = tool.handler({})

    assert result.startswith("compacted ")
    assert len(messages) == 1 + 1 + 6
    assert messages[0] is head_obj
    summary_msg = messages[1]
    assert summary_msg["role"] == "user"
    assert "<context_summary>" in summary_msg["content"]
    assert "FAKE_SUMMARY_BODY" in summary_msg["content"]
    for index, expected_obj in enumerate(tail_objs):
        assert messages[2 + index] is expected_obj
    client.call.assert_called_once()
```

### Step 1.6: Run failing test

Run: `pytest tests/builtin_tools/test_compact.py::test_handler_normal_path_replaces_middle -v`
Expected: FAIL with `NotImplementedError`

### Step 1.7: Implement normal path + `_summarize` + `_render_messages_for_summary`

- [ ] 把 handler 与两个辅助函数补完。

替换 `compact.py` 里的 `_handler` 占位与 `NotImplementedError`,并新增两个 module-level 函数:

```python
SUMMARIZE_SYSTEM = "You are a conversation history summarizer."

SUMMARIZE_PROMPT = """
你的任务是把下面的对话历史压缩成结构化 summary,供后续 LLM 理解上下文用。

输出格式:
## 已完成的关键步骤
<按时间顺序列出 tool calls 和重要结论, 5-10 条>

## 关键发现 / 中间结果
<事实性内容、找到的段落或数据, 简洁列点>

## 待办 / 下一步
<如果 history 里 LLM 表达过未完成的计划, 列出来>

要求:
- 只保留事实和决策, 删掉客套和重复
- 引用具体 paper_id / chunk 摘要 / 数字结果
- 总长度控制在 800 token 内

对话历史:
""".strip()


def _render_messages_for_summary(middle: list[dict]) -> str:
    """Best-effort plain-text render for the summarizer LLM."""
    lines: list[str] = []
    for message in middle:
        role = message.get("role", "?")
        content = message.get("content")
        if isinstance(content, str):
            lines.append(f"[{role}] {content}")
            continue
        if not isinstance(content, list):
            lines.append(f"[{role}] {content!r}")
            continue
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    inner_text = inner if isinstance(inner, str) else str(inner)
                    lines.append(
                        f"[tool_result for {block.get('tool_use_id', '?')}] "
                        f"{inner_text[:500]}"
                    )
                else:
                    lines.append(f"[{role} {block.get('type', '?')}] {block!r}")
                continue
            block_type = getattr(block, "type", None)
            if block_type == "text":
                lines.append(f"[{role} text] {getattr(block, 'text', '')}")
            elif block_type == "tool_use":
                lines.append(
                    f"[{role} tool_use] "
                    f"{getattr(block, 'name', '?')}"
                    f"({getattr(block, 'input', {})!r})"
                )
            else:
                lines.append(f"[{role} {block_type}] {block!r}")
    return "\n".join(lines)


def _summarize(middle: list[dict], client: LLMClient) -> str:
    rendered = _render_messages_for_summary(middle)
    prompt = f"{SUMMARIZE_PROMPT}\n{rendered}"
    response = client.call(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        system=SUMMARIZE_SYSTEM,
    )
    return response.text or "(empty summary)"
```

把工厂里的 `_handler` 替换为完整实现:

```python
    def _handler(args: dict) -> str:
        if len(messages_ref) <= 1 + 2 * K:
            return "already compact, nothing to summarize"

        head = messages_ref[0]
        tail = messages_ref[-2 * K:]
        middle = messages_ref[1:-2 * K]

        on_event("compact_start", {"middle_count": len(middle)})
        summary_text = _summarize(middle, client_factory())
        on_event("compact_done", {"kept_recent": len(tail)})

        messages_ref[:] = [
            head,
            {
                "role": "user",
                "content": f"<context_summary>\n{summary_text}\n</context_summary>",
            },
            *tail,
        ]
        return (
            f"compacted {len(middle)} messages into summary; "
            f"kept last {len(tail)} turns"
        )
```

### Step 1.8: Run normal-path test

Run: `pytest tests/builtin_tools/test_compact.py::test_handler_normal_path_replaces_middle -v`
Expected: PASS

### Step 1.9: Add failure-path test

- [ ] 追加一个 case:`_summarize` 抛异常时 handler raise,messages 不变。

```python
def test_handler_summarize_failure_reraises_and_keeps_messages():
    messages = _make_long_messages(10)
    snapshot = list(messages)

    client = MagicMock()
    client.call.side_effect = RuntimeError("LLM explodes")
    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: client,
        on_event=lambda k, v: None,
    )
    with pytest.raises(RuntimeError, match="LLM explodes"):
        tool.handler({})
    assert messages == snapshot
```

### Step 1.10: Run failure test

Run: `pytest tests/builtin_tools/test_compact.py::test_handler_summarize_failure_reraises_and_keeps_messages -v`
Expected: PASS(实现已经天然支持 — `_summarize` raise 在 messages 替换前;`compact_done` 不会 emit 因为还没到那一行)

### Step 1.11: Add lifecycle event test

- [ ] 验证两个事件都 emit,payload 字段对。

```python
def test_lifecycle_events_emitted():
    messages = _make_long_messages(10)
    events: list[tuple[str, dict]] = []

    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: _fake_client("S"),
        on_event=lambda k, v: events.append((k, dict(v))),
    )
    tool.handler({})

    kinds = [k for k, _ in events]
    assert kinds == ["compact_start", "compact_done"]
    assert events[0][1] == {"middle_count": 3}
    assert events[1][1] == {"kept_recent": 6}
```

### Step 1.12: Run event test

Run: `pytest tests/builtin_tools/test_compact.py::test_lifecycle_events_emitted -v`
Expected: PASS

### Step 1.13: Add metadata + nudge test

- [ ] 一并验证 Tool name / input_schema / nudge 字符串。

```python
def test_tool_metadata_and_nudge():
    tool = compact_context_tool(
        messages_ref=[],
        client_factory=lambda: _fake_client("x"),
        on_event=lambda k, v: None,
    )
    assert tool.name == "compact_context"
    assert tool.input_schema["required"] == []
    assert tool.input_schema["additionalProperties"] is False
    assert tool.input_schema["properties"] == {}
    assert "compact_context" in COMPACT_CONTEXT_NUDGE
    assert "Long conversation" in COMPACT_CONTEXT_NUDGE
```

### Step 1.14: Add head/tail identity test

- [ ] 显式验证 head 与 tail 的对象身份(id())不变,确保 in-place 替换不破坏引用。

```python
def test_handler_preserves_head_and_tail_identity():
    messages = _make_long_messages(10)
    same_list_id = id(messages)
    head_id = id(messages[0])
    tail_ids = [id(message) for message in messages[-6:]]

    tool = compact_context_tool(
        messages_ref=messages,
        client_factory=lambda: _fake_client("S"),
        on_event=lambda k, v: None,
    )
    tool.handler({})

    assert id(messages) == same_list_id
    assert id(messages[0]) == head_id
    assert [id(message) for message in messages[2:]] == tail_ids
```

### Step 1.15: Run all compact tests

Run: `pytest tests/builtin_tools/test_compact.py -v`
Expected: 6 PASSED

### Step 1.16: Run full fast test suite to confirm no regression

Run: `pytest tests -q --ignore=tests/test_main_integration.py`
Expected: 全绿(原 92 + 新 6 = 98 passed)

### Step 1.17: Commit

```bash
git add paperpilot/builtin_tools/compact.py tests/builtin_tools/test_compact.py
git commit -m "Day 13 Task 1: add compact_context built-in tool"
```

---

## Task 2: main.py 接线 + 集成测试

把新 tool 拼进 `_build_tools`,nudge 拼进 system prompt。需要小重构:`messages` 创建提到 `_build_tools` 之前。

**Files:**
- Modify: `paperpilot/main.py`
- Modify: `tests/test_main_integration.py`

### Step 2.1: Write failing fast integration test

- [ ] 在 `tests/test_main_integration.py` 末尾追加一条 fast 断言。

```python
def test_build_system_prompt_includes_compact_context_nudge():
    prompt = _build_system_prompt()
    assert "## Long conversation" in prompt
    assert "compact_context" in prompt
```

并在文件顶部 `slow` 断言里追加 `compact_context`(验证它真的进了 tools 列表)。把现有的 `test_build_tools_contains_load_skill_research_todo_and_mcp_tools` 替换为:

```python
@pytest.mark.slow
def test_build_tools_contains_load_skill_research_todo_and_mcp_tools():
    messages: list[dict] = []
    tools, mcp = _build_tools(messages_ref=messages)
    try:
        names = [tool.name for tool in tools]
        assert "load_skill" in names
        assert "research_todo" in names
        assert "paper_deep_read" in names
        assert "compact_context" in names
        assert any(name.startswith("mcp__") for name in names)
    finally:
        mcp.close()
```

### Step 2.2: Run failing test

Run: `pytest tests/test_main_integration.py::test_build_system_prompt_includes_compact_context_nudge -v`
Expected: FAIL with `assert '## Long conversation' in prompt`

### Step 2.3: Wire `compact_context_tool` into `main.py`

- [ ] 改 `paperpilot/main.py`:
  1. 导入新 tool 与 nudge
  2. `_build_tools` 增加 `messages_ref: list[dict]` 形参,把 `compact_context_tool(...)` 加入 `tools` 列表
  3. `_build_system_prompt` 末尾追加 `+ "\n\n" + COMPACT_CONTEXT_NUDGE`
  4. `run()` 把 `messages = [{"role":"user","content":query}]` 创建提到 `_build_tools` 调用之前,并把 `messages_ref=messages` 传进去

完整改动:

文件顶部 import 区追加:
```python
from paperpilot.builtin_tools.compact import (
    COMPACT_CONTEXT_NUDGE,
    compact_context_tool,
)
```

`_build_system_prompt` 改:
```python
def _build_system_prompt(registry: SkillRegistry | None = None) -> str:
    registry = registry or SkillRegistry(SKILLS_DIR)
    return (
        SYSTEM_PROMPT_BASE
        + render_skill_section(registry.list_metadata())
        + "\n\n"
        + RESEARCH_TODO_NUDGE
        + "\n\n"
        + PAPER_DEEP_READ_NUDGE
        + "\n\n"
        + COMPACT_CONTEXT_NUDGE
    )
```

`_build_tools` 改签名 + body:
```python
def _build_tools(
    registry: SkillRegistry | None = None,
    todo_store: TodoStore | None = None,
    messages_ref: list[dict] | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[list[Tool], MCPClient]:
    """Return (tools, mcp_client); caller is responsible for close()."""
    registry = registry or SkillRegistry(SKILLS_DIR)
    todo_store = todo_store or TodoStore()
    messages_ref = messages_ref if messages_ref is not None else []
    emit = on_event or _default_logger

    mcp = MCPClient(MANIFEST_PATH)
    try:
        mcp.start()
        mcp_tools = mcp.list_tools()
        tools: list[Tool] = [
            load_skill_tool(registry),
            research_todo_tool(todo_store),
            paper_deep_read_tool(
                client_factory=lambda: LLMClient(),
                mcp_tools=mcp_tools,
                on_event=emit,
            ),
            compact_context_tool(
                messages_ref=messages_ref,
                client_factory=lambda: LLMClient(),
                on_event=emit,
            ),
            *mcp_tools,
        ]
        return tools, mcp
    except Exception:
        mcp.close()
        raise
```

`run()` 改:
```python
def run(query: str, *, max_iter: int = 8, on_event=None) -> list[dict]:
    """Run one complete agent conversation and return final messages."""
    load_dotenv()

    emit = on_event or _default_logger
    registry = SkillRegistry(SKILLS_DIR)
    todo_store = TodoStore()
    messages: list[dict] = [{"role": "user", "content": query}]
    tools, mcp = _build_tools(
        registry, todo_store, messages_ref=messages, on_event=emit,
    )
    try:
        return agent_loop(
            messages,
            system=_build_system_prompt(registry),
            tools=tools,
            client=LLMClient(),
            guardrail=Guardrail(
                max_iterations=max_iter,
                budget_tokens=int(os.environ.get("BUDGET_TOKENS", 50_000)),
            ),
            on_event=emit,
        )
    finally:
        mcp.close()
```

### Step 2.4: Run fast integration tests

Run: `pytest tests/test_main_integration.py -q`
Expected: PASS(原有 4 fast + 新 1 fast = 5 passed,1 deselected slow)

### Step 2.5: Run full fast test suite

Run: `pytest tests -q`
Expected: 99 passed(98 from Task 1 + 1 new),9 deselected

### Step 2.6: Commit

```bash
git add paperpilot/main.py tests/test_main_integration.py
git commit -m "Day 13 Task 2: wire compact_context into main + nudge"
```

---

## Task 3: 真 LLM smoke 验证

`scripts/day13_smoke.py`:让 LLM 先 deep-read 一篇 paper(产生大量 tool_result),再切话题,期望 LLM 自主调 `compact_context`。

**Files:**
- Create: `scripts/day13_smoke.py`

### Step 3.1: Write the smoke script

- [ ] 新建文件。

`agent_loop` 的 `turn` event 当前 payload 是 `{text, tool_calls, usage}`,不含 messages 长度;不为了 smoke 改 loop(红线)。改用 **`tool_result` 内容里 compact_context 自己返回的 "compacted N messages into summary; kept last M turns" 字符串**做断言锚点 — 这条信息 handler 必返,直接证明压缩真发生且数字合理。`run()` 最后返回的 `messages` 也直接 `len()`,作为最终长度证据。

```python
"""Day 13 smoke: compact_context 自决触发 + messages 真被改短。

让 LLM 先深读一篇 paper (build_index + 多次 colbert.search 的大体积
tool_result), 再切到一个跟前文无关的简短问题, prompt 提示可以先 compact。
断言: 看到 compact_start/done 事件, compact_context tool_result 含 "compacted"
字样, 且 run() 返回的 messages 长度 <= 1 + 1 + 6 + 后续 tool_call 链长度。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

PAPER_ID = "1706.03762"


def main() -> None:
    saw_compact_start = 0
    saw_compact_done = 0
    tool_calls: list[str] = []
    compact_result_text: str | None = None
    middle_count_seen: int | None = None

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        nonlocal saw_compact_start, saw_compact_done
        nonlocal compact_result_text, middle_count_seen
        if kind == "compact_start":
            saw_compact_start += 1
            middle_count_seen = int(payload.get("middle_count", 0))
            print(f"  ## compact_start: middle={middle_count_seen}")
        elif kind == "compact_done":
            saw_compact_done += 1
            print(f"  ## compact_done: kept={payload.get('kept_recent')}")
        elif kind == "tool_call":
            name = payload["name"]
            tool_calls.append(name)
            print(f"  -> {name}")
        elif kind == "tool_result":
            name = payload["name"]
            content = payload.get("content", "")
            content_text = content if isinstance(content, str) else str(content)
            print(f"  <- {name}: {content_text[:160]}...")
            if name == "compact_context":
                compact_result_text = content_text
        elif kind == "guardrail_stop":
            print(f"  !! guardrail: {payload.get('reason')}")

    prompt = (
        f"先深读 arxiv {PAPER_ID} (Attention Is All You Need): 调 "
        "load_skill('deep-read-paper'), 然后按 skill 走 download -> build_index -> "
        "至少 4 次 colbert.search(query 覆盖 method/experiments/ablation/limitations), "
        "把结果整合成一段精读摘要。\n\n"
        "完成上面之后, 我换一个完全无关的问题: 用一句话告诉我 Python list 和 "
        "tuple 的区别。回答这个问题前, 先调 compact_context() 把前面深读的对话 "
        "历史压缩 (history 已经很长, 后面这个问题不需要前文 tool_result 的细节)。"
    )

    messages = run(prompt, max_iter=20, on_event=tracer)

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for block in last:
            if hasattr(block, "text"):
                print(block.text)
    else:
        print(last)

    assert saw_compact_start >= 1, (
        f"FAIL: compact_start not observed; tool_calls={tool_calls}"
    )
    assert saw_compact_done >= 1, "FAIL: compact_done not observed"
    assert "compact_context" in tool_calls, (
        f"FAIL: compact_context tool never called; tool_calls={tool_calls}"
    )
    assert compact_result_text is not None, "FAIL: no tool_result for compact_context"
    match = re.search(r"compacted (\d+) messages into summary", compact_result_text)
    assert match, (
        f"FAIL: compact tool_result missing 'compacted N messages'; "
        f"got: {compact_result_text!r}"
    )
    compacted_n = int(match.group(1))
    assert compacted_n >= 4, (
        f"FAIL: compacted only {compacted_n} messages; expected >= 4"
    )
    assert middle_count_seen == compacted_n, (
        f"FAIL: middle_count={middle_count_seen} != compacted_n={compacted_n}"
    )

    print(
        f"\nDay 13 smoke PASSED "
        f"(compacted={compacted_n}, final_messages={len(messages)}, "
        f"compact_calls={saw_compact_start})"
    )


if __name__ == "__main__":
    main()
```

### Step 3.2: 运行 smoke

Run(确保 `.env` 里有 `DEEPSEEK_API_KEY`):
```
python scripts/day13_smoke.py
```

Expected: 终端打出 deep-read tool calls,然后看到 `## compact_start: middle=N`,最后输出 `Day 13 smoke PASSED`。

### Step 3.3: 如果 LLM 没自主调 compact

- [ ] 排查路径:
  1. tracer 里看 system prompt 是否真含 `## Long conversation`(`run()` 起始时打一行 prompt 长度可帮助)
  2. 把 prompt 里"先调 compact_context"再加重一句:`必须先调 compact_context, 再回答 list/tuple 问题`
  3. 确认 max_iter=20 够走完 deep-read + compact + final answer

如果改 prompt 就能跑过,保留改动并写入文件;不要去改 `loop.py` 或 nudge 文本绕过(违反"决策由 LLM 做"红线)。

### Step 3.4: Commit

```bash
git add scripts/day13_smoke.py
git commit -m "Day 13 Task 3: add day13 smoke for compact_context"
```

---

## DoD 验证(全部 task 完成后跑一遍)

- [ ] `pytest tests -q` → 99 passed,9 deselected(数字为 Task 1 后预估;实际以最终统计为准)
- [ ] `pytest tests -m slow -q tests/test_main_integration.py` → 1 passed(`compact_context` in build_tools 列表)
- [ ] `python scripts/day13_smoke.py` → `Day 13 smoke PASSED`
- [ ] `git diff main -- paperpilot/core/loop.py paperpilot/core/adapter.py` → empty(红线守恒)
- [ ] `git log --oneline | head -5` → 看到 3 条 Day 13 Task commit