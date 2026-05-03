# Day 11 paper_deep_read Subagent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 PaperPilot 第 3 个 L2 内嵌 tool `paper_deep_read`,与 `load_skill` / `research_todo` 同层。主 agent 调用时,handler 通过 `ThreadPoolExecutor` 并发 spawn N 个独立 sync `agent_loop` (每个有独立 context / Guardrail / LLMClient),最后把每篇 markdown 摘要拼成 markdown tool_result 返主 agent。Claude Code 四引擎 (skill / todo / **subagent** / compact) 第三个落地。

**Architecture:** 新建 `paperpilot/builtin_tools/subagent.py`,内含 `paper_deep_read_tool(client_factory, mcp_tools, on_event)` 工厂 + `_run_one` worker + `_extract_last_text` + `_filter_subagent_tools` + `SUBAGENT_SYSTEM` + `PAPER_DEEP_READ_NUDGE` + `_render_results`。复用现有 sync `agent_loop` / `Guardrail`,**`loop.py` / `adapter.py` / `mcp_client.py` 一行不改**。`main.py` `_build_tools` 签名扩 `on_event`,`run()` 把 `on_event` 透传进去。

**Tech Stack:** Python stdlib (`concurrent.futures.ThreadPoolExecutor`, `threading.Lock`) + 现有 `paperpilot.core.adapter.Tool` / `LLMClient` / `Guardrail` + pytest。

**Spec:** `docs/superpowers/specs/2026-05-04-day11-paper-deep-read-design.md`

---

## File Structure

| 路径 | 状态 | 责任 |
|---|---|---|
| `paperpilot/builtin_tools/subagent.py` | 新 | `paper_deep_read_tool(client_factory, mcp_tools, on_event)` + `_run_one` + `_extract_last_text` + `_filter_subagent_tools` + `_render_results` + `SUBAGENT_SYSTEM` + `PAPER_DEEP_READ_NUDGE` + `MAX_PAPERS` / `SUBAGENT_MAX_ITER` / `THREAD_POOL_SIZE` 常量 |
| `paperpilot/main.py` | 改 | `_build_tools` 签名加 `on_event`; `run()` 把 `on_event` 透传; `_build_system_prompt` 末尾追加 `PAPER_DEEP_READ_NUDGE` |
| `tests/builtin_tools/test_subagent.py` | 新 | 16 单测 (filter / extract / run_one / handler / event / metadata) |
| `tests/test_main_integration.py` | 改 | 加 1 fast 测 (system prompt 含 `PAPER_DEEP_READ_NUDGE`); slow 测加 1 行 assertion |
| `scripts/day11_smoke.py` | 新 | 端到端真 LLM + 真 MCP, 直接 prompt 让 LLM 调 paper_deep_read 并发精读 3 篇 |

---

## Task 1: subagent.py + 16 单测

**Files:**
- Create: `paperpilot/builtin_tools/subagent.py`
- Create: `tests/builtin_tools/test_subagent.py`

- [ ] **Step 1.1: Write failing tests**

Write `tests/builtin_tools/test_subagent.py`:

```python
"""subagent (paper_deep_read) 单测。

mock LLMClient + 简化 Tool 验证: filter / extract / run_one / handler 聚合 /
event 透传 / metadata。所有测试纯内存, 不触发真 MCP / 真网络。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import pytest

from paperpilot.core.adapter import ParsedResponse, Tool, ToolCall
from paperpilot.builtin_tools.subagent import (
    MAX_PAPERS,
    PAPER_DEEP_READ_NUDGE,
    SUBAGENT_MAX_ITER,
    SUBAGENT_SYSTEM,
    _extract_last_text,
    _filter_subagent_tools,
    _run_one,
    paper_deep_read_tool,
)


# ---------- helpers ----------

class FakeClient:
    """Mock LLMClient 接受 responder(messages) -> ParsedResponse | raises。"""

    def __init__(self, responder: Callable[[list[dict]], ParsedResponse]):
        self._responder = responder
        self.call_count = 0

    def call(self, messages, tools, *, system):
        self.call_count += 1
        return self._responder(messages)

    def append_assistant_turn(self, messages, response):
        messages.append(
            {"role": "assistant", "content": response.text or ""}
        )

    def append_tool_results(self, messages, results):
        messages.append({
            "role": "user",
            "content": [
                {
                    "tool_use_id": r.id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in results
            ],
        })


def _text_response(text: str) -> ParsedResponse:
    return ParsedResponse(
        text=text, tool_calls=[], usage={"total_tokens": 50}, raw=None,
    )


def _tool_call_response(name: str, args: dict | None = None) -> ParsedResponse:
    return ParsedResponse(
        text=None,
        tool_calls=[ToolCall(id="tc-1", name=name, arguments=args or {})],
        usage={"total_tokens": 50},
        raw=None,
    )


def _stub_tool(name: str, return_value: str = "ok") -> Tool:
    return Tool(
        name=name, description="stub", input_schema={},
        handler=lambda args: return_value,
    )


# ---------- _filter_subagent_tools ----------

def test_filter_subagent_tools_keeps_three():
    tools = [
        _stub_tool("mcp__arxiv__search_papers"),
        _stub_tool("mcp__arxiv__download_paper"),
        _stub_tool("mcp__colbert__build_index"),
        _stub_tool("mcp__colbert__search"),
        _stub_tool("mcp__graph__build_graph"),
        _stub_tool("mcp__graph__get_common_citations"),
    ]
    kept = _filter_subagent_tools(tools)
    names = [t.name for t in kept]
    assert names == [
        "mcp__arxiv__download_paper",
        "mcp__colbert__build_index",
        "mcp__colbert__search",
    ]


def test_filter_subagent_tools_drops_paper_deep_read():
    tools = [
        _stub_tool("mcp__arxiv__download_paper"),
        _stub_tool("mcp__colbert__search"),
        _stub_tool("paper_deep_read"),
    ]
    kept = _filter_subagent_tools(tools)
    assert "paper_deep_read" not in [t.name for t in kept]


# ---------- _extract_last_text ----------

class _AnthropicTextBlock:
    """模拟 anthropic SDK content block。"""
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _AnthropicToolUseBlock:
    def __init__(self):
        self.type = "tool_use"


def test_extract_last_text_from_anthropic_blocks():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [
            _AnthropicTextBlock("Section A"),
            _AnthropicToolUseBlock(),
            _AnthropicTextBlock("Section B"),
        ]},
    ]
    assert _extract_last_text(messages) == "Section A\nSection B"


def test_extract_last_text_from_string_content():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "summary text"},
    ]
    assert _extract_last_text(messages) == "summary text"


def test_extract_last_text_returns_none_when_no_assistant():
    messages = [{"role": "user", "content": "hi"}]
    assert _extract_last_text(messages) is None


# ---------- handler 输入校验 ----------

def test_handler_rejects_zero_papers():
    tool = paper_deep_read_tool(
        client_factory=lambda: FakeClient(lambda m: _text_response("x")),
        mcp_tools=[], on_event=lambda k, v: None,
    )
    with pytest.raises(ValueError) as exc:
        tool.handler({"paper_ids": [], "user_query": "q"})
    assert "1.." in str(exc.value)


def test_handler_rejects_too_many_papers():
    tool = paper_deep_read_tool(
        client_factory=lambda: FakeClient(lambda m: _text_response("x")),
        mcp_tools=[], on_event=lambda k, v: None,
    )
    too_many = [f"p{i}" for i in range(MAX_PAPERS + 1)]
    with pytest.raises(ValueError) as exc:
        tool.handler({"paper_ids": too_many, "user_query": "q"})
    msg = str(exc.value)
    assert f"1..{MAX_PAPERS}" in msg
    assert f"got {MAX_PAPERS + 1}" in msg


# ---------- _run_one ----------

def test_run_one_returns_ok_status_on_success():
    summary = "## Core Method\nFoo\n## Key Findings\nBar\n## Relevance to Query\nBaz"
    client = FakeClient(lambda m: _text_response(summary))
    result = _run_one(
        "1706.03762", "what is attention",
        client_factory=lambda: client,
        tools=[],
        on_event=lambda k, v: None,
        max_iter=4,
    )
    assert result["paper_id"] == "1706.03762"
    assert result["status"] == "ok"
    assert "Core Method" in result["summary"]


def test_run_one_returns_max_iter_when_guardrail_stops():
    # 不停发 tool_call -> guardrail max_iter 撞顶
    client = FakeClient(lambda m: _tool_call_response("noop"))
    result = _run_one(
        "1706.03762", "q",
        client_factory=lambda: client,
        tools=[_stub_tool("noop")],
        on_event=lambda k, v: None,
        max_iter=2,
    )
    assert result["paper_id"] == "1706.03762"
    assert result["status"] == "max_iter_reached"


def test_run_one_returns_error_status_on_exception():
    def bad(messages):
        raise RuntimeError("boom")
    client = FakeClient(bad)
    result = _run_one(
        "1706.03762", "q",
        client_factory=lambda: client,
        tools=[],
        on_event=lambda k, v: None,
        max_iter=4,
    )
    assert result["paper_id"] == "1706.03762"
    assert result["status"].startswith("error: RuntimeError")
    assert "boom" in result["status"]


# ---------- handler 聚合 ----------

def test_handler_aggregates_three_papers():
    def factory():
        # 每个子 agent 都返回 1 轮 text
        return FakeClient(lambda m: _text_response("done"))

    tool = paper_deep_read_tool(
        client_factory=factory,
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    out = tool.handler({
        "paper_ids": ["A", "B", "C"],
        "user_query": "q",
    })
    assert isinstance(out, str)
    assert "Paper Deep Read Results (3 papers)" in out
    for pid in ("A", "B", "C"):
        assert f"### {pid}" in out


def test_handler_one_paper_failed_others_ok():
    def factory():
        # 工厂返回 client; client.call 根据 messages 第一条决定行为
        def responder(messages):
            user_msg = messages[0]["content"]
            if "B" in user_msg:
                raise RuntimeError("paper B explodes")
            return _text_response("ok done")
        return FakeClient(responder)

    tool = paper_deep_read_tool(
        client_factory=factory,
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    out = tool.handler({
        "paper_ids": ["A", "B", "C"],
        "user_query": "q",
    })
    # 三段都在 (失败的也有段, 写明 status)
    assert "### A (status: ok)" in out
    assert "### B (status: error" in out
    assert "### C (status: ok)" in out


def test_handler_preserves_input_order():
    """并发完成顺序无关, 输出按输入顺序排列。"""
    barrier = {"A": 0.03, "B": 0.0, "C": 0.015}

    def factory():
        def responder(messages):
            user_msg = messages[0]["content"]
            for pid, delay in barrier.items():
                if f" {pid}," in user_msg or f" {pid} " in user_msg:
                    time.sleep(delay)
                    return _text_response(f"summary for {pid}")
            return _text_response("?")
        return FakeClient(responder)

    tool = paper_deep_read_tool(
        client_factory=factory,
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    out = tool.handler({
        "paper_ids": ["A", "B", "C"],
        "user_query": "q",
    })
    idx_a = out.index("### A")
    idx_b = out.index("### B")
    idx_c = out.index("### C")
    assert idx_a < idx_b < idx_c


# ---------- event aggregation ----------

def test_event_aggregation_prefixes_paper_id():
    received: list[tuple[str, dict]] = []
    received_lock = threading.Lock()

    def main_emit(kind: str, payload: dict) -> None:
        with received_lock:
            received.append((kind, dict(payload)))

    def factory():
        return FakeClient(lambda m: _text_response("done"))

    tool = paper_deep_read_tool(
        client_factory=factory,
        mcp_tools=[],
        on_event=main_emit,
    )
    tool.handler({
        "paper_ids": ["A", "B"],
        "user_query": "q",
    })
    # 至少要看到 turn 事件且都带 subagent_paper_id
    turn_events = [p for k, p in received if k == "turn"]
    assert len(turn_events) >= 2
    for p in turn_events:
        assert p.get("subagent_paper_id") in {"A", "B"}


# ---------- render ----------

def test_handler_renders_markdown_segments_per_paper():
    def factory():
        return FakeClient(lambda m: _text_response("body content"))

    tool = paper_deep_read_tool(
        client_factory=factory,
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    out = tool.handler({
        "paper_ids": ["1706.03762"],
        "user_query": "q",
    })
    # 顶部标题 + paper 段标题 + body
    assert out.startswith("## Paper Deep Read Results (1 papers)")
    assert "### 1706.03762 (status: ok)" in out
    assert "body content" in out


# ---------- metadata ----------

def test_tool_metadata():
    tool = paper_deep_read_tool(
        client_factory=lambda: FakeClient(lambda m: _text_response("x")),
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    assert tool.name == "paper_deep_read"
    schema = tool.input_schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["paper_ids", "user_query"]
    paper_ids_schema = schema["properties"]["paper_ids"]
    assert paper_ids_schema["minItems"] == 1
    assert paper_ids_schema["maxItems"] == MAX_PAPERS
    assert paper_ids_schema["items"]["minLength"] == 1
    # 常量也校验下不被误改
    assert MAX_PAPERS == 8
    assert SUBAGENT_MAX_ITER == 8
    assert "## 多论文并发精读" in PAPER_DEEP_READ_NUDGE
    assert "Core Method" in SUBAGENT_SYSTEM
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `python -m pytest tests/builtin_tools/test_subagent.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.builtin_tools.subagent'`.

- [ ] **Step 1.3: Implement subagent.py**

Write `paperpilot/builtin_tools/subagent.py`:

```python
"""paper_deep_read built-in tool (L2 subagent layer).

Spawns N independent sync `agent_loop` invocations on a `ThreadPoolExecutor`,
each with its own conversation history / Guardrail / LLMClient. Sub-agent
events are forwarded to the main `on_event` with a `subagent_paper_id` field
under a `threading.Lock` so trace lines do not interleave.

Mirrors Claude Code s04 subagent semantics: independent context, structured
summary returned to parent. Concurrency is via threads (not asyncio) because
the existing `agent_loop` is sync and LLM HTTP calls are IO-bound.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from paperpilot.core.adapter import LLMClient, Tool
from paperpilot.core.guardrail import Guardrail
from paperpilot.core.loop import EventCallback, agent_loop


MAX_PAPERS = 8
SUBAGENT_MAX_ITER = 8
SUBAGENT_BUDGET_TOKENS = 20_000
THREAD_POOL_SIZE = 8

SUBAGENT_TOOL_SUFFIXES = (
    "__download_paper",
    "__build_index",
    "__search",
)


PAPER_DEEP_READ_NUDGE = """
## 多论文并发精读
当用户问题需要对比/综合 3-8 篇论文时, 调 paper_deep_read(paper_ids=[...],
user_query="...") 一次性并发精读, 每篇会由独立子 agent 处理并回传 markdown 摘要。
单篇深读 (用 deep-read-paper skill) / 不到 3 篇 / 只看摘要够时不必用。
""".rstrip()


SUBAGENT_SYSTEM = """你是论文精读子 agent。任务: 围绕用户问题, 精读指定的一篇论文, 提取核心方法 / 关键实验结果 / 与问题的相关性。

工作流程 (建议):
  1. mcp__arxiv__download_paper(arxiv_id="<paper_id>") 拿全文
  2. mcp__colbert__build_index(documents=[download_paper 返回值]) 建索引
  3. 多次 mcp__colbert__search(query="...", paper_ids=["<paper_id>"]) 查关键概念 / 方法 / 实验
  4. 综合 search 结果写最终摘要

最终输出格式 (markdown, ~500 token):
  ## Core Method
  <一段, 论文方法的核心要点>

  ## Key Findings
  <一段, 关键实验结果 / 数据 / 结论>

  ## Relevance to Query
  <一段, 与用户问题的关联点>

约束:
  - 只处理这一篇论文; 不要 download / search 其它 paper_id
  - 只用上述 3 个 tool; 没有其它工具可用
  - max 8 轮; 接近上限时直接收尾输出摘要
  - 不要返回 JSON, 不要返回 tool_use; 摘要写在最后一个 assistant turn 的 text 里
"""


def _filter_subagent_tools(mcp_tools: list[Tool]) -> list[Tool]:
    """从主 agent 工具集挑出子 agent 可用的 3 个: download_paper / build_index / search。"""
    return [t for t in mcp_tools if t.name.endswith(SUBAGENT_TOOL_SUFFIXES)]


def _extract_last_text(messages: list[dict]) -> str | None:
    """取最后一个 assistant turn 里的 text, 兼容 anthropic content block list 与 plain str。"""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content if content else None
        if isinstance(content, list):
            texts = [
                getattr(b, "text", "")
                for b in content
                if getattr(b, "type", None) == "text"
            ]
            joined = "\n".join(t for t in texts if t)
            if joined:
                return joined
    return None


def _run_one(
    paper_id: str,
    user_query: str,
    *,
    client_factory: Callable[[], LLMClient],
    tools: list[Tool],
    on_event: EventCallback,
    max_iter: int = SUBAGENT_MAX_ITER,
    budget_tokens: int = SUBAGENT_BUDGET_TOKENS,
) -> dict[str, str]:
    """单篇精读 worker。返回 {paper_id, summary, status}。"""
    sub_messages: list[dict] = [{
        "role": "user",
        "content": (
            f"精读论文 {paper_id}, 围绕用户问题『{user_query}』提取核心方法 / "
            f"关键实验结果 / 与问题的相关性。完成后输出 markdown 三段摘要。"
        ),
    }]
    guard = Guardrail(max_iterations=max_iter, budget_tokens=budget_tokens)
    try:
        agent_loop(
            sub_messages,
            system=SUBAGENT_SYSTEM,
            tools=tools,
            client=client_factory(),
            guardrail=guard,
            on_event=on_event,
        )
    except Exception as e:
        return {
            "paper_id": paper_id,
            "summary": _extract_last_text(sub_messages) or "",
            "status": f"error: {type(e).__name__}: {e}",
        }
    summary = _extract_last_text(sub_messages) or ""
    status = "max_iter_reached" if guard.stop_reason() else "ok"
    return {"paper_id": paper_id, "summary": summary, "status": status}


def _render_results(results: list[dict[str, str]]) -> str:
    lines = [f"## Paper Deep Read Results ({len(results)} papers)", ""]
    for r in results:
        lines.append(f"### {r['paper_id']} (status: {r['status']})")
        if r["summary"]:
            lines.append(r["summary"])
        else:
            lines.append("_(no summary produced)_")
        lines.append("")
    return "\n".join(lines).rstrip()


def paper_deep_read_tool(
    *,
    client_factory: Callable[[], LLMClient],
    mcp_tools: list[Tool],
    on_event: EventCallback,
) -> Tool:
    """构造 paper_deep_read 工具实例。

    Args:
        client_factory: 每个子 agent 调一次, 返回新 LLMClient。
        mcp_tools: 主 agent 的 mcp 工具列表; 子 agent 从中过滤出 3 个。
        on_event: 主 agent emit 回调; 子 agent 事件透传时加 subagent_paper_id。
    """
    subagent_tools = _filter_subagent_tools(mcp_tools)
    emit_lock = threading.Lock()

    def _make_sub_emit(pid: str) -> EventCallback:
        def sub_emit(kind: str, payload: dict[str, Any]) -> None:
            with emit_lock:
                on_event(kind, {**payload, "subagent_paper_id": pid})
        return sub_emit

    def _handler(args: dict[str, Any]) -> str:
        paper_ids = args["paper_ids"]
        user_query = args["user_query"]
        if not (1 <= len(paper_ids) <= MAX_PAPERS):
            raise ValueError(
                f"paper_ids count must be 1..{MAX_PAPERS}, got {len(paper_ids)}"
            )

        results: list[dict[str, str]] = []
        with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as ex:
            futures = {
                ex.submit(
                    _run_one,
                    pid,
                    user_query,
                    client_factory=client_factory,
                    tools=subagent_tools,
                    on_event=_make_sub_emit(pid),
                ): pid
                for pid in paper_ids
            }
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({
                        "paper_id": pid,
                        "summary": "",
                        "status": f"error: {type(e).__name__}: {e}",
                    })

        order = {pid: i for i, pid in enumerate(paper_ids)}
        results.sort(key=lambda r: order[r["paper_id"]])
        return _render_results(results)

    return Tool(
        name="paper_deep_read",
        description=(
            "并发精读多篇论文。每篇由独立子 agent 处理, 各有独立 context 窗口, "
            "最后只回传 ~500 token markdown 摘要 (Core Method / Key Findings / "
            "Relevance to Query 三段)。"
            "何时用: 识别出 3-8 篇值得精读的论文且需要对比/综合时。"
            "何时不用: 只看摘要够 (用 mcp__arxiv__search_papers); 单篇深读 "
            "(用 deep-read-paper skill); 论文不到 3 篇 (顺序读更省事)。"
            "输入: paper_ids (list[str], 1<=N<=8), user_query (str)。"
            "返回: markdown, 每篇一段, 含 paper_id / status / summary。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PAPERS,
                    "items": {"type": "string", "minLength": 1},
                },
                "user_query": {"type": "string", "minLength": 1},
            },
            "required": ["paper_ids", "user_query"],
            "additionalProperties": False,
        },
        handler=_handler,
    )
```

注意 `EventCallback` 是 `paperpilot.core.loop` 里已经定义好的 type alias (`Callable[[str, dict], None]`),复用不要重新定义。

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `python -m pytest tests/builtin_tools/test_subagent.py -v`

Expected: PASS, 16 tests passed.

调试提示 (按出错频率):
- `ImportError: cannot import name 'EventCallback' from 'paperpilot.core.loop'` — `loop.py` 里 `EventCallback` 已导出 (Day 4 起就有), 检查 import 拼写
- `test_run_one_returns_max_iter_when_guardrail_stops` 失败但 `status == "ok"` — 检查 `guard.stop_reason()` 在 `agent_loop` 正常退出 (LLM 自然停止) 时是否为 None; 若为 None 状态判定才走 ok 分支
- `test_handler_preserves_input_order` flaky — `time.sleep` 间隔太短被 GIL 调度抹平; 把 0.03s 拉到 0.1s, 0.015s 拉到 0.05s
- `test_event_aggregation_prefixes_paper_id` 抓不到 `turn` 事件 — `agent_loop` 在 `emit("turn", ...)` 之前先 `client.call()`, 子 agent 的 FakeClient 必须能产出 ParsedResponse; 检查 responder 是否 1 轮就 text-only 收尾

- [ ] **Step 1.5: Run all fast tests to confirm no regression**

Run: `python -m pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py -m "not slow"`

Expected: 87 passed, 5 deselected (原 71 fast + Task 1 新 16 fast)。slow 5 个 deselected 不变。

- [ ] **Step 1.6: Commit**

```bash
git add paperpilot/builtin_tools/subagent.py tests/builtin_tools/test_subagent.py
git commit -m "Day 11 Task 1: paper_deep_read subagent tool + 16 单测"
```

---

## Task 2: main.py 集成 + 集成测扩展

**Files:**
- Modify: `paperpilot/main.py`
- Modify: `tests/test_main_integration.py`

- [ ] **Step 2.1: Write failing fast integration test**

读现状: `tests/test_main_integration.py` (Day 10 末) 已含:
- fast: `test_build_system_prompt_includes_skills` (Day 9)
- fast: `test_build_system_prompt_includes_research_todo_nudge` (Day 10)
- slow: `test_build_tools_contains_load_skill_research_todo_and_mcp_tools` (Day 10)

**追加** 1 个 fast 测 (放 Day 10 fast 测之后):

```python
def test_build_system_prompt_includes_paper_deep_read_nudge():
    prompt = _build_system_prompt()
    assert "## 多论文并发精读" in prompt
    assert "paper_deep_read" in prompt
```

**修改** Day 10 那个 slow 测,在 names 检查里加 `"paper_deep_read"` 一行 (其它一字不改)。
旧 slow 测尾段:
```python
        assert "load_skill" in names
        assert "research_todo" in names
        assert any(name.startswith("mcp__") for name in names)
```
改成:
```python
        assert "load_skill" in names
        assert "research_todo" in names
        assert "paper_deep_read" in names
        assert any(name.startswith("mcp__") for name in names)
```

(不重命名 slow 测函数, 保 commit diff 最小; 函数名沿用 Day 10 的 `test_build_tools_contains_load_skill_research_todo_and_mcp_tools`。)

- [ ] **Step 2.2: Run new fast test to verify it fails**

Run: `python -m pytest tests/test_main_integration.py::test_build_system_prompt_includes_paper_deep_read_nudge -v`

Expected: FAIL with `AssertionError: assert '## 多论文并发精读' in prompt` (main.py 还没拼 `PAPER_DEEP_READ_NUDGE`)。

- [ ] **Step 2.3: Modify main.py**

读现状: `paperpilot/main.py` 顶部 import 段 (Day 10 末) 含:

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

**追加** 一段 import (按字母序插在 research_todo 与 skill_loader 之间):

```python
from paperpilot.builtin_tools.subagent import (
    PAPER_DEEP_READ_NUDGE,
    paper_deep_read_tool,
)
```

`from paperpilot.core import Guardrail, LLMClient, agent_loop` 已经存在, 不动。

**改 `_build_system_prompt`** (Day 10 末状态):

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

改成 (末尾再追加 `PAPER_DEEP_READ_NUDGE`):

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
    )
```

**改 `_build_tools`** (Day 10 末状态):

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

改成 (签名加 `on_event` 参数,在 mcp tools 构造完后注入 paper_deep_read):

```python
def _build_tools(
    registry: SkillRegistry | None = None,
    todo_store: TodoStore | None = None,
    on_event: Callable[[str, dict], None] | None = None,
) -> tuple[list[Tool], MCPClient]:
    """Return (tools, mcp_client); caller is responsible for close()."""
    registry = registry or SkillRegistry(SKILLS_DIR)
    todo_store = todo_store or TodoStore()
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
            *mcp_tools,
        ]
        return tools, mcp
    except Exception:
        mcp.close()
        raise
```

import `Callable`:在 main.py 顶部 `from typing import Callable` (放在已有 `from pathlib import Path` 后)。

**改 `run`** (Day 10 末状态):

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
            tools=tools,
            client=LLMClient(),
            guardrail=Guardrail(
                max_iterations=max_iter,
                budget_tokens=int(os.environ.get("BUDGET_TOKENS", 50_000)),
            ),
            on_event=on_event or _default_logger,
        )
    finally:
        mcp.close()
```

改成 (把 emit 算出来后传给 `_build_tools` 和 `agent_loop`,确保 paper_deep_read 子 agent emit 与主 trace 一致):

```python
def run(query: str, *, max_iter: int = 8, on_event=None) -> list[dict]:
    """Run one complete agent conversation and return final messages."""
    load_dotenv()

    emit = on_event or _default_logger
    registry = SkillRegistry(SKILLS_DIR)
    todo_store = TodoStore()
    tools, mcp = _build_tools(registry, todo_store, on_event=emit)
    try:
        messages = [{"role": "user", "content": query}]
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

`_default_logger` 函数本身一字不改 (Day 9 已固化)。

注意:`_default_logger(kind, payload)` 在 Day 9 写的是 fixed 行为 (`tool_call` / `tool_result` / `guardrail_stop` 三种)。子 agent 透传后会出现含 `subagent_paper_id` 字段的 `tool_call` 事件,`_default_logger` 不感知此字段是 OK 的 (它只读 `name` / `arguments` / `content`)。trace 视觉效果:子 agent 的 tool_call 看起来跟主 agent 一样,只是 turn 顺序穿插。Day 11 smoke 用自定义 tracer 显式取 `subagent_paper_id` 字段,主 trace 有可视化。

- [ ] **Step 2.4: Run new fast test to verify it passes**

Run: `python -m pytest tests/test_main_integration.py::test_build_system_prompt_includes_paper_deep_read_nudge -v`

Expected: PASS.

- [ ] **Step 2.5: Run all fast tests in test_main_integration.py**

Run: `python -m pytest tests/test_main_integration.py -v -m "not slow"`

Expected: 3 passed (Day 9 fast `test_build_system_prompt_includes_skills` + Day 10 fast + Day 11 fast)。

- [ ] **Step 2.6: Run modified slow test to verify**

Run: `python -m pytest tests/test_main_integration.py::test_build_tools_contains_load_skill_research_todo_and_mcp_tools -v -m slow`

Expected: PASS。真启 mcp servers (~5s),验证 `paper_deep_read` 真出现在 tools list。

- [ ] **Step 2.7: Run full default test suite to check for regressions**

Run: `python -m pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py`

Expected: 88 passed, 5 deselected (87 from Task 1 + 1 new fast = 88 fast; 5 slow deselected, slow 改的那个不算新增)。

- [ ] **Step 2.8: Commit**

```bash
git add paperpilot/main.py tests/test_main_integration.py
git commit -m "Day 11 Task 2: main 集成 paper_deep_read + nudge 注入 system prompt"
```

---

## Task 3: scripts/day11_smoke.py 端到端真 LLM + 真并发

**Files:**
- Create: `scripts/day11_smoke.py`

- [ ] **Step 3.1: Write day11_smoke.py**

Write `scripts/day11_smoke.py`:

```python
"""Day 11 smoke: main loop -> paper_deep_read -> 3 个并发子 agent -> markdown 摘要 -> 综合回答.

prompt 显式说"用 paper_deep_read", 验证 tool 本体 (并发 / 隔离 / 透传 / 失败处理),
不验证 LLM 自决用工具的能力 (那个验证留给 Day 12 skill 集成)。

assertions:
  - 主 agent 调 paper_deep_read 至少 1 次, paper_ids 长度 == 3
  - tool_result (markdown) 含 3 个 paper_id 段
  - 至少 2 个 paper status == "ok" (允许 1 个 max_iter 或 error 容差)
  - tracer 抓到子 agent 内部 mcp__colbert__search 调用 (带 subagent_paper_id)
  - 主 agent 最终回答长度 > 200 chars 且至少含 2 个 paper_id 提及
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

PAPER_IDS = ["1706.03762", "2010.11929", "2005.14165"]


def main() -> None:
    saw_main: set[str] = set()
    saw_subagent_search: set[str] = set()
    paper_deep_read_calls: list[dict[str, Any]] = []
    last_paper_deep_read_result: str | None = None

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        nonlocal last_paper_deep_read_result
        sub_pid = payload.get("subagent_paper_id")
        if kind == "tool_call":
            name = payload["name"]
            args = payload.get("arguments", {})
            if sub_pid:
                # 子 agent 的 tool_call
                if name.endswith("__search"):
                    saw_subagent_search.add(sub_pid)
                print(f"  [sub:{sub_pid}] -> {name}({_preview(args)})")
            else:
                saw_main.add(name)
                if name == "paper_deep_read":
                    paper_deep_read_calls.append(args)
                print(f"  -> {name}({_preview(args)})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            preview = (
                content[:160] if isinstance(content, str) else str(content)[:160]
            )
            tag = f"[sub:{sub_pid}] " if sub_pid else ""
            print(f"  {tag}<- {payload['name']}: {preview}...")
            if not sub_pid and payload.get("name") == "paper_deep_read":
                last_paper_deep_read_result = (
                    content if isinstance(content, str) else str(content)
                )
        elif kind == "guardrail_stop":
            tag = f"[sub:{sub_pid}] " if sub_pid else ""
            print(f"  {tag}!! guardrail: {payload['reason']}")
        elif kind == "tool_arg_repair":
            print(f"  ~ repaired {payload['name']}: {payload['repaired']}")

    prompt = (
        "请用 paper_deep_read 同时精读这 3 篇 arxiv 论文: "
        f"{PAPER_IDS[0]} / {PAPER_IDS[1]} / {PAPER_IDS[2]}, "
        "围绕『self-attention 在不同模态/规模下的设计差异』做精读对比, "
        "最后给出综合分析。"
    )
    messages = run(prompt, max_iter=10, on_event=tracer)

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    final_text = ""
    if isinstance(last, list):
        for block in last:
            if hasattr(block, "text"):
                print(block.text)
                final_text += block.text + "\n"
    else:
        print(last)
        final_text = str(last)

    # ---------- assertions ----------
    assert "paper_deep_read" in saw_main, (
        f"FAIL: paper_deep_read not called by main agent; saw {saw_main}"
    )
    assert len(paper_deep_read_calls) >= 1, (
        f"FAIL: paper_deep_read called 0 times"
    )
    first_call_args = paper_deep_read_calls[0]
    paper_ids_arg = first_call_args.get("paper_ids", [])
    assert len(paper_ids_arg) == 3, (
        f"FAIL: paper_deep_read first call paper_ids count = "
        f"{len(paper_ids_arg)}, expected 3; got {paper_ids_arg}"
    )

    assert last_paper_deep_read_result is not None, (
        "FAIL: did not capture paper_deep_read tool_result"
    )
    for pid in PAPER_IDS:
        assert f"### {pid}" in last_paper_deep_read_result, (
            f"FAIL: paper_deep_read result missing segment for {pid}"
        )
    ok_count = len(re.findall(
        r"### \S+ \(status: ok\)", last_paper_deep_read_result
    ))
    assert ok_count >= 2, (
        f"FAIL: only {ok_count} paper(s) reached status=ok; "
        f"expected >= 2 (1 max_iter / error 容差)"
    )

    assert len(saw_subagent_search) >= 1, (
        f"FAIL: no subagent emitted mcp__colbert__search "
        f"(expected event payload to contain subagent_paper_id)"
    )

    assert len(final_text.strip()) > 200, (
        f"FAIL: final answer too short ({len(final_text)} chars)"
    )
    pid_mentions = sum(1 for pid in PAPER_IDS if pid in final_text)
    assert pid_mentions >= 2, (
        f"FAIL: final answer mentions only {pid_mentions} paper_id(s); "
        "expected >= 2"
    )

    print("\nDay 11 smoke PASSED")


def _preview(args: dict[str, Any]) -> dict[str, Any]:
    if "paper_ids" in args and "user_query" in args:
        return {
            "paper_ids": args["paper_ids"],
            "user_query": args["user_query"][:60] + "...",
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

- [ ] **Step 3.2: Run smoke (real LLM + real MCP servers + real concurrency)**

Run: `python scripts/day11_smoke.py`

Expected:
- 主 trace 看到主 agent 调 `paper_deep_read(paper_ids=["1706.03762","2010.11929","2005.14165"], ...)`
- 之后 trace 出现 3 路并发子 agent (前缀 `[sub:1706.03762]` / `[sub:2010.11929]` / `[sub:2005.14165]`),每个子 agent 至少 download_paper + build_index + 1 次 search
- `paper_deep_read` 的 tool_result 是 markdown,含 3 个 `### <paper_id> (status: ...)` 段
- 主 agent 综合 3 篇摘要回答用户对比问题
- 退出码 0,stdout 末尾 `Day 11 smoke PASSED`
- API 实付:DeepSeek 主 agent ~3-5 turn + 3 个子 agent 各 ~5-8 turn = 总 ~$0.05-0.10

如果失败可能原因 + 调法:

1. **LLM 没调 paper_deep_read** (主 agent 直接回答):
   - 检查 prompt 显式说"用 paper_deep_read"是否被 LLM 识别
   - 看 system prompt 是否含 `## 多论文并发精读`
   - 加强 prompt:"必须用 paper_deep_read 工具,不要直接回答"

2. **子 agent 输出格式偏离** (没 ## Core Method 三段):
   - status 仍 = ok,只是 summary 文本不规整 — **不阻塞 smoke** (assertion 不校验段落格式)
   - 若想严控,改 SUBAGENT_SYSTEM 把"必须用以下三段标题"加重

3. **3 篇全 status = error** (colbert 并发崩):
   - 这是 spec 风险章节的 #1 风险
   - 第一兜底:把 `THREAD_POOL_SIZE` 改 1 (顺序模式),重跑;若通过 = 确定是 colbert 并发竞争
   - 第二兜底:在 `_filter_subagent_tools` 上层加 `threading.Lock` 包 build_index handler (但这就违反了"loop / mcp_client 一行不改",得另起 spec 讨论)
   - 临时:改 prompt 只精读 1-2 篇验证非并发路径正常

4. **subagent search 抓不到**:
   - 检查 `subagent_paper_id` 字段是否真的进了 payload (在 `subagent.py` `_make_sub_emit` 处加 `print` debug)
   - tracer 里 `payload.get("subagent_paper_id")` 是否拿到 (key 拼写对不对)

5. **`paper_deep_read_calls[0]["paper_ids"]` 长度不是 3**:
   - LLM 自己拆批分多次调用 — 不太可能 (3 篇正好是低限),若发生改 prompt 强调"一次调用,paper_ids 含全部 3 篇"

6. **arxiv 429 / Semantic Scholar 抖动**:
   - 等几分钟重跑;arxiv 限流通常 ~1 min cooldown

7. **token budget 撞顶** (最后回答被截):
   - `BUDGET_TOKENS` 默认 50k,主 agent + 3 个子 agent 总用量可能逼近;调到 80_000 重跑

- [ ] **Step 3.3: Verify regressions on prior smokes (optional)**

```bash
python scripts/day9_smoke.py
python scripts/day10_smoke.py
```

Expected: 都退出 0 + 各自 PASSED 字样。Day 11 改的部分 (subagent.py 新文件, main.py 加 nudge / on_event 透传) 不应影响 Day 9/10 链路 — 它们都不调 paper_deep_read。

(若 arxiv 抖动允许跳过,Day 11 通过即视为完工。)

- [ ] **Step 3.4: Commit**

```bash
git add scripts/day11_smoke.py
git commit -m "Day 11 Task 3: day11_smoke 端到端 (paper_deep_read 并发精读 3 篇)"
```

---

## Definition of Done

1. `python -m pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py` 全绿:
   - 88 fast 测 (原 71 + Task 1 新 16 + Task 2 新 1) 全 passed
   - 5 slow 测 deselected (Task 2 改 1 个 slow 测含 paper_deep_read assertion,不增加 slow 测数)
2. `python scripts/day9_smoke.py` 无回归 (可选,arxiv 抖动允许跳过)
3. `python scripts/day10_smoke.py` 无回归 (可选)
4. `python scripts/day11_smoke.py` 退出 0 + 打印 `Day 11 smoke PASSED`
5. day11_smoke tracer 必须满足:
   - 主 agent 调 `paper_deep_read` 至少 1 次,paper_ids 长度 == 3
   - tool_result 含 3 个 `### <paper_id>` 段
   - 至少 2 个 paper status == "ok"
   - 至少 1 个 subagent emit 含 `mcp__colbert__search` (payload 有 `subagent_paper_id`)
   - 主 agent 最终回答 > 200 chars,至少 2 个 paper_id 提及
6. `git grep -E "TODO|FIXME" paperpilot/builtin_tools/subagent.py scripts/day11_smoke.py` 空
7. 3 个 commit:Task 1 / Task 2 / Task 3 (与 Day 9 / Day 10 同节奏)

---

## 红线复核

| 红线 | Day 11 是否守住 |
|---|---|
| `loop.py` 一行不改 | ✓ subagent 复用现有 `agent_loop`, 通过 ThreadPoolExecutor spawn N 次独立调用 |
| `adapter.py` 一行不改 | ✓ 子 agent 用同一个 `LLMClient` / `Tool` / `ParsedResponse` 数据契约 |
| `mcp_client.py` 一行不改 | ✓ MCPClient 已是线程安全 (Day 5 dedicated asyncio loop), N 个 worker 直接共用同一个 MCPClient |
| 决策由 LLM 做 | ✓ 主 agent 自决何时 paper_deep_read; 子 agent 在专属 system prompt 下自决调哪些 tool |
| 不做推测性抽象 | ✓ 不留 cancel / retry / streaming / 子 agent 互通; YAGNI 严守 |
| Day 7 trip wire 不撞 | ✓ tool_result 是 str (markdown),不是 list[dict] |