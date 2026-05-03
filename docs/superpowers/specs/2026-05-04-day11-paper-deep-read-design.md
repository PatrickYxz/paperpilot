# PaperPilot Day 11 设计:`paper_deep_read` subagent 内嵌 tool

| 项 | 值 |
|---|---|
| 日期 | 2026-05-04 (Day 9 skill loading + Day 10 research_todo 完工后, 设计 Day 11 paper_deep_read subagent) |
| 范围 | (1) 新增 L2 内嵌 tool `paper_deep_read`, 与 `load_skill` / `research_todo` 同层; (2) 新增 `paperpilot/builtin_tools/subagent.py` 实现并发子 loop spawn; (3) ThreadPoolExecutor 跑 N 个独立 sync `agent_loop`; (4) 子 agent emit 加前缀转发主 trace; (5) 单测 + day11 smoke |
| 不在范围 | skill 集成 (放 Day 12); subagent 调用 subagent (递归 spawn); 子 agent 间共享 context / cache; 真正的 streaming UI; subagent 取消 / 中断; subagent 之间通信 |
| 状态 | Draft, 待用户 review |

---

## 1. 目标

让主 agent 在"识别出多篇论文需对比/综合"场景下, 通过 `paper_deep_read(paper_ids, user_query)` 一次 spawn N 个独立子 agent 并发精读, 每个子 agent 拥有**独立 context 窗口**, 完成后只回传 ~500 token markdown 摘要给主 agent。这是 PaperPilot 第三个 L2 内嵌 tool, 也是 Claude Code 四引擎 (skill / todo / **subagent** / compact) 中第三个落地, 项目最大差异化卖点。

同时严守方案 C / Day 5-10 锁定的红线:

- **`loop.py` / `adapter.py` / `mcp_client.py` 一行不改** —— subagent tool 走 ThreadPoolExecutor + 直接复用现有 sync `agent_loop`
- **server 之间不互通信** —— subagent 内嵌 tool 不知道任何 mcp server / skill 存在
- **决策由 LLM 做** —— 主 agent 自决何时用 paper_deep_read; 子 agent 自决调哪些 tool / 何时停
- **不做推测性抽象** —— 不留 cancel / retry / streaming / 子 agent 互通等口子
- **复用 Day 7 trip wire** —— `paper_deep_read` tool_result 是 str (markdown), 不撞 FastMCP list[dict] 监控

---

## 2. 关键设计决策 (Q1-Q5)

| ID | 决策 | 选项 | 主要理由 |
|---|---|---|---|
| Q1 | 并发实现方式 | **ThreadPoolExecutor** (vs asyncio.gather / 顺序 for) | LLM HTTP 是 IO-bound, GIL 自动释放, 线程并发足够; 现有 sync `agent_loop` 一行不改, 守 Day 5-10 红线; asyncio 路线要改 LLMClient + loop, blast radius 大且底层 HTTP 还得绕回 thread; 顺序 for 失去"并发"卖点 |
| Q2 | 子 agent 输入 | **`paper_ids: list[str]` + `user_query: str`** (vs 主 agent 预下载 text 直传) | 子 agent 自治, 不依赖主 agent 状态; 主 agent 完全不用先把 PDF text 加进 context (守 context 隔离); 主 agent 可能根本没下载过这些 paper (用户场景: "精读这 5 个 arxiv ID"); 网络抓取是 IO 并发场景, N 个并发 download 可接受 |
| Q3 | 子 agent 工具子集 | **`arxiv.download_paper` + `colbert.build_index` + `colbert.search` 三件** (vs 全集 / 仅 search) | 砍 `arxiv.search_papers` (子 agent 不做检索); 砍 `graph.*` (子 agent 不爬引用); 砍 `load_skill` / `research_todo` (子 agent 用专属 system prompt); **砍 `paper_deep_read` 自身** (防递归无限 spawn); 留三件够支撑"download → build_index → 多次 search → 总结"标准精读流 |
| Q4 | 触发场景 | **不集成 skill, day11_smoke 直接 prompt 验证** (vs 新写 compare-papers skill / 改造 find-classics) | Day 11 6h 做 subagent 本体已满档; skill 集成是 prose + 测试无技术风险, Day 12 单独做更清晰; smoke 用直接 prompt 验证最纯粹, 不混入 skill loading 归因问题 |
| Q5 | Event 聚合 | **B2: 子 agent emit 透传主 emit, 前缀 `[subagent:<paper_id>]`, threading.Lock 保 trace 不交错** (vs 只 emit 一次 `paper_deep_read_done` 总结) | 杀手锏卖点 = "看到 N 个子 agent 并行干活", B1 在 trace 里看不到内部进度; 工程量就是一把 Lock 包 emit 调用; demo 录像 / 简历讲解的钱在这条 trace 上 |

### 隐含决策 (已锁)

| 项 | 值 | 备注 |
|---|---|---|
| 内嵌 tool 模块路径 | `paperpilot/builtin_tools/subagent.py` | 与 `skill_loader.py` / `research_todo.py` 平级 |
| 子 agent system prompt | 静态常量 `SUBAGENT_SYSTEM` | 写"精读子 agent"角色 + download → build_index → search 工作流 + 输出 markdown 三段格式 |
| 子 agent max_iter | **8** | 与主 agent 默认一致 |
| `MAX_PAPERS` 上限 | **8** | 超 8 篇 handler 直接 raise, 让主 agent 拆批; 防 token / 资源爆炸 |
| `THREAD_POOL_SIZE` | **8** | 等于 MAX_PAPERS, 所有论文同时跑 |
| 子 agent 工具实例来源 | **复用主 agent 的 mcp 工具引用** (paper_deep_read handler 接受主 agent tools 子集) | 避免每次 spawn 都 start 新 MCPClient; MCPClient 已线程安全 (dedicated asyncio loop + run_coroutine_threadsafe) |
| 子 agent 之间隔离 | 各自独立 `messages` / `Guardrail` / `LLMClient` 实例 | 不共享 conversation / token budget |
| 失败模式 | 单篇异常 → 该项 `status = "error: <type>: <msg>"`, 不影响其他 | 整体 ThreadPoolExecutor.submit + .result() 包 try/except |
| 输出 schema | `list[{paper_id: str, summary: str, status: str}]` | summary 是 markdown 字符串, 不强制 JSON; status ∈ {"ok", "max_iter_reached", "error: ..."} |
| tool_result 形态 | **str** (markdown 拼接每篇 summary) | 不返 list[dict], 守 Day 7 FastMCP list[dict] trip wire |
| `additionalProperties` | False (input 顶层) | 阻止 LLM 误传额外字段 |

---

## 3. 架构

```
┌──────────────────────────────────────────────────────────────────┐
│  main.py 启动                                                    │
│    1. SkillRegistry(...)         # Day 9                         │
│    2. TodoStore()                 # Day 10                       │
│    3. mcp = MCPClient(...).start()                              │
│    4. tools = [load_skill_tool(reg),                            │
│                research_todo_tool(store),                        │
│                paper_deep_read_tool(                             │
│                    client_factory=lambda: LLMClient(),           │
│                    mcp_tools=mcp.list_tools(),                   │
│                    on_event=on_event,                            │
│                ),                                # Day 11 新     │
│                *mcp.list_tools()]                                │
│    5. system = base + skill_section + RESEARCH_TODO_NUDGE        │
│                       + PAPER_DEEP_READ_NUDGE   # Day 11 新, ~1段 │
│    6. agent_loop(messages, system, tools, ...)                  │
└──────────────────────────────────────────────────────────────────┘
                       │
                       ▼ LLM 调 paper_deep_read(paper_ids=[...], user_query="...")
┌──────────────────────────────────────────────────────────────────┐
│  paper_deep_read handler (主线程)                                │
│    1. 校验 1 <= len(paper_ids) <= MAX_PAPERS                    │
│    2. 构造子 agent 工具子集 (从主 mcp_tools 过滤出 download_paper│
│       / build_index / search 三个)                               │
│    3. ThreadPoolExecutor(max_workers=8) submit N 个 _run_one      │
│    4. as_completed 收集结果 (return_exceptions 模式)             │
│    5. 拼 markdown tool_result, return                            │
└──────────────────────────────────────────────────────────────────┘
                       │
                       ▼ N 个 worker 线程
┌──────────────────────────────────────────────────────────────────┐
│  _run_one(paper_id, user_query, ...) (worker 线程)               │
│    sub_messages = [{"role":"user", "content": SUBAGENT_PROMPT}]  │
│    sub_emit = lambda kind, payload:                              │
│        with lock: main_emit(kind, {**payload,                    │
│                                    "subagent_paper_id": paper_id})│
│    agent_loop(                                                   │
│        sub_messages,                                             │
│        system=SUBAGENT_SYSTEM,                                   │
│        tools=subagent_tools,         # 3 个 mcp tool             │
│        client=LLMClient(),           # 独立实例                  │
│        guardrail=Guardrail(max_iterations=8, ...),               │
│        on_event=sub_emit,                                        │
│    )                                                              │
│    summary = _extract_last_text(sub_messages)                    │
│    return {"paper_id": paper_id, "summary": summary,             │
│            "status": "ok" / "max_iter_reached" / "error: ..."}   │
└──────────────────────────────────────────────────────────────────┘
```

**关键边界:**

- `paper_deep_read` 是 L2 内嵌 tool (Python 函数 handler), **不走 MCP 协议** —— 与 `load_skill` / `research_todo` 同层
- 内嵌 tool 与 MCP tool 在 agent_loop 里**同等对待** (统一走 `Tool` dataclass + handler dispatch), `loop.py` / `adapter.py` / `mcp_client.py` 一行不改
- `paper_deep_read` 的 handler factory 接受 `(client_factory, mcp_tools, on_event)` 三个参数, 由 main.py 在 `run()` 拿到 on_event 之后构建 tools 时注入 (Day 11 main.py wiring 把 `_build_tools` 签名扩为 `(registry, todo_store, on_event)`)
- 一次 `paper_deep_read` 调用 = 一次 ThreadPoolExecutor 创建/销毁; 不维护全局线程池
- 子 agent **不能拿到** `paper_deep_read` 自身 (防递归); main.py 构造时把它从 mcp_tools 过滤即可 (反正 mcp_tools 里也没有, 它是内嵌工具)
- 主 emit 透传: 子 agent 内 `tool_call` / `tool_result` / `turn` / `guardrail_stop` 全部加 `subagent_paper_id` 字段后用 `threading.Lock` 包住调主 emit
- MCPClient 已是线程安全的 (Day 5 dedicated asyncio loop + `run_coroutine_threadsafe` 模式), N 个 worker 线程同时调 mcp tool 不撞

---

## 4. 文件布局

### 新增 / 改动

| 路径 | 状态 | 责任 | 预估行数 |
|---|---|---|---|
| `paperpilot/builtin_tools/subagent.py` | 新 | `paper_deep_read_tool(client_factory, mcp_tools, on_event)` 工厂 + `_run_one` worker + `_extract_last_text` + `_filter_subagent_tools` + `SUBAGENT_SYSTEM` + `PAPER_DEEP_READ_NUDGE` 常量 | ~150 |
| `paperpilot/builtin_tools/__init__.py` | 改 (可选) | re-export | +2 |
| `paperpilot/main.py` | 改 | 构造并注入 `paper_deep_read_tool`; system prompt 末尾拼 `PAPER_DEEP_READ_NUDGE` | +12 |
| `tests/builtin_tools/test_subagent.py` | 新 | mock LLMClient 跑 worker / 并发 / 失败 / event 透传 / 工具过滤 / MAX_PAPERS 校验 单测 | ~200 |
| `tests/test_main_integration.py` | 改 | fast: system prompt 含 `PAPER_DEEP_READ_NUDGE`; slow: tools 含 `paper_deep_read` | +6 |
| `scripts/day11_smoke.py` | 新 | 端到端真 LLM: 直接 prompt 让 LLM 调 paper_deep_read 并发精读 3 篇 | ~80 |

### 不动

- `paperpilot/core/loop.py` / `adapter.py` / `guardrail.py` — 一行不改
- `paperpilot/tools/mcp_client.py` — 一行不改 (Day 9 已稳)
- `paperpilot/builtin_tools/skill_loader.py` / `research_todo.py` — 一行不改
- 现有 mcp_servers (arxiv/colbert/graph) — 一行不改
- 现有 skills (`deep-read-paper.md` / `explore-citations.md` / `find-classics.md`) — 一行不改 (Day 12 才考虑 skill 集成)

---

## 5. 数据契约

### `paper_deep_read` tool schema

```python
Tool(
    name="paper_deep_read",
    description=(
        "并发精读多篇论文。每篇由独立子 agent 处理, 各有独立 context 窗口, "
        "最后只回传 ~500 token markdown 摘要 (Core Method / Key Findings / "
        "Relevance to Query 三段)。"
        "何时用: 识别出 3-8 篇值得精读的论文且需要对比/综合时。"
        "何时不用: 只看摘要够 (用 arxiv.search_papers); 单篇深读 (用 deep-read-paper "
        "skill); 论文不到 3 篇 (顺序读更省事)。"
        "输入: paper_ids (list[str], 1<=N<=8), user_query (str)。"
        "返回: markdown, 每篇一段, 含 paper_id / status / summary。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "paper_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {"type": "string", "minLength": 1},
            },
            "user_query": {"type": "string", "minLength": 1},
        },
        "required": ["paper_ids", "user_query"],
        "additionalProperties": False,
    },
    handler=...,  # 由 paper_deep_read_tool(client_factory, mcp_tools, on_event) 注入
)
```

### handler 行为

```python
def _handler(args: dict) -> str:
    paper_ids = args["paper_ids"]
    user_query = args["user_query"]
    if not (1 <= len(paper_ids) <= MAX_PAPERS):
        raise ValueError(
            f"paper_ids count must be 1..{MAX_PAPERS}, got {len(paper_ids)}"
        )

    subagent_tools = _filter_subagent_tools(mcp_tools)  # 留 3 个
    lock = threading.Lock()
    def make_sub_emit(pid: str):
        def sub_emit(kind: str, payload: dict) -> None:
            with lock:
                on_event(kind, {**payload, "subagent_paper_id": pid})
        return sub_emit

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as ex:
        futures = {
            ex.submit(
                _run_one,
                pid,
                user_query,
                client_factory=client_factory,
                tools=subagent_tools,
                on_event=make_sub_emit(pid),
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

    # 按输入顺序排序, 保结果稳定
    order = {pid: i for i, pid in enumerate(paper_ids)}
    results.sort(key=lambda r: order[r["paper_id"]])
    return _render_results(results)
```

### `_run_one` worker

```python
def _run_one(
    paper_id: str,
    user_query: str,
    *,
    client_factory: Callable[[], LLMClient],
    tools: list[Tool],
    on_event: EventCallback,
) -> dict:
    sub_messages = [{
        "role": "user",
        "content": (
            f"精读论文 {paper_id}, 围绕用户问题『{user_query}』提取核心方法 / "
            f"关键实验结果 / 与问题的相关性。完成后输出 markdown 三段摘要。"
        ),
    }]
    guard = Guardrail(max_iterations=SUBAGENT_MAX_ITER, budget_tokens=20_000)
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
```

### `_extract_last_text`

```python
def _extract_last_text(messages: list[dict]) -> str | None:
    """取最后一个 assistant turn 里的 text block, 忽略 tool_use 块。"""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [b.text for b in content if getattr(b, "type", None) == "text"]
            if texts:
                return "\n".join(texts)
    return None
```

### `_filter_subagent_tools`

```python
SUBAGENT_TOOL_SUFFIXES = (
    "__download_paper",
    "__build_index",
    "__search",   # mcp__colbert__search
)

def _filter_subagent_tools(mcp_tools: list[Tool]) -> list[Tool]:
    return [t for t in mcp_tools if t.name.endswith(SUBAGENT_TOOL_SUFFIXES)]
```

### `SUBAGENT_SYSTEM`

```
你是论文精读子 agent。任务: 围绕用户问题, 精读指定的一篇论文, 提取核心方法 /
关键实验结果 / 与问题的相关性。

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
  <一段, 与用户问题『{user_query}』的关联点>

约束:
  - 只处理这一篇论文; 不要 download / search 其它 paper_id
  - 只用上述 3 个 tool; 没有其它工具可用
  - max 8 轮; 接近上限时直接收尾输出摘要
  - 不要返回 JSON, 不要返回 tool_use; 摘要写在最后一个 assistant turn 的 text 里
```

### `PAPER_DEEP_READ_NUDGE` (拼到 system prompt)

```
## 多论文并发精读
当用户问题需要对比/综合 3-8 篇论文时, 调 paper_deep_read(paper_ids=[...],
user_query="...") 一次性并发精读, 每篇会由独立子 agent 处理并回传 markdown 摘要。
单篇深读 (用 deep-read-paper skill) / 不到 3 篇 / 只看摘要够时不必用。
```

### `_render_results` (tool_result 字符串)

```python
def _render_results(results: list[dict]) -> str:
    lines = [f"## Paper Deep Read Results ({len(results)} papers)", ""]
    for r in results:
        lines.append(f"### {r['paper_id']} (status: {r['status']})")
        if r["summary"]:
            lines.append(r["summary"])
        else:
            lines.append("_(no summary produced)_")
        lines.append("")
    return "\n".join(lines).rstrip()
```

---

## 6. 错误处理

按 Day 5/9/10 红线分两类:

### A. 启动期 hard-fail

无。`paper_deep_read_tool(...)` 是纯工厂, 无 IO / 无解析。

### B. 运行时 soft-fail

| 失败 | 触发 | LLM 看到 |
|---|---|---|
| `paper_ids` 超 MAX_PAPERS | LLM 一次传 9+ | `ValueError("paper_ids count must be 1..8, got 9")` → "我拆批" |
| `paper_ids` 空 | LLM 传 `[]` | JSON Schema `minItems: 1` 不通过, anthropic SDK / DeepSeek 报错 → agent_loop 转 is_error → LLM 补 |
| `paper_ids` 含空 string | LLM 传 `[""]` | JSON Schema `items.minLength: 1` 拦, 同上 |
| 单篇 `download_paper` 失败 (404) | arxiv id 不存在 | 该篇 `status = "error: ArxivNotFoundError: ..."`, 其他篇正常; 主 agent 看 status 自决要不要换 id |
| 单篇 max_iter 撞上限 | 子 agent 复杂查询 8 轮没收 | 该篇 `status = "max_iter_reached"`, summary 取最后一个 text block (可能为空); 主 agent 看 status 决定 |
| 单篇 budget 撞上限 | 子 agent token 超 20k | 同 max_iter, status = "max_iter_reached" (复用 Guardrail 路径) |
| 单篇内部异常 (网络 / 解析 / colbert build 错) | 任意未捕获异常 | `_run_one` 捕获后 `status = "error: <type>: <msg>"`, 其他篇不影响 |
| 全部失败 | N 篇都报错 | 主 agent 看到 N 个 status = error, 回答用户"全部失败, 原因…" |

**全部不做:**
- 自动重试单篇 (LLM 可自决再调 paper_deep_read)
- 自动 paper_id 模糊匹配修复
- 自动降级到顺序模式
- subagent 取消 / 超时强 kill (max_iter + budget 已是双保险)

---

## 7. 测试策略

### 层 1: 单元测试 `tests/builtin_tools/test_subagent.py` (秒级, mock LLMClient, CI 跑)

| 测试 | 测什么 |
|---|---|
| `test_filter_subagent_tools_keeps_three` | mcp_tools 含 search_papers / download_paper / build_index / search / graph 5 个时, 过滤后只剩 download_paper / build_index / search |
| `test_filter_subagent_tools_drops_paper_deep_read` | 即使 mcp_tools 里有 paper_deep_read (虽然实际不会), 也不留 |
| `test_extract_last_text_from_anthropic_blocks` | messages 末尾 assistant turn 含 text block list, 正确取出 |
| `test_extract_last_text_from_string_content` | content 是 plain str 时也能取 |
| `test_extract_last_text_returns_none_when_no_assistant` | 全 user 消息时返 None |
| `test_handler_rejects_zero_papers` | `paper_ids=[]` 抛 ValueError (handler 层兜底, schema 层之外) |
| `test_handler_rejects_too_many_papers` | `paper_ids=[*9]` 抛 ValueError, 含 "1..8" + "got 9" |
| `test_run_one_returns_ok_status_on_success` | mock client 让 agent_loop 1 轮直接 text 收尾, status="ok", summary 含 mock 内容 |
| `test_run_one_returns_max_iter_when_guardrail_stops` | mock client 让 LLM 一直发 tool_call, max_iter 撞上限, status="max_iter_reached" |
| `test_run_one_returns_error_status_on_exception` | mock client 抛异常, status 含 "error:" |
| `test_handler_aggregates_three_papers` | 3 个 paper_id, 全部 mock 成功, 返回 str 含全部 3 个 paper_id 段 |
| `test_handler_one_paper_failed_others_ok` | 3 个 paper_id, 中间 1 个抛, 其他 2 个 ok; 输出含全部 3 段, 失败那个 status 含 "error:" |
| `test_handler_preserves_input_order` | paper_ids 传入顺序 [B, A, C], 输出 markdown 段顺序也是 B / A / C (即使并发完成顺序不同) |
| `test_event_aggregation_prefixes_paper_id` | mock on_event, 子 agent 触发 turn / tool_call 时, 主 on_event 收到的 payload 含 `subagent_paper_id` 字段 |
| `test_event_aggregation_lock_prevents_interleave` | 双线程同时 emit 时, 主 on_event 调用串行 (用计数器或 thread-safety mock 验证) |
| `test_tool_metadata` | name == "paper_deep_read"; input_schema 含 maxItems=8 / minItems=1 / additionalProperties=False |

### 层 2: main.py 集成 `tests/test_main_integration.py` 加 1 fast + 扩展 1 slow

```python
def test_build_system_prompt_includes_paper_deep_read_nudge():
    prompt = _build_system_prompt()
    assert "## 多论文并发精读" in prompt
    assert "paper_deep_read" in prompt
    # Day 9/10 已有断言不动

# slow 测扩展: tools 里多 1 个 name
def test_build_tools_contains_all_builtin_and_mcp_tools():
    tools, mcp = _build_tools()
    try:
        names = [t.name for t in tools]
        assert "load_skill" in names
        assert "research_todo" in names
        assert "paper_deep_read" in names    # 新加
        assert any(n.startswith("mcp__") for n in names)
    finally:
        mcp.close()
```

### 层 3: Day 11 smoke `scripts/day11_smoke.py` (端到端真 LLM + 真 MCP + 真并发)

```
prompt: "请用 paper_deep_read 同时精读这 3 篇 arxiv 论文:
        1706.03762 / 2010.11929 / 2005.14165, 围绕『self-attention 在不同模态/规模
        下的设计差异』做精读对比。"

期望 LLM 行为:
  1. 看 system prompt 知道有 paper_deep_read → 直接调用,
     paper_ids=["1706.03762","2010.11929","2005.14165"], user_query="..."
  2. handler 启 ThreadPoolExecutor, 3 个子 agent 并发跑
  3. 每个子 agent: download → build_index → 多次 search → 写 markdown 摘要
  4. handler 拼 markdown tool_result 返主 agent
  5. 主 agent 综合 3 篇摘要回答用户对比问题

assertions:
  - paper_deep_read 被主 agent 调用至少 1 次
  - 该次调用 paper_ids 长度 == 3
  - tool_result (markdown) 含 3 个 paper_id 段
  - 至少 2 个 status == "ok" (允许 1 个 max_iter 或 error 容差)
  - tracer 抓到子 agent 内部至少 1 次 mcp__colbert__search 调用
    (验证 emit 透传, payload 含 subagent_paper_id)
  - 主 agent 最终回答 (last text block) 长度 > 200 字符且至少含 2 个 paper_id 提及
```

smoke prompt **显式说**"用 paper_deep_read"  —— Day 11 验证 tool 本体, 不验证 LLM 自决用工具的能力 (那个验证留给 Day 12 skill 集成)。

**编码** (沿用 Day 5/6/8/9/10 模式): smoke 顶部 `sys.stdout.reconfigure(encoding="utf-8")`。

---

## 8. 工作量预估 + 完工标志

| 阶段 | 预估 |
|---|---|
| 写 plan | ~30 min |
| Task 1: subagent.py + 16 单测 (filter / extract / run_one / handler / event aggregation / metadata) | ~2.5 h |
| Task 2: main.py 集成 (注入 paper_deep_read_tool + 拼 nudge + 集成测扩展) | ~50 min |
| Task 3: scripts/day11_smoke.py + 真 LLM 联调 | ~1.5 h |
| **合计** | **~5 小时** (预算 6-8h, 留 ~2-3h buffer 给并发风险点排障 / 子 agent prompt 调优) |

**完工标志 (Definition of Done)**:

1. `pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py` 全绿:
   - 原 71 (Day 10 末) + Task 1 新 16 + Task 2 新 1 fast = **88 fast 测**
   - slow 5 个 deselected; Task 2 给 slow `test_build_tools` 加 1 个 assertion 不增加 slow 测数
2. `python scripts/day9_smoke.py` 无回归 (deep-read 链路仍通)
3. `python scripts/day10_smoke.py` 无回归 (find-classics + research_todo 链路仍通)
4. `python scripts/day11_smoke.py` 退出 0 + stdout 末尾 `Day 11 smoke PASSED`
5. day11 smoke tracer 必须抓到:
   - `paper_deep_read` 主 agent 调用 1 次, paper_ids 长度 3
   - 子 agent emit (payload 含 `subagent_paper_id`) 至少 1 个 `mcp__colbert__search`
   - 至少 2 个 paper status == "ok"
6. `git grep -E "TODO|FIXME" paperpilot/builtin_tools/subagent.py` 空
7. 3 个 commit: Task 1 / Task 2 / Task 3 (与 Day 9/10 同节奏)

---

## 9. trip wire (Day 7 监控复用)

`paper_deep_read` tool_result 是 **str (markdown 文本)**, 不撞 Day 7 锁的 FastMCP list[dict] 监控。

新增以下监控条件 (作为"未来某天数据量级跳变"的提醒钉子, 第一版**不写**监控脚本):

| 触发条件 | 行动 |
|---|---|
| 同 session 内 LLM 反复调 paper_deep_read 同样 paper_ids > 2 次 | 主 agent 没消化子 agent 摘要; 检查 description 是否引导得不够明确 |
| 单次 paper_deep_read 调用里 ≥ 3 篇 status = error | 子 agent 工具 / system prompt 出问题; 看 trace 子 agent 调了什么 |
| 单次调用 wall-clock > 5 分钟 | 子 agent stuck 或 colbert build_index 在并发下严重退化; 可能要降 THREAD_POOL_SIZE 或加 wall-clock timeout |
| trace 看到子 agent 调 paper_deep_read 自身 | _filter_subagent_tools 漏过滤; 必须立刻修 (递归无限 spawn) |

---

## 10. 风险

| 风险 | 等级 | 缓解 |
|---|---|---|
| **colbert server `index_manager` 多线程并发安全性未验证** | 中 | 当前 colbert server 是单进程, 索引按 paper_id 维度存 dict; 不同 paper_id 写不同 key 通常 OK, 但 Python dict 写入不是原子 (CPython GIL 保护浅层但 build 流程中含 IO + 多步状态变更, 仍可能有竞争)。**Task 1 单测全 mock LLM, 不触发真 colbert; Task 3 day11_smoke 真并发 3 篇是首次验证。若 smoke 翻车, 第一兜底降 THREAD_POOL_SIZE=1 (顺序模式), 第二兜底给 colbert handler 加锁** |
| **arxiv `download_paper` 同 paper_id 并发命中本地 cache 写入竞争** | 低 | 不同 subagent 通常拿不同 paper_id; 极端情况主 agent 让两个子 agent 读同一个 id, 第一个写 cache_path 时第二个读到半截。**实际场景概率低; 真翻车给 download_paper 加 file lock** |
| **LLM 不调用 paper_deep_read** (description 没让 LLM 觉得该用) | 低 | Day 11 smoke prompt 显式说"用 paper_deep_read", 直接验证 tool 路径; 自决场景留 Day 12 skill 集成验证 |
| **子 agent 写不出预期格式 markdown** (没有 ## Core Method 三段) | 低-中 | system prompt 写得明确; 即使格式偏离, summary 字段仍是子 agent 的最后 text, 主 agent 拿到也能用; 不强制校验格式 (违反 YAGNI 校验易陷入 LLM retry 循环) |
| **子 agent 死循环消耗 token** | 低 | Guardrail max_iter=8 + budget_tokens=20_000 双保险; 单篇极限 ~$0.02 (DeepSeek 价位), 3 篇全炸也只 $0.06 |
| **emit Lock 用错 (deadlock 或漏锁)** | 低 | 单测 `test_event_aggregation_lock_prevents_interleave` 验; lock 只包 emit 一行, 不可能 deadlock |
| **ThreadPoolExecutor 异常吞 (上下文管理器内异常未传播)** | 低 | `as_completed` + `fut.result()` + try/except 显式捕获每篇异常; 单测 `test_handler_one_paper_failed_others_ok` 验 |

---

## 11. 不在范围 / 推迟到未来的事

| 项 | 推迟到何时触发 |
|---|---|
| skill 集成 (compare-papers / 改 find-classics) | **Day 12** (默认下一天就做) |
| 子 agent 调用子 agent (递归 spawn) | 永远不做 — 哲学上 subagent 是单层并发, 不是树形 |
| 子 agent 之间共享 context / cache 命中 | 真出现性能瓶颈再说 (现在每篇独立 download / build_index 是 OK 的) |
| 真 streaming UI (per-subagent 进度条) | demo 录视频要美化时再说 (CLI 用 trace 已够) |
| subagent 取消 / 中断 (Ctrl-C 单篇) | 永远不做 — max_iter + budget 已是双保险 |
| subagent 之间通信 (一篇结果喂另一篇) | 永远不做 — 杀手锏卖点恰恰是"独立 context", 互通就破坏了 |
| async loop 重写 | 永远不做 — Q1 决策已锁线程池路线 |
| `pdf-parse-mcp` (第 4 个 MCP server) | **Day 13+** (优先级: skill 集成 > pdf 入口) |
| `compact_context` 内嵌 tool | 长 PDF 真撞 token 上限时 |
| `vlm-mcp` + Qwen-VL | Week 3 末 |

---

## 附: 与已有 spec / 架构的衔接点

- **复用 Day 4 agent_loop** —— 子 agent 跑的就是同一个 sync `agent_loop`, 通过 ThreadPoolExecutor spawn N 个独立调用; 守"loop 一行不改"红线
- **复用 Day 5 mcp_client** —— `MCPClient` 已是线程安全 (dedicated asyncio loop + run_coroutine_threadsafe), N 个 worker 线程同时调 mcp tool 不撞; 守"mcp_client 一行不改"红线
- **复用 Day 7 trip wire** —— `paper_deep_read` tool_result 是 str (markdown), 不撞 FastMCP list[dict] 监控
- **复用 Day 8 graph-mcp** —— graph 工具被 `_filter_subagent_tools` 过滤掉, 子 agent 不爬引用网络 (与设计意图一致)
- **复用 Day 9 skill loading** —— `paper_deep_read` 与 `load_skill` / `research_todo` 同层 L2; Day 12 skill 集成时, 新 skill 自然 prose 调 paper_deep_read
- **复用 Day 10 research_todo** —— Day 12 集成时, find-classics / 新 compare-papers skill 可在 step 0 预填 todo 含 "paper_deep_read 精读 top-k", 与 todo 链路串起来
- **对齐 Claude Code s04 Subagent** —— "subagent-as-tool" 模式 + 独立 context + 结构化摘要回传, 是 s04 设计的直接对齐; 唯一差异是并发用线程池 (CC 用 asyncio), 简历可解释"针对 sync ReAct loop 做适配, IO-bound 场景线程并发等价"