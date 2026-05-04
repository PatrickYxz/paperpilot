# Day 13: `compact_context` 内嵌 tool 设计

**目标**:补齐 L2 内嵌 tool 三件套(`load_skill` / `research_todo` / `compact_context`),与 Claude Code s06 对齐。LLM 自决何时压缩对话历史,handler 真改主 agent 的 messages list,主 agent only。

**Why:** 长会话(连续 deep-read + 多轮 search + 综合回答)会让 messages 体积线性增长,逼近 LLM 单请求上下文上限并消耗 Guardrail token 预算。需要一个 LLM 可主动调用的工具把过往 turns 压成结构化 summary,腾出预算继续做下游任务。简历叙事:对齐 Claude Code 三引擎(skill loading + TodoWrite + Compact),CLI 场景做减法但语义保留。

**Scope:** 一个新内嵌 tool + 一段 system prompt nudge + main.py 接线 + 单元测试 + 真 LLM smoke。

---

## §1 设计决策(brainstorm 沉淀)

| Q | 选择 | 理由 |
|---|---|---|
| Q1: Day 13 方向 | C `compact_context` | 一天搞定补齐 L2 三件套,简历叙事最完整;Day 14+ 转 vlm-mcp |
| Q2: 触发方式 | A LLM 自决(纯 tool) | 守"反 if-else / 决策由 LLM 做"红线;loop.py 一行不改 |
| Q3: 压缩机制 | A handler 闭包持 messages 引用,真改 list | 唯一真省 token 的方案;Claude Code s06 原汤 |
| Q4: 范围 | A 只主 agent | subagent 短命且 budget 兜底,YAGNI |

**红线守恒:**
- "loop.py 不动":✅ tool 通过闭包改 caller 的 messages list,不动 agent_loop
- "决策由 LLM 做":✅ 触发完全由 LLM 自决,nudge 只描述适用场景不规则化

---

## §2 文件改动

| 路径 | 动作 | 责任 |
|---|---|---|
| `paperpilot/builtin_tools/compact.py` | 新建,~120 行 | `compact_context_tool` 工厂 + `_summarize` + `COMPACT_CONTEXT_NUDGE` |
| `paperpilot/main.py` | 改 ~10 行 | `messages` 提前到 `_build_tools` 之前;`_build_tools` 接 `messages_ref`;tool 加入列表;nudge 拼到 system prompt |
| `tests/builtin_tools/test_compact.py` | 新建 | handler 三路径 + lifecycle events + metadata,~6 个 case |
| `tests/test_main_integration.py` | 加 1 条 | fast 断 `compact_context` in tools + nudge in system prompt |
| `scripts/day13_smoke.py` | 新建 | 真 LLM 长对话:deep-read → 切话题 → 验证 LLM 自主调 compact_context |

---

## §3 `compact_context` 接口

### 3.1 工厂签名

```python
def compact_context_tool(
    *,
    messages_ref: list[dict],
    client_factory: Callable[[], LLMClient],
    on_event: EventCallback,
    keep_recent_turns: int = 3,
) -> Tool
```

- `messages_ref`:主 agent 的 messages list,handler 通过这个引用真删/真插。run() 把 messages 创建提到 `_build_tools` 之前,把同一 list 同时传给 `_build_tools` 和 `agent_loop`
- `client_factory`:每次 compact 时新建 LLMClient,沿用 paper_deep_read 模式
- `on_event`:emit `compact_start` / `compact_done`,与现有 lifecycle 事件命名风格一致
- `keep_recent_turns`:K=3,即保留最近 6 条 message(3 个 assistant + 3 个 user/tool_result)

### 3.2 Tool 元数据

```python
Tool(
    name="compact_context",
    description=(
        "Compress earlier conversation into a structured summary, freeing "
        "context budget. Call when the conversation is long and recent tool "
        "results have made earlier turns redundant."
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

无入参:LLM 调用时 args={};compact 范围、保留长度都是工厂层固定的。

---

## §4 Handler 算法

```python
def _handler(args: dict) -> str:
    K = keep_recent_turns
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

**关键不变量:**

1. `head = messages_ref[0]` 永远是用户原始 query。保留它让 LLM 始终能看到任务目标
2. `tail` 必含触发 compact 的 assistant turn(里面的 tool_use 等待下一轮的 tool_result)。`messages_ref[-2*K:]` 保证这点
3. `messages_ref[:] = [...]`:in-place 改 list,不创建新 list。caller(包括 agent_loop)持有的引用依然指向同一 list,改完即生效
4. handler 返回的字符串走正常 tool_result 通路 append 到 list 末尾,不破坏 tool_use/tool_result 配对

### 4.1 错误处理

`_summarize` 抛异常 → handler 直接 re-raise,不触碰 messages_ref。loop 把异常转成 `is_error=True` 的 tool_result;LLM 看到失败可决定重试或继续工作。

不在 handler 里加重试逻辑(LLMClient 内部已有);不做"压缩失败时降级压更短"这类 fallback(YAGNI)。

### 4.2 边界:too-short path

`len(messages_ref) <= 1 + 2*K`(K=3 → ≤ 7)时直接返回 `"already compact, nothing to summarize"`,messages 不动。LLM 拿到这个 tool_result 自己判断不再调。

---

## §5 `_summarize` 实现

```python
SUMMARIZE_PROMPT = """
你的任务是把下面的对话历史压缩成结构化 summary,供后续 LLM 理解上下文用。

输出格式:
## 已完成的关键步骤
<按时间顺序列出 tool calls 和重要结论,~5-10 条>

## 关键发现 / 中间结果
<事实性内容、找到的段落或数据,简洁列点>

## 待办 / 下一步
<如果 history 里 LLM 表达过未完成的计划,列出来>

要求:
- 只保留事实和决策,删掉客套和重复
- 引用具体 paper_id / chunk 摘要 / 数字结果
- 总长度控制在 800 token 内

对话历史:
""".strip()


def _summarize(middle: list[dict], client: LLMClient) -> str:
    rendered = _render_messages_for_summary(middle)
    prompt = f"{SUMMARIZE_PROMPT}\n{rendered}"
    response = client.call(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        system="You are a conversation history summarizer.",
    )
    return response.text or "(empty summary)"
```

`_render_messages_for_summary`:把 messages 里的 string content / tool_use / tool_result 都拍成纯文本,例如:
```
[user] {内容}
[assistant text] {文本}
[assistant tool_use] {tool_name}({arguments json})
[tool_result for {tool_use_id}] {内容前 500 字}
```

只为给 summary LLM 看,不参与正式协议交互。

---

## §6 Nudge

```python
COMPACT_CONTEXT_NUDGE = """
## Long conversation
When tool results pile up and earlier turns no longer matter, call
compact_context() with no arguments. It rewrites the history into a
structured summary and frees context budget. Do not call it on a short
conversation or when you are mid-step (e.g. just got a tool result and
are about to act on it).
""".strip()
```

拼到 `_build_system_prompt` 末尾,与 `RESEARCH_TODO_NUDGE` / `PAPER_DEEP_READ_NUDGE` 同位置。

---

## §7 main.py 接线

当前 `run()` 顺序需要小重构:

**改前:**
```python
def run(query, *, max_iter=8, on_event=None):
    ...
    tools, mcp = _build_tools(registry, todo_store, on_event=emit)
    try:
        messages = [{"role": "user", "content": query}]
        return agent_loop(messages, ..., tools=tools, ...)
```

**改后:**
```python
def run(query, *, max_iter=8, on_event=None):
    ...
    messages: list[dict] = [{"role": "user", "content": query}]
    tools, mcp = _build_tools(
        registry, todo_store, messages_ref=messages, on_event=emit,
    )
    try:
        return agent_loop(messages, ..., tools=tools, ...)
```

`_build_tools` 增加 `messages_ref` 参数,把它传给 `compact_context_tool(messages_ref=messages_ref, ...)`,其它 tool 不变。

`_build_system_prompt` 末尾追加 `+ "\n\n" + COMPACT_CONTEXT_NUDGE`。

---

## §8 测试矩阵

### 8.1 Fast unit (`tests/builtin_tools/test_compact.py`)

| 测试 | 覆盖 |
|---|---|
| `test_handler_too_short_returns_noop` | `len(messages) <= 1 + 2*K` → 返 "already compact",list 不动 |
| `test_handler_normal_path_replaces_middle` | 10 条 messages → handler 后变 1(head) + 1(summary) + 6(tail) = 8 条;summary 内容含 mock LLM 返的字符串 |
| `test_handler_preserves_head_and_tail_identity` | head 和 tail 的对象引用与原始 list 同 id |
| `test_handler_summarize_failure_reraises_and_keeps_messages` | mock LLMClient.call raise → handler raise,messages 长度不变 |
| `test_lifecycle_events_emitted` | `compact_start` payload 含 middle_count;`compact_done` 含 kept_recent |
| `test_tool_metadata_and_nudge` | name=="compact_context" / input_schema required==[] / nudge 含 "compact_context" |

### 8.2 Fast integration (`tests/test_main_integration.py` 增量)

| 测试 | 覆盖 |
|---|---|
| `test_compact_context_tool_and_nudge_present` | `compact_context` in tools 名字列表;`COMPACT_CONTEXT_NUDGE` 文本 in system prompt |

### 8.3 Smoke (`scripts/day13_smoke.py`,真 LLM)

链路:
1. Prompt 让 LLM 先 deep-read 一篇 paper(产生 ≥ 6-8 条 messages,含 build_index + 多次 colbert.search 的 tool_result)
2. 然后切到一个跟原 paper 无关的简短问题,nudge 暗示可以先 compact

断言:
- tracer 看到 `compact_start` 事件 ≥ 1 次
- 最终 messages list 长度 < 历史峰值
- 最终回答 quality 不崩(包含切换后问题的实质回答)
- 主 agent 不撞 guardrail

---

## §9 已知局限(Day 13 不修)

| 项 | 描述 | 后续 |
|---|---|---|
| K=3 硬编码 | 太激进会丢早期细节,太保守省不了多少 | Day 14+ 真用上撞问题再调 |
| Summary 自耗 token | 压缩 N 条要调一次 LLM,峰值 input ≈ N×平均;极端长 history 可能撞模型上下文上限 → handler raise,LLM 回退 | 不解决,文档点出 |
| Guardrail 不计 compact 内部 token | compact 自己的 LLM 调用走 LLMClient 直调,不进 agent_loop 累计 | 短期可接受;严格口径在后续加 token 上报 |
| Subagent 不享受 | subagent budget 120k 通常够;真撞墙就让它停 | YAGNI |
| Summary 渲染兜底 | `_render_messages_for_summary` 用 best-effort 平铺,极端 content 类型(图、二进制)会被字符串化 | PaperPilot 当前所有 tool 返字符串,够用 |

---

## §10 DoD

- [ ] `paperpilot/builtin_tools/compact.py` 新建,导出 `compact_context_tool` 与 `COMPACT_CONTEXT_NUDGE`
- [ ] `tests/builtin_tools/test_compact.py` 6 个 case 全绿
- [ ] `tests/test_main_integration.py` 新增 1 条 fast 断言绿
- [ ] `paperpilot/main.py` `_build_tools` 接 `messages_ref` 参数,run() 顺序调整;system prompt 含 `COMPACT_CONTEXT_NUDGE`
- [ ] `scripts/day13_smoke.py` 真 LLM 跑通,看到 ≥ 1 次 `compact_start` 事件
- [ ] 全仓 fast 测试不退化(当前 96 fast tests)
- [ ] 不动 `paperpilot/core/loop.py` / `paperpilot/core/adapter.py`(红线守恒)
