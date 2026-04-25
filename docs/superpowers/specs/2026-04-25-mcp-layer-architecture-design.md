# PaperPilot MCP 层架构设计

| 项 | 值 |
|---|---|
| 日期 | 2026-04-25(方案 C Day 4 完工日,设计 Day 5+ 起的 MCP 层) |
| 范围 | 完整 MCP 层骨架(5 个 server + mcp_client + manifest + main 入口);Day 5 落地 arxiv-mcp |
| 不在范围 | colbert / graph / pdf / vlm 这 4 个 server 各自的 tool 签名(各 server 上线日单独 brainstorm);L2 内嵌 tool(Week 2);UI(待定) |
| 状态 | Draft,待用户 review |

---

## 1. 目标

让 PaperPilot 的 Main Agent loop 通过 MCP 协议调用 5 个外部 server(`arxiv` / `colbert` / `graph` / `pdf` / `vlm`),同时:

1. **L1 基座层(`paperpilot/core/`)零污染** —— Day 4 已完工的 `loop.py` / `adapter.py` / `guardrail.py` 一行不改
2. **架构与 Claude Code / Claude Desktop 同构** —— 复用其 `mcpServers` JSON manifest schema、`mcp__<server>__<tool>` 命名约定、stdio + subprocess 部署形态;简历可直接讲"上游协议级兼容"
3. **不引入推测性抽象** —— 一次只解决当下问题,YAGNI 严守。每个未来扩展都标"何时触发新增"
4. **不引入规则路由 / 自动重试** —— 选 server、选 tool、出错后下一步动作,全由 LLM 决策;`mcp_client` 只传话不决策

---

## 2. 关键架构决策(Q1–Q6 + 方案 α)

| ID | 决策 | 选项 | 主要理由 |
|---|---|---|---|
| Q1 | 传输层 | **stdio + 客户端 spawn 子进程** | 与 Claude Code/Desktop 实际形态一致;本地零网络开销;5 个进程级隔离 |
| Q2 | 工具暴露 | **启动期全连 + 全暴露** | LLM 一次看见 ~19 个 tool 自由决策,纯粹符合"LLM 做决策"红线;tool schema 体积 3-5KB 可忽略 |
| Q3 | 命名 | **`mcp__<server>__<tool>` 前缀** | Claude Code 实际约定;名字自带 server 信息,避免撞名 |
| Q4 | 错误 | **启动 hard-fail / 运行 soft-fail / 60s 超时为边界** | 启动失败=配置 bug,必须吵醒;运行错误由 LLM 决策应对,不引入内置重试逻辑 |
| Q5 | 清单 | **JSON manifest,字段 `command/args/env`** | 与 Claude Desktop `claude_desktop_config.json` schema 一致 |
| Q6 | 边界 | 见下表 5 个 server 的"做/不做" | 单一职责;arxiv 只管元数据,pdf-parse 兼下载,vlm 只管看图 |
| α | sync↔async 桥接 | **`agent_loop` 保持同步,async 关在 `MCPClient` 后台线程内** | PaperPilot 无真并发需求;改全链路 async 是为 SDK 历史包袱付税 |

### 5 个 Server 的边界矩阵(Q6)

**命名约定**:`<short>` 是 manifest key + Python 模块名(`paperpilot.mcp_servers.<short>`);`<short>-mcp` 是本文档叙述性后缀,不出现在代码里。Tool 全名形如 `mcp__<short>__<tool>`(如 `mcp__pdf__extract`)。

| Server (`<short>`) | 核心职责 | 不做 |
|---|---|---|
| **`arxiv`** | 按关键词/作者/类目搜 arXiv;返回 metadata(标题/摘要/作者/pdf_url) | 不下载 PDF;不做语义匹配 |
| **`pdf`** | URL/路径 → PyMuPDF 解析,返回结构化文本(段落+章节);**顺手下载**(URL→缓存→解析) | 不做 OCR;不做图表抽取(给 vlm) |
| **`colbert`** | ColBERT 索引增量构建 + Late Interaction 检索(query → top-k 段落) | 不做 PDF 解析(吃 pdf 输出);不做 rerank 之上逻辑 |
| **`graph`** | 拉引用关系 → NetworkX 图;查引用/被引、最短路径、影响力子图 | 不做论文搜索;不做图可视化 |
| **`vlm`** | 图片(PDF 页或路径)→ Qwen-VL 返回 caption + 关键读数 | 不做 PDF 定位/抠图(pdf 给 bbox);不做检索 |

---

## 3. 文件布局

### 新增/改动(Day 5 落地范围)

| 路径 | 状态 | 作用 | 预估行数 |
|---|---|---|---|
| `paperpilot/tools/mcp_client.py` | 新建 | sync-facing MCP 客户端,封装后台 asyncio 线程 + 5 个 session | ~120 |
| `paperpilot/mcp_servers.json` | 新建 | server spawn 清单(Claude Code `mcpServers` schema) | ~5(Day 5 只 1 项) |
| `paperpilot/mcp_servers/arxiv.py` | 新建 | 第一个 MCP server | ~100 |
| `paperpilot/main.py` | 新建 | 顶层入口,合并 tools + 启动 agent_loop | ~50 |
| `scripts/day5_smoke.py` | 新建 | Day 5 端到端冒烟 | ~40 |
| `tests/fixtures/echo_server.py` | 新建(可选) | 测试专用微型 server | ~25 |
| `tests/test_mcp_client.py` | 新建(可选) | MCPClient 单元测试 4 个 case | ~80 |
| `requirements.txt` | 修改 | 解注释 `arxiv>=2.1.0` + `requests>=2.32.0`;按需加 `pytest` | +2~3 行 |

### 不动(L1 零污染纪律)

- `paperpilot/core/loop.py` —— 一行不改
- `paperpilot/core/adapter.py` —— 一行不改
- `paperpilot/core/guardrail.py` —— 一行不改
- `scripts/day4_smoke.py` —— 保留为 L1 回归基线

### 布局决策

1. `mcp_client.py` 放 `paperpilot/tools/`,不放 `core/` —— MCP 是"一种 tool 来源",与 L2 内嵌 tool 平级
2. `arxiv.py` 不带 `_server` 后缀 —— 在 `mcp_servers/` 目录下文件名冗余
3. `main.py` 放顶层包,不放 `scripts/` —— 它是生产入口
4. **暂不建 `paperpilot/mcp_servers/_base.py`** —— rule of three,等 Day 6/7 写完 colbert/graph 看是否真有重复 boilerplate 再抽

---

## 4. `MCPClient` 接口契约

### 公开 API(3 个方法)

```python
class MCPClient:
    def __init__(self, manifest_path: str | Path = "paperpilot/mcp_servers.json"):
        """只读取路径,不做任何 IO。"""

    def start(self) -> None:
        """阻塞直到所有 server spawn + initialize + list_tools 完成。
        任何失败 → 清理已起的 subprocess,raise MCPStartupError。"""

    def list_tools(self) -> list[Tool]:
        """返回包装好的 paperpilot.core.adapter.Tool 列表。
        每个 Tool.handler 是 sync callable,内部转发到后台 loop。
        start() 之前调用 → RuntimeError。"""

    def close(self) -> None:
        """幂等。aclose AsyncExitStack → stop loop → join 线程。
        建议 atexit.register。"""
```

`call_tool` 不公开 —— 通过 `Tool.handler` 闭包间接被 `agent_loop` 调用,使 `agent_loop` 完全无视 MCP 概念。

### 内部结构

```
MCPClient
├── _loop: asyncio.AbstractEventLoop      (后台 daemon 线程的持久 loop)
├── _thread: threading.Thread(daemon=True)
├── _exit_stack: AsyncExitStack           (管 5 个 stdio_client + ClientSession 生命周期)
├── _sessions: dict[str, ClientSession]   (server_name → session)
└── _tools: list[Tool]                    (start 后填充)
```

### Handler 闭包伪码

```python
def handler(args: dict) -> str:
    coro = self._sessions[server].call_tool(tool, args)
    fut  = asyncio.run_coroutine_threadsafe(coro, self._loop)
    try:
        result = fut.result(timeout=MCP_TOOL_TIMEOUT)  # 见下方常量
    except concurrent.futures.TimeoutError:
        fut.cancel()
        raise MCPToolTimeout(f"mcp__{server}__{tool} > 60s")
    if getattr(result, "isError", False):
        err = "\n".join(b.text for b in result.content if hasattr(b, "text"))
        raise MCPToolError(f"mcp__{server}__{tool} failed: {err}")
    return "\n".join(b.text for b in result.content if hasattr(b, "text"))
```

`MCPToolError` / `MCPToolTimeout` 都是简单 `class _(Exception)`;被 `agent_loop:68` 的 `except Exception` 自动捕获 → `is_error=True` tool_result → LLM 自决策。

**`MCPTOOL_TIMEOUT` 常量**:模块级 `MCP_TOOL_TIMEOUT = int(os.environ.get("MCP_TOOL_TIMEOUT", 60))`(秒)。`.env.example` 同步加占位。这是边界值,不可配置策略。

### 错误边界表

| 场景 | 谁抛 | 谁处理 | 用户看见 |
|---|---|---|---|
| manifest 缺失/格式错 | start() | 进程退出 | `MCPStartupError: manifest invalid: ...` |
| server spawn 失败 | start() + 清理 | 同上 | `MCPStartupError: failed to spawn '<name>': ...` |
| initialize 握手 >10s | start() + 清理 | 同上 | `MCPStartupError: '<name>' initialize timeout` |
| list_tools 返回空 | warn,不报错 | 继续 | stderr warning |
| 运行期 RPC 失败 | handler raise | agent_loop 通用 except | LLM 看 error tool_result 自决策 |
| 运行期超时 60s | handler raise MCPToolTimeout | 同上 | 同上 |
| server 进程死掉 | 下次 handler raise | 同上 | 不自动重连,LLM 看到连续 error 自然换工具 |

### 进程模型

```
 主线程                    后台 daemon 线程        OS subprocess
┌────────┐  threadsafe  ┌──────────┐         ┌───────────┐
│ main   │─────────────>│ asyncio  │──stdio─>│ arxiv     │
│ agent_ │              │ loop     │──stdio─>│ colbert   │
│ loop   │<── result ───│  5×      │──stdio─>│ graph     │
│ (sync) │              │ Session  │──stdio─>│ pdf       │
└────────┘              └──────────┘──stdio─>│ vlm       │
                                              └───────────┘
```

退出时:atexit → close() → aclose AsyncExitStack → 5 subprocess 收 SIGTERM → loop.stop → 线程 join。`fut.result(timeout=5)` 兜底防 aclose 卡死。

---

## 5. `mcp_servers.json` 清单

### Schema(锁死 3 字段,不为未来留口)

| Key | 必填 | 含义 |
|---|---|---|
| `command` | ✅ | 可执行文件名(`python` / `node` / 绝对路径) |
| `args` | ✅ | argv 列表;惯用 `-m paperpilot.mcp_servers.<name>` |
| `env` | 否 | server 进程级 env;合并继承父进程 env |

不加 `transport` / `port` / `url`。需要时再加。

### Day 5 实际内容

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "python",
      "args": ["-m", "paperpilot.mcp_servers.arxiv"]
    }
  }
}
```

Day 6/7/8/9 各 server 上线时,对应 commit 追加 entry,不预先放注释占位。

### Spawn 流程(`MCPClient.start()` 内部)

```
1. json.load(manifest_path)["mcpServers"]
2. 校验:dict 类型;每条有 command + args;**server name 不含 "__"**(因为它是 namespace 分隔符)
3. _async_init() 提交后台 loop:
     AsyncExitStack 开启
     for name, cfg in manifest:
         params = StdioServerParameters(command, args, env=父env|cfg.env)
         stdio  = await stack.enter_async_context(stdio_client(params))
         session = await stack.enter_async_context(ClientSession(*stdio))
         await asyncio.wait_for(session.initialize(), timeout=10)
         tools  = (await session.list_tools()).tools
         self._sessions[name] = session
         for t in tools:
             self._tools.append(Tool(
                 name=f"mcp__{name}__{t.name}",
                 description=t.description or "",
                 input_schema=t.inputSchema,
                 handler=self._make_handler(name, t.name),
             ))
4. 任何步抛异常 → AsyncExitStack 自动 aclose → re-raise
5. 全部成功 → start() 返回
```

### Server stderr 处理

子进程 stderr **直接继承到父进程 stderr**(`stdio_client` 默认行为)。stdout 是 JSON-RPC 通道,任何 print 必须走 stderr。Day 12 再考虑分流到文件。

---

## 6. `arxiv` server(Day 5 落地)

### SDK 选择:`mcp.server.fastmcp.FastMCP`

签名+类型注解+docstring 自动推导 input_schema,Google-style `Args:` 块解析为参数描述。~100 行 vs 低层 API ~250 行。

### Day 5 tool 列表:**1 个**

```
mcp__arxiv__search_papers(query, max_results=10, category=None, sort_by="relevance")
```

`get_paper_by_id` 等 Day 7 graph-mcp 出现"按 id 反查"真实需求时再加。

### `paperpilot/mcp_servers/arxiv.py`

```python
"""arxiv-mcp: arXiv 论文搜索。Day 5 起。

启动:python -m paperpilot.mcp_servers.arxiv
被 paperpilot.tools.mcp_client 通过 stdio 拉起。
"""
from __future__ import annotations

import arxiv
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arxiv")
_client = arxiv.Client(page_size=50, delay_seconds=3)


@mcp.tool()
def search_papers(
    query: str,
    max_results: int = 10,
    category: str | None = None,
    sort_by: str = "relevance",
) -> str:
    """按关键词搜索 arXiv 论文,返回元数据列表。

    Args:
        query: 搜索关键词,如 "LoRA fine-tuning"。支持 arXiv 查询语法:
            au:作者名 / ti:标题 / abs:摘要 / cat:类目。
        max_results: 返回结果数,默认 10,上限 50。
        category: 限定 arXiv 类目,如 "cs.CL"/"cs.LG"。None 不限。
        sort_by: "relevance" / "submittedDate" / "lastUpdatedDate"。

    Returns:
        每篇论文一段,字段:arxiv_id / title / authors / published /
        primary_category / pdf_url / abstract,多篇用 --- 分隔。
    """
    max_results = min(max_results, 50)
    full_q = f"({query}) AND cat:{category}" if category else query
    sort_enum = {
        "relevance": arxiv.SortCriterion.Relevance,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }.get(sort_by, arxiv.SortCriterion.Relevance)

    search = arxiv.Search(query=full_q, max_results=max_results, sort_by=sort_enum)
    parts = [
        f"arxiv_id: {r.get_short_id()}\n"
        f"title: {r.title}\n"
        f"authors: {', '.join(a.name for a in r.authors)}\n"
        f"published: {r.published.date()}\n"
        f"primary_category: {r.primary_category}\n"
        f"pdf_url: {r.pdf_url}\n"
        f"abstract: {r.summary.strip()}"
        for r in _client.results(search)
    ]
    return "\n---\n".join(parts) if parts else f"No papers found for: {full_q}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 设计决策

1. `arxiv.Client(delay_seconds=3)` —— arxiv API 礼貌延迟,边界值,模块级常量
2. `max_results` 内置 cap=50 —— 防 LLM 瞎填导致 token 撑爆
3. **不写 try/except** —— 异常由 FastMCP → `isError=True` → MCPToolError → agent_loop 通用 except,全链路无规则逻辑
4. 不暴露 `arxiv.Search` 的 `id_list` 等参数(YAGNI)
5. 返回纯文本(`key: value\n...---\n...`),不返 JSON —— LLM 对结构化文本鲁棒,FastMCP 自动包成 TextContent

### Day 5 收工 GO 条件

- [ ] `python -m paperpilot.mcp_servers.arxiv` 单跑能起来,stdio 等待正常
- [ ] `MCPClient().start()` 拉起 arxiv subprocess 无错误,`list_tools()` 返长度 1 列表,name=`mcp__arxiv__search_papers`
- [ ] `scripts/day5_smoke.py` 跑 "找 3 篇 LoRA 相关近期论文":观察到完整 ReAct 链路 + 真实 arxiv 数据返回
- [ ] `MCPClient.close()` 后 arxiv subprocess 在系统中消失

---

## 7. `paperpilot/main.py` 集成点

### 文件内容(完整,~50 行)

```python
"""PaperPilot 顶层入口。Day 5 起。

CLI:python -m paperpilot.main --query "查 3 篇 LoRA 相关近期论文"
库:from paperpilot.main import run; run(query, max_iter=8)
"""
from __future__ import annotations

import argparse, atexit, os
from pathlib import Path

from dotenv import load_dotenv

from paperpilot.core import Guardrail, LLMClient, agent_loop
from paperpilot.tools.mcp_client import MCPClient

MANIFEST_PATH = Path(__file__).parent / "mcp_servers.json"

SYSTEM_PROMPT = """你是 PaperPilot,一个学术论文研究助手。

工作原则:
- 有 tool 可用时优先调 tool;不要自己编造论文标题、作者或 arxiv id
- 一次只解决用户问的事,不主动扩展任务范围
- tool 报错时,根据错误信息决定:重试(换参数) / 换工具 / 告诉用户失败原因
""".strip()


def _default_logger(kind: str, payload: dict) -> None:
    if kind == "tool_call":
        print(f"  → {payload['name']}({payload['arguments']})")
    elif kind == "tool_result":
        print(f"  ← {payload['name']}: {payload['content'][:120]}...")
    elif kind == "guardrail_stop":
        print(f"  ⚠ guardrail: {payload['reason']}")


def run(query: str, *, max_iter: int = 8, on_event=None) -> list[dict]:
    load_dotenv()

    mcp = MCPClient(MANIFEST_PATH)
    mcp.start()
    atexit.register(mcp.close)

    # Day 5: tools 只来自 MCP。Day 8+ 加 L2 inline tools 时改这一行:
    #   tools = INLINE_TOOLS + mcp.list_tools()
    tools = mcp.list_tools()

    messages = [{"role": "user", "content": query}]
    return agent_loop(
        messages,
        system=SYSTEM_PROMPT,
        tools=tools,
        client=LLMClient(),
        guardrail=Guardrail(
            max_iterations=max_iter,
            budget_tokens=int(os.environ.get("BUDGET_TOKENS", 50_000)),
        ),
        on_event=on_event or _default_logger,
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="paperpilot")
    p.add_argument("--query", required=True)
    p.add_argument("--max-iter", type=int, default=8)
    args = p.parse_args()
    messages = run(args.query, max_iter=args.max_iter)
    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for b in last:
            if hasattr(b, "text"):
                print(b.text)
    else:
        print(last)


if __name__ == "__main__":
    main()
```

### 集成关键点

- **`tools = mcp.list_tools()`** 是 MCP / inline tool 合流点。Day 8 加第一个 L2 inline tool 时变成 `INLINE_TOOLS + mcp.list_tools()`
- **`atexit.register(mcp.close)`** 处理 Ctrl-C / 正常退出 / 未捕获异常退出。`close()` 幂等
- **`SYSTEM_PROMPT` 暂留 main.py**,Week 2 加 skill loading 膨胀后再迁 `paperpilot/agent/system_prompt.py`
- **CLI 走 `python -m paperpilot.main`**,不引入 console_script entry point
- **不引入 logging 模块、retry、streaming、history 持久化** —— YAGNI

---

## 8. 测试策略

### 三层金字塔

```
                   ┌────────────────────────┐
   Week 4          │  full pytest suite     │  不在 Day 5 范围
                   │  (~50 cases)           │
                   └────────────────────────┘
                              ↑
                   ┌────────────────────────┐
   Day 5 必做      │  end-to-end smoke       │  scripts/day5_smoke.py
                   └────────────────────────┘
                              ↑
                   ┌────────────────────────┐
   Day 5 可选      │  MCPClient unit tests   │  tests/test_mcp_client.py
                   │  (~4 cases via fixture) │
                   └────────────────────────┘
```

### Day 5 必做:`scripts/day5_smoke.py`

依赖:网络 + DEEPSEEK_API_KEY + 真实 arxiv API。成本 ~$0.01,耗时 5-15s。

断言哲学:**只断"链路触发",不断"返回 N 篇 LoRA 论文"**(LLM 输出文本不稳;arxiv 检索结果随时间变)。具体:
- 模型调用了 `mcp__arxiv__search_papers`(证 namespacing + tool 注入)
- 收到了 tool_result(证 stdio 往返 + subprocess 工作)
- 模型自然停止,没被 guardrail 截断

### Day 5 可选:`tests/test_mcp_client.py`(无 LLM 无网络)

通过 `tests/fixtures/echo_server.py`(~25 行,FastMCP `echo` + `boom`)做对端,4 个 case 覆盖:
1. **namespacing**:`mcp__echo__echo` / `mcp__echo__boom` 出现在 `list_tools`
2. **同步↔异步桥接**:`echo_tool.handler({"text":"hello"}) == "hello"`(α 方案核心)
3. **soft-fail**:`boom.handler({})` 抛 `MCPToolError`(Q4 运行期分支)
4. **hard-fail**:`MCPClient` 配错命令 → `start()` 抛异常(Q4 启动期分支)

未覆盖:超时(写起来需要慢 server,Week 4 补)。

依赖:`pytest`。requirements.txt 解注释。

### Week 4 backlog(只列不做)

- `test_arxiv_server.py` —— mock `arxiv.Client`,验证 query 拼接、cap、空结果
- `test_main_integration.py` —— mock `LLMClient.call`,验证 `run()` 全流程
- `test_guardrail_with_mcp.py` —— mock LLM 让其连续重复 mcp tool 调用,验证 guardrail
- LLM-as-Judge 评估(方案 C 文档)

### 不写

- 不 mock asyncio loop(用真子进程 fixture,fixture server ~3s,完全可接受)
- 不为 `_default_logger` / `SYSTEM_PROMPT` 写测试(纯 print / 字符串常量)
- 不为 manifest schema 校验单独写测试(`test_startup_hardfail` 已覆盖)

---

## 9. 简历/面试讲法(本设计的口径基线)

> "PaperPilot 的 MCP 层与 Claude Code / Claude Desktop 上游同构:复用其 `mcpServers` JSON manifest schema、`mcp__<server>__<tool>` 命名约定、stdio + subprocess 部署形态。Main Agent loop 保持同步(~80 行,与 Claude Code s01 对齐),MCP SDK 的 asyncio 复杂度被隔离在 `mcp_client.py`(~120 行)的后台线程 + 持久 event loop 内,L1 基座层零污染。所有错误处理(选 server / 选 tool / 出错应对)由 LLM 决策,client 只守边界(超时、namespace 校验、启动失败 hard-fail),不做规则路由或自动重试。"

---

## 10. 不做的事(防越界清单)

- ❌ 不引入 LLMAdapter ABC 或 OpenAI 兼容路径
- ❌ 不引入 server 端 `_base.py` 共享脚手架(rule of three 触发前)
- ❌ 不引入 `connect_mcp` meta-tool 让 LLM 选 server(伪懒加载,且引入额外回合)
- ❌ 不在 `mcp_client` / handler 内部做自动重试或退避
- ❌ 不在 manifest schema 加 `transport` / `port` / `url` 字段
- ❌ 不在 `main.py` 引入 `logging` 模块
- ❌ 不为 ImageContent / ResourceLink 等非文本 tool_result 设计接口(Day 15+ vlm-mcp 来时再说)
- ❌ 不做 streaming 输出
- ❌ 不做 conversation history 持久化(Week 2 多轮交互再设计)

---

## 11. 后续路标(供 writing-plans 参考)

| 触发日 | 动作 |
|---|---|
| Day 5 | 落地本设计 1.0;`arxiv` 上线;smoke pass |
| Day 6 | `colbert` brainstorm + 上线;manifest 追加 entry |
| Day 7 | `graph` brainstorm + 上线;**此时考虑加 `get_paper_by_id` 到 `arxiv`**(graph 反查需要) |
| Day 8 | `pdf` brainstorm + 上线;**此时复盘是否抽 `mcp_servers/_base.py`**(rule of three) |
| Day 9 | `vlm` brainstorm + 上线;考虑 ImageContent 在 mcp_client 的处理 |
| Week 2 | L2 inline tools 上线 → `main.py` 的 `tools=` 行追加 `INLINE_TOOLS` |
| Day 12 | 考虑 server stderr 分流到文件 |
| Week 4 | 形成 pytest 测试矩阵 + LLM-as-Judge 评估 |

---

*Design end. 评审通过后由 writing-plans 出 Day 5 实施计划。*