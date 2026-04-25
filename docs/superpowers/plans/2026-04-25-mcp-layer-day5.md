# MCP Layer Day 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PaperPilot 上落地完整 MCP 层骨架,Day 5 内端到端跑通"用户提问 → Main Agent loop → MCPClient → stdio → arxiv server → arXiv API → 返回"全链路。

**Architecture:** 参照 spec `docs/superpowers/specs/2026-04-25-mcp-layer-architecture-design.md`(α 方案):`agent_loop` 保持同步,async MCP SDK 关在 `MCPClient` 后台 daemon 线程的持久 asyncio loop 内。Tool 命名 `mcp__<server>__<tool>`,manifest 与 Claude Code/Desktop 的 `mcpServers` schema 兼容。L1 基座层(`paperpilot/core/`)零改动。

**Tech Stack:** Python 3.10+ / `mcp` SDK(FastMCP 高层 API + 客户端 stdio_client)/ `arxiv` 包 / `anthropic` SDK / `pytest`(可选)。

**先决条件:**
- Day 4 已完工:`paperpilot/core/{loop,adapter,guardrail}.py` 跑通(参照 `scripts/day4_smoke.py`)
- `.env` 已配 `DEEPSEEK_API_KEY`
- 工作目录:`C:\Users\Administrator\PycharmProjects\PaperPilot`
- Shell:bash(Unix 语法)

---

## 任务总览

| # | Task | 输出 | TDD |
|---|---|---|---|
| 1 | 依赖与配置 | `requirements.txt`、`.env.example` 更新 | 无 |
| 2 | 测试 fixture server | `tests/fixtures/echo_server.py` | 无 |
| 3 | `MCPClient` + 4 个单元测试 | `paperpilot/tools/mcp_client.py`、`tests/test_mcp_client.py` | 是 |
| 4 | `arxiv` server + manifest | `paperpilot/mcp_servers/{arxiv.py,__init__.py}`、`paperpilot/mcp_servers.json` | 是(集成层) |
| 5 | `paperpilot/main.py` 入口 | 顶层 entry | 否(集成代码) |
| 6 | Day 5 端到端 smoke | `scripts/day5_smoke.py` + 真实运行 | smoke |
| 7 | Daily log + spec 状态 | `daily_logs/2026-04-26.md` | 无 |

---

## Task 1: 依赖与配置

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`

- [ ] **Step 1.1: 更新 `requirements.txt`**

把 `arxiv` / `requests` / `pytest` 三行解注释 / 添加。完整文件:

```text
# ===== Day 4: Agent Loop + LLM Client =====
anthropic>=0.40.0
python-dotenv>=1.0.0

# ===== Day 5: MCP server + 第一个 server (arxiv) =====
mcp>=1.0.0
arxiv>=2.1.0
requests>=2.32.0
pytest>=8.0.0

# ===== Day 6+: PDF 解析 =====
# PyMuPDF>=1.24.0

# ===== Day 8+: ColBERT 检索 =====
# ragatouille>=0.0.9

# ===== Day 11+: 引用图谱 =====
# networkx>=3.2

# ===== Week 4: 评估 / 测试扩展 =====
```

- [ ] **Step 1.2: 更新 `.env.example` 加 MCP 超时占位**

在 "Loop 防护栏" 段落下追加。完整段落:

```env
# ===== Loop 防护栏 =====
MAX_ITERATIONS=8
BUDGET_TOKENS=50000

# ===== MCP 边界 =====
# tool 调用超时(秒)。这是边界值,不是策略——超时即 fail,LLM 自决策下一步
MCP_TOOL_TIMEOUT=60
```

- [ ] **Step 1.3: 安装新依赖**

Run:
```bash
pip install -r requirements.txt
```

Expected:`Successfully installed arxiv-x.x.x mcp-x.x.x pytest-x.x.x ...`(若已装则 `Requirement already satisfied`)。

- [ ] **Step 1.4: 验证 import**

Run:
```bash
python -c "import mcp.server.fastmcp; import arxiv; import pytest; print('ok')"
```

Expected:`ok`(stdout 单行)。

- [ ] **Step 1.5: 提交**

```bash
git add requirements.txt .env.example
git commit -m "Day 5 deps: 解注释 arxiv/pytest,加 MCP_TOOL_TIMEOUT 占位"
```

---

## Task 2: 测试 fixture server

**Files:**
- Create: `tests/fixtures/__init__.py`(空文件)
- Create: `tests/fixtures/echo_server.py`

**Why:** 给 `MCPClient` 单元测试一个轻量、可控、本地的对端 server,避免测试依赖真实 arxiv API / 网络 / LLM。`echo_server` 暴露 2 个 tool:`echo` 走成功路径,`boom` 走异常路径——刚好对应 spec §8 的 4 个 test case。

- [ ] **Step 2.1: 建 `tests/fixtures/__init__.py`**

```bash
touch tests/fixtures/__init__.py
```

(空文件,只为让 `tests/fixtures/` 成为 Python package。)

- [ ] **Step 2.2: 写 `tests/fixtures/echo_server.py`**

完整文件(~25 行):

```python
"""测试专用微型 MCP server。

只供 tests/test_mcp_client.py 用作对端,不进生产路径。
启动:python tests/fixtures/echo_server.py(stdio 模式,等待父进程)。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def echo(text: str) -> str:
    """原样返回输入文本。供测试 happy path。

    Args:
        text: 要回显的文本。
    """
    return text


@mcp.tool()
def boom() -> str:
    """故意抛异常,供测试 isError 错误传播路径。"""
    raise RuntimeError("intentional failure for testing")


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- [ ] **Step 2.3: 手动 sanity check**

Run:
```bash
python tests/fixtures/echo_server.py
```

Expected:进程不退出(stdio 等待中)。按 `Ctrl-C` 终止。无 traceback。

- [ ] **Step 2.4: 提交**

```bash
git add tests/fixtures/__init__.py tests/fixtures/echo_server.py
git commit -m "Day 5: 测试 fixture echo_server (echo + boom 两 tool,供 mcp_client 单测)"
```

---

## Task 3: `MCPClient` 实现 + 4 个单元测试(TDD 主体)

**Files:**
- Create: `paperpilot/tools/mcp_client.py`
- Create: `tests/test_mcp_client.py`

**TDD 节奏:** 4 个测试 4 个 commit。每个测试先写、跑失败、补实现、跑通过、提交。

### 3-A: namespacing + listing

- [ ] **Step 3A.1: 写测试 `test_namespacing_and_listing`**

新建 `tests/test_mcp_client.py`,完整内容:

```python
"""MCPClient 单元测试。用 tests/fixtures/echo_server.py 做对端。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperpilot.tools.mcp_client import MCPClient

FIXTURE = Path(__file__).parent / "fixtures" / "echo_server.py"


def _make_manifest(tmp_path: Path) -> Path:
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({
        "mcpServers": {
            "echo": {"command": "python", "args": [str(FIXTURE)]}
        }
    }))
    return m


def test_namespacing_and_listing(tmp_path):
    """tool 名加 mcp__echo__ 前缀;list_tools 返回正确数量。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        names = {t.name for t in c.list_tools()}
        assert "mcp__echo__echo" in names
        assert "mcp__echo__boom" in names
    finally:
        c.close()
```

- [ ] **Step 3A.2: 跑测试,期望 ImportError**

Run:
```bash
pytest tests/test_mcp_client.py::test_namespacing_and_listing -v
```

Expected:`ImportError: cannot import name 'MCPClient' from 'paperpilot.tools.mcp_client'`(或 `ModuleNotFoundError`)。

- [ ] **Step 3A.3: 写 `MCPClient` 主体**

新建 `paperpilot/tools/mcp_client.py`,完整内容(~140 行):

```python
"""MCPClient: 同步 facing 的 MCP 客户端,封装后台 asyncio 线程 + N 个 session。

设计依据:docs/superpowers/specs/2026-04-25-mcp-layer-architecture-design.md (§4)
- agent_loop 保持同步;async 复杂度关在本文件
- 每个 MCP tool 包成 paperpilot.core.adapter.Tool,handler 是 sync 闭包
- 错误:启动 hard-fail / 运行 soft-fail / 60s 超时为边界
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from paperpilot.core.adapter import Tool

MCP_TOOL_TIMEOUT = int(os.environ.get("MCP_TOOL_TIMEOUT", 60))
_INITIALIZE_TIMEOUT = 10  # 单个 server initialize 握手超时,边界值


class MCPStartupError(RuntimeError):
    """启动期任何失败(manifest 错 / spawn 失败 / initialize 超时)。"""


class MCPToolError(RuntimeError):
    """运行期 tool 调用失败(server 端 isError=True 或 RPC 错)。"""


class MCPToolTimeout(RuntimeError):
    """运行期 tool 调用超过 MCP_TOOL_TIMEOUT。"""


class MCPClient:
    def __init__(self, manifest_path: str | Path):
        self._manifest_path = Path(manifest_path)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[Tool] = []
        self._started = False
        self._closed = False

    def start(self) -> None:
        """同步阻塞,直到所有 server 起好。任何失败 raise MCPStartupError。"""
        if self._started:
            raise RuntimeError("MCPClient.start() called twice")

        loop_ready = threading.Event()

        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            loop_ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_runner, daemon=True, name="mcp-loop")
        self._thread.start()
        loop_ready.wait(timeout=5)
        if not self._loop or not self._loop.is_running():
            # 给 run_forever 一点点 tick 起来的时间
            time.sleep(0.05)
        if not self._loop or not self._loop.is_running():
            raise MCPStartupError("background event loop failed to start")

        try:
            fut = asyncio.run_coroutine_threadsafe(self._async_init(), self._loop)
            fut.result()  # 任何异常会在这里 raise
        except Exception as e:
            # 启动失败:把已经开起来的 loop 也关掉,避免泄漏线程
            self._shutdown_loop()
            if isinstance(e, MCPStartupError):
                raise
            raise MCPStartupError(str(e)) from e

        self._started = True

    async def _async_init(self) -> None:
        if not self._manifest_path.exists():
            raise MCPStartupError(f"manifest not found: {self._manifest_path}")

        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise MCPStartupError(f"manifest invalid JSON: {e}") from e

        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            raise MCPStartupError("manifest missing 'mcpServers' dict")

        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        for name, cfg in servers.items():
            if "__" in name:
                raise MCPStartupError(f"server name '{name}' must not contain '__'")
            if "command" not in cfg or "args" not in cfg:
                raise MCPStartupError(f"server '{name}' missing command/args")

            env = {**os.environ, **cfg.get("env", {})}
            params = StdioServerParameters(
                command=cfg["command"], args=list(cfg["args"]), env=env,
            )
            try:
                stdio = await self._exit_stack.enter_async_context(stdio_client(params))
                read, write = stdio
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await asyncio.wait_for(
                    session.initialize(), timeout=_INITIALIZE_TIMEOUT
                )
            except Exception as e:
                raise MCPStartupError(f"failed to start server '{name}': {e}") from e

            tools_resp = await session.list_tools()
            self._sessions[name] = session
            for t in tools_resp.tools:
                self._tools.append(Tool(
                    name=f"mcp__{name}__{t.name}",
                    description=t.description or "",
                    input_schema=t.inputSchema,
                    handler=self._make_handler(name, t.name),
                ))

    def _make_handler(self, server: str, tool: str):
        def handler(args: dict) -> str:
            assert self._loop is not None
            coro = self._sessions[server].call_tool(tool, args)
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
            try:
                result = fut.result(timeout=MCP_TOOL_TIMEOUT)
            except concurrent.futures.TimeoutError as e:
                fut.cancel()
                raise MCPToolTimeout(
                    f"mcp__{server}__{tool} > {MCP_TOOL_TIMEOUT}s"
                ) from e
            text = "\n".join(
                b.text for b in result.content if hasattr(b, "text")
            )
            if getattr(result, "isError", False):
                raise MCPToolError(f"mcp__{server}__{tool} failed: {text}")
            return text
        return handler

    def list_tools(self) -> list[Tool]:
        if not self._started:
            raise RuntimeError("MCPClient.list_tools() before start()")
        return list(self._tools)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop and self._loop.is_running() and self._exit_stack:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._exit_stack.__aexit__(None, None, None), self._loop,
                )
                fut.result(timeout=5)
            except Exception:
                pass  # 强行关闭,subprocess 会因 stdio 断开自然退出
        self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
```

- [ ] **Step 3A.4: 跑测试,期望 PASS**

Run:
```bash
pytest tests/test_mcp_client.py::test_namespacing_and_listing -v
```

Expected:`PASSED`(单条绿)。如果 hangs > 30s,先 Ctrl-C,检查 echo_server.py 能否独立启动(Task 2 step 2.3)。

- [ ] **Step 3A.5: 提交**

```bash
git add paperpilot/tools/mcp_client.py tests/test_mcp_client.py
git commit -m "Day 5: MCPClient 骨架 + namespacing 测试通过"
```

### 3-B: 同步↔异步桥接(handler roundtrip)

- [ ] **Step 3B.1: 在 `tests/test_mcp_client.py` 末尾追加测试**

```python
def test_tool_call_roundtrip(tmp_path):
    """handler 同步调用穿过线程边界,返回正确结果。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        echo_tool = next(t for t in c.list_tools() if t.name == "mcp__echo__echo")
        assert echo_tool.handler({"text": "hello"}).strip() == "hello"
    finally:
        c.close()
```

- [ ] **Step 3B.2: 跑测试,期望 PASS(handler 已在 3A 实现)**

Run:
```bash
pytest tests/test_mcp_client.py::test_tool_call_roundtrip -v
```

Expected:`PASSED`。如失败,可能是 `result.content` block 解码问题—— 检查 echo_server 是否真返回 `TextContent`。

- [ ] **Step 3B.3: 提交**

```bash
git add tests/test_mcp_client.py
git commit -m "Day 5: MCPClient handler 同步↔异步桥接测试"
```

### 3-C: 错误传播(isError → MCPToolError)

- [ ] **Step 3C.1: 在 `tests/test_mcp_client.py` 末尾追加**

```python
def test_server_error_raises_MCPToolError(tmp_path):
    """server 端 raise → CallToolResult.isError=True → handler 抛 MCPToolError。
       agent_loop 现有 except Exception 会接走,转成 is_error tool_result。"""
    from paperpilot.tools.mcp_client import MCPToolError
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        boom = next(t for t in c.list_tools() if t.name == "mcp__echo__boom")
        with pytest.raises(MCPToolError, match="intentional failure"):
            boom.handler({})
    finally:
        c.close()
```

- [ ] **Step 3C.2: 跑测试,期望 PASS(isError 检查已在 3A 实现)**

Run:
```bash
pytest tests/test_mcp_client.py::test_server_error_raises_MCPToolError -v
```

Expected:`PASSED`。如失败,在 handler 里 print `getattr(result, 'isError', None)` 和 `result.content` 调试。

- [ ] **Step 3C.3: 提交**

```bash
git add tests/test_mcp_client.py
git commit -m "Day 5: MCPClient 错误传播测试 (isError → MCPToolError)"
```

### 3-D: 启动期 hard-fail

- [ ] **Step 3D.1: 在 `tests/test_mcp_client.py` 末尾追加**

```python
def test_startup_hardfail_on_bad_command(tmp_path):
    """server 起不来 → start() 抛 MCPStartupError,不返回半完成状态。"""
    from paperpilot.tools.mcp_client import MCPStartupError
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({
        "mcpServers": {"bad": {"command": "no-such-binary-xyz", "args": []}}
    }))
    c = MCPClient(m)
    with pytest.raises(MCPStartupError):
        c.start()
```

- [ ] **Step 3D.2: 跑测试,期望 PASS**

Run:
```bash
pytest tests/test_mcp_client.py::test_startup_hardfail_on_bad_command -v
```

Expected:`PASSED`。`stdio_client` 内部对找不到的可执行会抛 `FileNotFoundError`,被 `_async_init` 的 try 捕获,包成 `MCPStartupError`。

- [ ] **Step 3D.3: 跑全部 4 个测试,验证整套绿**

Run:
```bash
pytest tests/test_mcp_client.py -v
```

Expected:4 PASSED,0 FAILED,耗时 < 15s。

- [ ] **Step 3D.4: 提交**

```bash
git add tests/test_mcp_client.py
git commit -m "Day 5: MCPClient 启动 hard-fail 测试 + 4 单测全绿"
```

---

## Task 4: `arxiv` server + manifest

**Files:**
- Create: `paperpilot/mcp_servers/arxiv.py`
- Create: `paperpilot/mcp_servers.json`

- [ ] **Step 4.1: 写 manifest `paperpilot/mcp_servers.json`**

完整文件(5 行,Day 5 只 1 项):

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

- [ ] **Step 4.2: 写 `paperpilot/mcp_servers/arxiv.py`**

完整文件(~80 行):

```python
"""arxiv-mcp: arXiv 论文搜索 server。Day 5 起。

启动:python -m paperpilot.mcp_servers.arxiv
通过 stdio 被 paperpilot.tools.mcp_client 拉起,manifest 见
paperpilot/mcp_servers.json。
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
        sort_by: "relevance"(默认) / "submittedDate" / "lastUpdatedDate"。

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

    search = arxiv.Search(
        query=full_q, max_results=max_results, sort_by=sort_enum,
    )
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

- [ ] **Step 4.3: 验证 server 单进程能起来**

Run:
```bash
python -m paperpilot.mcp_servers.arxiv
```

Expected:进程不退出(stdio 等待),stderr 无 traceback。按 `Ctrl-C` 终止。如果 import 报错(找不到 `paperpilot.mcp_servers.arxiv`),检查 `paperpilot/mcp_servers/__init__.py` 是否存在(应该已存在,Day 4 建过)。

- [ ] **Step 4.4: 写一个临时集成 sanity 脚本验证 MCPClient + arxiv 接通**

Run(一行 Python,不建文件):
```bash
python -c "
from paperpilot.tools.mcp_client import MCPClient
c = MCPClient('paperpilot/mcp_servers.json')
c.start()
try:
    print('tools:', [t.name for t in c.list_tools()])
finally:
    c.close()
"
```

Expected stdout:`tools: ['mcp__arxiv__search_papers']`,无异常,5s 内返回。

- [ ] **Step 4.5: 提交**

```bash
git add paperpilot/mcp_servers/arxiv.py paperpilot/mcp_servers.json
git commit -m "Day 5: arxiv MCP server (search_papers tool) + manifest"
```

---

## Task 5: `paperpilot/main.py` 入口

**Files:**
- Create: `paperpilot/main.py`

- [ ] **Step 5.1: 写 `paperpilot/main.py`**

完整文件(~55 行):

```python
"""PaperPilot 顶层入口。Day 5 起。

CLI:python -m paperpilot.main --query "查 3 篇 LoRA 相关近期论文"
库:from paperpilot.main import run; run(query, max_iter=8)
"""
from __future__ import annotations

import argparse
import atexit
import os
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
    """跑一次完整 agent 会话,返回最终 messages。"""
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

- [ ] **Step 5.2: import sanity check**

Run:
```bash
python -c "from paperpilot.main import run; print('import ok')"
```

Expected:`import ok`。如果 `ImportError: cannot import name 'agent_loop' from 'paperpilot.core'`,检查 `paperpilot/core/__init__.py` 的导出(Day 4 应已包含 `agent_loop`、`LLMClient`、`Guardrail`)。

- [ ] **Step 5.3: 提交**

```bash
git add paperpilot/main.py
git commit -m "Day 5: paperpilot/main.py 顶层入口 (CLI + run() 库函数)"
```

---

## Task 6: Day 5 端到端 smoke test

**Files:**
- Create: `scripts/day5_smoke.py`

- [ ] **Step 6.1: 写 `scripts/day5_smoke.py`**

完整文件(~45 行):

```python
"""Day 5 冒烟:Main Loop ←→ MCPClient ←→ stdio ←→ arxiv-mcp ←→ arXiv API
                  ←→ LLM(DeepSeek) 全链路验证。

跑一次 ~$0.01(DeepSeek),需联网 + 有效 DEEPSEEK_API_KEY。
用法:python scripts/day5_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

EXPECT_TOOL = "mcp__arxiv__search_papers"


def main() -> None:
    saw_tool_call = False
    saw_tool_result = False

    def check(kind: str, payload: dict) -> None:
        nonlocal saw_tool_call, saw_tool_result
        if kind == "tool_call" and payload["name"] == EXPECT_TOOL:
            saw_tool_call = True
        if kind == "tool_result" and payload["name"] == EXPECT_TOOL:
            saw_tool_result = True
        print(f"[{kind}] {payload}")

    messages = run(
        "找 3 篇 LoRA 微调相关的近期论文,告诉我标题、作者和一句话摘要。",
        max_iter=5,
        on_event=check,
    )

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for b in last:
            if hasattr(b, "text"):
                print(b.text)

    assert saw_tool_call, f"FAIL: 期望模型调用 {EXPECT_TOOL},未观察到"
    assert saw_tool_result, f"FAIL: 期望 {EXPECT_TOOL} 返回 result,未观察到"
    print("\n✅ Day 5 smoke PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: 跑 smoke**

Run:
```bash
python scripts/day5_smoke.py
```

Expected(顺序大致):
1. `[tool_call] {'name': 'mcp__arxiv__search_papers', 'arguments': {'query': 'LoRA ...', ...}}`
2. `[tool_result] {'name': 'mcp__arxiv__search_papers', 'content': 'arxiv_id: 2106.09685\\ntitle: LoRA: Low-Rank...'}`(arxiv_id 可能不同;关键是结构对)
3. `[turn]` 再次,LLM 综合输出 3 篇论文摘要
4. `=== FINAL ===` + 用户友好的 3 篇论文列表
5. `✅ Day 5 smoke PASSED`

总耗时 5-15 秒,~$0.01。

**故障排查:**
- 卡在 `c.start()`(>10s):多半是 arxiv server 起不来。手动跑 `python -m paperpilot.mcp_servers.arxiv` 看 stderr
- LLM 没调 tool 直接编 → 检查 system prompt 是否被截掉,或换 query("找几篇...")更明确指令性
- `MCPToolError: ... no module named 'arxiv'`:Task 1 step 1.3 的 pip install 没生效

- [ ] **Step 6.3: 验证 subprocess 已清理**

Run:
```bash
ps -ef | grep "mcp_servers.arxiv" | grep -v grep
```

Expected:无输出(arxiv subprocess 已被 close() 杀掉)。如果还有残留进程,手动 `kill -9 <pid>` 并检查 `MCPClient.close` 路径。

- [ ] **Step 6.4: 提交**

```bash
git add scripts/day5_smoke.py
git commit -m "Day 5: 端到端 smoke (Main Loop → MCPClient → arxiv → LLM)"
```

---

## Task 7: Daily log + 验收

**Files:**
- Create: `daily_logs/2026-04-26.md`(对应方案 C Day 5)

- [ ] **Step 7.1: 跑一次完整端到端验收**

把 Day 5 GO 条件清单逐条核对:

```bash
# (a) arxiv server 单跑
python -m paperpilot.mcp_servers.arxiv  # Ctrl-C 终止,无 traceback

# (b) MCPClient list_tools 返回正确
python -c "from paperpilot.tools.mcp_client import MCPClient; c=MCPClient('paperpilot/mcp_servers.json'); c.start(); print([t.name for t in c.list_tools()]); c.close()"
# 期望:['mcp__arxiv__search_papers']

# (c) 单元测试 4 全绿
pytest tests/test_mcp_client.py -v
# 期望:4 passed

# (d) 端到端 smoke
python scripts/day5_smoke.py
# 期望:✅ Day 5 smoke PASSED

# (e) CLI 入口能跑
python -m paperpilot.main --query "找 2 篇 attention is all you need 相关论文"
# 期望:正常返回,有 FINAL 块
```

5 条全过 → Day 5 收工。任何一条失败,回到对应 Task 修。

- [ ] **Step 7.2: 写 `daily_logs/2026-04-26.md`**

模板对照 `daily_logs/2026-04-25.md`,完整内容:

```markdown
# 2026-04-26 (对应方案 C Day 5)

## 今日目标
落地 MCP 层骨架,端到端跑通 arxiv-mcp:用户提问 → Main Loop → MCPClient → stdio → arxiv server → arXiv API → LLM 综合输出。

## 完成项

### 设计文档
- `docs/superpowers/specs/2026-04-25-mcp-layer-architecture-design.md` — 11 节完整 MCP 层骨架(Q1-Q6 + α 方案 + 5 server 边界 + 文件布局 + MCPClient 契约 + manifest schema + arxiv 实现 + 测试策略 + 防越界清单 + 后续路标)
- `docs/superpowers/plans/2026-04-25-mcp-layer-day5.md` — Day 5 实施计划(7 任务,TDD 4 单测)

### 代码
- `paperpilot/tools/mcp_client.py` — MCPClient (~140 行),后台 daemon 线程 + 持久 asyncio loop,把 5(目前 1) 个 stdio session 包成 sync `Tool` 暴露
- `paperpilot/mcp_servers/arxiv.py` — 第一个 MCP server (~80 行),FastMCP + arxiv 包,1 个 tool: search_papers
- `paperpilot/mcp_servers.json` — manifest,Claude Desktop 兼容 schema
- `paperpilot/main.py` — 顶层入口 (~55 行),CLI + run() 库函数双角色
- `scripts/day5_smoke.py` — 端到端冒烟,只断言"链路触发"
- `tests/fixtures/echo_server.py` — 测试专用微型 server
- `tests/test_mcp_client.py` — 4 单元测试,覆盖 namespacing / 同步异步桥接 / isError / startup hard-fail
- `requirements.txt` — 解注释 arxiv / pytest;加 requests
- `.env.example` — 加 MCP_TOOL_TIMEOUT 占位

### 验证
全部 5 条 GO 条件通过:
- 单元测试 4 PASSED(~5s,无外部依赖)
- 端到端 smoke PASSED(~10s,$0.01)
- arxiv subprocess 在程序退出后正常清理

## 关键决策(架构级)

1. **agent_loop 保持同步,async 关在 mcp_client 内** (α 方案) — PaperPilot 无真并发需求,改全链路 async 是为 SDK 历史包袱付税。后台 daemon 线程 + 持久 event loop + run_coroutine_threadsafe 桥接,~80 行隔离全部 async 复杂度,L1 基座层零改动
2. **tool 命名 mcp__<server>__<tool>** — 沿用 Claude Code 实际约定;名字自带 server 信息,避免撞名
3. **错误三段式:启动 hard-fail / 运行 soft-fail / 60s 超时为边界** — 启动失败=配置 bug 必须吵醒;运行错误由 LLM 决策应对,client 不内置重试逻辑(守住"LLM 做决策"红线)
4. **Day 5 只发 1 个 tool (search_papers)** — get_paper_by_id 等 Day 7 graph-mcp 真实需求出现时再加,不预留口

## 今日走过的弯路 / 教训
(留空,完工再补;如无弯路则写"无")

## 明天(2026-04-27,对应方案 C Day 6)

- 第二个 MCP server:colbert-mcp(本地 Late Interaction 检索)
- brainstorm session 设计 colbert tool 列表(build_index / search 之类)
- 落地 ColBERT 索引基础架构(ragatouille)

## 未解决项 / TODO

- (按实际情况填)
```

- [ ] **Step 7.3: 提交 daily log + plan + spec**

```bash
git add daily_logs/2026-04-26.md docs/superpowers/specs/2026-04-25-mcp-layer-architecture-design.md docs/superpowers/plans/2026-04-25-mcp-layer-day5.md
git commit -m "Day 5: spec + plan + daily log"
```

- [ ] **Step 7.4: 最终 git status 检查**

Run:
```bash
git status
git log --oneline -10
```

Expected:`git status` 无未提交变更(除可能的 `__pycache__/`);`git log` 看见 Day 5 的连续 8-10 个 commit(7 task 各 1+ commit)。

---

## 自审清单(我已检查)

**1. Spec 覆盖检查:**
- §3 文件布局 → Task 2/3/4/5/6 覆盖所有新建文件 ✓
- §4 MCPClient 契约(start/list_tools/close + handler 闭包 + 错误边界表) → Task 3 全覆盖,4 个测试对应错误表关键行 ✓
- §5 manifest schema + spawn 流程 → Task 4.1 + Task 3 实现的 `_async_init` ✓
- §6 arxiv server 完整代码 → Task 4.2 直接引用 spec 代码 ✓
- §7 main.py 完整代码 → Task 5.1 直接引用 spec 代码 ✓
- §8 测试策略(必做 smoke + 可选 4 单测)→ Task 3(单测)+ Task 6(smoke)全覆盖 ✓
- §10 防越界清单 → 实施代码无任何越界(无 logging / 无 retry / 无 streaming / 无 history)✓

**2. Placeholder 扫描:** 无 TBD / TODO / "fill in details"。`daily_logs` 模板里"今日走过的弯路"留空是模板字段,完工时再填,不算 placeholder。✓

**3. 类型/方法名一致性:**
- `MCPClient` / `MCPStartupError` / `MCPToolError` / `MCPToolTimeout` 在 mcp_client.py 定义,test_mcp_client.py 里 import,大小写完全一致 ✓
- `Tool` 字段(`name` / `description` / `input_schema` / `handler`)和 `paperpilot/core/adapter.py` 已存在的 `@dataclass` 完全一致(Task 0 已读过)✓
- `mcp__<server>__<tool>` 在所有出现处一致(spec § 各处、Task 3.1 测试断言、Task 4 manifest)✓
- `MCP_TOOL_TIMEOUT` 在 mcp_client.py / .env.example 字面一致 ✓

---

## Plan 完成,执行选项

**两种执行方式,选一个:**

**1. Subagent-Driven(推荐)** — 我对每个 task 派一个 fresh subagent,task 间做 review,迭代快、上下文干净。

**2. Inline Execution** — 在当前会话直接 task by task 执行,中间设 checkpoint 让你 review。

哪种?
