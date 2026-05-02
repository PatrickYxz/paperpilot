# PaperPilot Day 8 graph-mcp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PaperPilot 加第三个 MCP server `graph-mcp` (NetworkX 引用关系图), 接 Semantic Scholar Graph API 拉 references / citations, 暴露 4 个 tool (`build_graph` / `get_neighbors` / `get_shortest_path` / `get_common_citations`) 让 LLM 在 main loop 里能做基于引用拓扑的学术导航 (找两篇 paper 之间的引用桥梁、共引经典文献、邻居 paper)。

**Architecture:** 独立 MCP server 进程, 包式三层 (`server.py` 协议 / `graph_manager.py` NetworkX+pickle / `ss_client.py` httpx+retry); server 间不互通信, 串联只在 LLM 那一层; pickle 持久化到 `data/graph/citation_graph.pkl` (atomic rename); SS API key 从 `SEMANTIC_SCHOLAR_API_KEY` env 读, 可空; mcp_client 零改动 (Day 6 已设 180s 全局超时, build_graph 单次 SS batch <30s 不撞)。

**Tech Stack:** Python 3.12, mcp SDK (FastMCP, 已用), NetworkX 3.x (DiGraph + shortest_path), httpx (sync HTTP client), pytest。Spec: `docs/superpowers/specs/2026-05-02-graph-mcp-design.md`。

**Spec 与现状的几处对齐 (plan 决策):**
1. 子包式布局 (`paperpilot/mcp_servers/graph/{__init__.py, server.py, graph_manager.py, ss_client.py}`) — 与 colbert-mcp 一致, 与 arxiv 单文件不同
2. **不写独立 `manifest.json`** — FastMCP 自动从 docstring 抽 schema (沿用 arxiv/colbert 现状)
3. mcp_client 不改, mcp_servers.json 加一行 graph entry
4. fixture mode 用 env `PAPERPILOT_SS_FIXTURE_DIR` 切换 (集成测试 / smoke 之外的本地实验也用得上)
5. 异常类全在 ss_client.py 和 graph_manager.py 内部定义, server.py 只 catch / 透传

---

## Task 1: 加依赖 + 测试目录 + 真抓 SS API fixture

**Files:**
- Modify: `requirements.txt`
- Create: `tests/fixtures/ss_attention_bert.json` (一次性手抓 SS API response)

预估: ~15-25min (含 SS API 一次真请求)

- [ ] **Step 1.1: 改 `requirements.txt` 加 Day 8 依赖**

在 requirements.txt 末尾追加:

```
# ===== Day 8: 引用关系图 =====
# graph-mcp 用 NetworkX 存图 + httpx 调 Semantic Scholar Graph API
networkx>=3.2
httpx>=0.27
```

- [ ] **Step 1.2: 装依赖**

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Expected: 装 networkx + httpx (及 httpx 依赖 anyio/httpcore/h11 等; 都很轻, 几秒搞定; 大概率 anyio 已被 mcp SDK 装过, 仅 networkx + httpx + httpcore + h11 是真新增)。

Verify:
```bash
.venv/Scripts/python.exe -c "import networkx, httpx; print(f'nx={networkx.__version__} httpx={httpx.__version__}')"
```
Expected: 形如 `nx=3.x.y httpx=0.27.x`。

- [ ] **Step 1.3: 真调 SS API 一次, 抓 fixture**

SS API key 可空 (unauth shared pool, demo 量级足够)。如果你已有 key, 设环境变量再跑; 没有就直接跑。

Run (Windows PowerShell, 单行):

```powershell
.venv/Scripts/python.exe -c @"
import json, httpx, os
url = 'https://api.semanticscholar.org/graph/v1/paper/batch'
fields = 'title,year,authors,externalIds,references.externalIds,references.title,references.year,references.authors,citations.externalIds,citations.title,citations.year,citations.authors'
ids = ['ARXIV:1706.03762', 'ARXIV:1810.04805']
headers = {'Content-Type': 'application/json'}
api_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
if api_key:
    headers['x-api-key'] = api_key
r = httpx.post(url, params={'fields': fields}, json={'ids': ids}, headers=headers, timeout=30.0)
r.raise_for_status()
data = r.json()
with open('tests/fixtures/ss_attention_bert.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f'fetched {len(data)} papers; first: {data[0][\"title\"][:50] if data[0] else None}')
"@
```

(用 bash 等价: `python -c "..."` + 单引号包整段也行, 看 shell)

Expected:
- 文件 `tests/fixtures/ss_attention_bert.json` 创建, 大小 100-500KB (含两 paper 各自的 references / citations 列表, citations 部分可能很大)
- stdout 含 `fetched 2 papers; first: Attention Is All You Need`

如果 SS 返 429 (限速), 等 5min 再试, 或申请 API key。如果返 None 列表项, 说明 paper 不在 SS 数据库 — 极不可能 (Attention 和 BERT 都是 ML 经典, 必然在), 真出现就换 paper id。

- [ ] **Step 1.4: 验证 fixture 内容**

```bash
.venv/Scripts/python.exe -c @"
import json
with open('tests/fixtures/ss_attention_bert.json', encoding='utf-8') as f:
    data = json.load(f)
print(f'len={len(data)}')
for i, p in enumerate(data):
    if p is None:
        print(f'  [{i}] None')
        continue
    refs = p.get('references') or []
    cits = p.get('citations') or []
    arxiv_id = (p.get('externalIds') or {}).get('ArXiv')
    print(f'  [{i}] arxiv={arxiv_id} title={p[\"title\"][:40]!r} refs={len(refs)} cits={len(cits)}')
    if refs:
        first_ref_arxiv = (refs[0].get('externalIds') or {}).get('ArXiv') if refs[0] else None
        print(f'      ref[0] arxiv={first_ref_arxiv} title={refs[0][\"title\"][:40] if refs[0] else None!r}')
"@
```

Expected:
```
len=2
  [0] arxiv=1706.03762 title='Attention Is All You Need' refs=N cits=M
      ref[0] arxiv=... title=...
  [1] arxiv=1810.04805 title='BERT: Pre-training of...' refs=N cits=M
      ref[0] arxiv=... title=...
```

(具体 N/M 数字看 SS 当前数据; references 一般几十, citations 数千 — Attention 被引几万次, SS 默认会限制返回数量, 截到 1000 左右)

如果某 paper 的 `references` 全没有 `ArXiv` externalId — 说明 fixture 不适合, 后续测试只能验"skip 无 arxiv_id 邻居"行为, 没法验"加邻居节点"。**这个不太可能, Attention 的 references 大部分都是早期 ML 经典, 多有 arxiv id**。

- [ ] **Step 1.5: 创建 graph 子包目录结构 (空 __init__)**

```bash
mkdir -p paperpilot/mcp_servers/graph
touch paperpilot/mcp_servers/graph/__init__.py
```

(Windows PowerShell: `New-Item -ItemType Directory paperpilot/mcp_servers/graph -Force; New-Item -ItemType File paperpilot/mcp_servers/graph/__init__.py`)

- [ ] **Step 1.6: Commit**

```bash
git status     # 确认仅: requirements.txt 改 / tests/fixtures/ss_attention_bert.json 新 / paperpilot/mcp_servers/graph/__init__.py 新
git add requirements.txt tests/fixtures/ss_attention_bert.json paperpilot/mcp_servers/graph/__init__.py
git commit -m "Day 8 Task 1: 加 networkx + httpx 依赖, 抓 SS API fixture, 建 graph 子包目录"
```

---

## Task 2: ss_client.py — Semantic Scholar HTTP client + 7 个单测 (TDD)

**Files:**
- Create: `paperpilot/mcp_servers/graph/ss_client.py`
- Create: `tests/mcp_servers/test_ss_client.py`

预估: ~70-90min (TDD 节奏 + retry/backoff 边界条件较多)

**TDD 思路**: 先把 7 个测试全写出 (一个文件), 跑全 fail; 然后实现 ss_client.py 让测试一个个变绿。这样可以一次性把接口形态锁死再写实现。

- [ ] **Step 2.1: 写测试文件 `tests/mcp_servers/test_ss_client.py` (7 个 test, 全 fail)**

Create file:

```python
"""ss_client 单测。

测试范围:
- happy path (mock httpx 返 fixture)
- 429 retry success (第一次 429, 第二次成功)
- 429 retry exhaust (3 次都 429 → SSRateLimitError)
- 5xx retry (同 429 路径)
- 4xx 立即抛 (不 retry)
- API key 无 / 有 时的 header 行为

不真发 SS 请求 (CI 不能依赖外网)。
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import httpx
import pytest

from paperpilot.mcp_servers.graph.ss_client import (
    SSClient,
    SSAPIError,
    SSRateLimitError,
)


FIXTURE = Path(__file__).parent.parent / "fixtures" / "ss_attention_bert.json"


def _load_fixture() -> list[dict]:
    with FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def _mock_response(status_code: int, json_data: list[dict] | None = None) -> MagicMock:
    """构造一个 httpx Response-like mock。"""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or []
    if 400 <= status_code < 600:
        # raise_for_status 抛 HTTPStatusError
        req = httpx.Request("POST", "https://api.semanticscholar.org/graph/v1/paper/batch")
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=req, response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_fetch_papers_batch_happy(monkeypatch):
    """正常返 SS response, ss_client 解析并返回 list[dict]。"""
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    fixture_data = _load_fixture()
    client = SSClient()
    with patch.object(client._http, "post", return_value=_mock_response(200, fixture_data)) as post:
        result = client.fetch_papers_batch(["1706.03762", "1810.04805"])
    assert post.call_count == 1
    # 直接返回 SS 给的 list (含 None 表示 missing)
    assert result == fixture_data
    # 调用应该带 fields query + ids body
    call = post.call_args
    assert "params" in call.kwargs and "fields" in call.kwargs["params"]
    assert call.kwargs["json"] == {"ids": ["ARXIV:1706.03762", "ARXIV:1810.04805"]}


def test_fetch_429_retry_then_success(monkeypatch):
    """第一次 429, 退避后第二次 200。总调用 2 次。"""
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    fixture_data = _load_fixture()
    client = SSClient()
    responses = [_mock_response(429), _mock_response(200, fixture_data)]
    with patch.object(client._http, "post", side_effect=responses) as post, \
         patch("paperpilot.mcp_servers.graph.ss_client.time.sleep") as sleep:
        result = client.fetch_papers_batch(["1706.03762", "1810.04805"])
    assert post.call_count == 2
    assert sleep.call_count == 1
    assert sleep.call_args[0][0] == 2.0  # 第一次退避 2s
    assert result == fixture_data


def test_fetch_429_retry_exhaust(monkeypatch):
    """连续 3 次 429, 退避 2/4/8s, 最终抛 SSRateLimitError。"""
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    client = SSClient()
    responses = [_mock_response(429), _mock_response(429), _mock_response(429)]
    with patch.object(client._http, "post", side_effect=responses) as post, \
         patch("paperpilot.mcp_servers.graph.ss_client.time.sleep") as sleep:
        with pytest.raises(SSRateLimitError):
            client.fetch_papers_batch(["1706.03762"])
    assert post.call_count == 3
    # 退避序列: 2, 4 (第 3 次失败直接抛, 不再 sleep)
    sleep_args = [c[0][0] for c in sleep.call_args_list]
    assert sleep_args == [2.0, 4.0]


def test_fetch_5xx_retry(monkeypatch):
    """503 后 200, 路径同 429。"""
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    fixture_data = _load_fixture()
    client = SSClient()
    responses = [_mock_response(503), _mock_response(200, fixture_data)]
    with patch.object(client._http, "post", side_effect=responses), \
         patch("paperpilot.mcp_servers.graph.ss_client.time.sleep"):
        result = client.fetch_papers_batch(["1706.03762"])
    assert result == fixture_data


def test_fetch_4xx_no_retry(monkeypatch):
    """400 立即抛 SSAPIError, 不 retry (这是 bug, 不是网络抖)。"""
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    client = SSClient()
    with patch.object(client._http, "post", return_value=_mock_response(400)) as post, \
         patch("paperpilot.mcp_servers.graph.ss_client.time.sleep") as sleep:
        with pytest.raises(SSAPIError):
            client.fetch_papers_batch(["bad_id"])
    assert post.call_count == 1
    assert sleep.call_count == 0


def test_fetch_no_api_key_uses_unauth(monkeypatch):
    """env 不设 key, request header 不带 x-api-key。"""
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    client = SSClient()
    with patch.object(client._http, "post", return_value=_mock_response(200, [])) as post:
        client.fetch_papers_batch(["1706.03762"])
    headers = post.call_args.kwargs.get("headers", {})
    assert "x-api-key" not in headers


def test_fetch_with_api_key_sets_header(monkeypatch):
    """env 有 key, request header 带 x-api-key。"""
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "fake-test-key-123")
    client = SSClient()
    with patch.object(client._http, "post", return_value=_mock_response(200, [])) as post:
        client.fetch_papers_batch(["1706.03762"])
    headers = post.call_args.kwargs.get("headers", {})
    assert headers.get("x-api-key") == "fake-test-key-123"
```

- [ ] **Step 2.2: 跑测试, 全 fail**

```bash
.venv/Scripts/python.exe -m pytest tests/mcp_servers/test_ss_client.py -v
```

Expected: 7 个 test 全 ERROR (`ImportError: cannot import name 'SSClient'`), 因为 `paperpilot/mcp_servers/graph/ss_client.py` 还不存在。

- [ ] **Step 2.3: 写 ss_client.py 最小实现**

Create `paperpilot/mcp_servers/graph/ss_client.py`:

```python
"""Semantic Scholar Graph API HTTP client。唯一接触 httpx 的模块。

API doc: https://api.semanticscholar.org/api-docs/graph

retry/backoff 在本层封装:
- 429 / 5xx / network timeout → exp backoff (2s/4s/8s, max 3 次)
- 4xx 非 429 → 立即抛 (这是请求构造 bug, 不是网络抖)

不感知业务层语义; graph_manager 调用这一层时只看到 list[dict] 或抛出的异常。
"""
from __future__ import annotations

import os
import time

import httpx


SS_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SS_FIELDS = (
    "title,year,authors,externalIds,"
    "references.externalIds,references.title,references.year,references.authors,"
    "citations.externalIds,citations.title,citations.year,citations.authors"
)
RETRY_SLEEPS = [2.0, 4.0, 8.0]   # 第 N 次失败后的退避秒数; 最后一次失败直接抛
REQUEST_TIMEOUT = 30.0


class SSAPIError(RuntimeError):
    """SS API 服务端故障 / 4xx 请求构造错 / 5xx 重试耗尽。"""


class SSRateLimitError(SSAPIError):
    """SS API 限速重试耗尽 (3 次都 429)。"""


class SSClient:
    """同步 httpx client; 进程级单例由 graph_manager 持有。"""

    def __init__(self) -> None:
        self._http = httpx.Client(timeout=REQUEST_TIMEOUT)

    def close(self) -> None:
        self._http.close()

    def fetch_papers_batch(self, arxiv_ids: list[str]) -> list[dict | None]:
        """对一批 arxiv id 批量取 SS metadata + references + citations。

        Args:
            arxiv_ids: 不带版本后缀的 arxiv id 列表 (e.g. ["1706.03762"])

        Returns:
            SS 返回的 list, 长度与 input 一致; 未找到的 paper 对应位置为 None。

        Raises:
            SSRateLimitError: 429 重试耗尽
            SSAPIError: 4xx 构造错 / 5xx 重试耗尽 / 网络超时重试耗尽
        """
        body = {"ids": [f"ARXIV:{aid}" for aid in arxiv_ids]}
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        for attempt, sleep_s in enumerate(RETRY_SLEEPS):
            try:
                resp = self._http.post(
                    SS_BATCH_URL,
                    params={"fields": SS_FIELDS},
                    json=body,
                    headers=headers,
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 429:
                    if attempt == len(RETRY_SLEEPS) - 1:
                        raise SSRateLimitError(f"SS rate limit after {len(RETRY_SLEEPS)} retries")
                    time.sleep(sleep_s)
                    continue
                if 500 <= resp.status_code < 600:
                    if attempt == len(RETRY_SLEEPS) - 1:
                        raise SSAPIError(f"SS {resp.status_code} after {len(RETRY_SLEEPS)} retries")
                    time.sleep(sleep_s)
                    continue
                if 400 <= resp.status_code < 500:
                    # 4xx 非 429 不 retry, 立即抛
                    raise SSAPIError(f"SS {resp.status_code}: {resp.text[:200]}")
            except httpx.TimeoutException:
                if attempt == len(RETRY_SLEEPS) - 1:
                    raise SSAPIError(f"SS timeout after {len(RETRY_SLEEPS)} retries")
                time.sleep(sleep_s)
                continue

        # 不可达
        raise SSAPIError("ss_client: unreachable retry loop exit")
```

- [ ] **Step 2.4: 跑测试**

```bash
.venv/Scripts/python.exe -m pytest tests/mcp_servers/test_ss_client.py -v
```

Expected: 7 passed。

如果 `test_fetch_429_retry_exhaust` 失败 — 检查 sleep_args 数量。我的设计是: 第 1 次 429 → sleep 2s, 第 2 次 429 → sleep 4s, 第 3 次 429 → 不 sleep, 直接抛。所以 sleep_args == [2.0, 4.0] 长度 2。如果实现写成"每次都 sleep 后再判是否最后一次", 长度会是 3, 与 test 不符。代码里 `attempt == len(RETRY_SLEEPS) - 1` 的判定确保最后一次不 sleep。

- [ ] **Step 2.5: Commit**

```bash
git add paperpilot/mcp_servers/graph/ss_client.py tests/mcp_servers/test_ss_client.py
git commit -m "Day 8 Task 2: ss_client + 7 个单测 (happy/retry/auth)"
```

---

## Task 3: graph_manager.py — build_graph + pickle 持久化 + 单测 (TDD)

**Files:**
- Create: `paperpilot/mcp_servers/graph/graph_manager.py` (build + 持久化部分)
- Create: `tests/mcp_servers/test_graph_manager.py` (build 相关测试, query 相关在 Task 4 加)

预估: ~75min

**TDD 思路**: 先写"build + persist + load + corrupt" 6 个 test → 跑全 fail → 实现 → 全过 → commit。Query tool 的 5 个 test 留 Task 4 (单独 commit, 文件干净)。

- [ ] **Step 3.1: 写 build / persist / load 相关测试 (6 个)**

Create `tests/mcp_servers/test_graph_manager.py`:

```python
"""graph_manager 单测 — build + 持久化 + load 部分。

mock ss_client (不发真请求); 真跑 NetworkX (本地, 微秒级)。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from paperpilot.mcp_servers.graph.graph_manager import (
    GraphManager,
    GraphCorruptError,
    NodeNotFoundError,
)
from paperpilot.mcp_servers.graph.ss_client import SSAPIError


FIXTURE = Path(__file__).parent.parent / "fixtures" / "ss_attention_bert.json"


def _load_fixture() -> list[dict]:
    with FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def _make_manager(tmp_path):
    """构造 GraphManager, pickle 落盘到 tmp_path。"""
    return GraphManager(graph_path=tmp_path / "citation_graph.pkl")


def test_build_adds_main_and_neighbors(tmp_path):
    """build 后图含 2 个主节点 + N 个邻居 + 边。"""
    fixture_data = _load_fixture()
    mgr = _make_manager(tmp_path)
    mgr._ss_client = MagicMock()
    mgr._ss_client.fetch_papers_batch.return_value = fixture_data

    result = mgr.build(["1706.03762", "1810.04805"])

    assert result["added_count"] == 2
    assert result["skipped_count"] == 0
    assert result["missing"] == []
    assert result["total_nodes"] >= 2
    assert result["total_edges"] >= 2  # 至少 BERT 引 Attention 1 条
    assert mgr._graph.has_node("1706.03762")
    assert mgr._graph.has_node("1810.04805")
    assert mgr._graph.nodes["1706.03762"]["node_type"] == "downloaded"
    assert mgr._graph.nodes["1706.03762"]["title"]


def test_build_skips_existing_downloaded(tmp_path):
    """连续 build 同 id 两次, 第二次该 id 进 skipped, 不调 SS。"""
    fixture_data = _load_fixture()
    mgr = _make_manager(tmp_path)
    mgr._ss_client = MagicMock()
    mgr._ss_client.fetch_papers_batch.return_value = fixture_data

    mgr.build(["1706.03762", "1810.04805"])
    # 第二次 build 同样的 ids
    result = mgr.build(["1706.03762", "1810.04805"])

    assert result["added_count"] == 0
    assert result["skipped_count"] == 2
    # SS 第二次不该被调
    assert mgr._ss_client.fetch_papers_batch.call_count == 1


def test_build_promotes_neighbor_to_downloaded(tmp_path):
    """已存在的 neighbor 节点被 build 时升格为 downloaded, 调 SS 拉它的 references。"""
    fixture_data = _load_fixture()
    # 找 fixture 中 Attention 的某个 reference 的 arxiv_id (作为后续要"升格"的目标)
    refs = fixture_data[0].get("references") or []
    target_neighbor_arxiv = None
    for r in refs:
        if r and (r.get("externalIds") or {}).get("ArXiv"):
            target_neighbor_arxiv = r["externalIds"]["ArXiv"]
            break
    assert target_neighbor_arxiv, "fixture 必须含至少一个有 arxiv id 的 reference"

    mgr = _make_manager(tmp_path)
    mgr._ss_client = MagicMock()
    # 第一次 build Attention, target 进图为 neighbor
    mgr._ss_client.fetch_papers_batch.return_value = [fixture_data[0]]
    mgr.build(["1706.03762"])
    assert mgr._graph.nodes[target_neighbor_arxiv]["node_type"] == "neighbor"

    # 第二次 build target, 应升格 + 调 SS
    # mock SS 第二次返回的 paper (用 fixture[1] BERT 顶替, title 不重要, 只要结构正确)
    promoted_paper = dict(fixture_data[1])
    promoted_paper["externalIds"] = {"ArXiv": target_neighbor_arxiv}
    mgr._ss_client.fetch_papers_batch.return_value = [promoted_paper]
    result = mgr.build([target_neighbor_arxiv])

    assert result["added_count"] == 1   # 升格算 added
    assert result["skipped_count"] == 0
    assert mgr._graph.nodes[target_neighbor_arxiv]["node_type"] == "downloaded"
    # SS 应被调 2 次 (第一次给 Attention, 第二次给 target)
    assert mgr._ss_client.fetch_papers_batch.call_count == 2


def test_build_missing_paper_in_ss(tmp_path):
    """SS 返 None 表示该 paper 不在数据库, 加入 missing 列表, 不抛错。"""
    fixture_data = _load_fixture()
    mgr = _make_manager(tmp_path)
    mgr._ss_client = MagicMock()
    # 模拟: 第一个 paper 找到, 第二个不在 SS
    mgr._ss_client.fetch_papers_batch.return_value = [fixture_data[0], None]

    result = mgr.build(["1706.03762", "9999.99999"])

    assert result["added_count"] == 1
    assert "9999.99999" in result["missing"]
    assert mgr._graph.has_node("1706.03762")
    assert not mgr._graph.has_node("9999.99999")


def test_build_persists_pickle_atomic(tmp_path):
    """build 后 pickle 文件存在 + 不存在 .tmp 残留, load 回来节点数一致。"""
    fixture_data = _load_fixture()
    mgr = _make_manager(tmp_path)
    mgr._ss_client = MagicMock()
    mgr._ss_client.fetch_papers_batch.return_value = fixture_data

    mgr.build(["1706.03762", "1810.04805"])
    pkl = tmp_path / "citation_graph.pkl"
    tmp = tmp_path / "citation_graph.pkl.tmp"

    assert pkl.exists()
    assert not tmp.exists(), ".tmp 应该被 os.replace 重命名走"

    # load 回来
    with pkl.open("rb") as f:
        loaded = pickle.load(f)
    assert loaded.has_node("1706.03762")
    assert loaded.number_of_nodes() == mgr._graph.number_of_nodes()


def test_load_existing_graph_on_init(tmp_path):
    """init 时若 pickle 已存在, 直接 load (不空图开始)。"""
    fixture_data = _load_fixture()
    pkl = tmp_path / "citation_graph.pkl"

    # 先建一个图并落盘
    mgr1 = _make_manager(tmp_path)
    mgr1._ss_client = MagicMock()
    mgr1._ss_client.fetch_papers_batch.return_value = fixture_data
    mgr1.build(["1706.03762", "1810.04805"])
    nodes_before = mgr1._graph.number_of_nodes()

    # 新 manager 实例化, 应 load
    mgr2 = GraphManager(graph_path=pkl)
    assert mgr2._graph.has_node("1706.03762")
    assert mgr2._graph.number_of_nodes() == nodes_before


def test_load_corrupt_pickle_raises(tmp_path):
    """坏 pickle 文件 → init 抛 GraphCorruptError (hard-fail, 不静默清空)。"""
    pkl = tmp_path / "citation_graph.pkl"
    pkl.write_bytes(b"\x00\x01 not a valid pickle \x99")

    with pytest.raises(GraphCorruptError):
        GraphManager(graph_path=pkl)
```

- [ ] **Step 3.2: 跑测试, 全 fail**

```bash
.venv/Scripts/python.exe -m pytest tests/mcp_servers/test_graph_manager.py -v
```

Expected: 7 个 ERROR `ImportError: cannot import name 'GraphManager'`。

- [ ] **Step 3.3: 写 graph_manager.py (build + 持久化部分; query tool 留 Task 4 加)**

Create `paperpilot/mcp_servers/graph/graph_manager.py`:

```python
"""graph-mcp 的图管理层。唯一接触 NetworkX + pickle 的模块。

启动期 (__init__):
  1. mkdir -p data/graph/  (若 graph_path 父目录不存在)
  2. 尝试 pickle.load(graph_path); 不存在 → 空图开始; 损坏 → GraphCorruptError 抛 (hard-fail)
  3. 创建 SSClient (httpx session)

build(arxiv_ids):
  - 已是 downloaded 主节点 → skipped
  - 是 neighbor → 升格 + 调 SS
  - 全新 → 调 SS
  - 调 ss_client.fetch_papers_batch
  - 加 / 升格主节点 + 加邻居节点 + 加边
  - pickle.dump → atomic rename
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import networkx as nx

from paperpilot.mcp_servers.graph.ss_client import SSClient


GRAPH_DIR = Path("data/graph")
GRAPH_FILE = GRAPH_DIR / "citation_graph.pkl"


class GraphCorruptError(RuntimeError):
    """启动期 pickle.load 失败 (文件损坏)。要求人工删文件再重启。"""


class NodeNotFoundError(RuntimeError):
    """get_neighbors / get_shortest_path 时 arxiv_id 不在图中。"""


class GraphManager:
    def __init__(self, graph_path: Path = GRAPH_FILE) -> None:
        self._graph_path = Path(graph_path)
        self._graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._graph: nx.DiGraph = self._load_or_empty()
        self._ss_client = SSClient()

    def _load_or_empty(self) -> nx.DiGraph:
        if not self._graph_path.exists():
            return nx.DiGraph()
        try:
            with self._graph_path.open("rb") as f:
                g = pickle.load(f)
        except Exception as e:
            raise GraphCorruptError(
                f"failed to load {self._graph_path}; delete it and restart: {e}"
            ) from e
        if not isinstance(g, nx.DiGraph):
            raise GraphCorruptError(f"{self._graph_path} is not a DiGraph: {type(g)!r}")
        return g

    def _save(self) -> None:
        tmp = self._graph_path.with_suffix(self._graph_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self._graph, f)
        os.replace(tmp, self._graph_path)

    def build(self, arxiv_ids: list[str]) -> dict:
        if not arxiv_ids:
            raise ValueError("arxiv_ids must not be empty")
        for aid in arxiv_ids:
            if not isinstance(aid, str):
                raise ValueError(f"arxiv_ids must be list[str]: {aid!r}")

        skipped: list[str] = []
        to_fetch: list[str] = []
        for aid in arxiv_ids:
            if self._graph.has_node(aid) and self._graph.nodes[aid].get("node_type") == "downloaded":
                skipped.append(aid)
            else:
                to_fetch.append(aid)

        added_count = 0
        missing: list[str] = []
        if to_fetch:
            ss_results = self._ss_client.fetch_papers_batch(to_fetch)
            for aid, paper in zip(to_fetch, ss_results):
                if paper is None:
                    missing.append(aid)
                    continue
                self._upsert_main_node(aid, paper)
                added_count += 1
                self._add_neighbors(aid, paper.get("references") or [], edge_dir="ref")
                self._add_neighbors(aid, paper.get("citations") or [], edge_dir="cit")

        self._save()

        return {
            "added_count": added_count,
            "skipped_count": len(skipped),
            "missing": missing,
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
        }

    def _upsert_main_node(self, arxiv_id: str, paper: dict) -> None:
        attrs = {
            "node_type": "downloaded",
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
        }
        if self._graph.has_node(arxiv_id):
            # 升格: 保留已有属性 (e.g. title 可能在 neighbor 时已写过, 这里覆盖)
            self._graph.nodes[arxiv_id].update(attrs)
        else:
            self._graph.add_node(arxiv_id, **attrs)

    def _add_neighbors(self, main_id: str, neighbors: list[dict], edge_dir: str) -> None:
        """edge_dir: 'ref' 表示 main 引用了 neighbor (main → neighbor);
                    'cit' 表示 neighbor 引用了 main (neighbor → main)。"""
        for n in neighbors:
            if not n:
                continue
            ext = n.get("externalIds") or {}
            n_arxiv = ext.get("ArXiv")
            if not n_arxiv:
                continue   # skip 无 arxiv_id 邻居 (与 PaperPilot 全局 ID 体系不一致)
            if not self._graph.has_node(n_arxiv):
                self._graph.add_node(
                    n_arxiv,
                    node_type="neighbor",
                    arxiv_id=n_arxiv,
                    title=n.get("title", ""),
                    year=n.get("year"),
                    authors=[a.get("name", "") for a in (n.get("authors") or [])],
                )
            # 加边 (DiGraph add_edge 幂等, 重复加不报错)
            if edge_dir == "ref":
                self._graph.add_edge(main_id, n_arxiv)
            else:  # cit
                self._graph.add_edge(n_arxiv, main_id)
```

- [ ] **Step 3.4: 跑测试**

```bash
.venv/Scripts/python.exe -m pytest tests/mcp_servers/test_graph_manager.py -v
```

Expected: 7 passed。

调试常见问题:
- `test_build_promotes_neighbor_to_downloaded`: 如果 fixture 第一个 reference 没有 ArXiv externalId, 测试 setup 会 assert 失败。这种概率不高 (Attention 的 references 多是 ML 经典), 但若发生需切到 fixture 里能找到 arxiv_id 的 reference; 实在没有改 fixture (重抓 SS, 用 references 多的另一篇 paper)。
- `test_build_persists_pickle_atomic`: `os.replace` 在 Windows 上对目标文件已存在的情况是合法的 (与 POSIX 同行为)。

- [ ] **Step 3.5: Commit**

```bash
git add paperpilot/mcp_servers/graph/graph_manager.py tests/mcp_servers/test_graph_manager.py
git commit -m "Day 8 Task 3: graph_manager build + pickle 持久化 + 6 单测"
```

---

## Task 4: graph_manager.py — 3 个 query tool + 5 单测 (TDD)

**Files:**
- Modify: `paperpilot/mcp_servers/graph/graph_manager.py` (+ get_neighbors / get_shortest_path / get_common_citations)
- Modify: `tests/mcp_servers/test_graph_manager.py` (+ 5 个 query test)

预估: ~45min

- [ ] **Step 4.1: 在 test_graph_manager.py 末尾追加 5 个 query test**

在 `tests/mcp_servers/test_graph_manager.py` 末尾追加 (保留前面 7 个 test 不动):

```python


# ============================================================
# Query tool 测试 (Task 4)
# ============================================================

def _build_synthetic_graph(tmp_path):
    """手工建一个小图: A → B → C, A → D, E → A (E 引用 A)。
       不调 SS, 直接操作 graph_manager._graph。
       返回 manager 实例。
    """
    mgr = _make_manager(tmp_path)
    g = mgr._graph
    for aid, year in [("A", 2020), ("B", 2021), ("C", 2022), ("D", 2019), ("E", 2023)]:
        g.add_node(
            aid,
            node_type="downloaded" if aid in {"A", "B", "C"} else "neighbor",
            arxiv_id=aid,
            title=f"Paper {aid}",
            year=year,
            authors=[f"Author{aid}"],
        )
    g.add_edge("A", "B")  # A cites B
    g.add_edge("B", "C")  # B cites C
    g.add_edge("A", "D")  # A cites D
    g.add_edge("E", "A")  # E cites A
    return mgr


def test_get_neighbors_directions(tmp_path):
    mgr = _build_synthetic_graph(tmp_path)
    refs = mgr.get_neighbors("A", direction="references", limit=10)
    cits = mgr.get_neighbors("A", direction="citations", limit=10)
    both = mgr.get_neighbors("A", direction="both", limit=10)

    assert {n["arxiv_id"] for n in refs} == {"B", "D"}
    assert all(n["edge"] == "references" for n in refs)
    assert {n["arxiv_id"] for n in cits} == {"E"}
    assert all(n["edge"] == "citations" for n in cits)
    assert {n["arxiv_id"] for n in both} == {"B", "D", "E"}


def test_get_neighbors_node_missing(tmp_path):
    mgr = _build_synthetic_graph(tmp_path)
    with pytest.raises(NodeNotFoundError):
        mgr.get_neighbors("ZZZ", direction="both")


def test_get_neighbors_year_sort_and_limit(tmp_path):
    mgr = _build_synthetic_graph(tmp_path)
    # A 引 B(2021), D(2019); year 倒序 → B 先
    result = mgr.get_neighbors("A", direction="references", limit=1)
    assert len(result) == 1
    assert result[0]["arxiv_id"] == "B"


def test_shortest_path_happy(tmp_path):
    mgr = _build_synthetic_graph(tmp_path)
    result = mgr.get_shortest_path("A", "C")
    assert result["length"] == 2
    assert [p["arxiv_id"] for p in result["path"]] == ["A", "B", "C"]


def test_shortest_path_no_path(tmp_path):
    mgr = _build_synthetic_graph(tmp_path)
    # D 没有任何 out edge, 到 E 没路径
    result = mgr.get_shortest_path("D", "E")
    assert result["length"] == -1
    assert result["path"] == []


def test_shortest_path_node_missing(tmp_path):
    mgr = _build_synthetic_graph(tmp_path)
    with pytest.raises(NodeNotFoundError):
        mgr.get_shortest_path("A", "ZZZ")


def test_common_citations(tmp_path):
    """A 和 X 都引 B → B cited_by_count=2。"""
    mgr = _make_manager(tmp_path)
    g = mgr._graph
    for aid, year in [("A", 2020), ("X", 2020), ("B", 2018), ("Y", 2017), ("Z", 2016)]:
        g.add_node(aid, node_type="downloaded" if aid in {"A", "X"} else "neighbor",
                   arxiv_id=aid, title=f"P{aid}", year=year, authors=[])
    g.add_edge("A", "B")
    g.add_edge("A", "Y")
    g.add_edge("X", "B")  # 共引 B
    g.add_edge("X", "Z")

    result = mgr.get_common_citations(["A", "X"], top_k=10)
    arxiv_ids = [r["arxiv_id"] for r in result]
    assert "B" in arxiv_ids   # 共引
    assert "Y" not in arxiv_ids  # 仅 A 引
    assert "Z" not in arxiv_ids  # 仅 X 引
    b_entry = next(r for r in result if r["arxiv_id"] == "B")
    assert b_entry["cited_by_count"] == 2


def test_common_citations_min_2_input(tmp_path):
    mgr = _make_manager(tmp_path)
    with pytest.raises(ValueError):
        mgr.get_common_citations(["A"], top_k=10)


def test_common_citations_all_missing_returns_empty(tmp_path):
    """input 全部不在图 → 返空列表 (不抛错)。"""
    mgr = _make_manager(tmp_path)
    result = mgr.get_common_citations(["NOPE1", "NOPE2"], top_k=10)
    assert result == []
```

(注意: 我多写了 1 个 test_shortest_path_node_missing, 共 5 测 → 实际 9 测; 这个 spec 没列但 NodeNotFoundError 行为对称, 顺手补; 总 query 测试 9 个比 spec 列的 5 个多, 因为 spec 写的是核心场景, plan 实现时常发现要补对称测试 — 这是合理的。)

- [ ] **Step 4.2: 跑测试, query 部分全 fail**

```bash
.venv/Scripts/python.exe -m pytest tests/mcp_servers/test_graph_manager.py -v
```

Expected: 前 7 个 (Task 3 的) 仍 passed; 新加的 9 个 ERROR `AttributeError: 'GraphManager' object has no attribute 'get_neighbors'`。

- [ ] **Step 4.3: 在 graph_manager.py 加 3 个 query method**

在 `paperpilot/mcp_servers/graph/graph_manager.py` 类 `GraphManager` 内、`_add_neighbors` 之后追加:

```python
    def get_neighbors(
        self, arxiv_id: str, direction: str = "both", limit: int = 10
    ) -> list[dict]:
        if not self._graph.has_node(arxiv_id):
            raise NodeNotFoundError(
                f"{arxiv_id} not in graph; call build_graph first or paper not in graph"
            )

        out: list[tuple[str, str]] = []  # (neighbor_id, edge_label)
        if direction in ("references", "both"):
            for n in self._graph.successors(arxiv_id):
                out.append((n, "references"))
        if direction in ("citations", "both"):
            for n in self._graph.predecessors(arxiv_id):
                out.append((n, "citations"))

        # 按 year 倒序 (None year 视为 -inf), tie-break by arxiv_id
        def _sort_key(item):
            nid, _ = item
            year = self._graph.nodes[nid].get("year")
            return (-(year or -10000), nid)
        out.sort(key=_sort_key)

        result = []
        for nid, edge in out[:limit]:
            attrs = self._graph.nodes[nid]
            result.append({
                "arxiv_id": nid,
                "title": attrs.get("title", ""),
                "year": attrs.get("year"),
                "authors": attrs.get("authors", []),
                "edge": edge,
            })
        return result

    def get_shortest_path(self, from_id: str, to_id: str) -> dict:
        if not self._graph.has_node(from_id):
            raise NodeNotFoundError(f"{from_id} not in graph")
        if not self._graph.has_node(to_id):
            raise NodeNotFoundError(f"{to_id} not in graph")
        try:
            path = nx.shortest_path(self._graph, source=from_id, target=to_id)
        except nx.NetworkXNoPath:
            return {"path": [], "length": -1}
        return {
            "path": [
                {"arxiv_id": n, "title": self._graph.nodes[n].get("title", "")}
                for n in path
            ],
            "length": len(path) - 1,
        }

    def get_common_citations(
        self, arxiv_ids: list[str], top_k: int = 10
    ) -> list[dict]:
        if len(arxiv_ids) < 2:
            raise ValueError("get_common_citations needs >=2 input arxiv_ids")

        from collections import Counter
        counter: Counter[str] = Counter()
        for aid in arxiv_ids:
            if not self._graph.has_node(aid):
                continue
            for succ in self._graph.successors(aid):
                counter[succ] += 1

        # 仅保留 cited_by_count >= 2
        common = [(nid, cnt) for nid, cnt in counter.items() if cnt >= 2]
        # 按 cnt 倒序, tie-break by year 倒序
        def _sort_key(item):
            nid, cnt = item
            year = self._graph.nodes[nid].get("year")
            return (-cnt, -(year or -10000), nid)
        common.sort(key=_sort_key)

        result = []
        for nid, cnt in common[:top_k]:
            attrs = self._graph.nodes[nid]
            result.append({
                "arxiv_id": nid,
                "title": attrs.get("title", ""),
                "year": attrs.get("year"),
                "authors": attrs.get("authors", []),
                "cited_by_count": cnt,
            })
        return result
```

- [ ] **Step 4.4: 跑测试**

```bash
.venv/Scripts/python.exe -m pytest tests/mcp_servers/test_graph_manager.py -v
```

Expected: 16 passed (7 Task 3 + 9 Task 4)。

- [ ] **Step 4.5: Commit**

```bash
git add paperpilot/mcp_servers/graph/graph_manager.py tests/mcp_servers/test_graph_manager.py
git commit -m "Day 8 Task 4: graph_manager 3 query tool (neighbors/path/common) + 9 单测"
```

---

## Task 5: server.py 协议层 + mcp_servers.json + 协议单测 (TDD)

**Files:**
- Create: `paperpilot/mcp_servers/graph/server.py` (FastMCP, 4 个 tool handler, 协议层不接触 networkx/httpx)
- Create: `tests/mcp_servers/test_graph_server.py` (5 个协议层测试)
- Modify: `paperpilot/mcp_servers.json` (+ graph entry)

预估: ~30-40min

- [ ] **Step 5.1: 写协议层测试 (5 个 test)**

Create `tests/mcp_servers/test_graph_server.py`:

```python
"""graph server.py 协议层单测。

只测路由 + 输入校验。mock 掉 _manager (graph_manager 已在 test_graph_manager 充分验证)。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paperpilot.mcp_servers.graph import server as graph_server


@pytest.fixture(autouse=True)
def _setup_mock_manager(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(graph_server, "_manager", mock)
    return mock


def test_build_graph_routes_to_manager(_setup_mock_manager):
    _setup_mock_manager.build.return_value = {
        "added_count": 2, "skipped_count": 0, "missing": [],
        "total_nodes": 5, "total_edges": 4,
    }
    result = graph_server._build_graph_impl(["1706.03762", "1810.04805"])
    _setup_mock_manager.build.assert_called_once_with(["1706.03762", "1810.04805"])
    assert result["added_count"] == 2


def test_build_graph_empty_input_raises(_setup_mock_manager):
    with pytest.raises(ValueError):
        graph_server._build_graph_impl([])


def test_build_graph_non_str_input_raises(_setup_mock_manager):
    with pytest.raises(ValueError):
        graph_server._build_graph_impl([123, "1810.04805"])


def test_get_neighbors_routes(_setup_mock_manager):
    _setup_mock_manager.get_neighbors.return_value = []
    graph_server._get_neighbors_impl("1706.03762", direction="references", limit=5)
    _setup_mock_manager.get_neighbors.assert_called_once_with(
        "1706.03762", direction="references", limit=5
    )


def test_get_shortest_path_routes(_setup_mock_manager):
    _setup_mock_manager.get_shortest_path.return_value = {"path": [], "length": -1}
    graph_server._get_shortest_path_impl("1706.03762", "1810.04805")
    _setup_mock_manager.get_shortest_path.assert_called_once_with(
        "1706.03762", "1810.04805"
    )


def test_get_common_citations_routes(_setup_mock_manager):
    _setup_mock_manager.get_common_citations.return_value = []
    graph_server._get_common_citations_impl(["1706.03762", "1810.04805"], top_k=5)
    _setup_mock_manager.get_common_citations.assert_called_once_with(
        ["1706.03762", "1810.04805"], top_k=5
    )
```

- [ ] **Step 5.2: 跑测试, 全 fail**

```bash
.venv/Scripts/python.exe -m pytest tests/mcp_servers/test_graph_server.py -v
```

Expected: ERROR `ImportError: cannot import name 'server' from 'paperpilot.mcp_servers.graph'` 之类。

- [ ] **Step 5.3: 写 server.py**

Create `paperpilot/mcp_servers/graph/server.py`:

```python
"""graph-mcp: 引用关系图 server。Day 8 起。

启动: python -m paperpilot.mcp_servers.graph.server
通过 stdio 被 paperpilot.tools.mcp_client 拉起, manifest 见 paperpilot/mcp_servers.json。

协议层职责: 输入校验 + 路由到 GraphManager。**不接触 networkx / httpx**。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from paperpilot.mcp_servers.graph.graph_manager import (
    GraphManager,
    NodeNotFoundError,
)

mcp = FastMCP("graph")
_manager: GraphManager | None = None


@mcp.tool()
def build_graph(arxiv_ids: list[str]) -> dict:
    """对一批 arxiv paper 拉 Semantic Scholar 引用关系入图 (含一跳双向邻居)。

    增量语义: 已是 'downloaded' 主节点的 id 跳过; 之前是 'neighbor' 的 id 升格为
    'downloaded' 并补拉它的 references/citations (LLM 工作流: 看邻居 → 决定深挖)。

    调用示例: build_graph(arxiv_ids=["1706.03762", "1810.04805"])

    Args:
        arxiv_ids: list[str], 不带版本后缀。空 list 或非 str 元素抛 ValueError。

    Returns:
        dict 含 added_count / skipped_count / missing / total_nodes / total_edges。
    """
    return _build_graph_impl(arxiv_ids)


def _build_graph_impl(arxiv_ids: list[str]) -> dict:
    if not arxiv_ids:
        raise ValueError("arxiv_ids must not be empty")
    for aid in arxiv_ids:
        if not isinstance(aid, str):
            raise ValueError(f"arxiv_ids must be list[str]: {aid!r}")
    assert _manager is not None, "GraphManager not initialized"
    return _manager.build(arxiv_ids)


@mcp.tool()
def get_neighbors(arxiv_id: str, direction: str = "both", limit: int = 10) -> list[dict]:
    """查某 paper 的引用邻居 (引用了哪些 / 被哪些引)。

    Args:
        arxiv_id: 已 build_graph 过的 paper, 否则抛 NodeNotFoundError。
        direction: "references" (它引谁) / "citations" (谁引它) / "both"。默认 both。
        limit: 返回邻居上限, 按 year 倒序。默认 10。

    Returns:
        list[dict], 每项含 arxiv_id / title / year / authors / edge ("references" 或 "citations")。
    """
    return _get_neighbors_impl(arxiv_id, direction=direction, limit=limit)


def _get_neighbors_impl(arxiv_id: str, direction: str, limit: int) -> list[dict]:
    assert _manager is not None
    return _manager.get_neighbors(arxiv_id, direction=direction, limit=limit)


@mcp.tool()
def get_shortest_path(from_id: str, to_id: str) -> dict:
    """找两篇 paper 之间最短引用路径 (有向)。

    Args:
        from_id: 起点 arxiv_id, 必须在图中。
        to_id: 终点 arxiv_id, 必须在图中。

    Returns:
        dict 含 path (list[{arxiv_id, title}]) 和 length (边数, 无路径时 -1)。
    """
    return _get_shortest_path_impl(from_id, to_id)


def _get_shortest_path_impl(from_id: str, to_id: str) -> dict:
    assert _manager is not None
    return _manager.get_shortest_path(from_id, to_id)


@mcp.tool()
def get_common_citations(arxiv_ids: list[str], top_k: int = 10) -> list[dict]:
    """找一批 paper 共同引用的下游 paper (ground-truth 经典文献场景)。

    Args:
        arxiv_ids: 至少 2 个 arxiv_id (单 paper 求共引无意义)。
        top_k: 返回上限, 按 cited_by_count 倒序。默认 10。

    Returns:
        list[dict], 每项含 arxiv_id / title / year / authors / cited_by_count。
        cited_by_count 表示在 input 列表中有多少篇引用了它 (>=2)。
    """
    return _get_common_citations_impl(arxiv_ids, top_k=top_k)


def _get_common_citations_impl(arxiv_ids: list[str], top_k: int) -> list[dict]:
    assert _manager is not None
    return _manager.get_common_citations(arxiv_ids, top_k=top_k)


if __name__ == "__main__":
    _manager = GraphManager()
    mcp.run(transport="stdio")
```

- [ ] **Step 5.4: 跑测试**

```bash
.venv/Scripts/python.exe -m pytest tests/mcp_servers/test_graph_server.py -v
```

Expected: 6 passed。

- [ ] **Step 5.5: 加 mcp_servers.json entry**

Read `paperpilot/mcp_servers.json`, 在 `mcpServers` object 中加 graph entry。

修改后内容应该是:
```json
{
  "mcpServers": {
    "arxiv": {
      "command": "python",
      "args": ["-m", "paperpilot.mcp_servers.arxiv"]
    },
    "colbert": {
      "command": "python",
      "args": ["-m", "paperpilot.mcp_servers.colbert.server"]
    },
    "graph": {
      "command": "python",
      "args": ["-m", "paperpilot.mcp_servers.graph.server"]
    }
  }
}
```

注意 colbert entry 后加逗号 (JSON 严格)。

- [ ] **Step 5.6: 验证 graph server 能独立启动 (sanity check, 不入测试套)**

```bash
.venv/Scripts/python.exe -m paperpilot.mcp_servers.graph.server
```

Expected: 进程启动后阻塞等 stdio 输入 (因为 transport="stdio"); 没有 traceback。Ctrl+C 退出。

如果立即报错 (e.g. `OSError: [Errno 13] Permission denied: 'data/graph'`), 检查 `data/graph/` 目录权限。

- [ ] **Step 5.7: 跑全套单测确认无回归**

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: Day 6 的 12 passed + Day 8 新加的 7 (ss_client) + 16 (graph_manager) + 6 (graph_server) = **41 passed, 4 deselected** (deselected 是 slow 集成测试, 默认 deselect)。

如果 < 41, 看哪个新测失败逐个修。如果 Day 6 测试有回归, 说明改动溢出范围, 需排查。

- [ ] **Step 5.8: Commit**

```bash
git add paperpilot/mcp_servers/graph/server.py tests/mcp_servers/test_graph_server.py paperpilot/mcp_servers.json
git commit -m "Day 8 Task 5: graph server.py 协议层 + 6 单测 + mcp_servers.json 加 graph entry"
```

---

## Task 6: 集成测试 + day8_smoke + 端到端真 SS / 真 LLM

**Files:**
- Create: `tests/mcp_servers/test_graph_via_client.py` (3 个 slow 集成测试)
- Modify: `paperpilot/mcp_servers/graph/ss_client.py` (+ fixture mode env 切换)
- Create: `scripts/day8_smoke.py`

预估: ~60-90min (含真 SS 调用 + 真 LLM 跑 smoke 的不可控耗时)

- [ ] **Step 6.1: 给 ss_client.py 加 fixture mode env 切换**

Modify `paperpilot/mcp_servers/graph/ss_client.py` 的 `SSClient.fetch_papers_batch` — 在方法开头加 fixture short-circuit:

```python
    def fetch_papers_batch(self, arxiv_ids: list[str]) -> list[dict | None]:
        """对一批 arxiv id 批量取 SS metadata + references + citations。

        Args:
            arxiv_ids: 不带版本后缀的 arxiv id 列表 (e.g. ["1706.03762"])

        Returns:
            SS 返回的 list, 长度与 input 一致; 未找到的 paper 对应位置为 None。

        Raises:
            SSRateLimitError: 429 重试耗尽
            SSAPIError: 4xx 构造错 / 5xx 重试耗尽 / 网络超时重试耗尽
        """
        # Fixture mode: 集成测试 / 离线实验用; env 设了就读 fixture, 不发真请求
        fixture_dir = os.environ.get("PAPERPILOT_SS_FIXTURE_DIR")
        if fixture_dir:
            return self._fetch_from_fixture(arxiv_ids, fixture_dir)

        body = {"ids": [f"ARXIV:{aid}" for aid in arxiv_ids]}
        ... (其余不变)
```

并在类内加方法:

```python
    def _fetch_from_fixture(self, arxiv_ids: list[str], fixture_dir: str) -> list[dict | None]:
        """读 PAPERPILOT_SS_FIXTURE_DIR/ss_attention_bert.json, 按 arxiv_id 匹配返回。

        匹配策略: fixture 是 list[paper], 按 paper.externalIds.ArXiv 找; 找不到返 None。
        """
        import json as _json
        path = Path(fixture_dir) / "ss_attention_bert.json"
        with path.open(encoding="utf-8") as f:
            data = _json.load(f)
        index = {}
        for p in data:
            if not p:
                continue
            ext = p.get("externalIds") or {}
            arxiv = ext.get("ArXiv")
            if arxiv:
                index[arxiv] = p
        return [index.get(aid) for aid in arxiv_ids]
```

(顶部 import 加 `from pathlib import Path` 如果还没有。)

- [ ] **Step 6.2: 跑 ss_client 单测确认 fixture mode 不影响现有测**

```bash
.venv/Scripts/python.exe -m pytest tests/mcp_servers/test_ss_client.py -v
```

Expected: 7 passed (现有测全用 monkeypatch.delenv 清掉 SEMANTIC_SCHOLAR_API_KEY 但没清 PAPERPILOT_SS_FIXTURE_DIR; 若 CI 环境碰巧有这 env, 会触发 fixture mode 误测)。

如果 7 个 test 中任何一个开始用真 fixture 路径 (i.e. 行为变了), 在每个 test 顶部加 `monkeypatch.delenv("PAPERPILOT_SS_FIXTURE_DIR", raising=False)`。**预防式**编辑, 改 7 处。

- [ ] **Step 6.3: 写集成测试 `test_graph_via_client.py` (3 个 slow test)**

Create `tests/mcp_servers/test_graph_via_client.py`:

```python
"""graph-mcp 集成测试 — 真起子进程 + MCP 协议; SS 走 fixture mode (不发真请求)。

标 @pytest.mark.slow, 默认 deselect; 本地 `pytest -m slow tests/mcp_servers/test_graph_via_client.py` 跑。
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import pytest

from paperpilot.tools.mcp_client import MCPClient


pytestmark = pytest.mark.slow


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _spawn_client_with_graph(tmp_path, monkeypatch) -> MCPClient:
    """启动 MCPClient, 注入 fixture mode env, 让 graph-mcp 走 fixture。
       graph_path 用 tmp_path 隔离, 避免污染开发机的 data/graph/。

    MCPClient(manifest_path) 必须传参; 用 paperpilot.main.MANIFEST_PATH (absolute)
    确保 chdir 后仍能找到 mcp_servers.json。
    """
    from paperpilot.main import MANIFEST_PATH
    monkeypatch.setenv("PAPERPILOT_SS_FIXTURE_DIR", str(FIXTURE_DIR))
    # 让 graph-mcp 用 tmp_path 作 graph 落盘 (graph_manager 用相对路径 data/graph/, 受 cwd 影响)
    monkeypatch.chdir(tmp_path)

    client = MCPClient(MANIFEST_PATH)
    client.start()
    return client


def _get_tool(client: MCPClient, name: str):
    """从 list_tools() 找到指定 name 的 Tool 对象 (其 .handler 是 sync 闭包)。"""
    for t in client.list_tools():
        if t.name == name:
            return t
    raise AssertionError(f"tool {name!r} not found in client tools")


def test_build_and_query_via_client(tmp_path, monkeypatch):
    """端到端: 启动 graph-mcp 子进程, 经 mcp_client 跑 build_graph + 3 个 query。"""
    client = _spawn_client_with_graph(tmp_path, monkeypatch)
    try:
        names = {t.name for t in client.list_tools()}
        assert "mcp__graph__build_graph" in names
        assert "mcp__graph__get_neighbors" in names
        assert "mcp__graph__get_shortest_path" in names
        assert "mcp__graph__get_common_citations" in names

        # build_graph (走 fixture, 不发真 SS)
        build_tool = _get_tool(client, "mcp__graph__build_graph")
        result_text = build_tool.handler({"arxiv_ids": ["1706.03762", "1810.04805"]})
        result = json.loads(result_text)
        assert result["added_count"] == 2

        # get_neighbors
        nbr_tool = _get_tool(client, "mcp__graph__get_neighbors")
        nbr_text = nbr_tool.handler({
            "arxiv_id": "1706.03762", "direction": "references", "limit": 5,
        })
        # FastMCP list[dict] 序列化坑 (Day 7) — 用 raw_decode 循环解析
        from json import JSONDecoder
        decoder = JSONDecoder()
        nbrs = []
        idx = 0
        s = nbr_text.strip()
        while idx < len(s):
            obj, end = decoder.raw_decode(s, idx)
            nbrs.append(obj)
            idx = end
            while idx < len(s) and s[idx] in " \n\t":
                idx += 1
        # 至少有几个 ref 邻居
        assert len(nbrs) > 0
    finally:
        client.close()


def test_pickle_persists_across_restart(tmp_path, monkeypatch):
    """build_graph → close client → 重起 → graph load 回来, 节点仍在。"""
    # 第一次启动 build
    client1 = _spawn_client_with_graph(tmp_path, monkeypatch)
    try:
        build_tool = _get_tool(client1, "mcp__graph__build_graph")
        build_tool.handler({"arxiv_ids": ["1706.03762"]})
    finally:
        client1.close()

    # tmp_path/data/graph/citation_graph.pkl 应存在
    pkl = tmp_path / "data" / "graph" / "citation_graph.pkl"
    assert pkl.exists()

    # 第二次启动, 不 build 直接查
    client2 = _spawn_client_with_graph(tmp_path, monkeypatch)
    try:
        nbr_tool = _get_tool(client2, "mcp__graph__get_neighbors")
        # 不抛 NodeNotFoundError 即说明 1706.03762 load 回来了
        nbr_text = nbr_tool.handler({
            "arxiv_id": "1706.03762", "direction": "references", "limit": 1,
        })
        # 能正常返回 (具体内容不重要, 不抛错就行)
        assert nbr_text is not None
    finally:
        client2.close()


def test_corrupt_pickle_startup_fail(tmp_path, monkeypatch):
    """预写坏 pickle → 启动 graph-mcp 进程应 hard-fail (mcp_client startup 抛 MCPStartupError)。"""
    from paperpilot.main import MANIFEST_PATH
    from paperpilot.tools.mcp_client import MCPStartupError

    monkeypatch.setenv("PAPERPILOT_SS_FIXTURE_DIR", str(FIXTURE_DIR))
    monkeypatch.chdir(tmp_path)

    pkl_dir = tmp_path / "data" / "graph"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl = pkl_dir / "citation_graph.pkl"
    pkl.write_bytes(b"\x00\x01 not a valid pickle \x99")

    client = MCPClient(MANIFEST_PATH)
    try:
        with pytest.raises(MCPStartupError):
            client.start()
    finally:
        client.close()
```

**API 用法关键点 (与 mcp_client.py 对齐, 避免实施时返工)**:
- `MCPClient(manifest_path)` 构造**必须传** manifest_path 参数 (Path 类型)
- 没有 `get_tool_handler(name)` 方法 — Tool 对象要从 `client.list_tools()` 中按 name 筛选, 然后调 `tool.handler(args)` (handler 是 sync 闭包)
- 启动失败抛 `MCPStartupError` (从 `paperpilot.tools.mcp_client` 导)
- `client.close()` 是同步, 不抛错; finally 里调即可

- [ ] **Step 6.4: 跑集成测试**

```bash
.venv/Scripts/python.exe -m pytest -m slow tests/mcp_servers/test_graph_via_client.py -v
```

Expected: 3 passed (大约 30-60s/case, 因为要起子进程)。

如果 `test_corrupt_pickle_startup_fail` 子进程虽然 crash 但 mcp_client 没抛错 (e.g. 只是 stderr 红字然后 list_tools 是空) — 检查 mcp_client.start 的失败传播。如果 mcp_client 不传播 startup error, 这个测试需要换检查方式 (e.g. start() 后立即 list_tools() 看是否含 graph__build_graph; 不含说明 server 没起来)。

- [ ] **Step 6.5: 写 day8_smoke.py 端到端 (真 SS + 真 LLM)**

Create `scripts/day8_smoke.py`:

```python
"""Day 8 端到端 smoke: search → download → build_graph → query → LLM 答案。

对 graph-mcp 的"求职级 demo"场景: 用 attention 和 BERT 这两篇经典 paper 验证
LLM 真能用 graph 工具找它们之间的引用关系。

不走 fixture mode — 真发 SS API + 真调 LLM。预计:
  arxiv search/download ~10-20s
  build_graph (SS batch ~3-5s)
  3-4 轮 LLM tool_use ~5-10s
  共 ~30-60s, ~$0.05 DeepSeek
"""
from __future__ import annotations

import os
import sys

# Windows 控制台 UTF-8 输出 (沿用 Day 5/6 经验)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# 不设 PAPERPILOT_SS_FIXTURE_DIR, 走真 SS
if "PAPERPILOT_SS_FIXTURE_DIR" in os.environ:
    del os.environ["PAPERPILOT_SS_FIXTURE_DIR"]

from paperpilot.main import run


PROMPT = (
    "搜 'attention is all you need' 和 'BERT pre-training' 这两篇经典 NLP paper, "
    "下载它们的全文, 然后用 graph 工具看 BERT 是否引用了 attention 那篇 (找最短路径), "
    "再列出 attention 那篇的 5 个最重要 references。最后总结你看到的引用关系。"
)


def main() -> int:
    print(f"[Day 8 smoke] prompt: {PROMPT[:80]}...")
    try:
        result = run(PROMPT, max_iterations=15)
    except Exception as e:
        print(f"[Day 8 smoke] FAIL: run() raised {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    answer = result.get("answer") or ""
    print("\n=== FINAL ANSWER ===\n" + answer)

    # 弱断言: LLM 的最终答案至少要提到"BERT" 和"Attention" 两个名字 (引用关系陈述)
    answer_lower = answer.lower()
    must_have = ["attention", "bert"]
    missing = [k for k in must_have if k not in answer_lower]
    if missing:
        print(f"\n[Day 8 smoke] FAIL: answer missing keywords {missing}", file=sys.stderr)
        return 1

    # 检查 data/graph/citation_graph.pkl 存在 (build_graph 真跑过)
    from pathlib import Path
    if not Path("data/graph/citation_graph.pkl").exists():
        print("\n[Day 8 smoke] FAIL: data/graph/citation_graph.pkl 未生成", file=sys.stderr)
        return 1

    print("\n✅ Day 8 smoke PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6.6: 跑 day5/day6 smoke 确认无回归**

```bash
.venv/Scripts/python.exe scripts/day5_smoke.py
```

Expected: 退出 0 + `Day 5 smoke PASSED`。

```bash
.venv/Scripts/python.exe scripts/day6_smoke.py
```

Expected: 退出 0 + `Day 6 smoke PASSED`。

(Day 5/6 不依赖 graph-mcp, 它们应该天然不受影响, 但跑一遍确认 mcp_servers.json 改动没有副作用。)

- [ ] **Step 6.7: 跑 day8_smoke**

```bash
.venv/Scripts/python.exe scripts/day8_smoke.py
```

Expected:
- 中间打印 LLM tool_use 序列 (类似 Day 5/6 smoke 的 `[turn] / [tool_call] / [tool_result]`)
- LLM 至少调一次 `mcp__graph__build_graph` 和一次 `mcp__graph__get_shortest_path` (查日志确认)
- stdout 末尾 `✅ Day 8 smoke PASSED`
- 退出码 0

**常见失败 + 修复**:
1. **LLM 不调 graph 工具, 只用 colbert/arxiv 答** — system prompt 不够引导。如果 prompt 已显式说"用 graph 工具", LLM 还忽视, 检查 paperpilot/agent/loop.py 或 main.py 的 system prompt 是否提了所有 4 server。临时方案: 在 PROMPT 里加更强引导 ("必须调 mcp__graph__build_graph")。
2. **build_graph 报 "9999.99999 不在 SS"** — LLM 传错 arxiv_id (从 search 结果误抄)。LLM 自己应能看错误重试。
3. **SS 限速 429 后退避 8s+8s+8s, 总 24s smoke 跑慢** — 可接受, smoke 整体 < 90s。
4. **shortest_path 返 length=-1** — 可能 SS fixture 时代的 BERT 不引 Attention, 但真 SS 数据里应该引 (BERT 论文确实引了 Attention)。如果真不引, smoke prompt 改成"看它们之间是否有共同引用的 paper"。

- [ ] **Step 6.8: 跑全套测试最终回归**

```bash
.venv/Scripts/python.exe -m pytest tests/ -v
```

Expected: 41 passed, 4 deselected (Day 6 12 + Day 8 29 (ss 7 + manager 16 + server 6) - 集成测 3 deselected = 26+12=...实际计数: Day 6 12 + ss 7 + manager 16 + server 6 = **41 passed**)。

(注意: 集成测试也是 deselected 因为 @pytest.mark.slow, 总 deselected = 4 (Day 6 colbert slow) + 3 (Day 8 graph slow) = **7 deselected**)。

- [ ] **Step 6.9: Commit (smoke 通过后)**

```bash
git add tests/mcp_servers/test_graph_via_client.py paperpilot/mcp_servers/graph/ss_client.py scripts/day8_smoke.py
git commit -m "Day 8 Task 6: 集成测试 (fixture mode) + day8_smoke 端到端 + ss_client fixture mode 切换"
```

---

## Definition of Done (Day 8 完工标志)

1. ✅ `pytest tests/` **41 passed, 7 deselected** (slow 测试本地手跑)
2. ✅ `pytest -m slow tests/mcp_servers/test_graph_via_client.py` **3 passed**
3. ✅ `python scripts/day5_smoke.py` 无回归
4. ✅ `python scripts/day6_smoke.py` 无回归
5. ✅ `python scripts/day8_smoke.py` 退出 0 + 打印 `Day 8 smoke PASSED`
6. ✅ `data/graph/citation_graph.pkl` 存在 (smoke 跑后)
7. ✅ `git grep "TODO\|FIXME" paperpilot/mcp_servers/graph/` 空
8. ✅ git log 显示 Day 8 6 个原子 commit (Task 1-6)
9. ✅ `git push origin main` 推上去 (用户确认后)

---

## 架构边界回顾 (与 spec §1 红线对账)

| 红线 | 验证 |
|---|---|
| L1 基座层 + Day 5 mcp_client 零改动 | `paperpilot/tools/mcp_client.py` 在本 plan 中**不被任何 task 修改** (grep 任何 "Modify: paperpilot/tools/mcp_client.py" 应返空) |
| server 之间不互通信 | graph-mcp 只 import paperpilot.mcp_servers.graph.* + httpx + networkx; 不 import arxiv / colbert |
| 决策由 LLM 做 | server.py 协议层抛 ValueError / NodeNotFoundError, 不 retry / 不 fallback / 不路由分支; ss_client 内部的 retry 是网络层抗抖动, 不感知业务 |
| 不做推测性抽象 | 不抽 `BaseGraphServer` 类, 不留多 graph 命名空间, 不预留 SQLite 后端, 不抽 SS HTTP 公共类 |
| 复用 Day 7 trip wire | 4 个 query tool 全返 list[dict] (与 colbert.search 一致); spec §10 的 trip wire 触发条件未变 |

---

## 工时切分 (与 spec §9 对账)

| Task | spec 预估 | plan 细化估算 |
|---|---|---|
| Task 1 (依赖+fixture+目录) | (合并到 Task 1 of spec) | ~15-25min |
| Task 2 (ss_client + 7 单测) | 60min | 70-90min |
| Task 3 (graph_manager build+持久化) | (Task 2 of spec, 拆出 build) | 75min |
| Task 4 (graph_manager 3 query) | (Task 2 of spec, 拆出 query) | 45min |
| Task 5 (server.py 协议层) | 45min | 30-40min |
| Task 6 (集成+smoke+真 SS+真 LLM) | (Task 4+5 of spec) | 60-90min |
| **合计** | spec 4.5h | plan **~5-6h** (与 spec 预算 6h 对齐, 含 buffer) |

plan 比 spec 多拆出"加依赖" 和"build vs query 两阶段" 是 TDD 实施粒度的自然结果, 不是 scope 蔓延。

---

## 注意事项 (implementer 必读)

1. **TDD 严格走**: 每个 task 内部"写测 → 跑红 → 写实现 → 跑绿 → commit" 五步。不允许"先写实现回头补测"。
2. **每 task 独立 commit**: 6 个 commit, message 格式 `Day 8 Task N: <一句话总结>`, 不带 AI 署名 (memory `feedback_no_claude_commits`)。
3. **遇到 spec 与 plan 不一致**: 优先 spec; plan 是 spec 的细化, 不增 spec 没说的特性。spec 没说的需求 = YAGNI 跳过。
4. **Day 6 / Day 7 测试必须不回归**: 每 task 完工前跑 `pytest tests/` 确认 12 (Day 6 baseline) 仍 passed (Day 8 中途测数会逐步累加)。
5. **subagent dispatch 时**: 把本 plan 的单 task 段落 (`## Task N` ~ 下一个 `## Task N+1` 之前) 完整给 subagent, 不裁剪, 不省略代码块。
