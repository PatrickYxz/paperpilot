# Day 12: ColBERT per-paper 索引隔离 + compare-papers skill 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ColBERT MCP server 的全局单索引改为 per-paper 隔离索引(命中即跳过 + 跨 session 持久化),撕掉 `paper_deep_read` 的 Day 11 串行限制,并新增 `compare-papers` skill 让 LLM 在多论文对比场景自动触发 `paper_deep_read`。

**Architecture:** `IndexManager` 内部状态从单 PLAID 改成 `dict[paper_id, _IndexState]`,`build` 走"内存命中→磁盘命中→冷启动"三路径,`search` 必填 `paper_id` 入参做隔离。磁盘目录使用 `_paper_key(paper_id)` 做 filesystem-safe 转义,避免旧式 arXiv ID 里的 `/` 变成路径分隔。`THREAD_POOL_SIZE` 从 1 提到 `min(3, len(paper_ids))`,与 per-paper 索引配合解锁真并发。

**Tech Stack:** Python 3.12, PyLate (ColBERT), FastMCP, pytest, ThreadPoolExecutor。

**Spec:** `docs/superpowers/specs/2026-05-04-day12-per-paper-index-and-compare-papers-design.md`。

---

## 文件结构

| 路径 | 动作 | 责任 |
|---|---|---|
| `paperpilot/mcp_servers/colbert/index_manager.py` | 重写 | per-paper 索引核心(三路径 build / paper_id search / lazy load / chunks.json 落盘) |
| `paperpilot/mcp_servers/colbert/server.py` | 改 | `search` schema 加 `paper_id`;`build_index` docstring/return 标注 cached_papers |
| `paperpilot/builtin_tools/subagent.py` | 改 | `THREAD_POOL_SIZE = 3`;`SUBAGENT_SYSTEM` search 例子加 `paper_id="..."`;删 Day 11 保守注释;emit `subagent_start` / `subagent_done` |
| `paperpilot/skills/compare-papers.md` | 新建 | `compare-papers` skill 触发 `paper_deep_read` |
| `paperpilot/skills/deep-read-paper.md` | 改 | search 步骤补 `paper_id` 入参 |
| `paperpilot/main.py` | 改一行 | `SYSTEM_PROMPT_BASE` 加"search 必传 paper_id"规约 |
| `tests/mcp_servers/test_colbert_server.py` | 重写 | server schema 透传(search 入参 paper_id、build 返回结构) |
| `tests/mcp_servers/test_index_manager.py` | 新建 | IndexManager 三路径 + paper_id 隔离 + lazy load(PyLate mock) |
| `tests/builtin_tools/test_subagent.py` | 改 | `THREAD_POOL_SIZE == 3`;`SUBAGENT_SYSTEM` 含 `paper_id=`;验证 subagent lifecycle events |
| `tests/test_main_integration.py` | 改 | fast 加 `compare-papers in skill section` |
| `tests/mcp_servers/test_colbert_via_client.py` | 改(slow) | 加 paper A/B 隔离断言 + 同 paper_id 二次 build 走 disk-hit |
| `scripts/day12_smoke.py` | 新建 | compare-papers 链路真 LLM 端到端 + 真并发证据(2+ subagent 生命周期窗口重叠) |

---

## Task 1: IndexManager per-paper 重写(单测,PyLate mock)

最大改动,严格 TDD。在不引入真 PyLate 的情况下用 `MagicMock` 把 `_model` / `indexes.PLAID` / `retrieve.ColBERT` mock 掉,只验证我们自己的状态机与文件 IO 逻辑。

**Files:**
- Modify: `paperpilot/mcp_servers/colbert/index_manager.py` (重写,~150 行)
- Create: `tests/mcp_servers/test_index_manager.py`

### Step 1.1: Write failing test for cold-build single paper

- [ ] 新建测试文件,写第一个失败 case。

```python
# tests/mcp_servers/test_index_manager.py
"""IndexManager per-paper 索引核心单测。PyLate 全部 mock。"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from paperpilot.mcp_servers.colbert import index_manager as im_module


@pytest.fixture
def patched_root(tmp_path, monkeypatch):
    monkeypatch.setattr(im_module, "INDEX_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def patched_pylate(monkeypatch):
    fake_model = MagicMock()
    fake_model.tokenizer.encode.return_value = list(range(300))
    fake_model.tokenizer.decode.side_effect = lambda toks: f"chunk-of-{len(toks)}"
    fake_model.encode.return_value = [[0.0] * 4]

    monkeypatch.setattr(im_module.models, "ColBERT", lambda **kw: fake_model)

    plaid_instances: list[MagicMock] = []

    def _plaid(**kw):
        inst = MagicMock()
        inst.__init_kwargs__ = kw
        plaid_instances.append(inst)
        return inst

    monkeypatch.setattr(im_module.indexes, "PLAID", _plaid)

    fake_retr = MagicMock()
    fake_retr.retrieve.return_value = [[]]
    monkeypatch.setattr(im_module.retrieve, "ColBERT", lambda **kw: fake_retr)

    return {"model": fake_model, "plaid": plaid_instances, "retr": fake_retr}


def test_cold_build_creates_per_paper_dir_and_chunks_json(
    patched_root, patched_pylate
):
    mgr = im_module.IndexManager()
    out = mgr.build([{"paper_id": "p1", "text": "hello world"}])

    assert out["indexed_count"] == 1
    assert out["index_name"] == "paperpilot_current"
    assert out["fresh_papers"] == ["p1"]
    assert out["cached_papers"] == []

    paper_dir = patched_root / "p1"
    assert (paper_dir / "paperpilot_current").parent == paper_dir
    assert (paper_dir / "chunks.json").exists()

    chunks = json.loads((paper_dir / "chunks.json").read_text("utf-8"))
    assert all(cid.startswith("p1::chunk_") for cid in chunks.keys())


def test_paper_key_escapes_path_separators(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    out = mgr.build([{"paper_id": "cs/0501001", "text": "legacy arxiv id"}])

    assert out["fresh_papers"] == ["cs/0501001"]
    paper_dirs = [p for p in patched_root.iterdir() if p.is_dir()]
    assert len(paper_dirs) == 1
    assert "/" not in paper_dirs[0].name
    assert "\\" not in paper_dirs[0].name

    chunks = json.loads((paper_dirs[0] / "chunks.json").read_text("utf-8"))
    assert all(cid.startswith("cs/0501001::chunk_") for cid in chunks.keys())
```

### Step 1.2: Run test, verify import / interface fails

- [ ] 跑测试,预期 `INDEX_ROOT` 还在但其它 API 没对上。

Run: `pytest tests/mcp_servers/test_index_manager.py::test_cold_build_creates_per_paper_dir_and_chunks_json -v`
Expected: FAIL(`fresh_papers` / `cached_papers` 字段不存在,或 chunks.json 不写,或旧式 arXiv ID 路径未转义)

### Step 1.3: Rewrite IndexManager — 删旧、加状态字典、加 path helpers

- [ ] 整体替换 `paperpilot/mcp_servers/colbert/index_manager.py`:

```python
"""colbert-mcp 索引层。per-paper 隔离 + 跨 session 持久化。

每个 paper_id 一个子目录:
  data/colbert_index/<paper_id>/
    paperpilot_current/   # PLAID 内部目录(沿用常量名)
    chunks.json           # {chunk_id: chunk_text} for search 回填

build_index 走三路径:
  1. memory-hit: paper_id 在 self._states 里 → 跳过
  2. disk-hit:  paper_dir 存在 → lazy load PLAID + chunks.json
  3. cold:      encode + 写 PLAID + dump chunks.json

Windows DLL 顺序仍要求 pyarrow / datasets 在 torch 之前 import。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow  # noqa: F401 (Windows DLL order)
import datasets  # noqa: F401 (Windows DLL order)

from pylate import indexes, models, retrieve

INDEX_NAME = "paperpilot_current"
INDEX_ROOT = Path("data/colbert_index")
MODEL_NAME = "lightonai/colbertv2.0"
_SAFE_PAPER_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")


class IndexNotFoundError(RuntimeError):
    """search 时该 paper 的索引在内存与磁盘都不存在。"""


@dataclass
class _IndexState:
    index: object
    chunk_texts: dict[str, str] = field(default_factory=dict)


class IndexManager:
    CHUNK_SIZE = 256
    OVERLAP = 32

    def __init__(self) -> None:
        INDEX_ROOT.mkdir(parents=True, exist_ok=True)
        self._model = models.ColBERT(model_name_or_path=MODEL_NAME)
        self._states: dict[str, _IndexState] = {}

    def build(self, documents: list[dict]) -> dict:
        if not documents:
            raise ValueError("documents must not be empty")

        cached: list[str] = []
        fresh: list[str] = []
        for d in documents:
            pid = d["paper_id"]
            if pid in self._states:
                cached.append(pid)
                continue
            if self._try_lazy_load(pid):
                cached.append(pid)
                continue
            self._cold_build(pid, d["text"])
            fresh.append(pid)

        return {
            "indexed_count": len(documents),
            "index_name": INDEX_NAME,
            "cached_papers": cached,
            "fresh_papers": fresh,
        }

    def search(self, query: str, paper_id: str, top_k: int) -> list[dict]:
        state = self._states.get(paper_id)
        if state is None:
            if not self._try_lazy_load(paper_id):
                raise IndexNotFoundError(
                    f"no index for {paper_id!r}; call build_index first"
                )
            state = self._states[paper_id]

        q_emb = self._model.encode(
            [query], is_query=True, show_progress_bar=False
        )
        retr = retrieve.ColBERT(index=state.index)
        scores = retr.retrieve(queries_embeddings=q_emb, k=top_k)
        return [
            {
                "paper_id": r["id"].split("::", 1)[0],
                "chunk_text": state.chunk_texts[r["id"]],
                "score": float(r["score"]),
            }
            for r in scores[0]
        ]

    def _cold_build(self, paper_id: str, text: str) -> None:
        chunks = self._chunk(text)
        chunk_ids = [f"{paper_id}::chunk_{i}" for i in range(len(chunks))]
        embs = self._model.encode(
            chunks, is_query=False, show_progress_bar=False
        )
        index = indexes.PLAID(
            index_folder=str(self._paper_root(paper_id)),
            index_name=INDEX_NAME,
            override=True,
        )
        index.add_documents(documents_ids=chunk_ids, documents_embeddings=embs)

        chunk_texts = dict(zip(chunk_ids, chunks))
        self._paper_root(paper_id).mkdir(parents=True, exist_ok=True)
        self._chunks_path(paper_id).write_text(
            json.dumps(chunk_texts, ensure_ascii=False),
            encoding="utf-8",
        )
        self._states[paper_id] = _IndexState(index=index, chunk_texts=chunk_texts)

    def _try_lazy_load(self, paper_id: str) -> bool:
        if not self._index_path(paper_id).exists():
            return False
        if not self._chunks_path(paper_id).exists():
            return False
        try:
            chunk_texts = json.loads(
                self._chunks_path(paper_id).read_text("utf-8")
            )
            index = indexes.PLAID(
                index_folder=str(self._paper_root(paper_id)),
                index_name=INDEX_NAME,
                override=False,
            )
        except Exception:
            return False
        self._states[paper_id] = _IndexState(index=index, chunk_texts=chunk_texts)
        return True

    def _chunk(self, text: str) -> list[str]:
        tokens = self._model.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return []
        out: list[str] = []
        i = 0
        while i < len(tokens):
            sub = tokens[i : i + self.CHUNK_SIZE]
            out.append(self._model.tokenizer.decode(sub))
            if i + self.CHUNK_SIZE >= len(tokens):
                break
            i += self.CHUNK_SIZE - self.OVERLAP
        return out

    def _paper_root(self, paper_id: str) -> Path:
        return INDEX_ROOT / self._paper_key(paper_id)

    def _index_path(self, paper_id: str) -> Path:
        return self._paper_root(paper_id) / INDEX_NAME

    def _chunks_path(self, paper_id: str) -> Path:
        return self._paper_root(paper_id) / "chunks.json"

    def _paper_key(self, paper_id: str) -> str:
        safe = _SAFE_PAPER_KEY_RE.sub("_", paper_id).strip("._")
        digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:10]
        return f"{safe or 'paper'}-{digest}"
```

### Step 1.4: Run cold-build test, verify pass

- [ ] Run: `pytest tests/mcp_servers/test_index_manager.py::test_cold_build_creates_per_paper_dir_and_chunks_json -v`
Expected: PASS

### Step 1.5: Add memory-hit + disk-hit test

- [ ] 追加两个用例:

```python
def test_memory_hit_skips_encode(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    mgr.build([{"paper_id": "p1", "text": "hello"}])
    patched_pylate["model"].encode.reset_mock()

    out = mgr.build([{"paper_id": "p1", "text": "hello"}])

    assert out["cached_papers"] == ["p1"]
    assert out["fresh_papers"] == []
    patched_pylate["model"].encode.assert_not_called()


def test_disk_hit_loads_from_disk(patched_root, patched_pylate):
    mgr1 = im_module.IndexManager()
    mgr1.build([{"paper_id": "p1", "text": "hello"}])

    # 模拟 server 重启
    mgr2 = im_module.IndexManager()
    patched_pylate["model"].encode.reset_mock()

    out = mgr2.build([{"paper_id": "p1", "text": "hello"}])

    assert out["cached_papers"] == ["p1"]
    assert out["fresh_papers"] == []
    patched_pylate["model"].encode.assert_not_called()
    assert "p1" in mgr2._states
```

### Step 1.6: Run, verify pass(实现已就位)

- [ ] Run: `pytest tests/mcp_servers/test_index_manager.py -v`
Expected: 3 passed

### Step 1.7: Add disk-hit fallback to cold-build on corrupted chunks.json

- [ ] 追加用例:

```python
def test_disk_hit_falls_back_to_cold_when_chunks_corrupt(
    patched_root, patched_pylate
):
    mgr1 = im_module.IndexManager()
    mgr1.build([{"paper_id": "p1", "text": "hello"}])

    (patched_root / "p1" / "chunks.json").write_text("not-json", encoding="utf-8")

    mgr2 = im_module.IndexManager()
    patched_pylate["model"].encode.reset_mock()

    out = mgr2.build([{"paper_id": "p1", "text": "hello"}])

    assert out["fresh_papers"] == ["p1"]
    assert out["cached_papers"] == []
    assert patched_pylate["model"].encode.called
```

- [ ] Run: `pytest tests/mcp_servers/test_index_manager.py -v`
Expected: 4 passed

### Step 1.8: Add search paper_id required + IndexNotFoundError

- [ ] 追加用例:

```python
def test_search_requires_built_paper_id(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    with pytest.raises(im_module.IndexNotFoundError):
        mgr.search("q", paper_id="never-built", top_k=3)


def test_search_returns_only_matching_paper_chunks(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    mgr.build([
        {"paper_id": "pA", "text": "alpha alpha alpha"},
        {"paper_id": "pB", "text": "beta beta beta"},
    ])

    state_a = mgr._states["pA"]
    chunk_id_a = next(iter(state_a.chunk_texts.keys()))
    patched_pylate["retr"].retrieve.return_value = [
        [{"id": chunk_id_a, "score": 0.9}]
    ]

    out = mgr.search("alpha", paper_id="pA", top_k=1)
    assert len(out) == 1
    assert out[0]["paper_id"] == "pA"
    assert out[0]["chunk_text"] == state_a.chunk_texts[chunk_id_a]
```

- [ ] Run: `pytest tests/mcp_servers/test_index_manager.py -v`
Expected: 6 passed

### Step 1.9: Add multi-paper build returns mixed cached/fresh

- [ ] 追加用例:

```python
def test_build_mixed_cached_and_fresh(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    mgr.build([{"paper_id": "p1", "text": "x"}])

    out = mgr.build([
        {"paper_id": "p1", "text": "x"},
        {"paper_id": "p2", "text": "y"},
    ])
    assert out["indexed_count"] == 2
    assert out["cached_papers"] == ["p1"]
    assert out["fresh_papers"] == ["p2"]
```

- [ ] Run: `pytest tests/mcp_servers/test_index_manager.py -v`
Expected: 7 passed

### Step 1.10: Sanity-run full test suite, then commit

- [ ] Run: `pytest tests/ --ignore=tests/mcp_servers/test_graph_via_client.py 2>&1 | tail -5`
Expected: 旧 `test_colbert_server.py` 可能 4 个挂(server 还没改),其余应通过。先记录有多少挂,Task 2 修。

- [ ] Commit:

```bash
git add paperpilot/mcp_servers/colbert/index_manager.py tests/mcp_servers/test_index_manager.py
git commit -m "Day 12 Task 1: IndexManager per-paper 索引核心 + 三路径 build"
```

---

## Task 2: ColBERT MCP server schema 更新

`search` 加 `paper_id` 入参,`build_index` docstring/return 反映 cached_papers/fresh_papers,`test_colbert_server.py` 同步重写。

**Files:**
- Modify: `paperpilot/mcp_servers/colbert/server.py`
- Modify: `tests/mcp_servers/test_colbert_server.py`

### Step 2.1: Rewrite test_colbert_server.py for new search signature

- [ ] 整体替换:

```python
"""colbert-mcp server.py 协议层单测。
mock IndexManager (避免拉 ColBERT 模型),只验证路由 + 输入校验。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paperpilot.mcp_servers.colbert import server as colbert_server
from paperpilot.mcp_servers.colbert.index_manager import IndexNotFoundError


@pytest.fixture
def mock_manager(monkeypatch):
    m = MagicMock()
    monkeypatch.setattr(colbert_server, "_manager", m)
    return m


def test_build_routes_to_manager(mock_manager):
    mock_manager.build.return_value = {
        "indexed_count": 2,
        "index_name": "paperpilot_current",
        "cached_papers": [],
        "fresh_papers": ["p1", "p2"],
    }
    docs = [
        {"paper_id": "p1", "text": "hello"},
        {"paper_id": "p2", "text": "world"},
    ]

    result = colbert_server._build_index_impl(docs)

    mock_manager.build.assert_called_once_with(docs)
    assert result["fresh_papers"] == ["p1", "p2"]


def test_build_empty_documents_raises_value_error(mock_manager):
    with pytest.raises(ValueError, match="must not be empty"):
        colbert_server._build_index_impl([])
    mock_manager.build.assert_not_called()


def test_build_missing_field_raises_value_error(mock_manager):
    bad = [{"paper_id": "p1"}]
    with pytest.raises(ValueError, match="paper_id.*text"):
        colbert_server._build_index_impl(bad)
    mock_manager.build.assert_not_called()


def test_search_passes_paper_id_to_manager(mock_manager):
    mock_manager.search.return_value = [
        {"paper_id": "p1", "chunk_text": "x", "score": 0.5}
    ]
    out = colbert_server._search_impl("q", paper_id="p1", top_k=3)

    mock_manager.search.assert_called_once_with("q", "p1", 3)
    assert out[0]["paper_id"] == "p1"


def test_search_index_missing_raises(mock_manager):
    mock_manager.search.side_effect = IndexNotFoundError("no index for 'p1'")
    with pytest.raises(IndexNotFoundError):
        colbert_server._search_impl("q", paper_id="p1", top_k=5)
```

### Step 2.2: Run test, verify it fails on signature mismatch

- [ ] Run: `pytest tests/mcp_servers/test_colbert_server.py -v`
Expected: FAIL on `_search_impl` 不接受 `paper_id` 关键字

### Step 2.3: Update server.py — add paper_id to search

- [ ] 整体替换 server.py:

```python
"""colbert-mcp: ColBERT 段落级语义检索 server。

启动:python -m paperpilot.mcp_servers.colbert.server
通过 stdio 被 paperpilot.tools.mcp_client 拉起,manifest 见 paperpilot/mcp_servers.json。

协议层职责: 输入校验 + 路由到 IndexManager。**不接触 PyLate**。
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from paperpilot.mcp_servers.colbert.index_manager import (
    IndexManager,
    IndexNotFoundError,
)

mcp = FastMCP("colbert")
_manager: IndexManager | None = None


@mcp.tool()
def build_index(documents: list[dict]) -> dict:
    """对一组论文全文建立 ColBERT 索引(per-paper 隔离,跨 session 持久化)。

    每个 paper_id 一套独立索引;同 paper_id 重复 build 会命中缓存(磁盘或内存)
    自动跳过 encode。

    调用示例: build_index(documents=[{"paper_id": "2401.12345", "text": "...全文..."}])
    download_paper 的返回值可直接封装进列表传入。

    Args:
        documents: list,每项 dict 含 paper_id (str) 与 text (str) 字段,非空。

    Returns:
        dict: indexed_count(int)、index_name(str)、
              cached_papers(list[str], 跳过的)、fresh_papers(list[str], 真编码的)。
    """
    return _build_index_impl(documents)


def _build_index_impl(documents: list[dict]) -> dict:
    if not documents:
        raise ValueError("documents must not be empty")
    for d in documents:
        if not isinstance(d, dict) or "paper_id" not in d or "text" not in d:
            raise ValueError(
                f"each document must be dict with 'paper_id' and 'text': {d!r}"
            )
    assert _manager is not None, "IndexManager not initialized"
    return _manager.build(documents)


@mcp.tool()
def search(query: str, paper_id: str, top_k: int = 5) -> list[dict]:
    """在某 paper 的 ColBERT 索引上查询 top-k 段落。

    Args:
        query: 自然语言查询。
        paper_id: 必填。必须是已 build_index 过的 paper_id。
        top_k: 返回的段落数,默认 5。

    Returns:
        list,每项 dict 含 paper_id (str)、chunk_text (str)、score (float)。
        若 paper_id 无对应索引(未 build_index 或已被清理)则抛 IndexNotFoundError。
    """
    return _search_impl(query, paper_id, top_k)


def _search_impl(query: str, paper_id: str, top_k: int) -> list[dict]:
    assert _manager is not None, "IndexManager not initialized"
    return _manager.search(query, paper_id, top_k)


if __name__ == "__main__":
    _manager = IndexManager()
    mcp.run(transport="stdio")
```

### Step 2.4: Run server tests, verify pass

- [ ] Run: `pytest tests/mcp_servers/test_colbert_server.py -v`
Expected: 5 passed

### Step 2.5: Run default suite to check regression

- [ ] Run: `pytest tests/ --ignore=tests/mcp_servers/test_graph_via_client.py 2>&1 | tail -5`
Expected: subagent 测试可能挂(THREAD_POOL_SIZE 还是 1),Task 3 修。

### Step 2.6: Commit

- [ ] ```bash
git add paperpilot/mcp_servers/colbert/server.py tests/mcp_servers/test_colbert_server.py
git commit -m "Day 12 Task 2: colbert-mcp server schema 加 paper_id"
```

---

## Task 3: 撕掉 paper_deep_read 的 Day 11 串行限制

**Files:**
- Modify: `paperpilot/builtin_tools/subagent.py`
- Modify: `tests/builtin_tools/test_subagent.py`

### Step 3.1: Update subagent test for THREAD_POOL_SIZE = 3

- [ ] 改 `tests/builtin_tools/test_subagent.py` 末尾的 `test_tool_metadata_and_conservative_worker_count`:
  - 函数名重命名为 `test_tool_metadata_and_worker_count`
  - `assert THREAD_POOL_SIZE == 1` → `assert THREAD_POOL_SIZE == 3`
  - 新增断言 `assert "paper_id=" in SUBAGENT_SYSTEM`(workflow 第 3 步必须包含 paper_id 入参示例)

```python
def test_tool_metadata_and_worker_count():
    tool = paper_deep_read_tool(
        client_factory=lambda: FakeClient(lambda m: _text_response("x")),
        mcp_tools=[],
        on_event=lambda k, v: None,
    )
    assert tool.name == "paper_deep_read"
    assert tool.input_schema["additionalProperties"] is False
    assert tool.input_schema["required"] == ["paper_ids", "user_query"]
    assert tool.input_schema["properties"]["paper_ids"]["maxItems"] == MAX_PAPERS
    assert MAX_PAPERS == 8
    assert SUBAGENT_MAX_ITER == 8
    assert THREAD_POOL_SIZE == 3
    assert "paper_deep_read" in PAPER_DEEP_READ_NUDGE
    assert "paper_ids" not in SUBAGENT_SYSTEM
    assert "paper_id=" in SUBAGENT_SYSTEM
    assert "mcp__colbert__search(query" in SUBAGENT_SYSTEM
```

### Step 3.2: Run, verify it fails

- [ ] Run: `pytest tests/builtin_tools/test_subagent.py::test_tool_metadata_and_worker_count -v`
Expected: FAIL on `THREAD_POOL_SIZE == 3`

### Step 3.3: Update subagent.py — flip pool size, update prompt, scrub Day 11 wording

- [ ] 改文件顶部 docstring(删保守措辞):

```python
"""paper_deep_read built-in tool.

Each paper is read by an independent agent loop with isolated context. Workers
run in parallel via ThreadPoolExecutor; ColBERT MCP gives each paper its own
index so concurrent build/search do not race.
"""
```

- [ ] 改常量:

```python
THREAD_POOL_SIZE = 3
```

- [ ] 改 `SUBAGENT_SYSTEM` workflow 第 3 步:

```
3. Call mcp__colbert__search(query="...", paper_id="<paper_id>", top_k=5) several
   times for method, experiments, findings, limitations, and query-specific evidence.
```

- [ ] 改 tool description(删 "Day 11 runs workers serially..."句):

```python
return Tool(
    name="paper_deep_read",
    description=(
        "Deep-read 1-8 papers with isolated subagents and return markdown "
        "summaries for comparison or synthesis. Use for multi-paper "
        "questions where abstracts are not enough."
    ),
    ...
```

- [ ] 鍦ㄥ瓙 agent worker 生命周期 emit:

```python
on_event("subagent_start", {})
...
on_event("subagent_done", {"status": status})
```

这些事件会被现有 `make_sub_emit(paper_id)` 包装,所以主 tracer 会看到 `subagent_paper_id`。day12_smoke 用生命周期窗口重叠证明并发,不再用 search tool_call/tool_result 窗口,避免 MCP stdio/server 串行化导致 flaky。

### Step 3.4: Run subagent tests, verify pass

- [ ] Run: `pytest tests/builtin_tools/test_subagent.py -v`
Expected: 14 passed(原 14 个用例全过)

### Step 3.5: Grep 防回流

- [ ] Run: `git grep -n "Day 11 conservative\|THREAD_POOL_SIZE = 1\|runs workers serially" paperpilot tests`
Expected: no output

### Step 3.6: Commit

- [ ] ```bash
git add paperpilot/builtin_tools/subagent.py tests/builtin_tools/test_subagent.py
git commit -m "Day 12 Task 3: 撕掉 paper_deep_read Day 11 串行限制"
```

---

## Task 4: compare-papers skill + deep-read-paper skill 同步 + main system prompt

**Files:**
- Create: `paperpilot/skills/compare-papers.md`
- Modify: `paperpilot/skills/deep-read-paper.md`
- Modify: `paperpilot/main.py`(系统提示词加一行)
- Modify: `tests/test_main_integration.py`

### Step 4.1: Add failing test for compare-papers in skill section

- [ ] 改 `tests/test_main_integration.py::test_build_system_prompt_includes_skills`,加一行:

```python
def test_build_system_prompt_includes_skills():
    prompt = _build_system_prompt()
    assert "## 可用 skill" in prompt
    assert "deep-read-paper" in prompt
    assert "explore-citations" in prompt
    assert "find-classics" in prompt
    assert "compare-papers" in prompt
    assert "PaperPilot" in prompt
    assert "build_index" in prompt
```

- [ ] 同文件追加新用例:

```python
def test_build_system_prompt_mentions_search_paper_id_required():
    prompt = _build_system_prompt()
    assert "mcp__colbert__search" in prompt
    assert "paper_id" in prompt
```

### Step 4.2: Run, verify both fail

- [ ] Run: `pytest tests/test_main_integration.py -v`
Expected: 2 fast 用例 FAIL

### Step 4.3: Create compare-papers skill

- [ ] 新建 `paperpilot/skills/compare-papers.md`:

```markdown
---
name: compare-papers
description: 多论文深读对比:用 paper_deep_read 并发精读 3-8 篇,综合对比方法/发现/适用场景
when_to_use: 用户给定 3-8 个 arxiv id 或论文,要求对比、综合或并列分析
---

# Compare Papers

## 适用场景
- 用户明确给出 3-8 个 arxiv id 或论文标题,要求对比/综合/并列分析。
- 用户问"这几篇 paper 的 X 设计有什么异同"。

## 步骤
1. 调一次 `paper_deep_read(paper_ids=[...], user_query="<原问题>")`。
   - 每篇会被一个独立子 agent 精读,用 download + colbert.build_index + colbert.search 流程。
   - 工具返回 markdown,每篇一个 `### <paper_id>` section,含核心方法 / 关键发现 / 与查询相关性。
2. 综合对比:相同点 / 不同点 / 各自适用场景。
3. 引用每篇的 section 作为证据,**不要编造段落或结论**。

## 注意
- 单篇用 `deep-read-paper`,3 篇起才用本 skill。
- 用户只给主题没具体 id 时,先自己调 `mcp__arxiv__search_papers` 拿候选 id 再触发本 skill。
- 不要为了凑数把不相关的 paper 塞进去;少而准比多而散更有用。
```

### Step 4.4: Update deep-read-paper skill — search step requires paper_id

- [ ] 改 `paperpilot/skills/deep-read-paper.md` 步骤 3:

```markdown
3. 多轮检索: 针对用户问题里的关键概念调 `mcp__colbert__search(query="...", paper_id="<step 2 build 的 paper_id>", top_k=3)`。
   - paper_id 必传,值与 step 2 build_index 时的 paper_id 一致。
   - 一个 query 不够时,拆成多个具体 query 多搜几次。
   - query 应围绕用户真正关心的术语,例如 definition、architecture、experiment、ablation。
```

### Step 4.5: Update main.py SYSTEM_PROMPT_BASE — add search paper_id rule

- [ ] 改 `SYSTEM_PROMPT_BASE` 末尾追加一行:

```python
SYSTEM_PROMPT_BASE = """你是 PaperPilot,一个学术论文研究助手。
工作原则:
- 有 tool 可用时优先调 tool;不要自己编造论文标题、作者或 arxiv id
- 一次只解决用户问的事,不主动扩展任务范围
- tool 报错时,根据错误信息决定:重试(换参数 / 换工具) / 告诉用户失败原因
- 调 tool 时必须按 schema 传完整必填参数;如果错误提示缺字段,下一轮必须补齐字段,不要重复同一个空参数
- 调 mcp__colbert__build_index 时,documents 必须是非空列表,每项包含 paper_id 和 text;通常直接使用 mcp__arxiv__download_paper 返回的对象组成 documents=[download_result]
- 调 mcp__colbert__search 时,paper_id 必填,值必须是已经 build_index 过的同一个 paper_id
""".strip()
```

### Step 4.6: Run main integration tests, verify pass

- [ ] Run: `pytest tests/test_main_integration.py -v`
Expected: 4 passed(原 3 + 新 1)。slow 标记的不在这次范围。

### Step 4.7: Run full default suite

- [ ] Run: `pytest tests/ --ignore=tests/mcp_servers/test_graph_via_client.py 2>&1 | tail -10`
Expected: 全绿(fast 套件)

### Step 4.8: Commit

- [ ] ```bash
git add paperpilot/skills/compare-papers.md paperpilot/skills/deep-read-paper.md paperpilot/main.py tests/test_main_integration.py
git commit -m "Day 12 Task 4: compare-papers skill + search paper_id 规约"
```

---

## Task 5: 更新 slow 集成 + day12_smoke + 回归

`tests/mcp_servers/test_colbert_via_client.py` 现有 5 个 slow 用例**全部都会被新 schema 破坏**(search 没传 paper_id;`test_startup_clears_stale_index` 直接测我们删掉的行为)。先逐一改/删,再加新的 isolation+disk-hit 用例,最后写 day12_smoke。

**Files:**
- Modify: `tests/mcp_servers/test_colbert_via_client.py` (5 个用例改 + 1 个删 + 1 个新增)
- Create: `scripts/day12_smoke.py`

### Step 5.1: Patch test_build_and_search — 加 paper_id

- [ ] 改 `test_build_and_search` 的 search 调用 + 断言:

```python
        build_result = build.handler({"documents": docs})
        b = json.loads(build_result) if isinstance(build_result, str) else build_result
        assert b["indexed_count"] == 3
        assert b["index_name"] == "paperpilot_current"
        assert set(b["fresh_papers"]) == {"p1", "p2", "p3"}

        # 改 search: paper_id 必填,只查 p3 的索引
        search_result = search.handler({
            "query": "late interaction retrieval",
            "paper_id": "p3",
            "top_k": 3,
        })
        results = _decode_results(search_result)
        assert len(results) >= 1
        assert all(r["paper_id"] == "p3" for r in results), \
            f"per-paper 隔离: search(paper_id=p3) 应只返 p3, got {results}"
        top1 = results[0]
        assert top1["chunk_text"] != ""
        assert len(top1["chunk_text"]) > 30
        assert isinstance(top1["score"], float)
```

### Step 5.2: Patch test_consecutive_rebuild_releases_previous_index — 重命名 + 改用例语义

per-paper 索引下,build p1 / build p2 是两套独立索引并存,不再互相覆盖。原用例 "p2 覆盖 p1 search 仍返 p2" 的语义已不成立。改成 "可以分别查到":

- [ ] 整段替换:

```python
@pytest.mark.slow
def test_two_papers_coexist_after_consecutive_builds(tmp_path):
    """per-paper 索引: 连续 build p1/p2 后,二者各自可查。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        build = next(t for t in c.list_tools() if t.name == "mcp__colbert__build_index")
        search = next(t for t in c.list_tools() if t.name == "mcp__colbert__search")

        build.handler({
            "documents": [{
                "paper_id": "p1",
                "text": "Transformers use self attention for sequence transduction.",
            }],
        })
        build.handler({
            "documents": [{
                "paper_id": "p2",
                "text": "ColBERT retrieval uses late interaction over token embeddings.",
            }],
        })

        r1 = _decode_results(search.handler({
            "query": "self attention", "paper_id": "p1", "top_k": 3
        }))
        r2 = _decode_results(search.handler({
            "query": "late interaction", "paper_id": "p2", "top_k": 3
        }))
        assert r1 and all(r["paper_id"] == "p1" for r in r1)
        assert r2 and all(r["paper_id"] == "p2" for r in r2)
    finally:
        c.close()
```

### Step 5.3: Delete test_startup_clears_stale_index

我们显式删除了启动期清理逻辑(spec §"启动期清理:不清,持久化"),该用例不再适用。

- [ ] 整段删除 `test_startup_clears_stale_index` 函数(连同它的 `@pytest.mark.slow` 装饰器)。
- [ ] 同步删文件顶部的 `CURRENT_INDEX = INDEX_ROOT / "paperpilot_current"`(只在该用例使用)。

### Step 5.4: Patch test_search_without_build_fails — 加 paper_id

- [ ] 改 search 调用:

```python
        with pytest.raises(MCPToolError) as ei:
            search.handler({
                "query": "anything",
                "paper_id": "never-built",
                "top_k": 5,
            })
        msg = str(ei.value)
        assert ("IndexNotFoundError" in msg or "no index" in msg.lower()), \
            f"unexpected error message: {msg}"
```

### Step 5.5: Add test_per_paper_isolation_and_disk_hit

- [ ] 在文件末尾追加新用例:

```python
@pytest.mark.slow
def test_per_paper_isolation_and_disk_hit(tmp_path):
    """build paper A → build paper B → search 各自隔离;
    重 build 同 id 走 disk-hit(cached_papers 命中)。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        build = next(t for t in c.list_tools() if t.name == "mcp__colbert__build_index")
        search = next(t for t in c.list_tools() if t.name == "mcp__colbert__search")

        out_a = build.handler({"documents": [
            {"paper_id": "iso-A", "text": "alpha alpha alpha alpha alpha alpha"}
        ]})
        a = json.loads(out_a) if isinstance(out_a, str) else out_a
        assert "iso-A" in a["fresh_papers"]

        out_b = build.handler({"documents": [
            {"paper_id": "iso-B", "text": "beta beta beta beta beta beta"}
        ]})
        b = json.loads(out_b) if isinstance(out_b, str) else out_b
        assert "iso-B" in b["fresh_papers"]

        ra = _decode_results(search.handler({
            "query": "alpha", "paper_id": "iso-A", "top_k": 3
        }))
        rb = _decode_results(search.handler({
            "query": "beta", "paper_id": "iso-B", "top_k": 3
        }))
        assert all(r["paper_id"] == "iso-A" for r in ra)
        assert all(r["paper_id"] == "iso-B" for r in rb)

        # 重 build A: 内存命中(同 server session)
        out_a2 = build.handler({"documents": [
            {"paper_id": "iso-A", "text": "alpha alpha alpha alpha alpha alpha"}
        ]})
        a2 = json.loads(out_a2) if isinstance(out_a2, str) else out_a2
        assert "iso-A" in a2["cached_papers"]
        assert a2["fresh_papers"] == []
    finally:
        c.close()
```

### Step 5.6: Run all slow colbert tests

- [ ] Run: `pytest tests/mcp_servers/test_colbert_via_client.py -v -m slow 2>&1 | tail -20`
Expected: 4 passed, 1 skipped(`test_startup_hard_fail_when_index_root_unwritable` 仍被 skip)。失败先调,不进 commit。

### Step 5.7: Create scripts/day12_smoke.py

- [ ] 新建,基于 day11_smoke 复用结构 + 加并发证据:

```python
"""Day 12 smoke: compare-papers skill + paper_deep_read true parallelism.

Use subagent lifecycle events for concurrency evidence. Do not use
mcp__colbert__search call/result windows for this assertion because the MCP
stdio/server layer may serialize tool calls even when worker loops overlap.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

PAPER_IDS = ["1706.03762", "2010.11929", "2005.14165"]


def main() -> None:
    saw_main: set[str] = set()
    saw_load_skill_args: list[dict] = []
    paper_deep_read_calls: list[dict[str, Any]] = []
    main_guardrails: list[str] = []
    sub_search_calls: list[tuple[str, str]] = []
    subagent_windows: dict[str, list[float | None]] = {}

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        sub_pid = payload.get("subagent_paper_id")
        if kind == "subagent_start" and sub_pid:
            subagent_windows[sub_pid] = [time.time(), None]
            print(f"  [sub:{sub_pid}] == start")
            return
        if kind == "subagent_done" and sub_pid:
            subagent_windows.setdefault(sub_pid, [time.time(), None])[1] = time.time()
            print(f"  [sub:{sub_pid}] == done: {payload.get('status')}")
            return
        if kind == "tool_call":
            name = payload["name"]
            args = payload.get("arguments", {})
            if sub_pid:
                print(f"  [sub:{sub_pid}] -> {name}({_preview(args)})")
                if name == "mcp__colbert__search":
                    sub_search_calls.append((sub_pid, args.get("paper_id", "")))
            else:
                saw_main.add(name)
                if name == "load_skill":
                    saw_load_skill_args.append(args)
                if name == "paper_deep_read":
                    paper_deep_read_calls.append(args)
                print(f"  -> {name}({_preview(args)})")
        elif kind == "tool_result":
            name = payload["name"]
            content = payload.get("content", "")
            content_text = content if isinstance(content, str) else str(content)
            tag = f"[sub:{sub_pid}] " if sub_pid else ""
            print(f"  {tag}<- {name}: {content_text[:160]}...")
        elif kind == "guardrail_stop":
            reason = payload["reason"]
            if sub_pid:
                print(f"  [sub:{sub_pid}] !! guardrail: {reason}")
            else:
                main_guardrails.append(reason)
                print(f"  !! guardrail: {reason}")

    prompt = (
        "I want a comparative deep read of 3 arXiv papers. Please use the "
        "compare-papers skill (load it first, then follow it). The papers are: "
        f"{PAPER_IDS[0]} (Attention Is All You Need), "
        f"{PAPER_IDS[1]} (ViT), and {PAPER_IDS[2]} (GPT-3). "
        "Compare how self-attention is designed or used across them. "
        "Mention at least two paper IDs in the final answer."
    )
    messages = run(prompt, max_iter=12, on_event=tracer)

    final_text = _extract_text(messages[-1].get("content"))
    deep_read_result = _find_tool_result(messages, "paper_deep_read") or ""

    assert any(a.get("name") == "compare-papers" for a in saw_load_skill_args), (
        f"FAIL: did not load compare-papers skill; load_skill args = {saw_load_skill_args}"
    )
    assert "paper_deep_read" in saw_main, "FAIL: paper_deep_read not called"
    assert paper_deep_read_calls and len(paper_deep_read_calls[0]["paper_ids"]) == 3

    ok_count = len(re.findall(r"### \S+ \(status: ok\)", deep_read_result))
    assert ok_count >= 2, f"FAIL: only {ok_count} subagent(s) ok"

    for sub_pid, arg_pid in sub_search_calls:
        assert sub_pid == arg_pid, (
            f"FAIL: subagent {sub_pid} called search with paper_id={arg_pid}"
        )

    assert _has_overlapping_windows(subagent_windows), (
        f"FAIL: no concurrent subagent lifecycle windows; windows = {subagent_windows}"
    )

    pid_mentions = sum(1 for pid in PAPER_IDS if pid in final_text)
    assert pid_mentions >= 2, f"FAIL: final answer mentions only {pid_mentions} ids"
    assert not main_guardrails, f"FAIL: main guardrail = {main_guardrails}"

    print("\nDay 12 smoke PASSED")


def _has_overlapping_windows(windows: dict[str, list[float | None]]) -> bool:
    flat: list[tuple[float, float, str]] = []
    for pid, window in windows.items():
        if len(window) != 2 or window[0] is None or window[1] is None:
            continue
        flat.append((float(window[0]), float(window[1]), pid))
    for i, (s1, e1, p1) in enumerate(flat):
        for s2, e2, p2 in flat[i + 1:]:
            if p1 != p2 and s1 < e2 and s2 < e1:
                return True
    return False

# Reuse _find_tool_result, _extract_text, and _preview helpers from day11_smoke.py.
```

### Step 5.8: Run day12_smoke

- [ ] Run: `python scripts/day12_smoke.py 2>&1 | tail -80`
Expected: 末行 "Day 12 smoke PASSED"

### Step 5.9: Run regression smokes

- [ ] Run: `python scripts/day9_smoke.py 2>&1 | tail -10`
Expected: PASSED(deep-read-paper 单 paper 路径,加了 paper_id 仍工作)

- [ ] Run: `python scripts/day10_smoke.py 2>&1 | tail -10`
Expected: PASSED(research_todo + find-classics 路径不变)

- [ ] Run: `python scripts/day11_smoke.py 2>&1 | tail -10`
Expected: PASSED(paper_deep_read 直接调用路径仍工作,新版应更快)

### Step 5.10: 防回流 grep

- [ ] Run: `git grep -n "Day 11 conservative\|paperpilot_current; call build_index first\|THREAD_POOL_SIZE = 1\|runs workers serially" paperpilot tests scripts`
Expected: no output

- [ ] Run: `git grep -nE "TODO|FIXME" paperpilot/mcp_servers/colbert paperpilot/builtin_tools/subagent.py paperpilot/skills/compare-papers.md scripts/day12_smoke.py`
Expected: no output

### Step 5.11: Commit

- [ ] ```bash
git add tests/mcp_servers/test_colbert_via_client.py scripts/day12_smoke.py
git commit -m "Day 12 Task 5: day12_smoke + colbert per-paper 隔离 slow 集成"
```

---

## DoD

- [ ] fast 套件 100% 绿(`pytest tests/ --ignore=tests/mcp_servers/test_graph_via_client.py` 全过)
- [ ] slow 套件 100% 绿(`pytest -m slow tests/mcp_servers/test_colbert_via_client.py tests/test_main_integration.py` 全过)
- [ ] `day9_smoke` / `day10_smoke` / `day11_smoke` / `day12_smoke` 4 个真 LLM 端到端全部通过
- [ ] day12_smoke 输出含"Day 12 smoke PASSED"且并发证据成立(2+ subagent 生命周期窗口重叠)
- [ ] `git grep "Day 11 conservative\|THREAD_POOL_SIZE = 1\|runs workers serially"` 无命中
- [ ] 触碰文件无 `TODO|FIXME` 残留
- [ ] 5 个 commit 切分干净,message 与本计划 Task 1-5 对应
