# Day 6 Second Half Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Day 6 colbert-mcp 收尾 —— 同步 spec/plan 中 ragatouille → PyLate 残留;实现 `IndexManager.build / search`(含 chunking + chunk_text 映射);拉 mcp_client 超时;跑端到端 smoke。

**Architecture:** 4 task 流水。Task A 纯文档同步,把上一 session 切换 PyLate 后未来得及改的 spec/plan 段落机械替换。Task B 完成 Task 4 真实现 + slow 集成测试(IndexManager 内部 chunking + 内存 chunk_text 字典)。Task C 改一行超时常量。Task D 写端到端 smoke 脚本并真跑。

**Tech Stack:** Python 3.12 (`.venv/`),PyLate 1.4+(ColBERT v2 后端),PyMuPDF,pytest。

**Specs:**
- 主 spec(Day 6 整体):`docs/superpowers/specs/2026-04-25-colbert-mcp-design.md`
- 子 spec(Task 4 内部 chunking + chunk_text 映射):`docs/superpowers/specs/2026-04-27-colbert-task4-chunking-design.md`

**与老 plan 的关系**:本 plan 取代老 plan `docs/superpowers/plans/2026-04-25-colbert-mcp.md` 的 Task 4 / 5 / 6。老 plan Task 1-3 已 committed(`60a369e` / `3b6300e` / `05ce7da`),不动。

**当前测试基线**(本 plan 开工时):`pytest tests/` 12 passed(4 mcp_client + 4 arxiv_download + 4 colbert_server)。本 plan 完工时基线 12 passed (默认) + 3 passed 1 skipped (slow)。

---

## Task A: 同步 spec / plan 中 ragatouille → PyLate 残留(mechanical)

**Files:**
- Modify: `docs/superpowers/specs/2026-04-25-colbert-mcp-design.md`(§2 决策表、§5 build/search 行为段、§7.A startup 表)
- Modify: `docs/superpowers/plans/2026-04-25-colbert-mcp.md`(Task 4 step 4.2/4.3/4.4 代码段)

**目的**:Task B 实现期不会再有 implementer 抄到旧 ragatouille 代码;两份历史文档自身也指向当前真相。

- [ ] **Step A.1: 在 spec §2 隐含决策表里追加两行**

打开 `docs/superpowers/specs/2026-04-25-colbert-mcp-design.md`,定位 §2 末尾的 "隐含决策(已锁)" 表(目前最后一行是 `| download_paper 批量 | 否,一次一篇 | 不变 |`)。在该行**下方**追加两行:

```
| chunk_text 持久化 | `IndexManager._chunk_texts: dict[chunk_id, str]`(进程内,与 PyLate index 同生命周期) | PyLate 索引层只存 embedding+id,不存原文;LLM 引用回答必须看到 chunk 原文(子 spec Q1) |
| Chunking 参数 | 固定 token 滑窗 size=256, overlap=32, 用 `_model.tokenizer` 切 | PyLate 不自动 chunk;ColBERT 多向量鲁棒于切割位置;PyMuPDF 输出的 `\n\n` 不可靠所以不用段落切(子 spec Q2) |
```

- [ ] **Step A.2: 替换 spec §5 `colbert.build_index` 代码块**

在 `docs/superpowers/specs/2026-04-25-colbert-mcp-design.md` 找 §5 中 `### \`colbert.build_index\`(Day 6 新增)` 段。该段当前包含 ragatouille 代码:

```
  2. RAGPretrainedModel.index(
       collection=[d["text"] for d in documents],
       document_ids=[d["paper_id"] for d in documents],
       index_name="paperpilot_current",
       index_root="data/colbert_index/",
       overwrite="force",       # 方案 1 的核心
     )
```

把整个 `行为:` 代码 fence 替换为:

```
行为:
  1. 校验 documents 非空、每个 dict 含 paper_id + text 字段(空 → ValueError)
  2. 对每篇 paper 用 self._model.tokenizer 切 256/overlap=32 token 滑窗
       chunk_id 命名 f"{paper_id}::chunk_{i}"
       text 为空 / 全空白的 paper 跳过(不产生 chunk,不抛错)
  3. self._model.encode(all_chunks, is_query=False) → 多向量 embedding
  4. pylate.indexes.PLAID(index_folder=str(INDEX_ROOT),
                          index_name="paperpilot_current",
                          override=True)
     index.add_documents(documents_ids=all_chunk_ids,
                         documents_embeddings=embs)
  5. 原子替换 self._index = index;self._chunk_texts = dict(zip(ids, texts))
  6. 返回 {"indexed_count": len(documents),  # 按 paper 数,不是 chunk 数
           "index_name": "paperpilot_current"}
```

- [ ] **Step A.3: 替换 spec §5 `colbert.search` 代码块**

同文件 §5 `### \`colbert.search\`(Day 6 新增)` 段。当前 `行为:` 代码 fence:

```
  1. 若 data/colbert_index/colbert/indexes/paperpilot_current/ 不存在
     → raise IndexNotFoundError("must call build_index first")
  2. 复用启动期加载的 RAGPretrainedModel 实例(若 build_index 跑过则已绑定到当前索引)
  3. .search(query=query, k=top_k) → 转换格式返回
```

替换为:

```
  1. 若 self._index is None 或 data/colbert_index/paperpilot_current/ 不存在
     → raise IndexNotFoundError("must call build_index first")
  2. q_emb = self._model.encode([query], is_query=True)
  3. retr = pylate.retrieve.ColBERT(index=self._index)
     scores = retr.retrieve(queries_embeddings=q_emb, k=top_k)
       # shape: list[list[{id, score}]] — 外层 query (len=1),内层 top-k
  4. 每条 r:
       paper_id = r["id"].split("::", 1)[0]
       chunk_text = self._chunk_texts[r["id"]]   # 用 [] 不掩盖 KeyError
       score = float(r["score"])
     返回 list[{paper_id, chunk_text, score}]
```

- [ ] **Step A.4: 替换 spec §7.A 启动 hard-fail 表里 ragatouille 行**

同文件 §7.A 表里目前有这一行:

```
| `RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")` 失败 | HF cache miss + 网络断 |
```

替换为:

```
| `pylate.models.ColBERT(model_name_or_path="lightonai/colbertv2.0")` 失败 | HF cache miss + 网络断 |
```

同表第一行 `import ragatouille / import fitz 失败 | 依赖装漏` 替换为:

```
| `import pylate` / `import fitz` 失败 | 依赖装漏 |
```

- [ ] **Step A.5: 替换老 plan Task 4 step 4.2 实现代码**

打开 `docs/superpowers/plans/2026-04-25-colbert-mcp.md`,定位 `**Step 4.2: 实现 \`IndexManager.build\`**` 段。当前给的替换代码块是 ragatouille `self._model.index(collection=..., overwrite="force")`(line ~622-635)。

整个 "替换为:" 之后那个 ```python ... ``` 代码块,整体换成:

```python
    def build(self, documents: list[dict]) -> dict:
        """对 documents 建立 ColBERT 索引;固定 index_name=paperpilot_current,强制覆盖。

        chunk 在内部完成(256 token 滑窗,overlap 32);chunk_text 留在
        self._chunk_texts 供 search 时回填。LLM 视角是 paper 级,不感知 chunk。
        """
        if not documents:
            raise ValueError("documents must not be empty")

        all_ids: list[str] = []
        all_texts: list[str] = []
        for d in documents:
            for i, ck in enumerate(self._chunk(d["text"])):
                all_ids.append(f"{d['paper_id']}::chunk_{i}")
                all_texts.append(ck)

        embs = self._model.encode(
            all_texts, is_query=False, show_progress_bar=False
        )

        index = indexes.PLAID(
            index_folder=str(INDEX_ROOT),
            index_name=INDEX_NAME,
            override=True,
        )
        index.add_documents(documents_ids=all_ids, documents_embeddings=embs)

        self._index = index
        self._chunk_texts = dict(zip(all_ids, all_texts))

        return {"indexed_count": len(documents), "index_name": INDEX_NAME}

    def _chunk(self, text: str) -> list[str]:
        """固定 token 滑窗;空 text 返空 list(跳过)。"""
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
```

(注意 plan 此 step 现在多说一段 `_chunk` —— 对应子 spec §3.2;旧 plan 没有它。)

- [ ] **Step A.6: 替换老 plan Task 4 step 4.3 实现代码**

同文件 `**Step 4.3: 实现 \`IndexManager.search\`**` 段。当前给的代码是 ragatouille `self._model.search(query=...)` + `r.get("document_id")`(line ~648-666)。

整个 "替换为:" 之后那个 ```python ... ``` 代码块,整体换成:

```python
    def search(self, query: str, top_k: int) -> list[dict]:
        """在 paperpilot_current 索引上查 top_k 段落;chunk_text 从内存映射回填。"""
        if self._index is None or not self._index_path().exists():
            raise IndexNotFoundError(
                "no index at paperpilot_current; call build_index first"
            )

        q_emb = self._model.encode(
            [query], is_query=True, show_progress_bar=False
        )
        retr = retrieve.ColBERT(index=self._index)
        scores = retr.retrieve(queries_embeddings=q_emb, k=top_k)
        # shape: list[list[{id, score}]] — 外层 query (len=1),内层 top-k

        return [
            {
                "paper_id": r["id"].split("::", 1)[0],
                "chunk_text": self._chunk_texts[r["id"]],
                "score": float(r["score"]),
            }
            for r in scores[0]
        ]
```

- [ ] **Step A.7: 替换老 plan Task 4 step 4.4 集成测试中索引路径**

同文件 step 4.4 内 `CURRENT_INDEX = INDEX_ROOT / "colbert" / "indexes" / "paperpilot_current"` 行。该路径是 ragatouille 的嵌套结构;PyLate 直接落到 `INDEX_ROOT/<index_name>/`。改为:

```python
CURRENT_INDEX = INDEX_ROOT / "paperpilot_current"
```

(本 plan 自己的 Task B.4 集成测试用同样的 PyLate 路径,不会回踩。)

- [ ] **Step A.8: Commit**

```bash
git status   # 应只显示 specs/2026-04-25-colbert-mcp-design.md + plans/2026-04-25-colbert-mcp.md 两文件 modified
git add docs/superpowers/specs/2026-04-25-colbert-mcp-design.md docs/superpowers/plans/2026-04-25-colbert-mcp.md
git commit -m "Day 6 同步: spec/plan 残留 ragatouille → PyLate (Task 4 实施前置)"
```

---

## Task B: `IndexManager.build / search` 真实现 + slow 集成测试(TDD)

**Files:**
- Modify: `paperpilot/mcp_servers/colbert/index_manager.py`(实现 `build` / `search` / `_chunk`,改 `__init__` 末尾加 init,改 `_index_path()`,加 import `indexes` / `retrieve`)
- Create: `tests/mcp_servers/test_colbert_via_client.py`(3 个 slow 集成 + 1 个 skipped)
- Create: `pytest.ini`(注册 `slow` mark)

**子 spec 参考**:`docs/superpowers/specs/2026-04-27-colbert-task4-chunking-design.md` §3 全部代码块。

- [ ] **Step B.1: 创建 `pytest.ini` 注册 slow mark**

仓库根目录暂无 `pytest.ini` 或 `pyproject.toml`(已 verified: `ls pytest.ini pyproject.toml` 无输出)。

Create `pytest.ini`:

```ini
[pytest]
markers =
    slow: 慢测试(本地手跑;真起 colbert-mcp + 真跑 ColBERT;~60-180s)
addopts = -m "not slow"
```

`addopts = -m "not slow"` 让默认 `pytest tests/` 跳过 slow,只 `pytest -m slow ...` 时才跑。

- [ ] **Step B.2: 写集成测试 `tests/mcp_servers/test_colbert_via_client.py`(预期 fail,因为 build/search 还是 NotImplementedError)**

Create file with content:

```python
"""colbert-mcp 集成测试(标 slow)。真起 colbert-mcp 进程,真跑 ColBERT。
本地: pytest -m slow tests/mcp_servers/test_colbert_via_client.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperpilot.tools.mcp_client import (
    MCPClient,
    MCPToolError,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_ROOT = REPO_ROOT / "data" / "colbert_index"
CURRENT_INDEX = INDEX_ROOT / "paperpilot_current"


def _make_manifest(tmp_path: Path) -> Path:
    """生成 manifest 仅含 colbert 一个 server,便于测试隔离。"""
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({
        "mcpServers": {
            "colbert": {
                "command": "python",
                "args": ["-m", "paperpilot.mcp_servers.colbert.server"],
            }
        }
    }))
    return m


@pytest.mark.slow
def test_build_and_search(tmp_path):
    """3 篇短 dummy text → build → search → 命中关键词且 chunk_text 非空。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        build = next(t for t in c.list_tools() if t.name == "mcp__colbert__build_index")
        search = next(t for t in c.list_tools() if t.name == "mcp__colbert__search")

        docs = [
            {"paper_id": "p1",
             "text": "Attention is all you need. The Transformer uses multi-head self-attention to model long-range dependencies in language."},
            {"paper_id": "p2",
             "text": "BERT pre-training uses masked language modeling on bidirectional transformers to learn contextual representations."},
            {"paper_id": "p3",
             "text": "ColBERT performs late interaction between query and document token embeddings for efficient passage retrieval at scale."},
        ]
        build_result = build.handler({"documents": docs})
        # FastMCP serialize dict 为 JSON 字符串
        b = json.loads(build_result) if isinstance(build_result, str) else build_result
        assert b["indexed_count"] == 3
        assert b["index_name"] == "paperpilot_current"

        search_result = search.handler({"query": "late interaction retrieval", "top_k": 3})
        results = json.loads(search_result) if isinstance(search_result, str) else search_result
        assert len(results) >= 1
        assert any(r["paper_id"] == "p3" for r in results), \
            f"expected p3 (ColBERT) in top-3, got {results}"
        # 子 spec §5 测试调整:chunk_text 必须真有内容,不能空
        top1 = results[0]
        assert top1["chunk_text"] != "", "chunk_text 不应为空(回归 _chunk_texts 映射链路)"
        assert len(top1["chunk_text"]) > 50, \
            f"chunk_text 太短不像真段落: {top1['chunk_text']!r}"
        assert isinstance(top1["score"], float)
    finally:
        c.close()


@pytest.mark.slow
def test_startup_clears_stale_index(tmp_path):
    """Q6: 启动期 rm -rf paperpilot_current/ 把上 session 残留干净清掉。"""
    CURRENT_INDEX.mkdir(parents=True, exist_ok=True)
    (CURRENT_INDEX / "stale_marker.txt").write_text("from previous session")
    assert (CURRENT_INDEX / "stale_marker.txt").exists()

    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        # IndexManager.__init__ 应已 rmtree paperpilot_current/
        assert not CURRENT_INDEX.exists() or not (CURRENT_INDEX / "stale_marker.txt").exists(), \
            "Q6 violation: stale marker survived startup"
    finally:
        c.close()


@pytest.mark.slow
def test_search_without_build_fails(tmp_path):
    """启动后没 build 直接 search → IndexNotFoundError 透传成 MCPToolError。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        search = next(t for t in c.list_tools() if t.name == "mcp__colbert__search")
        with pytest.raises(MCPToolError) as ei:
            search.handler({"query": "anything", "top_k": 5})
        msg = str(ei.value)
        assert ("IndexNotFoundError" in msg or "no index" in msg.lower()
                or "must call build_index" in msg.lower()), \
            f"unexpected error message: {msg}"
    finally:
        c.close()


@pytest.mark.slow
def test_startup_hard_fail_when_index_root_unwritable(tmp_path):
    """跨平台模拟"只读 INDEX_ROOT"在 Windows + Linux 上策略不一,
    且加 env 覆盖入口违反 YAGNI。该启动 hard-fail 路径在 Day 6 smoke 端到端兜底。"""
    pytest.skip(
        "INDEX_ROOT 当前为常量(子 spec §3 锁);跨平台只读模拟不稳。"
        "启动 hard-fail 路径在 day6_smoke 真实启动中验证。"
    )
```

- [ ] **Step B.3: 跑集成测试,预期 fail(NotImplementedError)**

Run: `pytest tests/mcp_servers/test_colbert_via_client.py -v -m slow`
Expected: 至少 `test_build_and_search` fail / error,理由 `NotImplementedError: Task 4 实现 - 使用 self._model.encode + indexes.PLAID`(从 `IndexManager.build` 抛出);`test_search_without_build_fails` 可能已 PASS(`_index is None` 检查在 NotImplementedError 之前)。`test_startup_clears_stale_index` 应 PASS(Task 3 已落地的清理逻辑)。`test_startup_hard_fail_*` SKIPPED。

如果 `test_search_without_build_fails` 也 fail —— 可能是 `_index_path()` 还指向 ragatouille 嵌套路径(Task 3 留的 `INDEX_ROOT / "colbert" / "indexes" / INDEX_NAME`)。Step B.4 会顺手修。

- [ ] **Step B.4: 改 `paperpilot/mcp_servers/colbert/index_manager.py`(实现 `_chunk` / `build` / `search` + `__init__` 追加 + `_index_path` 改 PyLate 布局 + 顶层 import)**

打开 `paperpilot/mcp_servers/colbert/index_manager.py`。当前文件结构(参 Task 3 commit `05ce7da`):
- 顶层 import `pyarrow`、`datasets`、`from pylate import models`
- 模块常量 `INDEX_NAME / INDEX_ROOT / MODEL_NAME`
- `IndexNotFoundError`
- `IndexManager.__init__ / build (raise NotImplementedError) / search (raise NotImplementedError) / _clear_stale_index / _index_path`

四处改动:

**(a) 顶层 import 加 `indexes` 和 `retrieve`**

把:

```python
from pylate import models
```

改为:

```python
from pylate import indexes, models, retrieve
```

**(b) `_index_path` 路径改 PyLate 布局(去掉 `colbert/indexes/` 嵌套)**

把:

```python
    def _index_path(self) -> Path:
        return INDEX_ROOT / "colbert" / "indexes" / INDEX_NAME
```

改为:

```python
    def _index_path(self) -> Path:
        return INDEX_ROOT / INDEX_NAME
```

**(c) `IndexManager` 内顶部加类常量,`__init__` 末尾追加 `_index` / `_chunk_texts` 初始化**

把:

```python
class IndexManager:
    def __init__(self) -> None:
        self._clear_stale_index()
        INDEX_ROOT.mkdir(parents=True, exist_ok=True)
        self._model = models.ColBERT(model_name_or_path=MODEL_NAME)
```

改为:

```python
class IndexManager:
    CHUNK_SIZE = 256
    OVERLAP = 32

    def __init__(self) -> None:
        self._clear_stale_index()
        INDEX_ROOT.mkdir(parents=True, exist_ok=True)
        self._model = models.ColBERT(model_name_or_path=MODEL_NAME)
        self._index: indexes.PLAID | None = None
        self._chunk_texts: dict[str, str] = {}
```

**(d) 实现 `build` + `_chunk`,替换 NotImplementedError**

把:

```python
    def build(self, documents: list[dict]) -> dict:
        raise NotImplementedError("Task 4 实现 - 使用 self._model.encode + indexes.PLAID")
```

替换为(注意先 build,再 _chunk —— 让 build 与 search 在文件中靠近,_chunk 是 build 的辅助放后面):

```python
    def build(self, documents: list[dict]) -> dict:
        """对 documents 建立 ColBERT 索引;固定 index_name=paperpilot_current,强制覆盖。

        chunk 在内部完成(256 token 滑窗,overlap 32);chunk_text 留在
        self._chunk_texts 供 search 时回填。LLM 视角是 paper 级,不感知 chunk。
        """
        if not documents:
            raise ValueError("documents must not be empty")

        all_ids: list[str] = []
        all_texts: list[str] = []
        for d in documents:
            for i, ck in enumerate(self._chunk(d["text"])):
                all_ids.append(f"{d['paper_id']}::chunk_{i}")
                all_texts.append(ck)

        embs = self._model.encode(
            all_texts, is_query=False, show_progress_bar=False
        )

        index = indexes.PLAID(
            index_folder=str(INDEX_ROOT),
            index_name=INDEX_NAME,
            override=True,
        )
        index.add_documents(documents_ids=all_ids, documents_embeddings=embs)

        self._index = index
        self._chunk_texts = dict(zip(all_ids, all_texts))

        return {"indexed_count": len(documents), "index_name": INDEX_NAME}
```

**(e) 实现 `search`,替换 NotImplementedError**

把:

```python
    def search(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError("Task 4 实现 - 使用 self._model.encode + retrieve.ColBERT")
```

替换为:

```python
    def search(self, query: str, top_k: int) -> list[dict]:
        """在 paperpilot_current 索引上查 top_k 段落;chunk_text 从内存映射回填。"""
        if self._index is None or not self._index_path().exists():
            raise IndexNotFoundError(
                "no index at paperpilot_current; call build_index first"
            )

        q_emb = self._model.encode(
            [query], is_query=True, show_progress_bar=False
        )
        retr = retrieve.ColBERT(index=self._index)
        scores = retr.retrieve(queries_embeddings=q_emb, k=top_k)
        # shape: list[list[{id, score}]] — 外层 query (len=1),内层 top-k

        return [
            {
                "paper_id": r["id"].split("::", 1)[0],
                "chunk_text": self._chunk_texts[r["id"]],
                "score": float(r["score"]),
            }
            for r in scores[0]
        ]
```

**(f) 在 `_clear_stale_index` **之前**(即 `search` 之后)插入 `_chunk` 私有方法**

```python

    def _chunk(self, text: str) -> list[str]:
        """固定 token 滑窗。空 text 返空 list(该 paper 不产生 chunk)。"""
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
```

- [ ] **Step B.5: 跑集成测试,预期 3 passed 1 skipped**

Run: `pytest tests/mcp_servers/test_colbert_via_client.py -v -m slow`
Expected: `3 passed, 1 skipped`,总耗时 ~60-180s(主要 build 一次 + ColBERT 模型加载;若 Task 1 step 1.6 已预热模型,启动 ~5-10s)。

排错:
- `test_build_and_search` fail "p3 not in top-3" → 子 spec 假设 ColBERT 真把 "late interaction retrieval" 查到 p3 排第一;若实际 p1/p2 也可能上榜,放宽断言到 "p3 在 top-3 内"(已是当前断言,如还 fail 说明 ColBERT 真排错,打印 results 看分布);
- `test_build_and_search` fail "chunk_text 太短" → 检查 `_chunk` 是否真切了 256 token(短 dummy text 只产生 1 个 chunk,该 chunk 接近原文整段,长度应 >50 char);如 chunk_text 是空串,说明 `self._chunk_texts` 在 search 用 `[]` 被 KeyError 但被 FastMCP 包成 isError —— 检查 chunk_id 命名一致性(build 与 search 的 split 必须对得上);
- `test_search_without_build_fails` fail → 检查 `_index_path()` 是否已改成 `INDEX_ROOT / INDEX_NAME`(Step B.4.b)。

- [ ] **Step B.6: 跑默认 pytest(slow 跳)确认无回归**

Run: `pytest tests/ -v`
Expected: `12 passed, 4 deselected`(Day 5 4 个 mcp_client + Task 2 4 个 arxiv_download + Task 3 4 个 colbert_server;新加的 4 个 slow 都被 deselect)。

- [ ] **Step B.7: Commit**

```bash
git status
git add paperpilot/mcp_servers/colbert/index_manager.py \
        tests/mcp_servers/test_colbert_via_client.py \
        pytest.ini
git commit -m "Day 6 Task 4: IndexManager.build/search 真实现 + chunking + 集成测试 (slow)"
```

---

## Task C: `mcp_client` tool 超时 60 → 180s + Day 5 smoke utf-8 修复

**Files:**
- Modify: `paperpilot/tools/mcp_client.py`(line 6 docstring 注释 + line 24 默认值)
- Modify: `scripts/day5_smoke.py`(顶部加 utf-8 reconfigure)

**理由**:子 spec 主 §7.C —— ColBERT `build_index` 真实耗时 30-90s,撞 Day 5 锁的 60s 上限。env 覆盖行为不动(用户拉短可覆盖)。

- [ ] **Step C.1: 改 `paperpilot/tools/mcp_client.py` line 24 默认值 60 → 180**

打开 `paperpilot/tools/mcp_client.py`。当前 line 24:

```python
MCP_TOOL_TIMEOUT = int(os.environ.get("MCP_TOOL_TIMEOUT", 60))
```

改为:

```python
MCP_TOOL_TIMEOUT = int(os.environ.get("MCP_TOOL_TIMEOUT", 180))
```

(只改 60 → 180;`os.environ.get` 行为不动。)

同文件 line 6 docstring 内目前有 `- 错误:启动 hard-fail / 运行 soft-fail / 60s 超时为边界`,把 `60s` 改 `180s` 保持文档同步:

```python
- 错误:启动 hard-fail / 运行 soft-fail / 180s 超时为边界
```

- [ ] **Step C.2: 改 `scripts/day5_smoke.py` 顶部加 utf-8 reconfigure**

打开 `scripts/day5_smoke.py`。在文件**开头**(模块 docstring 之后,任何其他 import 之前)插入:

```python
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
```

如果文件已有 `import sys`,把上面的 `import sys` 行去掉,保留 `if hasattr(...)` 这两行,放在 docstring 之后、其他 import 之前。

(如 `scripts/day5_smoke.py` 已经做过 utf-8 reconfigure,跳过此 step,在下一 step commit message 里说明 "day5 smoke 已有 utf-8 reconfigure,无改动"。)

- [ ] **Step C.3: 跑默认 pytest 确认无回归**

Run: `pytest tests/ -v`
Expected: `12 passed`(超时常量改不影响测试,echo_server / arxiv 单测 mock 行为不变)。

- [ ] **Step C.4: Commit**

```bash
git status
git add paperpilot/tools/mcp_client.py scripts/day5_smoke.py
git commit -m "Day 6 Task 5: mcp_client tool 超时 60→180s + day5 smoke utf-8 reconfigure"
```

---

## Task D: `scripts/day6_smoke.py` 端到端冒烟

**Files:**
- Create: `scripts/day6_smoke.py`

**前置**:`.env` 含有效 LLM API key(项目当前 LLM provider 决定);联网;ColBERT 模型已 cache(Task 1 step 1.6 已预热)。

- [ ] **Step D.1: 写 `scripts/day6_smoke.py`**

Create file with content:

```python
"""Day 6 冒烟: Main Loop ←→ MCPClient ←→ stdio ←→ {arxiv-mcp, colbert-mcp} ←→ LLM
全链路验证。

跑一次 ~$0.02-0.05(LLM 多轮 tool_use); 需联网 + 有效 LLM API key。
首次跑会触发 ColBERT 模型加载(预热过的话 ~10s)。
用法: python scripts/day6_smoke.py
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

EXPECT_TOOLS = {
    "mcp__arxiv__search_papers",
    "mcp__arxiv__download_paper",
    "mcp__colbert__build_index",
    "mcp__colbert__search",
}


def main() -> None:
    saw: set[str] = set()

    def tracer(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            saw.add(payload["name"])
            print(f"  → {payload['name']}({payload.get('arguments', {})})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            preview = content[:160] if isinstance(content, str) else str(content)[:160]
            print(f"  ← {payload['name']}: {preview}...")
        elif kind == "guardrail_stop":
            print(f"  ⚠ guardrail: {payload['reason']}")

    messages = run(
        "搜一篇 attention 相关的 arxiv 论文(最近一两年的就行),下载它的全文,"
        "然后在全文里查 multi-head attention 是怎么定义的,用一段话回答我。"
        "回答必须基于 colbert.search 返回的具体段落,不要只看 abstract。",
        max_iter=10,
        on_event=tracer,
    )

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for b in last:
            if hasattr(b, "text"):
                print(b.text)
    else:
        print(last)

    missing = EXPECT_TOOLS - saw
    assert not missing, f"FAIL: 期望调用的 tool 缺失 {missing};实际只见 {saw}"
    print("\n✅ Day 6 smoke PASSED")


if __name__ == "__main__":
    main()
```

注意 `paperpilot.main.run` 签名(Day 5 commit `ac43dcf`):若实际签名与本脚本不符(参数名不是 `max_iter` 或 `on_event` 等),开 `paperpilot/main.py` 看真签名再调整。

- [ ] **Step D.2: 端到端跑 smoke**

确认 `.env` 有效。

Run: `python scripts/day6_smoke.py`

预期事件序列(stdout 实时打印):
- `→ mcp__arxiv__search_papers(...)` 1 次
- `→ mcp__arxiv__download_paper(...)` 至少 1 次
- `→ mcp__colbert__build_index(...)` 1 次,**该步同步阻塞 30-90s**(ColBERT 真索引)
- `→ mcp__colbert__search(...)` 1+ 次
- 末尾打印 `=== FINAL ===` + LLM 答案(含 multi-head attention 定义)
- 末尾 `✅ Day 6 smoke PASSED`
- 退出码 0

排错:
- `MCPToolTimeout` on build_index → 单篇 paper 太长。临时:`MCP_TOOL_TIMEOUT=300 python scripts/day6_smoke.py`,记到 daily log,后续考虑下次 spec 微调
- `tool 缺失 {'mcp__colbert__build_index'}` → LLM 看 abstract 觉得够答了。prompt 里已加"不要只看 abstract" 强制要求,若仍跳:把 prompt 进一步收紧,例如指定具体术语让它必须查
- 启动卡 60+ 秒静默 → 模型未预热(回头跑老 plan Task 1 step 1.6)

- [ ] **Step D.3: 验证完工标志**

```bash
# 1) 默认 pytest 全绿
pytest tests/ -v

# 2) slow 集成测试 3 passed 1 skipped
pytest tests/mcp_servers/test_colbert_via_client.py -v -m slow

# 3) Day 5 smoke 不回归
python scripts/day5_smoke.py

# 4) 索引目录存在且 ≥10MB(单篇 paper 索引可能比 5 篇估算小)
ls -la data/colbert_index/paperpilot_current/

# 5) data/papers/ 至少 1 个 .txt
ls -la data/papers/
```

Expected:
- `12 passed` 默认
- `3 passed, 1 skipped` slow
- Day 5 smoke `✅ Day 5 smoke PASSED`
- 索引目录非空
- `data/papers/` 至少 1 个 `.txt`(几十 KB ~ 几百 KB)

- [ ] **Step D.4: 检查 `.gitignore` 是否忽略了 `data/`**

```bash
git status
```

如 `data/papers/` 或 `data/colbert_index/` 出现在 untracked,加进 `.gitignore`:

```
data/papers/
data/colbert_index/
```

并 `git add .gitignore && git commit -m "Day 6: 忽略 data/papers, data/colbert_index"`(放在 Step D.5 之前或合并)。

- [ ] **Step D.5: Final commit**

```bash
git status
git add scripts/day6_smoke.py
# 若 Step D.4 改了 .gitignore:
# git add .gitignore
git commit -m "Day 6 Task 6: day6_smoke 端到端冒烟 (arxiv search→download→colbert build→search→LLM 答案)"
```

- [ ] **Step D.6: Verify final clean state**

```bash
git log --oneline -10
git status
```

Expected:
- 顶 4 个 commits 是本 plan 的 Task A→D(commit message 形如 `Day 6 同步...` / `Day 6 Task 4: ...` / `Day 6 Task 5: ...` / `Day 6 Task 6: ...`)
- 之前是 Task 1-3 的 3 个 + 老 spec/plan 与 ragatouille 切换的 commit
- working tree clean (除 ignored 的 data/、__pycache__/)

---

## 完工标志(Definition of Done)

1. ✅ `pytest tests/` → `12 passed`(slow 自动 deselect)
2. ✅ `pytest -m slow tests/mcp_servers/test_colbert_via_client.py` → `3 passed, 1 skipped`
3. ✅ `python scripts/day5_smoke.py` 无回归 + 不再 emoji UnicodeError
4. ✅ `python scripts/day6_smoke.py` 退出 0 + 打印 `✅ Day 6 smoke PASSED`
5. ✅ `data/colbert_index/paperpilot_current/` 存在且非空
6. ✅ `data/papers/` 至少 1 个 `.txt`
7. ✅ git log Day 6 第二 session 含 4 个原子 commit (Task A / B / C / D)

---

## 自审要点(implementer 在每 Task 末尾自检)

1. **commit 粒度**:每 Task 一个 commit;开始下个 Task 前先 `git status` 确认 working tree clean
2. **不留 NotImplementedError**:Task B 完成后 `grep -rn "NotImplementedError" paperpilot/mcp_servers/colbert/` 应 0 处
3. **不引入推测性抽象**:不写 base class、不抽 `_chunk` 公共模块、不加 plugin hook
4. **不引入重试 / 退避 / 降级**:download_paper / build_index / search 失败原样向上抛;LLM 决策
5. **commit message 不出现 AI 署名**(用户 feedback rule)

---

## 与 Spec 的覆盖核对

| Spec 段 | 覆盖在 |
|---|---|
| 主 spec §2 隐含决策表(chunk_text + chunking 两行新加) | Task A.1 |
| 主 spec §5 build_index 行为 | Task A.2(spec 同步)+ Task B.4 (d) (代码实现) |
| 主 spec §5 search 行为 | Task A.3 + Task B.4 (e) |
| 主 spec §7.A 启动 hard-fail 表 | Task A.4(spec 同步);实际 hard-fail 路径在 Task B 测试 + Task D smoke 兜底验证 |
| 主 spec §7.C 超时调整 | Task C.1 |
| 主 spec §8 测试层 2(slow) | Task B.2 / B.5 |
| 主 spec §8 测试层 3(smoke) | Task D.1 / D.2 |
| 子 spec §3.1 内部状态 | Task B.4 (c) |
| 子 spec §3.2 `_chunk` | Task B.4 (f) |
| 子 spec §3.3 `build` | Task B.4 (d) |
| 子 spec §3.4 `__init__` 追加 | Task B.4 (c) |
| 子 spec §3.5 `search` | Task B.4 (e) |
| 子 spec §4 同步清单 | Task A.1-A.7 |
| 子 spec §5 测试调整(chunk_text 非空 + len > 50) | Task B.2(`test_build_and_search` 末两断言) |
