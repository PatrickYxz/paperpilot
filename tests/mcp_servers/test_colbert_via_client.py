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
        b = json.loads(build_result) if isinstance(build_result, str) else build_result
        assert b["indexed_count"] == 3
        assert b["index_name"] == "paperpilot_current"

        search_result = search.handler({"query": "late interaction retrieval", "top_k": 3})
        # FastMCP serializes list[dict] as N text blocks joined by "\n",
        # giving concatenated JSON objects rather than a JSON array.
        if isinstance(search_result, str):
            dec = json.JSONDecoder()
            results = []
            s = search_result.strip()
            idx = 0
            while idx < len(s):
                obj, end = dec.raw_decode(s, idx)
                results.append(obj)
                idx = end + len(s[end:]) - len(s[end:].lstrip())
        else:
            results = search_result
        assert len(results) >= 1
        assert any(r["paper_id"] == "p3" for r in results), \
            f"expected p3 (ColBERT) in top-3, got {results}"
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
