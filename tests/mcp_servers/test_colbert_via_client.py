"""colbert-mcp 集成测试(标 slow)。真起 colbert-mcp 进程,真跑 ColBERT。
本地: pytest -m slow tests/mcp_servers/test_colbert_via_client.py -v
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from paperpilot.tools.mcp_client import (
    MCPClient,
    MCPToolError,
)


def _make_manifest(tmp_path: Path) -> Path:
    """生成 manifest 仅含 colbert 一个 server,便于测试隔离。"""
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({
        "mcpServers": {
            "colbert": {
                "command": "python",
                "args": ["-m", "paperpilot.mcp_servers.colbert.server"],
                "env": {
                    "HF_HUB_OFFLINE": "1",
                },
            }
        }
    }))
    return m


def _decode_results(search_result):
    if not isinstance(search_result, str):
        return search_result
    dec = json.JSONDecoder()
    results = []
    s = search_result.strip()
    idx = 0
    while idx < len(s):
        obj, end = dec.raw_decode(s, idx)
        results.append(obj)
        idx = end + len(s[end:]) - len(s[end:].lstrip())
    return results


def _paper_prefix(tmp_path: Path) -> str:
    return f"{tmp_path.name}-{uuid.uuid4().hex[:8]}"


@pytest.mark.slow
def test_build_and_search(tmp_path):
    """3 篇短 dummy text → build → search → 命中关键词且 chunk_text 非空。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        build = next(t for t in c.list_tools() if t.name == "mcp__colbert__build_index")
        search = next(t for t in c.list_tools() if t.name == "mcp__colbert__search")
        prefix = _paper_prefix(tmp_path)

        docs = [
            {"paper_id": f"{prefix}-p1",
             "text": "Attention is all you need. The Transformer uses multi-head self-attention to model long-range dependencies in language."},
            {"paper_id": f"{prefix}-p2",
             "text": "BERT pre-training uses masked language modeling on bidirectional transformers to learn contextual representations."},
            {"paper_id": f"{prefix}-p3",
             "text": "ColBERT performs late interaction between query and document token embeddings for efficient passage retrieval at scale."},
        ]
        build_result = build.handler({"documents": docs})
        b = json.loads(build_result) if isinstance(build_result, str) else build_result
        assert b["indexed_count"] == 3
        assert b["index_name"] == "paperpilot_current"
        assert set(b["fresh_papers"]) == {f"{prefix}-p1", f"{prefix}-p2", f"{prefix}-p3"}

        search_result = search.handler({
            "query": "late interaction retrieval",
            "paper_id": f"{prefix}-p3",
            "top_k": 3,
        })
        results = _decode_results(search_result)
        assert len(results) >= 1
        assert all(r["paper_id"] == f"{prefix}-p3" for r in results), \
            f"per-paper isolation failed for search(paper_id={prefix}-p3): {results}"
        top1 = results[0]
        assert top1["chunk_text"] != "", "chunk_text 不应为空(回归 _chunk_texts 映射链路)"
        assert len(top1["chunk_text"]) > 30, \
            f"chunk_text 太短不像真段落: {top1['chunk_text']!r}"
        assert isinstance(top1["score"], float)
    finally:
        c.close()


@pytest.mark.slow
def test_planned_retrieval_tool_runs(tmp_path):
    """build_index 后 planned_retrieval 能返回 evidence_pool 结构。"""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        tools = c.list_tools()
        build = next(t for t in tools if t.name == "mcp__colbert__build_index")
        planned = next(
            t for t in tools if t.name == "mcp__colbert__planned_retrieval"
        )
        prefix = _paper_prefix(tmp_path)
        paper_id = f"{prefix}-planned"

        build_result = build.handler({
            "documents": [{
                "paper_id": paper_id,
                "text": (
                    "The experiments use WikiHop as the main dataset. "
                    "The baseline comparison includes BiDAF and a graph model. "
                    "Results are reported in the evaluation section. "
                ) * 4,
            }]
        })
        b = json.loads(build_result) if isinstance(build_result, str) else build_result
        assert paper_id in b["fresh_papers"]

        raw = planned.handler({
            "question": "What dataset was used?",
            "paper_id": paper_id,
            "paper_title": "A WikiHop Paper",
            "abstract": "The paper evaluates on WikiHop.",
            "top_k_each": 2,
            "summary_k": 3,
        })
        payload = json.loads(raw) if isinstance(raw, str) else raw
        assert "fallback_used" in payload["query_plan_meta"]
        assert payload["evidence_pool"]["stats"]["query_count"] >= 1
        assert payload["evidence_pool"]["stats"]["raw_result_count"] >= 1
        assert payload["evidence_pool"]["summary_items"]
        assert "Planned retrieval completed." in payload["summary_text"]
    finally:
        c.close()


@pytest.mark.slow
def test_two_papers_coexist_after_consecutive_builds(tmp_path):
    """per-paper indexes: consecutive p1/p2 builds remain separately searchable."""
    c = MCPClient(_make_manifest(tmp_path))
    c.start()
    try:
        build = next(t for t in c.list_tools() if t.name == "mcp__colbert__build_index")
        search = next(t for t in c.list_tools() if t.name == "mcp__colbert__search")
        prefix = _paper_prefix(tmp_path)
        p1 = f"{prefix}-p1"
        p2 = f"{prefix}-p2"

        build.handler({
            "documents": [{
                "paper_id": p1,
                "text": "Transformers use self attention for sequence transduction.",
            }],
        })
        build.handler({
            "documents": [{
                "paper_id": p2,
                "text": "ColBERT retrieval uses late interaction over token embeddings.",
            }],
        })

        r1 = _decode_results(search.handler({
            "query": "self attention",
            "paper_id": p1,
            "top_k": 3,
        }))
        r2 = _decode_results(search.handler({
            "query": "late interaction",
            "paper_id": p2,
            "top_k": 3,
        }))
        assert r1 and all(result["paper_id"] == p1 for result in r1)
        assert r2 and all(result["paper_id"] == p2 for result in r2)
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
            search.handler({
                "query": "anything",
                "paper_id": "never-built",
                "top_k": 5,
            })
        msg = str(ei.value)
        assert ("IndexNotFoundError" in msg or "no index" in msg.lower()), \
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


@pytest.mark.slow
def test_per_paper_isolation_and_cache_hits(tmp_path):
    """Search stays isolated; rebuilds hit memory and then disk across sessions."""
    manifest = _make_manifest(tmp_path)
    c = MCPClient(manifest)
    prefix = _paper_prefix(tmp_path)
    paper_a = f"{prefix}-iso-A"
    paper_b = f"{prefix}-iso-B"
    alpha_text = (
        "Alpha retrieval systems rank candidate passages by matching query terms "
        "with contextual document evidence. The method discusses token-level "
        "signals, passage scoring, and repeated alpha examples for a retrieval "
        "pipeline that remains distinct from unrelated beta evidence. "
    ) * 4
    beta_text = (
        "Beta retrieval systems evaluate document passages with a different "
        "semantic signal. The method emphasizes beta scoring, contrastive "
        "examples, and passage evidence that should remain isolated from the "
        "alpha retrieval paper during search. "
    ) * 4
    c.start()
    try:
        build = next(t for t in c.list_tools() if t.name == "mcp__colbert__build_index")
        search = next(t for t in c.list_tools() if t.name == "mcp__colbert__search")

        out_a = build.handler({"documents": [
            {"paper_id": paper_a, "text": alpha_text}
        ]})
        a = json.loads(out_a) if isinstance(out_a, str) else out_a
        assert paper_a in a["fresh_papers"]

        out_b = build.handler({"documents": [
            {"paper_id": paper_b, "text": beta_text}
        ]})
        b = json.loads(out_b) if isinstance(out_b, str) else out_b
        assert paper_b in b["fresh_papers"]

        ra = _decode_results(search.handler({
            "query": "alpha",
            "paper_id": paper_a,
            "top_k": 3,
        }))
        rb = _decode_results(search.handler({
            "query": "beta",
            "paper_id": paper_b,
            "top_k": 3,
        }))
        assert all(result["paper_id"] == paper_a for result in ra)
        assert all(result["paper_id"] == paper_b for result in rb)

        out_a2 = build.handler({"documents": [
            {"paper_id": paper_a, "text": alpha_text}
        ]})
        a2 = json.loads(out_a2) if isinstance(out_a2, str) else out_a2
        assert paper_a in a2["cached_papers"]
        assert a2["fresh_papers"] == []
    finally:
        c.close()

    c2 = MCPClient(manifest)
    c2.start()
    try:
        build2 = next(t for t in c2.list_tools() if t.name == "mcp__colbert__build_index")
        out_a3 = build2.handler({"documents": [
            {"paper_id": paper_a, "text": alpha_text}
        ]})
        a3 = json.loads(out_a3) if isinstance(out_a3, str) else out_a3
        assert paper_a in a3["cached_papers"]
        assert a3["fresh_papers"] == []
    finally:
        c2.close()
