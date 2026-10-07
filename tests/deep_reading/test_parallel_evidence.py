"""Parallel evidence worker tests (fake MCP + fake model)."""
from __future__ import annotations

from concurrent import futures
from unittest.mock import MagicMock, patch

from paperpilot.deep_reading.parallel_evidence import (
    MAX_PARALLEL_WORKERS,
    build_parallel_evidence_tool,
)
from paperpilot.papers import PaperCandidate


def _candidate(eid):
    return PaperCandidate(
        external_id=eid, title=f"Paper {eid}", authors=["A"],
        abstract="abs", source_url=f"https://arxiv.org/abs/{eid}",
    )


def _evidence_item(item_id, text, paper_id):
    class _I:
        pass
    item = _I()
    item.id = item_id
    item.chunk_text = text
    item.paper_external_id = paper_id
    return item


def _build(prepared, evidence, pool_items, note="Relevant evidence found."):
    context = MagicMock()
    context.model = _fake_model(note)

    def call_mcp_json(*, name, arguments, stage):
        pid = arguments["paper_id"]
        return {
            "evidence_pool": {
                "items": pool_items.get(pid, []),
                "summary_text": "pool",
            },
            "summary_text": "pool",
        }

    def decode(payload, candidate, *, question, top_k_each, summary_k):
        pid = candidate.external_id
        items = [
            _evidence_item(it["id"], it["chunk_text"], pid)
            for it in pool_items.get(pid, [])
        ]
        return items, [i.id for i in items]

    tool = build_parallel_evidence_tool(
        context,
        prepared=prepared,
        evidence=evidence,
        call_mcp_json=call_mcp_json,
        decode_evidence_pool=decode,
        contract_error=ValueError,
        emit_tool_call=lambda *a, **k: None,
        clip=lambda v: v,
        required_id=lambda v, f: str(v),
        require_agent_limit=lambda v, f: None,
    )
    return tool


class _fake_model:
    def __init__(self, note):
        self._note = note

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        from paperpilot.deep_reading.parallel_evidence import _PaperNote

        return _PaperNote(
            paper_id="x", note=self._note, selected_evidence_ids=["ev-1"]
        )


def test_two_papers_fan_out_and_merge_ledger():
    prepared = {"2401.00001v1": _candidate("2401.00001v1"),
                "2401.00002v1": _candidate("2401.00002v1")}
    evidence = {}
    pool = {
        "2401.00001v1": [{"id": "ev-1", "chunk_text": "alpha evidence"}],
        "2401.00002v1": [{"id": "ev-2", "chunk_text": "beta evidence"}],
    }
    tool = _build(prepared, evidence, pool)
    import json

    out = json.loads(tool.invoke({"question": "compare approaches"}))
    papers = {p["paper_id"]: p for p in out["papers"]}
    assert set(papers) == set(prepared)
    assert "Relevant evidence" in papers["2401.00001v1"]["note"]
    # evidence merged into the shared ledger under global ids
    assert "ev-1" in evidence and "ev-2" in evidence
    # selected preview exposes id + text preview but ledger holds the full text
    assert papers["2401.00001v1"]["selected_evidence"][0]["id"] == "ev-1"


def test_worker_failure_isolated_per_paper():
    prepared = {"2401.00001v1": _candidate("2401.00001v1"),
                "2401.00002v1": _candidate("2401.00002v1")}
    evidence = {}

    def call_mcp_json(*, name, arguments, stage):
        if arguments["paper_id"] == "2401.00001v1":
            raise RuntimeError("mcp down")
        return {"evidence_pool": {"items": [{"id": "ev-2", "chunk_text": "beta"}]}}

    context = MagicMock()
    context.model = _fake_model("note")
    tool = build_parallel_evidence_tool(
        context, prepared=prepared, evidence=evidence,
        call_mcp_json=call_mcp_json,
        decode_evidence_pool=lambda p, c, **k: (
            [_evidence_item(i["id"], i["chunk_text"], c.external_id)
             for i in (p.get("evidence_pool", {}).get("items", []))], []),
        contract_error=ValueError,
        emit_tool_call=lambda *a, **k: None,
        clip=lambda v: v,
        required_id=lambda v, f: str(v),
        require_agent_limit=lambda v, f: None,
    )
    import json

    out = json.loads(tool.invoke({"question": "q"}))
    papers = {p["paper_id"]: p for p in out["papers"]}
    assert "worker failed" in papers["2401.00001v1"]["error"]
    assert "note" in papers["2401.00002v1"]


def test_no_prepared_papers_returns_guidance():
    tool = _build({}, {}, {})
    out = tool.invoke({"question": "q"})
    assert "no papers prepared" in out["error"]


def test_worker_cap_bounded():
    assert 1 <= MAX_PARALLEL_WORKERS <= 8
