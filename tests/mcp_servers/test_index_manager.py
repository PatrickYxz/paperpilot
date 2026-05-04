"""IndexManager per-paper core tests. PyLate is fully mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

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

    paper_dir = mgr._paper_root("p1")
    assert paper_dir.parent == patched_root
    assert (paper_dir / "paperpilot_current").exists()
    assert (paper_dir / "chunks.json").exists()

    chunks = json.loads((paper_dir / "chunks.json").read_text("utf-8"))
    assert all(chunk_id.startswith("p1::chunk_") for chunk_id in chunks)


def test_paper_key_escapes_path_separators(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    out = mgr.build([{"paper_id": "cs/0501001", "text": "legacy arxiv id"}])

    assert out["fresh_papers"] == ["cs/0501001"]
    paper_dirs = [p for p in patched_root.iterdir() if p.is_dir()]
    assert len(paper_dirs) == 1
    assert "/" not in paper_dirs[0].name
    assert "\\" not in paper_dirs[0].name

    chunks = json.loads((paper_dirs[0] / "chunks.json").read_text("utf-8"))
    assert all(chunk_id.startswith("cs/0501001::chunk_") for chunk_id in chunks)


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

    mgr2 = im_module.IndexManager()
    patched_pylate["model"].encode.reset_mock()

    out = mgr2.build([{"paper_id": "p1", "text": "hello"}])

    assert out["cached_papers"] == ["p1"]
    assert out["fresh_papers"] == []
    patched_pylate["model"].encode.assert_not_called()
    assert "p1" in mgr2._states


def test_disk_hit_falls_back_to_cold_when_chunks_corrupt(
    patched_root, patched_pylate
):
    mgr1 = im_module.IndexManager()
    mgr1.build([{"paper_id": "p1", "text": "hello"}])
    mgr1._chunks_path("p1").write_text("not-json", encoding="utf-8")

    mgr2 = im_module.IndexManager()
    patched_pylate["model"].encode.reset_mock()

    out = mgr2.build([{"paper_id": "p1", "text": "hello"}])

    assert out["fresh_papers"] == ["p1"]
    assert out["cached_papers"] == []
    assert patched_pylate["model"].encode.called


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
    chunk_id_a = next(iter(state_a.chunk_texts))
    patched_pylate["retr"].retrieve.return_value = [
        [{"id": chunk_id_a, "score": 0.9}]
    ]

    out = mgr.search("alpha", paper_id="pA", top_k=1)
    assert len(out) == 1
    assert out[0]["paper_id"] == "pA"
    assert out[0]["chunk_text"] == state_a.chunk_texts[chunk_id_a]


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
