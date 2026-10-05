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
    fake_model.tokenizer.encode.side_effect = (
        lambda text, add_special_tokens=False: list(range(len(text.split())))
    )
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

    chunks_payload = json.loads(
        (paper_dir / "chunks.json").read_text("utf-8")
    )
    assert chunks_payload["version"] == 2
    chunks = chunks_payload["chunks"]
    assert all(chunk_id.startswith("p1::chunk_") for chunk_id in chunks)
    first = next(iter(chunks.values()))
    assert first["text"].startswith("hello world")
    assert first["meta"]["prefix"].startswith("[source: paper p1")
    assert "char_span" in first["meta"]


def test_paper_key_escapes_path_separators(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    out = mgr.build([{"paper_id": "cs/0501001", "text": "legacy arxiv id"}])

    assert out["fresh_papers"] == ["cs/0501001"]
    paper_dirs = [p for p in patched_root.iterdir() if p.is_dir()]
    assert len(paper_dirs) == 1
    assert "/" not in paper_dirs[0].name
    assert "\\" not in paper_dirs[0].name

    chunks_payload = json.loads((paper_dirs[0] / "chunks.json").read_text("utf-8"))
    chunks = chunks_payload["chunks"]
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
    assert out[0]["chunk_id"] == chunk_id_a.split("::", 1)[1]
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


def _long_paragraph(n_sentences: int, tag: str) -> str:
    return " ".join(
        f"{tag} sentence {i} discusses retrieval fusion." for i in range(n_sentences)
    )


def test_search_hybrid_returns_rank_fields(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    text = f"{_long_paragraph(60, 'alpha')}\n\n{_long_paragraph(60, 'beta')}"
    mgr.build([{"paper_id": "p1", "text": text}])

    all_ids = list(mgr._states["p1"].chunk_texts)
    assert len(all_ids) >= 2
    patched_pylate["retr"].retrieve.return_value = [
        [{"id": fid, "score": 0.5} for fid in all_ids[:20]]
    ]

    out = mgr.search("retrieval fusion", paper_id="p1", top_k=3)
    assert len(out) == 3
    fused_scores = [hit["fused_score"] for hit in out]
    assert fused_scores == sorted(fused_scores, reverse=True)
    for hit in out:
        assert "dense_rank" in hit and "sparse_rank" in hit
        assert hit["context_prefix"].startswith("[source: paper p1")
        # Overlap-carrying chunks may span the paragraph boundary; verify the
        # first and last sentence both come from the original text.
        sentences = hit["chunk_text"].split("\n")
        assert sentences[0] in text
        assert sentences[-1] in text


def test_search_hybrid_missing_dense_hit_recovered_by_sparse(
    patched_root, patched_pylate
):
    mgr = im_module.IndexManager()
    text = f"{_long_paragraph(60, 'alpha')}\n\n{_long_paragraph(60, 'beta')}"
    mgr.build([{"paper_id": "p1", "text": text}])
    all_ids = list(mgr._states["p1"].chunk_texts)

    # Dense returns nothing; BM25 alone must still rank beta-tagged chunks.
    patched_pylate["retr"].retrieve.return_value = [[]]
    out = mgr.search("beta sentence", paper_id="p1", top_k=2)
    assert len(out) == 2
    assert all(hit["dense_rank"] is None for hit in out)
    assert all(hit["sparse_rank"] is not None for hit in out)


def test_search_dense_mode_keeps_colbert_score(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    mgr.build([{"paper_id": "p1", "text": "alpha alpha alpha"}])
    chunk_id = next(iter(mgr._states["p1"].chunk_texts))
    patched_pylate["retr"].retrieve.return_value = [
        [{"id": chunk_id, "score": 0.9}]
    ]

    out = mgr.search("alpha", paper_id="p1", top_k=1, mode="dense")
    assert out[0]["score"] == 0.9
    assert "fused_score" not in out[0]


def test_search_rejects_unknown_mode(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    mgr.build([{"paper_id": "p1", "text": "hello world"}])
    with pytest.raises(ValueError, match="mode"):
        mgr.search("q", paper_id="p1", top_k=1, mode="sparse")


def test_lazy_load_accepts_v1_chunks_payload(patched_root, patched_pylate):
    mgr = im_module.IndexManager()
    paper_dir = mgr._paper_root("p1")
    mgr._index_path("p1").mkdir(parents=True, exist_ok=True)
    mgr._chunks_path("p1").write_text(
        json.dumps({"p1::chunk_0": "legacy text"}), encoding="utf-8"
    )

    out = mgr.search("legacy", paper_id="p1", top_k=1)
    assert len(out) == 1
    assert out[0]["chunk_text"] == "legacy text"
    assert out[0]["context_prefix"] == ""
    assert out[0]["sparse_rank"] is not None


def test_custom_prefix_provider_is_persisted(patched_root, patched_pylate):
    mgr = im_module.IndexManager(
        prefix_provider=lambda paper_id, record, total: f"<{paper_id}#{record.index}>"
    )
    mgr.build([{"paper_id": "p1", "text": "one two three"}])

    payload = json.loads(mgr._chunks_path("p1").read_text("utf-8"))
    first = next(iter(payload["chunks"].values()))
    assert first["meta"]["prefix"] == "<p1#0>"
