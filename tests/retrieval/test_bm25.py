"""BM25 index and tokenizer tests. All expectations are hand-computable."""
from __future__ import annotations

import math

import pytest

from paperpilot.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_ascii_words_lowercase():
    assert tokenize("The BM25 Algorithm") == ["the", "bm25", "algorithm"]


def test_tokenize_cjk_bigrams():
    # Two-char run becomes one bigram; three-char run becomes two bigrams.
    assert tokenize("模型蒸馏") == ["模型", "型蒸", "蒸馏"]
    assert tokenize("猫") == ["猫"]


def test_tokenize_mixed():
    tokens = tokenize("Transformer 模型的蒸馏")
    assert "transformer" in tokens
    assert "模型" in tokens
    assert "蒸馏" in tokens


def _scores(index: BM25Index, query: str) -> dict[str, float]:
    return dict(index.search(query, top_k=10))


def test_idf_rare_term_outranks_common_term():
    index = BM25Index().build({
        "d1": "model distillation model distillation model",
        "d2": "model model",
    })

    # "distillation" appears in 1/2 docs, "model" in 2/2. With identical TF
    # patterns the rarer term must carry strictly more weight per occurrence.
    scores = _scores(index, "distillation")
    assert scores["d1"] > 0.0
    assert scores.get("d2", 0.0) == 0.0

    # Query only the common term: both score, but the shorter doc wins —
    # TF saturation (3 vs 2 hits gains little) plus length normalization
    # favor d2 over the longer d1.
    common = _scores(index, "model")
    assert common["d2"] > common["d1"] > 0.0


def test_idf_matches_lucene_variant_hand_computed():
    # 3 docs, term in exactly 1: idf = ln(1 + (3 - 1 + 0.5) / (1 + 0.5))
    index = BM25Index().build({
        "a": "quantum",
        "b": "classical computing",
        "c": "neural networks",
    })
    scores = _scores(index, "quantum")
    n, df = 3, 1
    idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
    tf, dl = 1, 1
    avgdl = 5 / 3  # doc lengths are 1, 2, 2
    expected = idf * tf * (index.k1 + 1.0) / (
        tf + index.k1 * (1.0 - index.b + index.b * dl / avgdl)
    )
    assert scores["a"] == pytest.approx(expected, rel=1e-9)


def test_term_frequency_saturation():
    # Doubling TF must less than double the score (k1 saturation).
    index = BM25Index().build({
        "once": "cat cat dog",
        "twice": "cat cat cat cat dog",
        "filler": "bird bird bird bird",
    })
    once = _scores(index, "cat")["once"]
    twice = _scores(index, "cat")["twice"]
    assert twice < 2 * once


def test_length_normalization_penalizes_padding():
    # Same TF, but one doc is padded with distinct filler terms.
    index = BM25Index().build({
        "short": "cat",
        "padded": "cat unrelated words padding here",
    })
    scores = _scores(index, "cat")
    assert scores["short"] > scores["padded"]


def test_search_respects_top_k_and_skips_unknown_terms():
    index = BM25Index().build({
        "a": "cat",
        "b": "cat cat",
        "c": "dog",
    })
    assert [doc for doc, _ in index.search("cat zebra", top_k=1)] == ["b"]
    assert index.search("zebra", top_k=5) == []


def test_build_replaces_previous_corpus():
    index = BM25Index().build({"a": "cat"}).build({"z": "dog"})
    assert index.doc_count == 1
    assert _scores(index, "cat") == {}
    assert [doc for doc, _ in index.search("dog", top_k=1)] == ["z"]
