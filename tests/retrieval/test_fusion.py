"""RRF fusion tests."""
from __future__ import annotations

from paperpilot.retrieval.fusion import rrf_fuse


def test_doc_ranked_first_everywhere_wins():
    fused = rrf_fuse(
        {"dense": ["a", "b"], "sparse": ["a", "c"]},
        k=60,
    )
    assert [doc for doc, _ in fused] == ["a", "b", "c"]


def test_fused_score_is_sum_of_reciprocal_ranks():
    fused = dict(
        rrf_fuse({"dense": ["a", "b"], "sparse": ["b", "a"]}, k=60)
    )
    # Both docs appear at rank 1 in one list and rank 2 in the other.
    expected = 1 / 61 + 1 / 62
    assert fused["a"] == expected
    assert fused["b"] == expected


def test_top_k_truncates():
    fused = rrf_fuse({"dense": ["a", "b", "c"]}, k=60, top_k=2)
    assert [doc for doc, _ in fused] == ["a", "b"]


def test_ties_break_deterministically_by_doc_id():
    fused = rrf_fuse({"dense": ["b"], "sparse": ["a"]}, k=60)
    assert [doc for doc, _ in fused] == ["a", "b"]


def test_single_ranking_preserves_order():
    fused = rrf_fuse({"only": ["x", "y", "z"]}, k=60, top_k=3)
    assert [doc for doc, _ in fused] == ["x", "y", "z"]


def test_empty_rankings_yield_empty():
    assert rrf_fuse({"dense": [], "sparse": []}) == []
