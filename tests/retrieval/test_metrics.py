"""Retrieval metric tests."""
from __future__ import annotations

from paperpilot.retrieval.metrics import mrr, ndcg, recall_at_k


def test_recall_at_k_counts_hits_within_window():
    retrieved = ["a", "b", "c", "d", "e"]
    assert recall_at_k(retrieved, {"b", "d"}, k=5) == 1.0
    assert recall_at_k(retrieved, {"b", "z"}, k=5) == 0.5
    # k window truncates: "e" only counts when k >= 5
    assert recall_at_k(retrieved, {"e"}, k=4) == 0.0


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["a"], set(), k=3) == 0.0


def test_mrr_first_relevant_position():
    assert mrr(["a", "b", "c"], {"a"}) == 1.0
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert mrr(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg(["a", "b", "x"], {"a", "b"}, k=3) == 1.0


def test_ndcg_penalizes_late_hits():
    good = ndcg(["a", "x", "x"], {"a"}, k=3)
    late = ndcg(["x", "x", "a"], {"a"}, k=3)
    assert good > late > 0.0
