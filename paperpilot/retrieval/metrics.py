"""Retrieval quality metrics (recall@k, MRR, nDCG)."""
from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs that appear in the first k results."""
    if not relevant:
        return 0.0
    hits = set(retrieved[:k]) & relevant
    return len(hits) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant doc (0 when none is retrieved)."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant:
        return 0.0

    def _dcg(ranking: Sequence[str]) -> float:
        total = 0.0
        for rank, doc_id in enumerate(ranking[:k], start=1):
            rel = 1.0 if doc_id in relevant else 0.0
            total += rel / math.log2(rank + 1)
        return total

    ideal = sorted(relevant, key=lambda d: (d not in retrieved, d))
    denominator = _dcg(ideal)
    return _dcg(retrieved) / denominator if denominator else 0.0
