"""Score fusion for hybrid retrieval pipelines."""
from __future__ import annotations

from typing import Mapping, Sequence


def rrf_fuse(
    rankings: Mapping[str, Sequence[str]],
    k: int = 60,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over named result rankings.

    Each ranking maps to an ordered list of doc ids (best first). A doc's
    fused score is ``sum(1 / (k + rank))`` across rankings. Ties break by
    doc id so the output is deterministic.
    """
    scores: dict[str, float] = {}
    for ranked_ids in rankings.values():
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        return ordered[:top_k]
    return ordered
