"""Ranking for user long-term memories: BM25 relevance times recency.

Memories are append-only (Mem0 v3 style), so conflicts between stale and
fresh facts are resolved at read time: equally relevant entries sort by
recency, and lexical relevance still dominates via a normalized BM25 score.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from paperpilot.retrieval.bm25 import BM25Index
from paperpilot.web.store.records import UserMemoryRecord

DEFAULT_RECENCY_HALF_LIFE_DAYS = 45.0


@dataclass(frozen=True)
class ScoredMemory:
    record: UserMemoryRecord
    relevance: float
    recency: float
    score: float


class MemorySearchIndex:
    """In-memory index over one user's memories (typically tens to hundreds)."""

    def __init__(
        self,
        records: Sequence[UserMemoryRecord],
        *,
        recency_half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
        now: datetime | None = None,
    ) -> None:
        self._records = list(records)
        self._half_life_days = recency_half_life_days
        self._now = now or datetime.now(timezone.utc)
        self._index = BM25Index().build(
            {
                record.memory_id: f"{record.kind}\n{record.content}"
                for record in self._records
            }
        )

    def search(self, query: str, top_k: int) -> list[ScoredMemory]:
        if top_k <= 0 or not self._records:
            return []

        by_id = {record.memory_id: record for record in self._records}
        lexical = dict(self._index.search(query, top_k=len(self._records)))
        max_lexical = max(lexical.values(), default=0.0)

        scored: list[ScoredMemory] = []
        for record in self._records:
            raw = lexical.get(record.memory_id, 0.0)
            # Normalize per-query so the lexical term stays in [0, 1] no
            # matter the BM25 magnitude; recency only breaks near-ties.
            relevance = raw / max_lexical if max_lexical > 0 else 0.0
            recency = self._recency(record.created_at)
            scored.append(
                ScoredMemory(
                    record=record,
                    relevance=relevance,
                    recency=recency,
                    score=relevance * (1.0 + recency),
                )
            )

        scored.sort(key=lambda item: (-item.score, -item.recency, item.record.memory_id))
        if max_lexical == 0.0:
            # No lexical signal at all (e.g. list-style query): fall back to
            # newest-first so the agent still sees the latest facts.
            scored.sort(key=lambda item: (-item.recency, item.record.memory_id))
        return scored[:top_k]

    def _recency(self, created_at: str) -> float:
        try:
            stamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        age_days = max((self._now - stamp).total_seconds() / 86400.0, 0.0)
        return 0.5 ** (age_days / max(self._half_life_days, 1e-9))


def search_user_memories(
    records: Sequence[UserMemoryRecord],
    query: str,
    top_k: int,
    *,
    now: datetime | None = None,
) -> list[ScoredMemory]:
    """Convenience wrapper: index one user's records and search once."""
    return MemorySearchIndex(records, now=now).search(query, top_k)


def turn_summary_prefix(summary) -> str:
    """Contextual prefix anchoring one turn digest to its conversation."""
    return (
        f"[conversation {summary.conversation_id}; {summary.created_at[:10]}; "
        f"question: {summary.question[:80]}]"
    )


def search_turn_summaries(
    summaries: Sequence,
    query: str,
    top_k: int,
) -> list[tuple[object, float]]:
    """BM25 search over prefixed turn digests; returns (summary, score)."""
    if not summaries or top_k <= 0:
        return []
    index = BM25Index().build(
        {
            summary.user_message_id: (
                f"{turn_summary_prefix(summary)}\n"
                f"{summary.narrative or ''}\n{summary.question}"
            )
            for summary in summaries
        }
    )
    by_id = {summary.user_message_id: summary for summary in summaries}
    return [
        (by_id[doc_id], score) for doc_id, score in index.search(query, top_k)
    ]
