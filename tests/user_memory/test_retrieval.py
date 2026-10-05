"""Memory ranking tests: lexical relevance dominates, recency breaks ties."""
from __future__ import annotations

from paperpilot.user_memory.retrieval import search_user_memories
from paperpilot.web.store.records import UserMemoryRecord

NOW = "2026-10-05T00:00:00+00:00"


def _record(memory_id, content, created_at, kind="fact"):
    return UserMemoryRecord(
        memory_id=memory_id,
        user_id="user-1",
        kind=kind,
        content=content,
        context={},
        source_conversation_id="conv-1",
        source_task_id="task-1",
        source_message_id="msg-1",
        support_span="span",
        created_at=created_at,
        status="active",
    )


FACTS = [
    _record("old", "Works on federated learning aggregation.",
            "2026-04-01T00:00:00+00:00"),
    _record("new", "Works on federated learning aggregation and privacy budgets.",
            "2026-10-01T00:00:00+00:00"),
    _record("unrelated", "Prefers survey papers as entry points.",
            "2026-10-04T00:00:00+00:00"),
]


def test_lexical_relevance_beats_recency():
    hits = search_user_memories(FACTS, "federated aggregation", top_k=3, now=None)
    ids = [hit.record.memory_id for hit in hits]
    assert ids[0] in {"old", "new"}
    assert "unrelated" == ids[-1] or "unrelated" not in ids[:1]


def test_recency_breaks_ties_between_equally_relevant():
    hits = search_user_memories(
        FACTS[:2], "federated learning aggregation", top_k=2
    )
    assert [hit.record.memory_id for hit in hits] == ["new", "old"]


def test_no_lexical_signal_falls_back_to_newest_first():
    hits = search_user_memories(FACTS, "zzz nonexistent topic", top_k=3)
    # Fallback ordering is strictly newest-first by created_at.
    assert [hit.record.memory_id for hit in hits] == [
        "unrelated",
        "new",
        "old",
    ]


def test_chinese_query_matches_via_cjk_bigrams():
    hits = search_user_memories(
        [
            _record("zh", "研究方向是检索增强生成 RAG。", "2026-10-01T00:00:00+00:00"),
            _record("en", "Prefers survey papers.", "2026-10-02T00:00:00+00:00"),
        ],
        "检索增强",
        top_k=2,
    )
    assert hits[0].record.memory_id == "zh"


def test_top_k_zero_returns_empty():
    assert search_user_memories(FACTS, "federated", top_k=0) == []


def test_turn_summary_search_matches_question_and_prefix():
    from paperpilot.user_memory.retrieval import search_turn_summaries

    class _Summary:
        def __init__(self, mid, conversation_id, question, narrative, created_at):
            self.user_message_id = mid
            self.conversation_id = conversation_id
            self.question = question
            self.narrative = narrative
            self.created_at = created_at

    summaries = [
        _Summary("m1", "conv-a", "How does LoRA rank affect accuracy?",
                 "Found rank 8 sufficient.", "2026-09-01T00:00:00Z"),
        _Summary("m2", "conv-b", "Summarize the federated aggregation paper.",
                 None, "2026-09-20T00:00:00Z"),
    ]
    hits = search_turn_summaries(summaries, "lora rank", top_k=1)
    assert hits[0][0].user_message_id == "m1"

    # Prefix clue only: conversation id is not in the raw texts.
    hits = search_turn_summaries(summaries, "conversation conv-b", top_k=1)
    assert hits[0][0].user_message_id == "m2"
