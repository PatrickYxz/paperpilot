"""Sleep-learning consolidation tests (fake models)."""
from __future__ import annotations

from unittest.mock import MagicMock

from paperpilot.user_memory.consolidation import (
    ConsolidationOutput,
    ConsolidationProposal,
    apply_proposals,
    build_consolidation_proposals,
    consolidation_due,
    review_proposals,
)
from paperpilot.user_memory.reviewer import CardReview, CardReviewOutput
from paperpilot.web.store.records import UserMemoryRecord


def _mem(mid, content, created="2026-10-01T00:00:00Z"):
    return UserMemoryRecord(
        memory_id=mid, user_id="u1", kind="fact", content=content,
        context={}, source_conversation_id="c1", source_task_id="t1",
        source_message_id="m1", support_span=f"span {mid}",
        created_at=created, status="active",
    )


def _proposer(returning):
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = (
        ConsolidationOutput(proposals=returning)
    )
    return model


def test_proposals_drop_unknown_ids_and_double_claims():
    memories = [_mem("a", "works on compression"), _mem("b", "compression focus")]
    proposals = build_consolidation_proposals(
        memories=memories,
        model=_proposer([
            ConsolidationProposal(action="merge", source_memory_ids=["a", "ghost"],
                                  merged_content="User works on compression.",
                                  reason="same fact"),
            ConsolidationProposal(action="archive", source_memory_ids=["a"],
                                  reason="dup"),
        ]),
    )
    # The merge loses its unknown member, degenerates to a single-card
    # merge, and is dropped whole; the archive (valid id, unclaimed)
    # survives.
    assert len(proposals) == 1
    assert proposals[0].action == "archive"
    assert proposals[0].source_memory_ids == ["a"]


def test_merge_requires_two_sources_and_content():
    proposals = build_consolidation_proposals(
        memories=[_mem("a", "x"), _mem("b", "y")],
        model=_proposer([
            ConsolidationProposal(action="merge", source_memory_ids=["a"],
                                  merged_content="merged", reason="r"),
            ConsolidationProposal(action="merge", source_memory_ids=["a", "b"],
                                  merged_content="  ", reason="r"),
        ]),
    )
    assert proposals == []


def test_review_rejects_and_fails_open():
    proposals = [
        ConsolidationProposal(action="archive", source_memory_ids=["a"], reason="stale"),
        ConsolidationProposal(action="archive", source_memory_ids=["b"], reason="dup"),
    ]
    reviewer = MagicMock()
    reviewer.with_structured_output.return_value.invoke.return_value = (
        CardReviewOutput(reviews=[CardReview(index=0, verdict="reject", reason="still durable")])
    )
    approved = review_proposals(proposals, [_mem("a", "x"), _mem("b", "y")], reviewer)
    assert [p.source_memory_ids for p in approved] == [["b"]]

    broken = MagicMock()
    broken.with_structured_output.return_value.invoke.side_effect = RuntimeError
    assert review_proposals(proposals, [], broken) == proposals


def test_consolidation_due_threshold_and_cooldown(monkeypatch):
    monkeypatch.setattr("paperpilot.user_memory.consolidation._last_consolidation_at", {})
    assert consolidation_due("u", 14) is False
    assert consolidation_due("u", 15) is True
    from paperpilot.user_memory.consolidation import _last_consolidation_at
    import time
    _last_consolidation_at["u"] = time.monotonic()
    assert consolidation_due("u", 30) is False


class _FakeStore:
    def __init__(self):
        self.archived = []
        self.appended = []

    def archive_user_memory(self, memory_id, *, reason):
        self.archived.append((memory_id, reason))
        return True

    def append_user_memory(self, *, record):
        self.appended.append(record)
        return record


def test_apply_merge_writes_provenance_and_archives_sources():
    store = _FakeStore()
    counts = apply_proposals(
        store=store,
        user_id="u1",
        proposals=[ConsolidationProposal(
            action="merge", source_memory_ids=["a", "b"],
            merged_content="User works on compression (since 2026).",
            reason="same fact, two wordings",
        )],
        memories=[_mem("a", "compression"), _mem("b", "compression focus")],
    )
    assert counts == {"merged": 1, "archived": 2}
    assert len(store.appended) == 1
    record = store.appended[0]
    assert record.context["merged_from"] == ["a", "b"]
    assert "consolidation_reason" in record.context
    assert {mid for mid, _ in store.archived} == {"a", "b"}
