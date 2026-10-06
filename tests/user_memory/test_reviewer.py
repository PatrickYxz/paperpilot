"""Reviewer approval-stage tests (fake models; real Qwen covered by manual runs)."""
from __future__ import annotations

from unittest.mock import MagicMock

from paperpilot.user_memory.extractor import MemoryCandidate
from paperpilot.user_memory.reviewer import CardReview, CardReviewOutput, review_memory_cards

USER_TEXT = "I work on model compression and only read papers with code."
ASSISTANT_TEXT = "The paper proposes multi-head attention."


def _card(content="User works on model compression.", span="I work on model compression"):
    return MemoryCandidate(
        kind="fact", content=content, support_span=span,
    )


def _model_returning(reviews):
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = (
        CardReviewOutput(reviews=reviews)
    )
    return model


def test_approves_clean_cards():
    approved, reasons = review_memory_cards(
        cards=[_card()], user_text=USER_TEXT,
        assistant_text=ASSISTANT_TEXT,
        model=_model_returning([CardReview(index=0, verdict="approve")]),
    )
    assert len(approved) == 1 and reasons == []


def test_rejected_cards_are_dropped_with_reasons():
    approved, reasons = review_memory_cards(
        cards=[_card(), _card("User lives in Tokyo.", span="I work on model compression")],
        user_text=USER_TEXT, assistant_text=ASSISTANT_TEXT,
        model=_model_returning([
            CardReview(index=0, verdict="approve"),
            CardReview(index=1, verdict="reject", reason="span does not entail content"),
        ]),
    )
    assert len(approved) == 1
    assert "span does not entail content" in reasons[0]


def test_missing_review_index_defaults_to_approve():
    approved, _ = review_memory_cards(
        cards=[_card()], user_text=USER_TEXT, assistant_text=ASSISTANT_TEXT,
        model=_model_returning([]),
    )
    assert len(approved) == 1


def test_reviewer_failure_fails_open():
    model = MagicMock()
    model.with_structured_output.return_value.invoke.side_effect = RuntimeError("down")
    approved, reasons = review_memory_cards(
        cards=[_card()], user_text=USER_TEXT, assistant_text=ASSISTANT_TEXT,
        model=model,
    )
    assert len(approved) == 1 and reasons == []


def test_empty_cards_short_circuit():
    assert review_memory_cards(
        cards=[], user_text="", assistant_text="", model=MagicMock()
    ) == ([], [])
