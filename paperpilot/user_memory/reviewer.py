"""Independent reviewer approval for extracted memory cards (book ch.4).

The reviewer is the second, risk-oriented perspective on the proposer's
cards: same rules, opposite bias. Fail-open on infrastructure errors — a
missing reviewer must not break the best-effort memory pipeline, and the
deterministic span verification still gates every card before this stage.
"""
from __future__ import annotations

import logging
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from paperpilot.user_memory.extractor import MemoryCandidate

_LOGGER = logging.getLogger("paperpilot.user_memory")

_REVIEWER_SYSTEM_PROMPT = (
    "You are the risk-control reviewer of a two-stage memory pipeline. "
    "Another model proposed long-term memory cards about the user from one "
    "conversation turn. Apply the same rules the proposer was given — keep "
    "only durable, abstracted, self-contained facts — but with the opposite "
    "bias: when in doubt, reject. Reject a card when its support span does "
    "not actually entail the content (over-generalization or speculation), "
    "when it is a one-off detail of this turn rather than a durable trait, "
    "or when the content is not useful in future conversations. Approve "
    "only cards a future conversation could rely on."
)


class CardReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    verdict: Literal["approve", "reject"]
    reason: str = Field(default="", max_length=300)


class CardReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviews: list[CardReview]


def review_memory_cards(
    *,
    cards: Sequence[MemoryCandidate],
    user_text: str,
    assistant_text: str,
    model: object,
) -> tuple[list[MemoryCandidate], list[str]]:
    """Return (approved cards, rejection reasons). Fail-open on errors."""
    if not cards:
        return list(cards), []
    try:
        from langchain.messages import HumanMessage, SystemMessage

        numbered = "\n".join(
            f"#{i}: [{card.kind}] {card.content}\n  support_span: "
            f"{card.support_span}\n  (subject={card.subject}, "
            f"topic={card.topic})"
            for i, card in enumerate(cards)
        )
        runnable = model.with_structured_output(CardReviewOutput)
        output = runnable.invoke(
            [
                SystemMessage(content=_REVIEWER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "Conversation turn:\n\n"
                        f"USER:\n{user_text}\n\n"
                        f"ASSISTANT:\n{assistant_text}\n\n"
                        f"Proposed cards:\n{numbered}\n\n"
                        "Review each card by its index."
                    )
                ),
            ]
        )
        reviews = CardReviewOutput.model_validate(output).reviews
    except Exception:
        _LOGGER.warning(
            "memory card review failed; approving all span-verified cards",
            exc_info=True,
        )
        return list(cards), []

    verdicts = {review.index: review for review in reviews}
    approved: list[MemoryCandidate] = []
    reasons: list[str] = []
    for index, card in enumerate(cards):
        review = verdicts.get(index)
        if review is None or review.verdict == "approve":
            approved.append(card)
        else:
            reasons.append(f"#{index} {card.content[:60]!r}: {review.reason}")
    return approved, reasons
