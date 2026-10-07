"""Sleep-learning consolidation for user memory cards (book ch.8).

The proposer LLM only PROPOSES (merge/archive/keep); a reviewer approves
each proposal; execution is append-only — merges write a new card with
provenance and archive the sources by status flip, so every change is
reversible and auditable. The in-process cooldown prevents re-running the
consolidator every turn when the active count stays above the threshold;
a fresh process re-runs once, which is idempotent.
"""
from __future__ import annotations

import logging
import time
from typing import Literal

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from paperpilot.web.store.records import NewUserMemory, UserMemoryRecord

_LOGGER = logging.getLogger("paperpilot.user_memory")

CONSOLIDATION_THRESHOLD = 15
CONSOLIDATION_COOLDOWN_SECONDS = 3600.0
_last_consolidation_at: dict[str, float] = {}

_PROPOSER_SYSTEM_PROMPT = (
    "You consolidate a user's long-term memory cards. Cards were "
    "extracted from separate conversations and may repeat or fragment the "
    "same fact. Enumerate ALL consolidation opportunities before "
    "proposing: merge EVERY group of cards stating the same underlying "
    "fact, even if worded differently (fold them into one self-contained "
    "card); when two cards conflict (e.g. old and new address), merge "
    "them into one card stating the timeline rather than picking one; "
    "archive cards that are stale, superseded, or have no long-term "
    "value. Do not merge unrelated facts. Each card may appear in at "
    "most one proposal."
)


class ConsolidationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["merge", "archive", "keep"]
    source_memory_ids: list[str] = Field(min_length=1)
    merged_content: str | None = Field(default=None, max_length=400)
    reason: str = Field(default="", max_length=300)


class ConsolidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[ConsolidationProposal]


def consolidation_due(user_id: str, active_count: int) -> bool:
    if active_count < CONSOLIDATION_THRESHOLD:
        return False
    last = _last_consolidation_at.get(user_id, 0.0)
    return (time.monotonic() - last) >= CONSOLIDATION_COOLDOWN_SECONDS


def build_consolidation_proposals(
    *,
    memories: list[UserMemoryRecord],
    model: object,
) -> list[ConsolidationProposal]:
    """Ask the proposer for proposals; invalid references are dropped and
    each card is claimed by at most one proposal (first come, first kept)."""
    try:
        runnable = model.with_structured_output(ConsolidationOutput)
        catalog = "\n".join(
            f"- {m.memory_id} [{m.kind}] {m.content} "
            f"(recorded {m.created_at[:10]})"
            for m in memories
        )
        output = runnable.invoke(
            [
                SystemMessage(content=_PROPOSER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Active memory cards:\n{catalog}\n\n"
                    "Propose consolidation; 'keep' proposals are optional."
                ),
            ]
        )
        parsed = ConsolidationOutput.model_validate(output).proposals
    except Exception:
        _LOGGER.warning("memory consolidation proposal failed", exc_info=True)
        return []

    known_ids = {m.memory_id for m in memories}
    claimed: set[str] = set()
    accepted: list[ConsolidationProposal] = []
    for proposal in parsed:
        if proposal.action == "keep":
            continue
        ids = [mid for mid in proposal.source_memory_ids if mid in known_ids]
        if not ids or set(ids) & claimed:
            continue
        if proposal.action == "merge":
            if len(ids) < 2 or not (proposal.merged_content or "").strip():
                continue
        claimed.update(ids)
        accepted.append(
            ConsolidationProposal(
                action=proposal.action,
                source_memory_ids=ids,
                merged_content=proposal.merged_content,
                reason=proposal.reason,
            )
        )
    return accepted


def review_proposals(
    proposals: list[ConsolidationProposal],
    memories: list[UserMemoryRecord],
    model: object,
) -> list[ConsolidationProposal]:
    """Reviewer approval per proposal (fail-open only on infra errors)."""
    if not proposals:
        return []
    from paperpilot.user_memory.reviewer import CardReview, CardReviewOutput

    by_id = {m.memory_id: m for m in memories}
    try:
        runnable = model.with_structured_output(CardReviewOutput)
        catalog = "\n".join(
            f"#{i}: {p.action} {p.source_memory_ids} -> "
            f"{p.merged_content or '(archive)'} | reason: {p.reason}"
            for i, p in enumerate(proposals)
        )
        source_lines = "\n".join(
            f"- {m.memory_id}: {m.content[:120]}" for m in memories
        )
        output = runnable.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the risk-control reviewer of memory "
                        "consolidation proposals. Reject a proposal when "
                        "it merges cards that are not the same fact, when "
                        "a merged card loses a material condition or "
                        "detail, or when an archive targets a card that "
                        "still looks durable. Approve minimal, faithful "
                        "changes. Review by index."
                    )
                ),
                HumanMessage(
                    content=f"Cards:\n{source_lines}\n\nProposals:\n{catalog}"
                ),
            ]
        )
        reviews = CardReviewOutput.model_validate(output).reviews
    except Exception:
        _LOGGER.warning("consolidation review failed; approving", exc_info=True)
        return proposals

    verdicts = {review.index: review for review in reviews}
    approved = []
    for index, proposal in enumerate(proposals):
        review = verdicts.get(index)
        if review is None or review.verdict == "approve":
            approved.append(proposal)
        else:
            _LOGGER.info(
                "consolidation proposal rejected: %s %s: %s",
                proposal.action,
                proposal.source_memory_ids,
                review.reason,
            )
    del by_id
    return approved


def apply_proposals(
    *,
    store,
    user_id: str,
    proposals: list[ConsolidationProposal],
    memories: list[UserMemoryRecord],
) -> dict[str, int]:
    """Execute approved proposals append-only. Returns action counts."""
    by_id = {m.memory_id: m for m in memories}
    counts = {"merged": 0, "archived": 0}
    for proposal in proposals:
        if proposal.action == "archive":
            for memory_id in proposal.source_memory_ids:
                store.archive_user_memory(
                    memory_id,
                    reason=f"consolidation: {proposal.reason[:200]}",
                )
                counts["archived"] += 1
        elif proposal.action == "merge":
            sources = [by_id[m] for m in proposal.source_memory_ids]
            longest = max(sources, key=lambda m: len(m.support_span))
            new_id = f"{sources[0].source_task_id}-cons-{int(time.time())}"
            store.append_user_memory(
                record=NewUserMemory(
                    memory_id=new_id,
                    user_id=user_id,
                    kind=sources[0].kind,
                    content=proposal.merged_content or "",
                    context={
                        "extraction": "consolidated",
                        "merged_from": [m.memory_id for m in sources],
                        "consolidation_reason": proposal.reason,
                    },
                    source_conversation_id=sources[0].source_conversation_id,
                    source_task_id=sources[0].source_task_id,
                    source_message_id=sources[0].source_message_id,
                    support_span=longest.support_span,
                    created_at=longest.created_at,
                )
            )
            for memory_id in proposal.source_memory_ids:
                store.archive_user_memory(
                    memory_id,
                    reason=f"merged into {new_id}: {proposal.reason[:160]}",
                )
                counts["archived"] += 1
            counts["merged"] += 1
    return counts
