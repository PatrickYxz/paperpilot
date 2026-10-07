"""Idempotent extraction pipeline from one published turn to memories."""
from __future__ import annotations

import logging

from paperpilot.user_memory.reviewer_model import build_reviewer_model_from_env
from paperpilot.user_memory.consolidation import (
    apply_proposals,
    build_consolidation_proposals,
    consolidation_due,
    review_proposals,
)
from paperpilot.user_memory.profile import (
    generate_profile,
    profile_due_for_refresh,
)
from paperpilot.user_memory.reviewer import review_memory_cards
from paperpilot.user_memory.extractor import (
    extract_memory_candidates,
    verify_candidates,
)
from paperpilot.web.store.records import NewUserMemory, ResearchTask
from paperpilot.web.store.helpers import utc_now


def run_memory_extraction(
    *,
    store,
    model: object,
    task: ResearchTask,
    user_message_id: str,
    user_text: str,
    assistant_text: str,
) -> int:
    """Extract, verify, and append memories for one task. Returns written count.

    Idempotent on the source task: if any memory already exists for it, the
    pipeline is a no-op. Raises propagate to the caller (best-effort there).
    """
    if task.user_id is None or task.conversation_id is None:
        return 0
    if store.count_task_memories(task.id) > 0:
        return 0

    candidates = extract_memory_candidates(
        user_text=user_text,
        assistant_text=assistant_text,
        model=model,
    )
    verified = verify_candidates(candidates, [user_text, assistant_text])
    reviewer_model = build_reviewer_model_from_env()
    if reviewer_model is not None and verified:
        verified, rejection_reasons = review_memory_cards(
            cards=verified,
            user_text=user_text,
            assistant_text=assistant_text,
            model=reviewer_model,
        )
        for reason in rejection_reasons:
            logging.getLogger("paperpilot.user_memory").info(
                "memory card rejected by reviewer: %s", reason
            )

    written = 0
    for index, candidate in enumerate(verified, start=1):
        store.append_user_memory(
            record=NewUserMemory(
                memory_id=f"{task.id}-mem-{index}",
                user_id=task.user_id,
                kind=candidate.kind,
                content=candidate.content,
                context={
                    "extraction": "card-v2",
                    "subject": candidate.subject,
                    "relationship": candidate.relationship,
                    "topic": candidate.topic,
                    "backstory": candidate.backstory,
                },
                source_conversation_id=task.conversation_id,
                source_task_id=task.id,
                source_message_id=user_message_id,
                support_span=candidate.support_span,
                created_at=utc_now(),
            )
        )
        written += 1
    if written:
        refresh_user_profile_if_due(
            store=store, model=model, user_id=task.user_id
        )
        consolidate_user_memories_if_due(
            store=store, model=model, user_id=task.user_id
        )
    return written


def refresh_user_profile_if_due(*, store, model, user_id: str) -> bool:
    """Regenerate the overview profile once enough new memories exist."""
    _LOGGER = logging.getLogger("paperpilot.user_memory")
    try:
        active = store.list_user_memories(user_id)
        profile = store.get_user_profile(user_id)
        if not profile_due_for_refresh(
            active_memory_count=len(active),
            profiled_memory_count=(
                profile.source_memory_count if profile else None
            ),
        ):
            return False
        text = generate_profile(model=model, memories=active)
        if not text:
            return False
        store.upsert_user_profile(
            user_id=user_id,
            profile_text=text,
            source_memory_count=len(active),
        )
        return True
    except Exception:
        _LOGGER.warning("user profile refresh failed", exc_info=True)
        return False


def consolidate_user_memories_if_due(*, store, model, user_id: str) -> dict | None:
    """Sleep-learning pass: propose, review, apply (book ch.8)."""
    from paperpilot.user_memory.consolidation import (
        _last_consolidation_at,
    )
    import time as _time

    active = store.list_user_memories(user_id)
    if not consolidation_due(user_id, len(active)):
        return None
    _last_consolidation_at[user_id] = _time.monotonic()

    proposals = build_consolidation_proposals(memories=active, model=model)
    if not proposals:
        return {"merged": 0, "archived": 0, "proposed": 0}

    from paperpilot.user_memory.reviewer_model import build_reviewer_model_from_env

    reviewer = build_reviewer_model_from_env()
    if reviewer is not None:
        proposals = review_proposals(proposals, active, reviewer)
    counts = apply_proposals(
        store=store, user_id=user_id, proposals=proposals, memories=active
    )
    counts["proposed"] = len(proposals)
    logging.getLogger("paperpilot.user_memory").info(
        "memory consolidation: %s", counts
    )
    return counts
