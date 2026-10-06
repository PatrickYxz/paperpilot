"""Idempotent extraction pipeline from one published turn to memories."""
from __future__ import annotations

import logging

from paperpilot.user_memory.profile import (
    generate_profile,
    profile_due_for_refresh,
)
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
