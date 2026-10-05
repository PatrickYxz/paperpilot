"""Idempotent extraction pipeline from one published turn to memories."""
from __future__ import annotations

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
                context={"extraction": "turn-v1"},
                source_conversation_id=task.conversation_id,
                source_task_id=task.id,
                source_message_id=user_message_id,
                support_span=candidate.support_span,
                created_at=utc_now(),
            )
        )
        written += 1
    return written
