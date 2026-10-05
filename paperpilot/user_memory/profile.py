"""Materialized user research profile: the always-visible overview layer."""
from __future__ import annotations

from typing import Sequence

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from paperpilot.web.store.records import UserMemoryRecord

PROFILE_REFRESH_DELTA = 5
_PROFILE_WORD_BUDGET = 120

_PROFILE_SYSTEM_PROMPT = (
    "You maintain a compact research profile of one user from their "
    "long-term memories. Merge duplicates, prefer the newest fact when "
    "memories conflict, drop stale one-off details, and keep the profile "
    "under 120 words. Write plain sentences about: research focus, "
    "method preferences, ongoing projects, and notable papers they studied."
)


class UserProfileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str = Field(min_length=10, max_length=2000)


def generate_profile(
    *,
    model: object,
    memories: Sequence[UserMemoryRecord],
) -> str | None:
    """Distill memories into one profile paragraph; None on any failure."""
    if not memories:
        return None
    try:
        runnable = model.with_structured_output(UserProfileOutput)
        response = runnable.invoke(
            [
                SystemMessage(content=_PROFILE_SYSTEM_PROMPT),
                HumanMessage(
                    content="\n".join(
                        f"- [{record.kind}] {record.content} "
                        f"(recorded {record.created_at[:10]})"
                        for record in memories
                    )
                ),
            ]
        )
        parsed = UserProfileOutput.model_validate(response)
    except Exception:
        return None
    return clip_profile(parsed.profile)


def clip_profile(profile: str) -> str:
    """Hard word budget so the overview layer stays cheap."""
    words = profile.split()
    return " ".join(words[:_PROFILE_WORD_BUDGET])


def profile_due_for_refresh(
    *,
    active_memory_count: int,
    profiled_memory_count: int | None,
) -> bool:
    """True when enough new memories accumulated since the last profile."""
    if profiled_memory_count is None:
        return active_memory_count > 0
    return active_memory_count - profiled_memory_count >= PROFILE_REFRESH_DELTA
