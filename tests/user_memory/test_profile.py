"""Profile generation and refresh-threshold tests."""
from __future__ import annotations

from unittest.mock import MagicMock

from paperpilot.user_memory.profile import (
    UserProfileOutput,
    clip_profile,
    generate_profile,
    profile_due_for_refresh,
)
from tests.user_memory.test_retrieval import _record


def test_generate_profile_happy_path():
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = (
        UserProfileOutput(profile="PhD student working on RAG.")
    )
    text = generate_profile(model=model, memories=[_record("m1", "x", "2026-10-01T00:00:00+00:00")])
    assert text == "PhD student working on RAG."


def test_generate_profile_failure_returns_none():
    model = MagicMock()
    model.with_structured_output.return_value.invoke.side_effect = RuntimeError
    assert generate_profile(model=model, memories=[_record("m", "x", "2026-10-01T00:00:00+00:00")]) is None
    assert generate_profile(model=model, memories=[]) is None


def test_clip_profile_enforces_word_budget():
    assert len(clip_profile(" ".join(["word"] * 500)).split()) == 120


def test_refresh_due_threshold():
    assert profile_due_for_refresh(active_memory_count=1, profiled_memory_count=None)
    assert not profile_due_for_refresh(active_memory_count=0, profiled_memory_count=None)
    assert not profile_due_for_refresh(active_memory_count=4, profiled_memory_count=0)
    assert profile_due_for_refresh(active_memory_count=5, profiled_memory_count=0)
