"""Extractor prompt-to-candidates and span verification tests."""
from __future__ import annotations

from unittest.mock import MagicMock

from paperpilot.user_memory.extractor import (
    MemoryCandidate,
    MemoryExtractionOutput,
    extract_memory_candidates,
    verify_candidates,
)

USER_TEXT = "I work on retrieval-augmented generation. Please skip survey papers."
ASSISTANT_TEXT = "Focused on method papers with code; skipping surveys."


def _model_returning(memories):
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = (
        MemoryExtractionOutput(memories=memories)
    )
    return model


def test_extracts_candidates_through_structured_output():
    model = _model_returning(
        [
            MemoryCandidate(
                kind="fact",
                content="User works on retrieval-augmented generation.",
                support_span="I work on retrieval-augmented generation.",
            )
        ]
    )
    candidates = extract_memory_candidates(
        user_text=USER_TEXT, assistant_text=ASSISTANT_TEXT, model=model
    )
    assert len(candidates) == 1
    assert candidates[0].kind == "fact"


def test_extractor_failures_yield_no_candidates():
    model = MagicMock()
    model.with_structured_output.return_value.invoke.side_effect = RuntimeError(
        "provider down"
    )
    assert (
        extract_memory_candidates(
            user_text=USER_TEXT, assistant_text=ASSISTANT_TEXT, model=model
        )
        == []
    )


def test_verify_keeps_spans_present_in_source_texts():
    candidates = [
        MemoryCandidate(
            kind="preference",
            content="User dislikes survey papers.",
            support_span="Please skip survey papers.",
        ),
        MemoryCandidate(
            kind="fact",
            content="User lives in Tokyo.",
            support_span="I live in Tokyo.",
        ),
    ]
    verified = verify_candidates(candidates, [USER_TEXT, ASSISTANT_TEXT])
    assert [c.content for c in verified] == ["User dislikes survey papers."]


def test_verify_normalizes_whitespace_and_case():
    verified = verify_candidates(
        [
            MemoryCandidate(
                kind="fact",
                content="Works on RAG.",
                support_span="i work ON retrieval-augmented   generation.",
            )
        ],
        [USER_TEXT],
    )
    assert len(verified) == 1
