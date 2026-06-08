"""Bulk user paper input detector tests."""
from __future__ import annotations

from paperpilot.bulk_input import BulkPaperInputDetector


def _paper_text(target: str = "2402.13718") -> str:
    return (
        f"Please compare this paper with {target} for similarity.\n\n"
        "Abstract\nThis paper studies retrieval augmented generation.\n\n"
        "Introduction\n" + ("Long context retrieval. " * 60) + "\n\n"
        "References\n[1] Prior work."
    )


def test_detector_finds_long_paper_similarity_request():
    detector = BulkPaperInputDetector(min_chars=200)

    result = detector.detect(_paper_text())

    assert result is not None
    assert result.target_paper_id == "2402.13718"
    assert "Abstract" in result.document_text


def test_detector_requires_similarity_intent():
    detector = BulkPaperInputDetector(min_chars=200)
    text = _paper_text().replace(
        "Please compare this paper with 2402.13718 for similarity.",
        "Please summarize this paper.",
    )

    assert detector.detect(text) is None


def test_detector_requires_target_paper_id():
    detector = BulkPaperInputDetector(min_chars=200)
    text = _paper_text().replace("2402.13718", "the other paper")

    assert detector.detect(text) is None
