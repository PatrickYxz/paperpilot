"""Tests for semantic audit summary generation."""
from scripts.day20_summarize_semantic_audit import (
    _label_counts,
    _matrix_counts,
    _render_summary,
)


ROWS = [
    {
        "case_id": "case-1",
        "baseline": "paperpilot",
        "strict_pass": True,
        "semantic_label": "correct",
    },
    {
        "case_id": "case-2",
        "baseline": "paperpilot",
        "strict_pass": True,
        "semantic_label": "partial",
    },
    {
        "case_id": "case-3",
        "baseline": "paperpilot",
        "strict_pass": False,
        "semantic_label": "correct",
    },
    {
        "case_id": "case-4",
        "baseline": "paperpilot",
        "strict_pass": False,
        "semantic_label": "incorrect",
    },
]


def test_label_counts() -> None:
    counts = _label_counts(ROWS)

    assert counts["correct"] == 2
    assert counts["partial"] == 1
    assert counts["incorrect"] == 1


def test_matrix_counts() -> None:
    matrix = _matrix_counts(ROWS)

    assert matrix[(True, "correct")] == 1
    assert matrix[(True, "partial")] == 1
    assert matrix[(False, "correct")] == 1
    assert matrix[(False, "incorrect")] == 1


def test_render_summary_contains_false_positive_and_negative_counts() -> None:
    text = _render_summary(ROWS, baseline="paperpilot")

    assert "# Semantic Audit Summary" in text
    assert "| total audited cases | 4 |" in text
    assert "| estimated strict false positives | 1 |" in text
    assert "| estimated strict false negatives | 1 |" in text
    assert "| pass | partial | 1 |" in text
    assert "| fail | correct | 1 |" in text
