"""Tests for semantic failure-mode classification."""
from pathlib import Path

from scripts.day22_classify_failure_modes import (
    build_failure_rows,
    classify_failure_modes,
    diagnostic_cases,
    render_markdown,
)


def test_diagnostic_cases_selects_non_perfect_manual_reviews() -> None:
    rows = [
        {"case_id": "case-1", "review_decision": "accept_as_correct"},
        {"case_id": "case-2", "review_decision": "keep_partial"},
        {"case_id": "case-3", "review_decision": "downgrade_to_incorrect"},
    ]

    selected = diagnostic_cases(rows)

    assert [row["case_id"] for row in selected] == ["case-2", "case-3"]


def test_classify_overbroad_scope_from_review_notes() -> None:
    modes = classify_failure_modes(
        {
            "strict_pass": True,
            "semantic_label": "partial",
            "review_notes": "Answer includes them but adds separate baseline methods that blur scope.",
        }
    )

    assert "overbroad_scope" in modes


def test_classify_missing_required_part_from_review_notes() -> None:
    modes = classify_failure_modes(
        {
            "strict_pass": False,
            "semantic_label": "partial",
            "review_notes": "Covers several components but misses the exact gold component.",
        }
    )

    assert "missing_required_part" in modes


def test_classify_wrong_numeric_or_fact_from_review_notes() -> None:
    modes = classify_failure_modes(
        {
            "strict_pass": False,
            "semantic_label": "partial",
            "review_notes": "Question asks how much improvement; the central percentages differ from the gold answer.",
        }
    )

    assert "wrong_numeric_or_fact" in modes


def test_classify_strict_keyword_false_positive() -> None:
    modes = classify_failure_modes(
        {
            "strict_pass": True,
            "semantic_label": "incorrect",
            "review_notes": "Gold says WikiHop; prediction says HotpotQA.",
        }
    )

    assert "strict_keyword_false_positive" in modes
    assert "wrong_numeric_or_fact" in modes


def test_generic_gold_wording_does_not_trigger_wrong_or_ambiguity() -> None:
    modes = classify_failure_modes(
        {
            "strict_pass": False,
            "semantic_label": "partial",
            "review_decision": "keep_partial",
            "review_notes": "Includes several correct improvements but misses the RoBERTa TRANSLATE-TEST lag required by gold.",
        }
    )

    assert modes == ["missing_required_part"]


def test_build_failure_rows_attaches_modes() -> None:
    rows = [
        {
            "case_id": "case-1",
            "category": "partial_medium",
            "review_decision": "keep_partial",
            "strict_pass": False,
            "semantic_label": "partial",
            "confidence": "medium",
            "question": "What methods?",
            "review_notes": "Omits one method.",
        }
    ]

    failure_rows = build_failure_rows(rows)

    assert failure_rows[0]["case_id"] == "case-1"
    assert failure_rows[0]["failure_modes"] == ["missing_required_part"]


def test_render_markdown_contains_counts_and_case_details() -> None:
    all_rows = [
        {
            "case_id": "case-1",
            "category": "partial_medium",
            "review_decision": "keep_partial",
            "strict_pass": False,
            "semantic_label": "partial",
            "confidence": "medium",
            "question": "What methods?",
            "review_notes": "Omits one method.",
        }
    ]
    failure_rows = build_failure_rows(all_rows)

    text = render_markdown(
        all_rows=all_rows,
        failure_rows=failure_rows,
        source_path=Path("input.jsonl"),
    )

    assert "# Semantic Failure Modes" in text
    assert "| missing_required_part | 1 |" in text
    assert "`case-1`" in text
    assert "It is a diagnosis layer; it does not change the calibrated score." in text
