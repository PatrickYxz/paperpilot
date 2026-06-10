"""Tests for enriched QASPER subset profiling."""
from scripts.day19_profile_enriched_eval import (
    _answer_type_counts,
    _question_type,
    _render_profile,
)


def test_question_type_numeric() -> None:
    assert _question_type("How many comments were used?") == "numeric"
    assert _question_type("What percentage of examples are labeled?") == "numeric"


def test_question_type_boolean() -> None:
    assert _question_type("Does the paper report macro F1?") == "boolean"


def test_question_type_fact_or_list() -> None:
    assert _question_type("Which languages are used?") == "fact_or_list"
    assert _question_type("What labels are available?") == "fact_or_list"


def test_answer_type_counts() -> None:
    rows = [
        {
            "answers": [
                {
                    "extractive_spans": ["A"],
                    "free_form_answer": "",
                    "yes_no": None,
                    "unanswerable": False,
                    "evidence": ["E"],
                    "highlighted_evidence": [],
                }
            ]
        },
        {
            "answers": [
                {
                    "extractive_spans": [],
                    "free_form_answer": "Free",
                    "yes_no": True,
                    "unanswerable": False,
                    "evidence": [],
                    "highlighted_evidence": ["H"],
                }
            ]
        },
    ]

    counts = _answer_type_counts(rows)

    assert counts["extractive"] == 1
    assert counts["free_form"] == 1
    assert counts["yes_no"] == 1
    assert counts["unanswerable"] == 0
    assert counts["with_evidence"] == 1
    assert counts["with_highlighted_evidence"] == 1


def test_render_profile_contains_core_sections() -> None:
    rows = [
        {
            "case_id": "case-1",
            "question": "Which labels are available?",
            "oracle_spans": ["positive", "negative"],
            "answers": [
                {
                    "extractive_spans": ["positive", "negative"],
                    "free_form_answer": "",
                    "yes_no": None,
                    "unanswerable": False,
                    "evidence": ["The labels are positive and negative."],
                    "highlighted_evidence": ["positive and negative"],
                }
            ],
        }
    ]

    text = _render_profile(rows)

    assert "# QASPER Enriched Subset Profile" in text
    assert "| cases | 1 |" in text
    assert "| fact_or_list | 1 |" in text
    assert "| with_evidence | 1 |" in text
