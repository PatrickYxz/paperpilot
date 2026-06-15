"""Tests for manual calibration candidate preparation."""
import json
from pathlib import Path

from scripts.day21_prepare_calibration import (
    build_candidates,
    candidate_category,
    render_markdown,
    write_jsonl,
)


def test_candidate_category_selects_partials_and_severe_false_positives() -> None:
    assert (
        candidate_category(
            {
                "strict_pass": False,
                "semantic_label": "partial",
                "confidence": "medium",
            }
        )
        == "partial_medium"
    )
    assert (
        candidate_category(
            {
                "strict_pass": True,
                "semantic_label": "incorrect",
                "confidence": "high",
            }
        )
        == "strict_pass_semantic_bad"
    )
    assert (
        candidate_category(
            {
                "strict_pass": False,
                "semantic_label": "incorrect",
                "confidence": "high",
            }
        )
        is None
    )
    assert (
        candidate_category(
            {
                "strict_pass": True,
                "semantic_label": "correct",
                "confidence": "high",
            }
        )
        is None
    )


def test_build_candidates_joins_audit_results_and_gold_metadata() -> None:
    candidates = build_candidates(
        audit_rows=[
            {
                "case_id": "case-1",
                "baseline": "paperpilot",
                "strict_pass": True,
                "semantic_label": "partial",
                "confidence": "high",
                "reason": "Core answer present, extra unsupported item.",
            },
            {
                "case_id": "case-2",
                "baseline": "paperpilot",
                "strict_pass": False,
                "semantic_label": "correct",
                "confidence": "high",
                "reason": "Fine.",
            },
        ],
        result_rows=[
            {
                "case_id": "case-1",
                "question": "What metrics?",
                "oracle_spans": ["AUC"],
                "predicted": "AUC and accuracy",
                "trace_path": "trace.jsonl",
            }
        ],
        enriched_rows=[
            {
                "case_id": "case-1",
                "question": "What metrics?",
                "oracle_spans": ["AUC"],
                "answers": [
                    {
                        "extractive_spans": ["AUC"],
                        "free_form_answer": "",
                        "yes_no": None,
                        "unanswerable": False,
                        "evidence": ["We used AUC."],
                    }
                ],
            }
        ],
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["case_id"] == "case-1"
    assert candidate["category"] == "partial_high"
    assert candidate["review_decision"] == "TODO"
    assert candidate["question"] == "What metrics?"
    assert candidate["oracle_spans"] == ["AUC"]
    assert candidate["predicted"] == "AUC and accuracy"
    assert candidate["gold_answers"][0]["evidence"] == ["We used AUC."]


def test_render_markdown_and_write_jsonl(tmp_path: Path) -> None:
    candidates = [
        {
            "case_id": "case-1",
            "category": "partial_high",
            "review_decision": "TODO",
            "strict_pass": True,
            "semantic_label": "partial",
            "confidence": "high",
            "question": "What metrics?",
            "oracle_spans": ["AUC"],
            "predicted": "AUC and accuracy",
            "judge_reason": "Core answer present.",
        }
    ]

    markdown = render_markdown(candidates)
    assert "# Semantic Calibration Candidates" in markdown
    assert "partial_high: 1" in markdown
    assert "AUC and accuracy" in markdown

    out = tmp_path / "candidates.jsonl"
    write_jsonl(out, candidates)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == candidates
