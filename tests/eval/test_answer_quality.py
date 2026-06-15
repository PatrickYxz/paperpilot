"""Tests for eval answer quality checks and repair prompt helpers."""
from paperpilot.eval.answer_quality import (
    build_repair_prompt,
    evaluate_answer_quality,
    should_repair_answer,
)


def test_internal_trace_marker_is_severe() -> None:
    quality = evaluate_answer_quality(
        "Step 4: synthesize.\n\nShort answer: WikiHop.\n\nEvidence: The study uses WikiHop.",
    )

    assert quality["severity"] == "severe"
    assert should_repair_answer(quality) is True
    assert quality["issues"][0]["code"] == "internal_trace_marker"


def test_missing_sections_are_severe() -> None:
    quality = evaluate_answer_quality("The answer is WikiHop.")

    assert quality["severity"] == "severe"
    assert {issue["code"] for issue in quality["issues"]} == {
        "missing_short_answer_section",
        "missing_evidence_section",
    }


def test_short_answer_too_long_is_severe() -> None:
    long_answer = "A" * 701

    quality = evaluate_answer_quality(
        f"Short answer: {long_answer}\n\n"
        "Evidence: The evidence repeats the answer with enough supporting detail.",
    )

    assert _codes(quality) == {"short_answer_too_long"}
    assert quality["repair_required"] is True


def test_numeric_claim_must_be_supported_in_evidence() -> None:
    quality = evaluate_answer_quality(
        "Short answer: The model improves by 12.5%.\n\n"
        "Evidence: The paper reports a clear improvement over the baseline.",
    )

    assert "numeric_not_supported_in_evidence" in _codes(quality)
    assert should_repair_answer(quality) is True


def test_matching_number_in_evidence_passes_numeric_check() -> None:
    quality = evaluate_answer_quality(
        "Short answer: The model improves by 12.5%.\n\n"
        "Evidence: The paper reports a 12.5% improvement over the baseline.",
    )

    assert "numeric_not_supported_in_evidence" not in _codes(quality)


def test_list_dumping_is_risk_only() -> None:
    quality = evaluate_answer_quality(
        "Short answer: A, B, C, D, E, F, G, H are listed.\n\n"
        "Evidence: The paper directly lists A, B, C, D, E, F, G, H in the methods section.",
    )

    assert quality["severity"] == "risk"
    assert quality["passed"] is True
    assert quality["repair_required"] is False
    assert should_repair_answer(quality) is False
    assert _codes(quality) == {"list_dumping_risk"}


def test_thin_evidence_is_risk_only() -> None:
    quality = evaluate_answer_quality(
        "Short answer: The dataset is WikiHop.\n\nEvidence: WikiHop.",
    )

    assert quality["severity"] == "risk"
    assert "evidence_too_thin" in _codes(quality)
    assert should_repair_answer(quality) is False


def test_repair_prompt_is_narrow() -> None:
    quality = evaluate_answer_quality("Step 4\nShort answer: 10%.\n\nEvidence: improvement.")

    prompt = build_repair_prompt(
        question="How much did it improve?",
        raw_answer="Step 4\nShort answer: 10%.\n\nEvidence: improvement.",
        quality=quality,
    )

    assert "Do not call tools." in prompt
    assert "Do not add new facts." in prompt
    assert "How much did it improve?" in prompt
    assert "internal_trace_marker" in prompt


def _codes(quality: dict) -> set[str]:
    return {issue["code"] for issue in quality["issues"]}
