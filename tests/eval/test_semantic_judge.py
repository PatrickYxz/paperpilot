"""Tests for semantic judge prompt and response parsing."""
import pytest

from paperpilot.eval.semantic_judge import (
    build_judge_prompt,
    create_audit_record,
    parse_judge_response,
)


def _enriched_case() -> dict:
    return {
        "case_id": "qasper-1-q0",
        "paper_title": "A Test Paper",
        "arxiv_id": "1234.56789",
        "question": "What labels are available?",
        "oracle_spans": ["negative", "positive"],
        "full_text": (
            "The dataset has negative and positive labels. "
            "The appendix states: 15 positive words and 15 negative words were used."
        ),
        "answers": [
            {
                "extractive_spans": ["negative", "positive"],
                "free_form_answer": "The labels are negative and positive.",
                "yes_no": None,
                "unanswerable": False,
                "evidence": ["The dataset has negative and positive labels."],
                "highlighted_evidence": ["negative and positive labels"],
            }
        ],
    }


def _result_row() -> dict:
    return {
        "case_id": "qasper-1-q0",
        "baseline": "paperpilot",
        "predicted": "The available labels are negative and positive.",
        "passed": True,
    }


def test_build_judge_prompt_includes_gold_evidence_and_prediction() -> None:
    prompt = build_judge_prompt(_enriched_case(), _result_row())

    assert "What labels are available?" in prompt
    assert "negative and positive labels" in prompt
    assert "The available labels are negative and positive." in prompt
    assert "Strict scorer result (diagnostic only): pass" in prompt
    assert "Paper title: A Test Paper" in prompt
    assert "Additional full-paper context" in prompt
    assert "The dataset has negative and positive labels" in prompt
    assert "Return only JSON" in prompt


def test_build_judge_prompt_includes_full_text_context_for_predicted_quotes() -> None:
    result = _result_row()
    result["predicted"] = 'The answer is "15 positive words and 15 negative words".'

    prompt = build_judge_prompt(_enriched_case(), result)

    assert "15 positive words and 15 negative words were used" in prompt


def test_build_judge_prompt_allows_semantically_matching_extra_detail() -> None:
    prompt = build_judge_prompt(_enriched_case(), _result_row())

    assert "Do not require exact wording" in prompt
    assert "Extra detail is acceptable" in prompt
    assert "not present in the gold evidence" in prompt
    assert "more specific than the gold answer" in prompt
    assert "This is a semantic audit" in prompt
    assert "Do not apply substring" in prompt
    assert "core definition or a direct paraphrase" in prompt
    assert "specific count or list of examples" in prompt


def test_build_judge_prompt_marks_empty_prediction_explicitly() -> None:
    result = _result_row()
    result["predicted"] = ""

    prompt = build_judge_prompt(_enriched_case(), result)

    assert "Predicted answer being judged, verbatim:" in prompt
    assert "```text\n(empty prediction)\n```" in prompt
    assert "Never infer or reconstruct a predicted answer" in prompt


def test_parse_judge_response_accepts_plain_json() -> None:
    parsed = parse_judge_response(
        '{"semantic_label":"correct","confidence":"high","reason":"Matches."}'
    )

    assert parsed == {
        "semantic_label": "correct",
        "confidence": "high",
        "reason": "Matches.",
    }


def test_parse_judge_response_accepts_fenced_json() -> None:
    parsed = parse_judge_response(
        '```json\n{"semantic_label":"partial","confidence":"medium","reason":"Missing one item."}\n```'
    )

    assert parsed["semantic_label"] == "partial"
    assert parsed["confidence"] == "medium"


def test_parse_judge_response_rejects_invalid_label() -> None:
    with pytest.raises(ValueError, match="invalid semantic_label"):
        parse_judge_response(
            '{"semantic_label":"almost","confidence":"high","reason":"No."}'
        )


def test_parse_judge_response_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="invalid confidence"):
        parse_judge_response(
            '{"semantic_label":"correct","confidence":"certain","reason":"No."}'
        )


def test_create_audit_record_uses_legacy_passed_field() -> None:
    record = create_audit_record(
        result_row=_result_row(),
        judge_payload={
            "semantic_label": "correct",
            "confidence": "high",
            "reason": "The prediction matches the gold evidence.",
        },
        audit_model="deepseek-chat",
        audit_version="v1",
    )

    assert record.case_id == "qasper-1-q0"
    assert record.baseline == "paperpilot"
    assert record.strict_pass is True
    assert record.semantic_label == "correct"
    assert record.used_gold_evidence is True
