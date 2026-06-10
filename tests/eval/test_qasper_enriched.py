"""Tests for enriched QASPER eval loading."""
from dataclasses import asdict
from pathlib import Path

from paperpilot.eval.qasper_loader import (
    EnrichedEvalCase,
    QasperAnswer,
    load_qasper_enriched_cases,
)

FIXTURE = Path(__file__).parent / "fixtures" / "qasper_mini.json"


def test_load_qasper_enriched_preserves_answer_metadata() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)
    q1 = next(c for c in cases if c.case_id == "qasper-2001.12345-q0")

    assert isinstance(q1, EnrichedEvalCase)
    assert q1.oracle_spans == ("span-1a",)
    assert len(q1.answers) == 1

    answer = q1.answers[0]
    assert isinstance(answer, QasperAnswer)
    assert answer.annotation_id == "q1-a1"
    assert answer.extractive_spans == ("span-1a",)
    assert answer.free_form_answer == "Free answer 1."
    assert answer.yes_no is None
    assert answer.unanswerable is False
    assert answer.evidence == ("Evidence paragraph 1.",)
    assert answer.highlighted_evidence == ("Highlighted sentence 1.",)


def test_load_qasper_enriched_preserves_multiple_answers() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)
    q3 = next(c for c in cases if c.case_id == "qasper-2001.12345-q2")

    assert q3.oracle_spans == ("span-3a", "span-3b")
    assert len(q3.answers) == 2
    assert q3.answers[0].extractive_spans == ("span-3a",)
    assert q3.answers[1].extractive_spans == ("span-3b",)
    assert q3.answers[1].annotation_id == "q3-a2"
    assert q3.answers[1].evidence == ("Evidence paragraph 3b.",)


def test_load_qasper_enriched_matches_existing_filter_policy() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)

    assert {c.arxiv_id for c in cases} == {"2001.12345"}
    assert [c.case_id for c in cases] == [
        "qasper-2001.12345-q0",
        "qasper-2001.12345-q1",
        "qasper-2001.12345-q2",
    ]


def test_enriched_case_is_json_serializable_via_asdict() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)
    row = asdict(cases[0])

    assert row["case_id"] == "qasper-2001.12345-q0"
    assert row["oracle_spans"] == ("span-1a",)
    assert row["answers"][0]["annotation_id"] == "q1-a1"
    assert row["answers"][0]["extractive_spans"] == ("span-1a",)
