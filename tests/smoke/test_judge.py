"""LLM-as-judge with injected invokes: parsing, normalization, error folding."""
from __future__ import annotations

import pytest

from paperpilot.smoke.judge import JudgeError, judge_answer, run_judge_check


def _judge_invoke(payload: str):
    return lambda prompt: payload


def test_judge_answer_parses_and_normalizes_verdict() -> None:
    payload = '{"covered": ["p1"], "missing": ["p2"], "reason": "p2 not stated"}'
    verdict = judge_answer(
        "q", ("p1", "p2"), "answer text", model_name="m", invoke=_judge_invoke(payload)
    )
    assert verdict["covered"] == ["p1"]
    assert verdict["missing"] == ["p2"]
    assert verdict["reason"] == "p2 not stated"


def test_judge_answer_rejects_non_json_output() -> None:
    with pytest.raises(JudgeError, match="no JSON"):
        judge_answer(
            "q", ("p1",), "a", model_name="m", invoke=_judge_invoke("I cannot grade.")
        )


def test_judge_answer_rejects_non_list_fields() -> None:
    with pytest.raises(JudgeError, match="covered/missing"):
        judge_answer(
            "q",
            ("p1",),
            "a",
            model_name="m",
            invoke=_judge_invoke('{"covered": "p1", "missing": [], "reason": ""}'),
        )


def test_judge_answer_requires_expected_points() -> None:
    with pytest.raises(JudgeError, match="expected_points"):
        judge_answer("q", (), "a", model_name="m", invoke=_judge_invoke("{}"))


def test_run_judge_check_full_coverage_passes(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(scenario=make_scenario(expected_points=("CoNLL-2003",)))
    payload = '{"covered": ["CoNLL-2003"], "missing": [], "reason": "all stated"}'
    check = run_judge_check(
        outcome.scenario, outcome, model_name="m", invoke=_judge_invoke(payload)
    )
    assert check.name == "judge_points_covered"
    assert check.passed is True


def test_run_judge_check_reports_missing_points(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(scenario=make_scenario(expected_points=("CoNLL-2003", "GLUE")))
    payload = '{"covered": ["CoNLL-2003"], "missing": ["GLUE"], "reason": "GLUE absent"}'
    check = run_judge_check(
        outcome.scenario, outcome, model_name="m", invoke=_judge_invoke(payload)
    )
    assert check.passed is False
    assert "GLUE" in check.detail


def test_run_judge_check_converts_invoke_failure(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(scenario=make_scenario(expected_points=("CoNLL-2003",)))

    def _boom(prompt: str) -> str:
        raise ConnectionError("judge unreachable")

    check = run_judge_check(outcome.scenario, outcome, model_name="m", invoke=_boom)
    assert check.passed is False
    assert "judge error" in check.detail
