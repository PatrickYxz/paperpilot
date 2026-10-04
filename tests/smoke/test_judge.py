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


# ---------- 拒绝判定（诱导题） ----------


def _refusal_invoke(payload: str):
    return lambda prompt: payload


def test_refusal_judge_declines_passes(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(
        scenario=make_scenario(expect_refusal=True, expected_points=()),
    )
    payload = '{"verdict": "declines", "reason": "answer states no such metric is reported"}'
    check = run_judge_check(
        outcome.scenario, outcome, model_name="m", invoke=_refusal_invoke(payload)
    )
    assert check.name == "judge_declines"
    assert check.passed is True


def test_refusal_judge_fabricates_fails(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(
        scenario=make_scenario(expect_refusal=True, expected_points=()),
    )
    payload = '{"verdict": "fabricates", "reason": "answer invents 76.4 top-1"}'
    check = run_judge_check(
        outcome.scenario, outcome, model_name="m", invoke=_refusal_invoke(payload)
    )
    assert check.passed is False
    assert "fabricates" in check.detail


def test_refusal_judge_rejects_bad_verdict(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(
        scenario=make_scenario(expect_refusal=True, expected_points=()),
    )
    with pytest.raises(JudgeError, match="declines/fabricates"):
        judge_answer(
            "q",
            (),
            "a",
            model_name="m",
            invoke=_refusal_invoke('{"verdict": "maybe"}'),
            refusal=True,
        )


def test_refusal_judge_needs_no_points(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(
        scenario=make_scenario(expect_refusal=True, expected_points=()),
    )
    payload = '{"verdict": "declines", "reason": "ok"}'
    check = run_judge_check(
        outcome.scenario, outcome, model_name="m", invoke=_refusal_invoke(payload)
    )
    assert check.passed is True  # 无 expected_points 的拒绝判定合法


# ---------- 真实构造路径（2026-10-04 真跑暴露的签名 bug 回归） ----------


def test_real_judge_invoke_builds_model_with_keyword_args(monkeypatch) -> None:
    import paperpilot.deep_reading.runner as runner_module
    from paperpilot.smoke.judge import _build_judge_invoke

    calls: list[dict] = []

    class _FakeModel:
        def invoke(self, messages):
            calls.append({"messages": [type(m).__name__ for m in messages]})
            return type("R", (), {"content": '{"covered": [], "missing": ["p"], "reason": "x"}'})()

    def _recorder(**kwargs):
        calls.append(kwargs)
        return _FakeModel()

    monkeypatch.setattr(runner_module, "build_deep_reading_model", _recorder)
    invoke = _build_judge_invoke("judge-model")
    raw = invoke("grade this")

    assert calls[0] == {
        "model_name": "judge-model",
        "research_max_output_tokens": 2048,
    }
    assert "HumanMessage" in calls[1]["messages"]
    assert "missing" in raw
