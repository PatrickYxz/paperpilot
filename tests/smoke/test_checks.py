"""Programmatic checklist evaluation."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from paperpilot.smoke.checks import (
    TurnObservation,
    evaluate_checks,
    extract_citations,
    extract_evidence_ids,
)


def test_extract_evidence_ids_reads_research_result_pool() -> None:
    message = {
        "metadata": {
            "research_result": {
                "evidence_items": [{"id": "evg_1"}, {"id": "evg_2"}],
            }
        }
    }
    assert extract_evidence_ids(message) == ["evg_1", "evg_2"]
    assert extract_evidence_ids({"metadata": {}}) == []
    assert extract_evidence_ids(None) == []


def test_extract_citations_from_nested_metadata() -> None:
    message = {"metadata": {"research_result": {"citations": [{"evidence_id": "e1"}, {"evidence_id": "e2"}]}}}
    assert len(extract_citations(message)) == 2


def test_extract_citations_direct_key_and_missing() -> None:
    assert extract_citations({"metadata": {"citations": [1, 2, 3]}}) == [1, 2, 3]
    assert extract_citations({"metadata": {}}) == []
    assert extract_citations(None) == []


def test_checks_all_pass_on_healthy_outcome(passing_outcome) -> None:
    checks = evaluate_checks(passing_outcome(), context_management=True)
    by_name = {c.name: c for c in checks}
    assert set(by_name) == {
        "task_completed",
        "turns_completed",
        "assistant_answer_present",
        "citations_gte",
        "citations_anchored",
        "context_artifacts_present",
        "no_retry_events",
        "no_secret_leak",
    }
    assert all(c.passed for c in checks)


def test_check_task_failed(passing_outcome) -> None:
    outcome = passing_outcome(task={"status": "failed"})
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["task_completed"]
    assert check.passed is False
    assert "0/1" in check.detail


def test_check_answer_missing(passing_outcome) -> None:
    outcome = passing_outcome(assistant_message={"content": "   "})
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["assistant_answer_present"]
    assert check.passed is False


def test_check_citations_below_expectation(passing_outcome) -> None:
    outcome = passing_outcome(citations=[{"evidence_id": "only-one"}])
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["citations_gte"]
    assert check.passed is False
    assert "1 (expect >= 2)" in check.detail


def test_check_artifacts_only_when_context_management(passing_outcome) -> None:
    outcome = passing_outcome(artifacts=[])
    with_cm = {c.name: c for c in evaluate_checks(outcome, context_management=True)}
    without_cm = {c.name: c for c in evaluate_checks(outcome, context_management=False)}
    assert with_cm["context_artifacts_present"].passed is False
    assert "context_artifacts_present" not in without_cm


def test_check_retry_event_detected(passing_outcome) -> None:
    outcome = passing_outcome(events=[{"type": "task_retry_scheduled"}])
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["no_retry_events"]
    assert check.passed is False


def test_check_secret_leak_detected(passing_outcome) -> None:
    outcome = passing_outcome(
        events=[{"type": "task_started", "detail": "key sk-abcdef0123456789 rejected"}]
    )
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["no_secret_leak"]
    assert check.passed is False


def test_check_secret_scan_ignores_business_ids_like_task_ids(passing_outcome) -> None:
    outcome = passing_outcome(
        task={"id": "task-1", "status": "completed"},
        events=[{"type": "task_started", "task_id": "task-1"}],
    )
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["no_secret_leak"]
    assert check.passed is True


def test_check_harness_error_recorded(passing_outcome) -> None:
    outcome = passing_outcome(error="RuntimeError: boom")
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["no_harness_error"]
    assert check.passed is False


# ---------- 要点覆盖（程序化严格层） ----------


def test_expected_points_covered_when_stated_in_answer(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(scenario=make_scenario(expected_points=("CoNLL-2003",)))
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["expected_points_covered"]
    assert check.passed is True


def test_expected_points_missing_listed_in_detail(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(scenario=make_scenario(expected_points=("CoNLL-2003", "GLUE")))
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["expected_points_covered"]
    assert check.passed is False
    assert "GLUE" in check.detail and "CoNLL-2003" not in check.detail


def test_expected_points_check_absent_without_points(passing_outcome) -> None:
    checks = evaluate_checks(passing_outcome(), context_management=False)
    assert "expected_points_covered" not in {c.name for c in checks}


# ---------- 引用锚定（真实性校验） ----------


def test_citations_anchored_passes_on_healthy_outcome(passing_outcome) -> None:
    check = {
        c.name: c
        for c in evaluate_checks(passing_outcome(), context_management=False)
    }["citations_anchored"]
    assert check.passed is True
    assert "anchored: 3" in check.detail


def test_fabricated_citation_id_fails_and_names_it(passing_outcome) -> None:
    outcome = passing_outcome(
        citations=[{"evidence_id": "e0"}, {"evidence_id": "evg_fabricated"}],
    )
    check = {
        c.name: c
        for c in evaluate_checks(outcome, context_management=False)
    }["citations_anchored"]
    assert check.passed is False
    assert "evg_fabricated (turn 1)" in check.detail


def test_missing_evidence_pool_fails_closed(passing_outcome) -> None:
    outcome = passing_outcome(
        assistant_message={
            "content": "answer with citations",
            "metadata": {"citations": [{"evidence_id": "e0"}]},
        },
    )
    check = {
        c.name: c
        for c in evaluate_checks(outcome, context_management=False)
    }["citations_anchored"]
    assert check.passed is False
    assert "evidence pool missing (turn 1)" in check.detail


def test_no_citations_is_vacuously_anchored(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(
        citations=[], scenario=make_scenario(expect_citations_gte=0)
    )
    check = {
        c.name: c
        for c in evaluate_checks(outcome, context_management=False)
    }["citations_anchored"]
    assert check.passed is True


def test_malformed_citation_flagged(passing_outcome) -> None:
    outcome = passing_outcome(citations=["not-a-dict"])
    check = {
        c.name: c
        for c in evaluate_checks(outcome, context_management=False)
    }["citations_anchored"]
    assert check.passed is False
    assert "<malformed citation>" in check.detail


def test_anchoring_is_per_turn(make_scenario, passing_outcome) -> None:
    first = passing_outcome()
    follow_up = TurnObservation(
        task={"status": "completed"},
        assistant_message={"content": "follow up", "metadata": {}},
        citations=[{"evidence_id": "evg_from_turn2"}],
    )
    outcome = replace(
        first,
        scenario=make_scenario(follow_ups=("q2?",)),
        turns=[first.turns[0], follow_up],
    )
    check = {
        c.name: c
        for c in evaluate_checks(outcome, context_management=False)
    }["citations_anchored"]
    # 第一轮锚定正常；第二轮引用无池 → fail-closed 并标轮次
    assert check.passed is False
    assert "evidence pool missing (turn 2)" in check.detail


def test_real_payload_shape_regression() -> None:
    """2026-10-04 真实载荷形状：池条目键为 id，引用条目键为 evidence_id。"""
    message = {
        "content": "…findings are based on datasets coming from twitter [1][2].",
        "metadata": {
            "citations": [
                {"evidence_id": "evg_4b333aa9624c8b777881e354", "label": "[1]"},
                {"evidence_id": "evg_6e8a1cb7b48711b5b813cb47", "label": "[2]"},
            ],
            "research_result": {
                "evidence_items": [
                    {"id": "evg_6e8a1cb7b48711b5b813cb47", "chunk_text": "…"},
                    {"id": "evg_4b333aa9624c8b777881e354", "chunk_text": "…"},
                ],
                "used_papers": [],
                "limitations": [],
            },
        },
    }
    assert extract_evidence_ids(message) == [
        "evg_6e8a1cb7b48711b5b813cb47",
        "evg_4b333aa9624c8b777881e354",
    ]
    citations = extract_citations(message)
    assert {c["evidence_id"] for c in citations} <= set(extract_evidence_ids(message))


# ---------- 诱导题（trap terms 程序化层） ----------


def test_trap_terms_absent_passes(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(
        scenario=make_scenario(trap_terms=("88.55", "76%")),
    )
    outcome.turns[0].assistant_message = {
        "content": "the paper does not report an ImageNet accuracy",
        "metadata": {"citations": [], "research_result": {"evidence_items": []}},
    }
    outcome.turns[0].citations = []
    check = {
        c.name: c
        for c in evaluate_checks(outcome, context_management=False)
    }["no_trap_terms"]
    assert check.passed is True


def test_trap_term_present_fails_with_term_and_turn(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome(scenario=make_scenario(trap_terms=("CONLL", "88.55")))
    outcome.turns[0].assistant_message["content"] = (
        "the dataset is CoNLL-2003 with 88.55 accuracy"
    )
    check = {
        c.name: c
        for c in evaluate_checks(outcome, context_management=False)
    }["no_trap_terms"]
    assert check.passed is False
    assert "CONLL (turn 1)" in check.detail and "88.55 (turn 1)" in check.detail


def test_trap_check_absent_without_trap_terms(passing_outcome) -> None:
    checks = evaluate_checks(passing_outcome(), context_management=False)
    assert "no_trap_terms" not in {c.name for c in checks}


# ---------- 多轮断言 ----------


def _multi_turn_outcome(make_scenario, passing_outcome) -> Any:
    first = passing_outcome()
    follow_up = TurnObservation(
        task={"status": "completed"},
        assistant_message={"content": "accuracy improved by 2 points", "metadata": {}},
        citations=[{"evidence_id": "f1"}],
        events=[],
        artifacts=[],
    )
    return replace(
        first,
        scenario=make_scenario(follow_ups=("numeric results?",)),
        turns=[first.turns[0], follow_up],
    )


def test_turns_completed_passes_when_all_follow_ups_ran(make_scenario, passing_outcome) -> None:
    outcome = _multi_turn_outcome(make_scenario, passing_outcome)
    checks = {c.name: c for c in evaluate_checks(outcome, context_management=False)}
    assert checks["turns_completed"].passed is True
    assert checks["task_completed"].detail == "completed turns: 2/2"


def test_turns_completed_fails_when_follow_up_missing(make_scenario, passing_outcome) -> None:
    outcome = passing_outcome()
    outcome.scenario = make_scenario(follow_ups=("numeric results?", "limitations?"))
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["turns_completed"]
    assert check.passed is False
    assert "1 (expect 3)" in check.detail


def test_citations_and_points_bind_to_first_turn(make_scenario, passing_outcome) -> None:
    outcome = _multi_turn_outcome(make_scenario, passing_outcome)
    checks = {c.name: c for c in evaluate_checks(outcome, context_management=False)}
    # 第一轮 3 条引用满足 >=2；第二轮仅 1 条不影响判定
    assert checks["citations_gte"].passed is True


def test_turn_error_surfaces_as_harness_error(passing_outcome) -> None:
    outcome = passing_outcome()
    outcome.turns[0].error = "HTTPError: 409 conversation head has changed"
    check = {c.name: c for c in evaluate_checks(outcome, context_management=False)}["no_harness_error"]
    assert check.passed is False
    assert "409" in check.detail


@pytest.mark.parametrize("depth", ["quick", "standard", "deep"])
def test_scenario_depths_accepted(tmp_path, depth: str) -> None:
    from paperpilot.smoke.scenarios import load_scenarios

    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        f'{{"id": "a", "question": "q", "depth": "{depth}", "paper_external_id": "1v1"}}\n',
        encoding="utf-8",
    )
    assert load_scenarios(cases)[0].depth == depth
