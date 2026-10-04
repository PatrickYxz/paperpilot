"""Scenario execution and report-row construction."""
from __future__ import annotations

from paperpilot.smoke.adapter import ConversationApi
from paperpilot.smoke.checks import evaluate_checks
from paperpilot.smoke.runner import build_report_row, run_scenario


def test_run_scenario_end_to_end_with_fake_client(fake_client, make_scenario) -> None:
    client = fake_client(
        artifacts=[{"id": 1}],
        assistant={
            "id": "m-a",
            "role": "assistant",
            "content": "CoNLL-2003 is the dataset.",
            "metadata": {
                "citations": [{"evidence_id": "e1"}, {"evidence_id": "e2"}, {"evidence_id": "e3"}],
                "research_result": {
                    "evidence_items": [{"id": f"e{i}"} for i in (1, 2, 3)],
                    "used_papers": [],
                    "limitations": [],
                },
            },
        },
    )
    outcome, checks, elapsed = run_scenario(
        ConversationApi(client), make_scenario(), context_management=True, timeout_s=5
    )
    assert outcome.error == ""
    assert elapsed >= 0
    assert all(c.passed for c in checks), [c.detail for c in checks if not c.passed]


def test_run_scenario_survives_adapter_failure(fake_client, make_scenario) -> None:
    client = fake_client(register_status=503)
    outcome, checks, _ = run_scenario(
        ConversationApi(client), make_scenario(), context_management=False, timeout_s=5
    )
    assert "register failed" in outcome.error
    assert {c.name: c for c in checks}["no_harness_error"].passed is False


def test_report_row_carries_metadata_and_rollup(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome()
    checks = evaluate_checks(outcome, context_management=True)
    row = build_report_row(
        make_scenario(), checks, 12.34, context_management=True, outcome=outcome
    )
    assert row["git_commit"]
    assert row["scenario_id"] == "s1"
    assert row["model"]
    assert row["passed"] is True
    assert row["citations_count"] == 3  # 第一轮（主问题）引用数
    assert row["turns"] == [
        {"status": "completed", "answer_chars": len("the dataset is CoNLL-2003"), "citations_count": 3}
    ]
    assert {c["name"] for c in row["checks"]} == {c.name for c in checks}


def test_report_row_rollup_fails_on_any_failed_check(passing_outcome, make_scenario) -> None:
    outcome = passing_outcome()
    checks = evaluate_checks(outcome, context_management=True)
    checks[0].passed = False
    row = build_report_row(
        make_scenario(), checks, 1.0, context_management=True, outcome=outcome
    )
    assert row["passed"] is False


def test_run_scenario_executes_follow_ups_with_chained_head(fake_client, make_scenario) -> None:
    client = fake_client(
        turns=[
            {
                "final_status": "completed",
                "artifacts": [{"id": 1}],
                "assistant": {
                    "role": "assistant",
                    "content": "the dataset is CoNLL-2003.",
                    "metadata": {
                        "citations": [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
                        "research_result": {
                            "evidence_items": [{"id": "e1"}, {"id": "e2"}],
                            "used_papers": [],
                            "limitations": [],
                        },
                    },
                },
            },
            {
                "final_status": "completed",
                "artifacts": [{"id": 2}],
                "assistant": {
                    "role": "assistant",
                    "content": "accuracy improved by 2 points.",
                    "metadata": {
                        "citations": [{"evidence_id": "e3"}],
                        "research_result": {
                            "evidence_items": [{"id": "e3"}],
                            "used_papers": [],
                            "limitations": [],
                        },
                    },
                },
            },
        ]
    )
    scenario = make_scenario(follow_ups=("What numeric results are reported?",))
    outcome, checks, _ = run_scenario(
        ConversationApi(client), scenario, context_management=True, timeout_s=5
    )
    assert outcome.error == ""
    assert len(outcome.turns) == 2
    by_name = {c.name: c for c in checks}
    assert by_name["turns_completed"].passed is True
    assert by_name["task_completed"].passed is True
    assert by_name["citations_gte"].passed is True  # 第一轮 2 条 >= 2
    message_posts = [c for c in client.calls if c[1].endswith("/messages") and c[0] == "POST"]
    assert len(message_posts) == 2
    assert message_posts[1][2]["expected_head_message_id"] == "m-a-1"


def test_run_scenario_stops_follow_ups_after_failed_turn(fake_client, make_scenario) -> None:
    client = fake_client(
        turns=[{"final_status": "failed", "assistant": None}]
    )
    scenario = make_scenario(follow_ups=("never asked?",))
    outcome, checks, _ = run_scenario(
        ConversationApi(client), scenario, context_management=False, timeout_s=5
    )
    assert len(outcome.turns) == 1
    by_name = {c.name: c for c in checks}
    assert by_name["turns_completed"].passed is False  # 1 轮观测，期望 2
    assert by_name["task_completed"].passed is False
    message_posts = [c for c in client.calls if c[1].endswith("/messages") and c[0] == "POST"]
    assert len(message_posts) == 1  # 追问未发出
