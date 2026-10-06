"""Failure attribution category tests over synthetic event streams."""
from __future__ import annotations

from paperpilot.smoke.attribution import (
    CATEGORY_BUDGET,
    CATEGORY_CHECK,
    CATEGORY_CONTRACT,
    CATEGORY_INFRA,
    CATEGORY_TOOL_LOOP,
    attribute_failure,
)


def _event(kind: str, eid: int, payload: dict | None = None) -> dict:
    return {"id": eid, "type": kind, "payload": payload or {}}


def test_contract_failure_wins_with_decision_evidence():
    events = [
        _event("tool_call", 1),
        _event("research_contract_failure", 2, {
            "error": "selected evidence ID is not in the retrieval ledger: ev-9",
            "decision": {"selected_evidence_ids": ["ev-9"]},
        }),
        _event("failed", 3, {"error_type": "ResearchContractError"}),
    ]
    a = attribute_failure(events)
    assert a.category == CATEGORY_CONTRACT
    assert a.first_error_event_id == 2
    assert "ev-9" in a.summary
    assert a.evidence["selected_evidence_ids"] == ["ev-9"]


def test_terminal_error_type_maps_category():
    for error_type, expected in [
        ("AgentBudgetExceededError", CATEGORY_BUDGET),
        ("OperationalError", CATEGORY_INFRA),
    ]:
        a = attribute_failure([_event("failed", 7, {"error_type": error_type})])
        assert a.category == expected, error_type


def test_repetition_warning_attributes_tool_loop():
    a = attribute_failure([
        _event("tool_call", 1),
        _event("tool_repetition_warning", 5, {
            "name": "retrieve_paper_evidence", "repeats": 3,
            "fingerprint": "retrieve_paper_evidence:{...}",
        }),
    ])
    assert a.category == CATEGORY_TOOL_LOOP
    assert "x3" in a.summary


def test_completed_but_check_failed():
    a = attribute_failure([], failed_checks=["expected_points_covered", "citations_gte"])
    assert a.category == CATEGORY_CHECK
    assert "expected_points_covered" in a.summary


def test_first_error_wins_over_later_signals():
    events = [
        _event("tool_repetition_warning", 4, {"name": "t", "repeats": 3}),
        _event("failed", 9, {"error_type": "ResearchContractError"}),
    ]
    assert attribute_failure(events).first_error_event_id == 4
