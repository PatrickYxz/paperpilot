"""Scenario JSONL loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from paperpilot.smoke.scenarios import ScenarioError, load_scenarios


def test_load_scenarios_parses_rows_and_flags(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "# comment line\n"
        '{"id": "a", "question": "q1", "depth": "quick", "paper_external_id": "1v1", "active": true}\n'
        '{"id": "b", "question": "q2", "depth": "deep", "paper_external_id": "2v1", "expect_citations_gte": 3}\n',
        encoding="utf-8",
    )
    scenarios = load_scenarios(cases)
    assert [s.id for s in scenarios] == ["a", "b"]
    assert scenarios[0].active is True
    assert scenarios[1].active is False
    assert scenarios[1].expect_citations_gte == 3


@pytest.mark.parametrize(
    ("row", "problem"),
    [
        ('{"id": "a", "depth": "quick"}', "missing fields"),
        ('{"id": "a", "question": "q", "depth": "instant", "paper_external_id": "1"}', "depth"),
        ('{"id": "a", "question": "q", "depth": "quick", "paper_external_id": "1"}\n'
         '{"id": "a", "question": "q2", "depth": "quick", "paper_external_id": "1"}', "duplicate"),
        ("{not json}", "invalid JSON"),
    ],
)
def test_load_scenarios_rejects_bad_rows(tmp_path: Path, row: str, problem: str) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match=problem):
        load_scenarios(cases)


def test_load_scenarios_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="not found"):
        load_scenarios(tmp_path / "absent.jsonl")


def test_load_scenarios_parses_expected_points(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"id": "a", "question": "q", "depth": "quick", "paper_external_id": "1v1",'
        ' "expected_points": ["CoNLL-2003", "GLUE"]}\n',
        encoding="utf-8",
    )
    scenario = load_scenarios(cases)[0]
    assert scenario.expected_points == ("CoNLL-2003", "GLUE")


def test_load_scenarios_rejects_non_string_points(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"id": "a", "question": "q", "depth": "quick", "paper_external_id": "1v1",'
        ' "expected_points": ["ok", 42]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="expected_points"):
        load_scenarios(cases)


def test_load_scenarios_parses_follow_ups(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"id": "a", "question": "q", "depth": "standard", "paper_external_id": "1v1",'
        ' "follow_ups": ["numeric results?", "limitations?"]}\n',
        encoding="utf-8",
    )
    scenario = load_scenarios(cases)[0]
    assert scenario.follow_ups == ("numeric results?", "limitations?")


def test_load_scenarios_rejects_bad_follow_ups(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"id": "a", "question": "q", "depth": "quick", "paper_external_id": "1v1",'
        ' "follow_ups": ["ok", ""]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match="follow_ups"):
        load_scenarios(cases)


def test_load_scenarios_parses_tags_refusal_and_traps(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"id": "a", "question": "q", "depth": "quick", "paper_external_id": "1v1",'
        ' "tags": ["adversarial", "honesty"], "expect_refusal": true,'
        ' "trap_terms": ["80.5", "76%"]}\n',
        encoding="utf-8",
    )
    scenario = load_scenarios(cases)[0]
    assert scenario.tags == ("adversarial", "honesty")
    assert scenario.expect_refusal is True
    assert scenario.trap_terms == ("80.5", "76%")


@pytest.mark.parametrize("bad_field", ["tags", "trap_terms"])
def test_load_scenarios_rejects_non_string_tag_lists(tmp_path: Path, bad_field: str) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        '{"id": "a", "question": "q", "depth": "quick", "paper_external_id": "1v1",'
        f' "{bad_field}": ["ok", 3]}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ScenarioError, match=bad_field):
        load_scenarios(cases)


def test_default_cases_file_ships_with_the_package() -> None:
    from paperpilot.smoke.scenarios import DEFAULT_CASES_PATH

    scenarios = load_scenarios(DEFAULT_CASES_PATH)
    assert any(s.active for s in scenarios)
