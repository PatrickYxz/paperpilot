"""smoke_compare 契约：回归/改善/新增/移除判定、退出码、容错与输出格式。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paperpilot.smoke.compare import (
    compare_reports,
    load_report_rows,
    main,
    render_text,
)


def _row(
    scenario_id: str,
    *,
    passed: bool = True,
    failed_checks: tuple[str, ...] = (),
    elapsed: float = 10.0,
    citations: int | None = 3,
) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "passed": passed,
        "elapsed_s": elapsed,
        "citations_count": citations,
        "checks": [
            {"name": name, "passed": False, "detail": ""}
            for name in failed_checks
        ],
    }


def _write_report(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_all_pass_same_reports_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_report(tmp_path / "b.jsonl", [_row("s1"), _row("s2")])
    current = _write_report(tmp_path / "c.jsonl", [_row("s1"), _row("s2")])
    assert main([str(baseline), str(current)]) == 0
    out = capsys.readouterr().out
    assert "SAME" in out and "REGRESSION" not in out


def test_regression_exits_one_and_lists_newly_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_report(tmp_path / "b.jsonl", [_row("s1")])
    current = _write_report(
        tmp_path / "c.jsonl", [_row("s1", passed=False, failed_checks=("citations_gte",))]
    )
    assert main([str(baseline), str(current)]) == 1
    out = capsys.readouterr().out
    assert "REGRESSION" in out
    assert "newly failed: citations_gte" in out


def test_improvement_marked_and_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_report(
        tmp_path / "b.jsonl", [_row("s1", passed=False, failed_checks=("task_completed",))]
    )
    current = _write_report(tmp_path / "c.jsonl", [_row("s1")])
    assert main([str(baseline), str(current)]) == 0
    out = capsys.readouterr().out
    assert "IMPROVED" in out
    assert "fixed: task_completed" in out


def test_more_failures_than_baseline_is_regression(tmp_path: Path) -> None:
    baseline = _write_report(
        tmp_path / "b.jsonl", [_row("s1", passed=False, failed_checks=("a",))]
    )
    current = _write_report(
        tmp_path / "c.jsonl", [_row("s1", passed=False, failed_checks=("a", "b"))]
    )
    result = compare_reports(load_report_rows(baseline), load_report_rows(current))
    assert result["entries"][0]["verdict"] == "REGRESSION"
    assert result["summary"]["ok"] is False


def test_same_failure_set_is_same_not_regression(tmp_path: Path) -> None:
    baseline = _write_report(
        tmp_path / "b.jsonl", [_row("s1", passed=False, failed_checks=("a",))]
    )
    current = _write_report(
        tmp_path / "c.jsonl", [_row("s1", passed=False, failed_checks=("a",))]
    )
    result = compare_reports(load_report_rows(baseline), load_report_rows(current))
    assert result["entries"][0]["verdict"] == "SAME"
    assert result["summary"]["ok"] is False  # 仍在失败，只是没有退步


def test_new_and_removed_scenarios(tmp_path: Path) -> None:
    baseline = _write_report(tmp_path / "b.jsonl", [_row("old")])
    current = _write_report(tmp_path / "c.jsonl", [_row("new")])
    result = compare_reports(load_report_rows(baseline), load_report_rows(current))
    verdicts = {e["scenario_id"]: e["verdict"] for e in result["entries"]}
    assert verdicts == {"new": "NEW", "old": "REMOVED"}


def test_old_report_without_new_fields_tolerated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old_row = {"scenario_id": "s1", "passed": True, "elapsed_s": 5.0, "checks": []}
    baseline = _write_report(tmp_path / "b.jsonl", [old_row])
    current = _write_report(tmp_path / "c.jsonl", [_row("s1", citations=7)])
    assert main([str(baseline), str(current)]) == 0
    out = capsys.readouterr().out
    assert "s1" in out  # 缺 citations_count 的旧行渲染为 "-"，不报错


def test_json_output_parseable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_report(tmp_path / "b.jsonl", [_row("s1")])
    current = _write_report(tmp_path / "c.jsonl", [_row("s1")])
    assert main([str(baseline), str(current), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["ok"] is True
    assert result["entries"][0]["scenario_id"] == "s1"


def test_render_text_includes_elapsed_delta_and_citations() -> None:
    result = compare_reports({"s1": _row("s1", elapsed=10.0)}, {"s1": _row("s1", elapsed=12.5, citations=5)})
    text = render_text(result)
    assert "+2.5" in text
    assert "5" in text


def test_missing_report_file_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = _write_report(tmp_path / "b.jsonl", [_row("s1")])
    assert main([str(baseline), str(tmp_path / "absent.jsonl")]) == 2
    assert "not found" in capsys.readouterr().err


def test_empty_report_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no report rows"):
        load_report_rows(empty)


def test_duplicate_scenario_id_rejected(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "dup.jsonl", [_row("s1"), _row("s1")])
    with pytest.raises(ValueError, match="duplicate"):
        load_report_rows(report)
