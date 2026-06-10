"""Tests for the semantic audit runner script."""
import json
from pathlib import Path

from paperpilot.eval.semantic_audit import SemanticAuditRecord
from scripts.day20_run_semantic_audit import (
    _existing_case_ids,
    run_semantic_audit,
)


class FakeJudge:
    def audit(self, enriched_case: dict, result_row: dict) -> SemanticAuditRecord:
        return SemanticAuditRecord(
            case_id=result_row["case_id"],
            baseline=result_row["baseline"],
            strict_pass=result_row["passed"],
            semantic_label="correct",
            confidence="high",
            reason=f"Reviewed {enriched_case['question']}",
            used_gold_evidence=True,
            audit_model="fake",
            audit_version="v1",
        )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_existing_case_ids_reads_output_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "audit.jsonl"
    _write_jsonl(out, [{"case_id": "case-1"}, {"case_id": "case-2"}])

    assert _existing_case_ids(out) == {"case-1", "case-2"}


def test_run_semantic_audit_writes_joined_rows(tmp_path: Path) -> None:
    enriched = tmp_path / "enriched.jsonl"
    results = tmp_path / "results.jsonl"
    out = tmp_path / "audit.jsonl"
    _write_jsonl(enriched, [
        {"case_id": "case-1", "question": "Q1?", "answers": []},
        {"case_id": "case-2", "question": "Q2?", "answers": []},
    ])
    _write_jsonl(results, [
        {"case_id": "case-1", "baseline": "paperpilot", "predicted": "A1", "passed": True},
        {"case_id": "case-2", "baseline": "paperpilot", "predicted": "A2", "passed": False},
    ])

    count = run_semantic_audit(
        enriched_path=enriched,
        results_path=results,
        out_path=out,
        judge=FakeJudge(),
        baseline="paperpilot",
        limit=None,
        resume=False,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    assert [row["case_id"] for row in rows] == ["case-1", "case-2"]
    assert rows[0]["semantic_label"] == "correct"
    assert rows[1]["strict_pass"] is False


def test_run_semantic_audit_resume_skips_existing_case_ids(tmp_path: Path) -> None:
    enriched = tmp_path / "enriched.jsonl"
    results = tmp_path / "results.jsonl"
    out = tmp_path / "audit.jsonl"
    _write_jsonl(enriched, [
        {"case_id": "case-1", "question": "Q1?", "answers": []},
        {"case_id": "case-2", "question": "Q2?", "answers": []},
    ])
    _write_jsonl(results, [
        {"case_id": "case-1", "baseline": "paperpilot", "predicted": "A1", "passed": True},
        {"case_id": "case-2", "baseline": "paperpilot", "predicted": "A2", "passed": False},
    ])
    _write_jsonl(out, [
        {
            "case_id": "case-1",
            "baseline": "paperpilot",
            "strict_pass": True,
            "semantic_label": "correct",
            "confidence": "high",
            "reason": "Already done.",
            "used_gold_evidence": True,
            "audit_model": "fake",
            "audit_version": "v1",
        }
    ])

    count = run_semantic_audit(
        enriched_path=enriched,
        results_path=results,
        out_path=out,
        judge=FakeJudge(),
        baseline="paperpilot",
        limit=None,
        resume=True,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert count == 1
    assert [row["case_id"] for row in rows] == ["case-1", "case-2"]
