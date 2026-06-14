"""Evaluation snapshot helpers for the Web workbench."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_PATH = Path("data/eval/semantic_audit_paperpilot_full_20260611.jsonl")
DEFAULT_CALIBRATION_PATH = Path(
    "data/eval/semantic_calibration_candidates_20260614.jsonl"
)
MAX_CANDIDATE_TEXT_CHARS = 1400
LABEL_ORDER = [
    "correct",
    "partial",
    "incorrect",
    "contradictory",
    "unverifiable",
    "judge_uncertain",
]
DECISION_ORDER = [
    "accept_as_correct",
    "keep_partial",
    "downgrade_to_incorrect",
    "judge_error",
    "TODO",
]


def build_eval_snapshot(
    *,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
) -> dict[str, Any]:
    """Build a compact eval snapshot for the local Web dashboard."""
    if not audit_path.exists():
        return {
            "available": False,
            "message": f"Audit file not found: {audit_path}",
            "audit_path": str(audit_path),
            "calibration_path": str(calibration_path),
        }

    audit_rows = _load_jsonl(audit_path)
    calibration_rows = _load_jsonl(calibration_path) if calibration_path.exists() else []
    calibration_by_id = {str(row["case_id"]): row for row in calibration_rows}

    total = len(audit_rows)
    strict_pass = sum(1 for row in audit_rows if row.get("strict_pass") is True)
    semantic_counts = Counter(str(row.get("semantic_label") or "") for row in audit_rows)
    calibrated_counts = Counter(
        _calibrated_label(row, calibration_by_id) for row in audit_rows
    )
    decision_counts = Counter(
        str(row.get("review_decision") or "TODO") for row in calibration_rows
    )

    semantic_correct = semantic_counts["correct"]
    semantic_weighted = semantic_counts["correct"] + 0.5 * semantic_counts["partial"]
    calibrated_correct = calibrated_counts["correct"]
    calibrated_weighted = (
        calibrated_counts["correct"] + 0.5 * calibrated_counts["partial"]
    )

    return {
        "available": True,
        "message": "Evaluation snapshot loaded.",
        "audit_path": str(audit_path),
        "calibration_path": str(calibration_path),
        "total_cases": total,
        "strict": {
            "pass_count": strict_pass,
            "rate": _rate(strict_pass, total),
        },
        "semantic": {
            "correct_count": semantic_correct,
            "correct_rate": _rate(semantic_correct, total),
            "weighted_count": semantic_weighted,
            "weighted_rate": _rate(semantic_weighted, total),
            "label_counts": _ordered_counts(semantic_counts, LABEL_ORDER),
        },
        "calibrated": {
            "available": bool(calibration_rows),
            "candidate_count": len(calibration_rows),
            "correct_count": calibrated_correct,
            "correct_rate": _rate(calibrated_correct, total),
            "weighted_count": calibrated_weighted,
            "weighted_rate": _rate(calibrated_weighted, total),
            "label_counts": _ordered_counts(calibrated_counts, LABEL_ORDER),
            "decision_counts": _ordered_counts(decision_counts, DECISION_ORDER),
        },
    }


def list_calibration_candidates(
    *,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    category: str | None = None,
    review_decision: str | None = None,
) -> dict[str, Any]:
    """Return calibration candidates for the Web read-only browser."""
    if not calibration_path.exists():
        return {
            "available": False,
            "message": f"Calibration file not found: {calibration_path}",
            "calibration_path": str(calibration_path),
            "total_candidates": 0,
            "candidates": [],
        }

    rows = _load_jsonl(calibration_path)
    filtered = [
        row
        for row in rows
        if (category is None or row.get("category") == category)
        and (
            review_decision is None
            or row.get("review_decision") == review_decision
        )
    ]

    return {
        "available": True,
        "message": "Calibration candidates loaded.",
        "calibration_path": str(calibration_path),
        "total_candidates": len(rows),
        "filtered_count": len(filtered),
        "candidates": [_candidate_for_web(row) for row in filtered],
    }


def _calibrated_label(
    audit_row: dict[str, Any],
    calibration_by_id: dict[str, dict[str, Any]],
) -> str:
    calibration = calibration_by_id.get(str(audit_row.get("case_id")))
    decision = str((calibration or {}).get("review_decision") or "")
    if decision == "accept_as_correct":
        return "correct"
    if decision == "keep_partial":
        return "partial"
    if decision == "downgrade_to_incorrect":
        return "incorrect"
    return str(audit_row.get("semantic_label") or "")


def _candidate_for_web(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "category": row.get("category"),
        "review_decision": row.get("review_decision"),
        "review_notes": row.get("review_notes") or "",
        "strict_pass": row.get("strict_pass"),
        "semantic_label": row.get("semantic_label"),
        "confidence": row.get("confidence"),
        "question": row.get("question"),
        "oracle_spans": row.get("oracle_spans") or [],
        "predicted_excerpt": _truncate_text(row.get("predicted") or ""),
        "judge_reason": _truncate_text(row.get("judge_reason") or ""),
        "trace_path": row.get("trace_path"),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _rate(count: int | float, total: int) -> float:
    if total == 0:
        return 0.0
    return round(float(count) / total, 4)


def _ordered_counts(counter: Counter[str], order: list[str]) -> dict[str, int]:
    result = {key: counter[key] for key in order if counter[key]}
    for key in sorted(counter):
        if key and key not in result:
            result[key] = counter[key]
    return result


def _truncate_text(value: str, *, max_chars: int = MAX_CANDIDATE_TEXT_CHARS) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"
