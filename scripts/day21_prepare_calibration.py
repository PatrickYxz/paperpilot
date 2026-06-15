"""Prepare manual calibration candidates from semantic audit output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_EVAL_DIR = Path("data/eval")
DEFAULT_ENRICHED_PATH = DEFAULT_EVAL_DIR / "qasper_subset_enriched.jsonl"
DEFAULT_AUDIT_PATH = DEFAULT_EVAL_DIR / "semantic_audit_paperpilot_full_20260611.jsonl"
DEFAULT_RESULTS_PATH = DEFAULT_EVAL_DIR / "results_paperpilot.jsonl"
DEFAULT_JSONL_OUT = DEFAULT_EVAL_DIR / "semantic_calibration_candidates_20260614.jsonl"
DEFAULT_MD_OUT = Path("docs/semantic_calibration_candidates_20260614.md")

SEVERE_FALSE_POSITIVE_LABELS = {"incorrect", "contradictory"}
REVIEW_DECISIONS = [
    "TODO",
    "accept_as_correct",
    "keep_partial",
    "downgrade_to_incorrect",
    "judge_error",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        sys.exit(f"FAIL: {path} not found.")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index_by_case_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in rows}


def candidate_category(audit_row: dict[str, Any]) -> str | None:
    """Return the manual-review category for a semantic audit row."""
    semantic_label = str(audit_row.get("semantic_label") or "")
    confidence = str(audit_row.get("confidence") or "")
    strict_pass = bool(audit_row.get("strict_pass"))

    if semantic_label == "partial":
        return f"partial_{confidence or 'unknown'}"
    if strict_pass and semantic_label in SEVERE_FALSE_POSITIVE_LABELS:
        return "strict_pass_semantic_bad"
    return None


def build_candidates(
    *,
    audit_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    enriched_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join semantic audit rows with QASPER metadata and PaperPilot outputs."""
    results_by_id = _index_by_case_id(result_rows)
    enriched_by_id = _index_by_case_id(enriched_rows)

    candidates: list[dict[str, Any]] = []
    for audit_row in audit_rows:
        category = candidate_category(audit_row)
        if category is None:
            continue

        case_id = str(audit_row["case_id"])
        result_row = results_by_id.get(case_id, {})
        enriched_row = enriched_by_id.get(case_id, {})
        candidates.append(
            {
                "case_id": case_id,
                "category": category,
                "review_decision": "TODO",
                "review_notes": "",
                "baseline": audit_row.get("baseline"),
                "strict_pass": audit_row.get("strict_pass"),
                "semantic_label": audit_row.get("semantic_label"),
                "confidence": audit_row.get("confidence"),
                "question": result_row.get("question") or enriched_row.get("question"),
                "oracle_spans": enriched_row.get("oracle_spans")
                or result_row.get("oracle_spans")
                or [],
                "gold_answers": _compact_gold_answers(enriched_row.get("answers") or []),
                "predicted": result_row.get("predicted") or "",
                "judge_reason": audit_row.get("reason") or "",
                "trace_path": result_row.get("trace_path"),
            }
        )

    return sorted(
        candidates,
        key=lambda row: (
            _category_sort_key(str(row["category"])),
            str(row["case_id"]),
        ),
    )


def render_markdown(candidates: list[dict[str, Any]]) -> str:
    """Render a human-readable calibration review document."""
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[str(candidate["category"])] = counts.get(str(candidate["category"]), 0) + 1

    lines: list[str] = [
        "# Semantic Calibration Candidates",
        "",
        "Date: 2026-06-14",
        "",
        "This file mirrors `data/eval/semantic_calibration_candidates_20260614.jsonl`.",
        "Use it to fill manual review decisions, then copy those decisions back into the JSONL.",
        "",
        "Allowed `review_decision` values:",
        "",
    ]
    lines.extend(f"- `{decision}`" for decision in REVIEW_DECISIONS)
    lines.extend(["", "## Summary", ""])
    lines.append(f"- Total candidates: {len(candidates)}")
    for category in sorted(counts, key=_category_sort_key):
        lines.append(f"- {category}: {counts[category]}")

    current_category = None
    index_in_category = 0
    for candidate in candidates:
        category = str(candidate["category"])
        if category != current_category:
            current_category = category
            index_in_category = 0
            lines.extend(["", f"## {category}", ""])
        index_in_category += 1
        lines.extend(_render_candidate(candidate, index_in_category))

    return "\n".join(lines).rstrip() + "\n"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _compact_gold_answers(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for answer in answers:
        compact.append(
            {
                "extractive_spans": answer.get("extractive_spans") or [],
                "free_form_answer": answer.get("free_form_answer") or "",
                "yes_no": answer.get("yes_no"),
                "unanswerable": bool(answer.get("unanswerable")),
                "evidence": answer.get("evidence") or [],
            }
        )
    return compact


def _render_candidate(candidate: dict[str, Any], index: int) -> list[str]:
    predicted = _truncate_block(str(candidate.get("predicted") or ""), max_chars=1600)
    if not predicted:
        predicted = "(empty prediction)"
    return [
        f"### {index}. `{candidate['case_id']}`",
        "",
        f"- Review decision: `{candidate['review_decision']}`",
        f"- Strict pass: `{str(candidate.get('strict_pass')).lower()}`",
        f"- Semantic label: `{candidate.get('semantic_label')}`",
        f"- Confidence: `{candidate.get('confidence')}`",
        "",
        "**Question**",
        "",
        _one_line(candidate.get("question")),
        "",
        "**Oracle spans**",
        "",
        _format_list(candidate.get("oracle_spans") or []),
        "",
        "**PaperPilot predicted excerpt**",
        "",
        "```text",
        predicted.replace("```", "'''"),
        "```",
        "",
        "**Judge reason**",
        "",
        _one_line(candidate.get("judge_reason"), max_chars=900),
        "",
        "**Manual notes**",
        "",
        f"- {_one_line(candidate.get('review_notes'))}",
        "",
    ]


def _format_list(items: list[Any]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- `{_one_line(item, max_chars=240)}`" for item in items[:12])


def _one_line(value: Any, *, max_chars: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ...[truncated]"


def _truncate_block(text: str, *, max_chars: int) -> str:
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 15].rstrip() + "\n...[truncated]"


def _category_sort_key(category: str) -> tuple[int, str]:
    order = {
        "partial_high": 0,
        "partial_medium": 1,
        "partial_low": 2,
        "partial_unknown": 3,
        "strict_pass_semantic_bad": 4,
    }
    return order.get(category, 99), category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--enriched-path", type=Path, default=DEFAULT_ENRICHED_PATH)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    candidates = build_candidates(
        audit_rows=_load_jsonl(args.audit_path),
        result_rows=_load_jsonl(args.results_path),
        enriched_rows=_load_jsonl(args.enriched_path),
    )
    write_jsonl(args.jsonl_out, candidates)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(candidates), encoding="utf-8-sig")
    print(f"Wrote {len(candidates)} candidates to {args.jsonl_out}")
    print(f"Wrote review document to {args.md_out}")


if __name__ == "__main__":
    main()
