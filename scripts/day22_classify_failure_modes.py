"""Classify manually reviewed calibration cases into failure modes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_CANDIDATES_PATH = Path("data/eval/semantic_calibration_candidates_20260614.jsonl")
DEFAULT_MD_OUT = Path("docs/semantic_failure_modes_20260614.md")

DIAGNOSIS_DECISIONS = {"keep_partial", "downgrade_to_incorrect"}
STRICT_BAD_LABELS = {"incorrect", "contradictory"}

FAILURE_MODE_ORDER = [
    "overbroad_scope",
    "missing_required_part",
    "wrong_numeric_or_fact",
    "strict_keyword_false_positive",
    "answer_format_noise",
    "judge_or_gold_ambiguity",
    "unknown",
]

FAILURE_MODE_DESCRIPTIONS = {
    "overbroad_scope": "Answer includes extra methods, datasets, metrics, baselines, or unrelated context beyond the question.",
    "missing_required_part": "Answer covers some gold content but misses at least one required component.",
    "wrong_numeric_or_fact": "Answer contains a wrong number, dataset name, method name, label, or central factual claim.",
    "strict_keyword_false_positive": "Strict scorer passed because of keyword/span overlap, but review says the answer is not fully correct.",
    "answer_format_noise": "Answer is harder to judge because it is noisy, verbose, list-heavy, or mixes reasoning with final output.",
    "judge_or_gold_ambiguity": "The main issue is ambiguous wording, narrow gold spans, or judge over-penalization.",
    "unknown": "The reviewed fields are insufficient for a confident failure-mode assignment.",
}

ENGINEERING_IMPLICATIONS = {
    "overbroad_scope": "Tighten answer synthesis: answer the asked scope first, then isolate optional context.",
    "missing_required_part": "Add explicit checklist coverage for multi-part questions before final synthesis.",
    "wrong_numeric_or_fact": "Add stricter numeric/entity verification against retrieved evidence before final output.",
    "strict_keyword_false_positive": "Do not trust span hits alone; require answer-level semantic consistency for pass labels.",
    "answer_format_noise": "Separate internal reasoning, candidate spans, final answer, and evidence more cleanly.",
    "judge_or_gold_ambiguity": "Keep manual calibration available because some QASPER oracle spans are narrow or underspecified.",
    "unknown": "Needs manual inspection before it can drive product work.",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON rows."""
    if not path.exists():
        sys.exit(f"FAIL: {path} not found.")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def diagnostic_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return manually reviewed cases that still need product diagnosis."""
    return [
        row
        for row in rows
        if str(row.get("review_decision") or "") in DIAGNOSIS_DECISIONS
    ]


def classify_failure_modes(row: dict[str, Any]) -> list[str]:
    """Assign transparent, multi-label failure modes to a reviewed row."""
    text = _review_text(row)
    decision = str(row.get("review_decision") or "")
    modes: list[str] = []

    if _is_strict_keyword_false_positive(row):
        modes.append("strict_keyword_false_positive")

    if _contains_any(
        text,
        [
            "adds separate",
            "adds clustering",
            "adds",
            "extra",
            "blur scope",
            "scope noisy",
            "unrelated",
            "beyond",
            "different pos subtasks",
        ],
    ):
        modes.append("overbroad_scope")

    if _contains_any(
        text,
        [
            "misses",
            "miss ",
            "missing",
            "omits",
            "omit",
            "does not clearly name",
            "does not state",
            "incomplete",
            "not all",
            "some but not all",
        ],
    ):
        modes.append("missing_required_part")

    if _contains_any(
        text,
        [
            "wrong",
            "different numbers",
            "different percentages",
            "percentages differ",
            "gold says",
            "prediction says",
            "prediction answers",
            "contradict",
            "contradicts",
        ],
    ) or (
        decision == "downgrade_to_incorrect"
        and _contains_any(text, ["gold answer is", "instead", "different"])
    ):
        modes.append("wrong_numeric_or_fact")

    if _contains_any(
        text,
        [
            "noisy",
            "list-dumping",
            "verbose",
            "hard to judge",
            "mixed reasoning",
            "garbled",
        ],
    ):
        modes.append("answer_format_noise")

    if _contains_any(
        text,
        [
            "ambiguous",
            "narrow gold",
            "judge over",
            "judge error",
            "oracle too narrow",
        ],
    ):
        modes.append("judge_or_gold_ambiguity")

    if not modes:
        modes.append("unknown")
    return _dedupe_in_order(modes)


def build_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach failure modes to diagnostic cases."""
    failure_rows = []
    for row in diagnostic_cases(rows):
        failure_rows.append(
            {
                "case_id": str(row.get("case_id") or ""),
                "category": str(row.get("category") or ""),
                "review_decision": str(row.get("review_decision") or ""),
                "strict_pass": bool(row.get("strict_pass")),
                "semantic_label": str(row.get("semantic_label") or ""),
                "confidence": str(row.get("confidence") or ""),
                "question": _one_line(row.get("question"), max_chars=180),
                "review_notes": _one_line(row.get("review_notes"), max_chars=260),
                "failure_modes": classify_failure_modes(row),
            }
        )
    return failure_rows


def render_markdown(
    *,
    all_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    source_path: Path,
) -> str:
    """Render a Markdown failure-mode report."""
    mode_counts = _mode_counts(failure_rows)
    decision_counts = Counter(row["review_decision"] for row in failure_rows)

    lines = [
        "# Semantic Failure Modes",
        "",
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"Source: `{source_path}`",
        "",
        "This report classifies manually reviewed non-perfect calibration cases.",
        "It is a diagnosis layer; it does not change the calibrated score.",
        "",
        "## Scope",
        "",
        f"- Reviewed calibration candidates: {len(all_rows)}",
        f"- Diagnostic cases: {len(failure_rows)}",
        f"- Decisions included: `{', '.join(sorted(DIAGNOSIS_DECISIONS))}`",
        "",
        "## Counts By Failure Mode",
        "",
        "| failure_mode | count | meaning | engineering implication |",
        "|---|---:|---|---|",
    ]
    for mode in FAILURE_MODE_ORDER:
        lines.append(
            "| "
            + mode
            + " | "
            + str(mode_counts[mode])
            + " | "
            + FAILURE_MODE_DESCRIPTIONS[mode]
            + " | "
            + ENGINEERING_IMPLICATIONS[mode]
            + " |"
        )

    lines.extend(
        [
            "",
            "## Counts By Review Decision",
            "",
            "| review_decision | count |",
            "|---|---:|",
        ]
    )
    for decision in sorted(decision_counts):
        lines.append(f"| {decision} | {decision_counts[decision]} |")

    lines.extend(
        [
            "",
            "## Case Details",
            "",
            "| case_id | decision | semantic | strict | failure_modes | review_notes |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in failure_rows:
        strict = "pass" if row["strict_pass"] else "fail"
        modes = ", ".join(f"`{mode}`" for mode in row["failure_modes"])
        lines.append(
            "| "
            + f"`{row['case_id']}`"
            + " | "
            + f"`{row['review_decision']}`"
            + " | "
            + f"`{row['semantic_label']}`"
            + " | "
            + strict
            + " | "
            + modes
            + " | "
            + _escape_table(row["review_notes"])
            + " |"
        )

    lines.extend(
        [
            "",
            "## What This Suggests Next",
            "",
            "1. Prioritize answer-scope control before changing retrieval. Several partial cases found the relevant evidence but included extra or differently scoped material.",
            "2. Add a final-answer checklist for questions asking for multiple methods, datasets, components, or metrics.",
            "3. Add stricter numeric/entity verification for percentage, dataset-name, and method-name questions.",
            "4. Keep strict score as a regression guard, but avoid treating it as the only product-quality metric.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _review_text(row: dict[str, Any]) -> str:
    notes = str(row.get("review_notes") or "").strip()
    if notes:
        return notes.lower()
    return str(row.get("judge_reason") or "").lower()


def _is_strict_keyword_false_positive(row: dict[str, Any]) -> bool:
    return bool(row.get("strict_pass")) and str(row.get("semantic_label") or "") in STRICT_BAD_LABELS


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _dedupe_in_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in FAILURE_MODE_ORDER:
        if item in items and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _mode_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["failure_modes"])
    return counts


def _one_line(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ...[truncated]"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-path", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    rows = load_jsonl(args.candidates_path)
    failure_rows = build_failure_rows(rows)
    report = render_markdown(
        all_rows=rows,
        failure_rows=failure_rows,
        source_path=args.candidates_path,
    )
    write_report(args.md_out, report)
    print(f"Wrote {len(failure_rows)} diagnostic cases to {args.md_out}")


if __name__ == "__main__":
    main()
