"""Summarize semantic audit JSONL output."""
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

DEFAULT_EVAL_DIR = Path("data/eval")
LABEL_ORDER = [
    "correct",
    "partial",
    "incorrect",
    "contradictory",
    "unverifiable",
    "judge_uncertain",
]
FALSE_POSITIVE_LABELS = {"partial", "incorrect", "contradictory"}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        sys.exit(f"FAIL: {path} not found. Run semantic audit first.")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _label_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("semantic_label", "")) for row in rows)


def _matrix_counts(rows: list[dict[str, Any]]) -> Counter[tuple[bool, str]]:
    return Counter(
        (bool(row.get("strict_pass")), str(row.get("semantic_label", "")))
        for row in rows
    )


def _render_summary(rows: list[dict[str, Any]], *, baseline: str) -> str:
    label_counts = _label_counts(rows)
    matrix = _matrix_counts(rows)
    false_positives = sum(
        1
        for row in rows
        if row.get("strict_pass") is True
        and row.get("semantic_label") in FALSE_POSITIVE_LABELS
    )
    false_negatives = sum(
        1
        for row in rows
        if row.get("strict_pass") is False and row.get("semantic_label") == "correct"
    )
    uncertain = label_counts["judge_uncertain"]

    lines: list[str] = []
    lines.append("# Semantic Audit Summary")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**Baseline**: `{baseline}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| total audited cases | {len(rows)} |")
    lines.append(f"| estimated strict false positives | {false_positives} |")
    lines.append(f"| estimated strict false negatives | {false_negatives} |")
    lines.append(f"| judge uncertain | {uncertain} |")
    lines.append("")
    lines.append("## Semantic Labels")
    lines.append("")
    lines.append("| label | count |")
    lines.append("|---|---:|")
    for label in LABEL_ORDER:
        lines.append(f"| {label} | {label_counts[label]} |")
    lines.append("")
    lines.append("## Strict Score X Semantic Label")
    lines.append("")
    lines.append("| strict | semantic_label | count |")
    lines.append("|---|---|---:|")
    for strict_pass in [True, False]:
        strict_name = "pass" if strict_pass else "fail"
        for label in LABEL_ORDER:
            count = matrix[(strict_pass, label)]
            if count:
                lines.append(f"| {strict_name} | {label} | {count} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="paperpilot")
    parser.add_argument("--audit-path", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    audit_path = args.audit_path or DEFAULT_EVAL_DIR / f"semantic_audit_{args.baseline}.jsonl"
    out_path = args.out_path or DEFAULT_EVAL_DIR / "semantic_audit_summary.md"
    rows = _load_rows(audit_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_summary(rows, baseline=args.baseline), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
