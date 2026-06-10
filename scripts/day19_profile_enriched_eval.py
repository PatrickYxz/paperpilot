"""Profile the enriched QASPER subset."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IN_PATH = Path("data/eval/qasper_subset_enriched.jsonl")
OUT_PATH = Path("data/eval/qasper_subset_profile.md")


def _question_type(question: str) -> str:
    q = question.strip().lower()
    if re.match(r"^(how many|how much|what percentage|what percent|how big)\b", q):
        return "numeric"
    if re.match(r"^(does|do|did|is|are|was|were|can|could)\b", q):
        return "boolean"
    if re.match(
        r"^(which|what are|what were|what is|what data|what dataset|"
        r"what baselines|what metrics|what methods|what languages|what labels)\b",
        q,
    ):
        return "fact_or_list"
    if re.match(r"^(how|why)\b", q):
        return "mechanism_or_explanation"
    return "other"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        sys.exit(
            f"FAIL: {path} not found. "
            "Run scripts/day19_prepare_enriched_eval.py first."
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _answer_type_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for answer in row.get("answers") or []:
            if answer.get("extractive_spans"):
                counts["extractive"] += 1
            if answer.get("free_form_answer"):
                counts["free_form"] += 1
            if answer.get("yes_no") is not None:
                counts["yes_no"] += 1
            if answer.get("unanswerable"):
                counts["unanswerable"] += 1
            if answer.get("evidence"):
                counts["with_evidence"] += 1
            if answer.get("highlighted_evidence"):
                counts["with_highlighted_evidence"] += 1
    return counts


def _render_profile(rows: list[dict[str, Any]]) -> str:
    qtypes = Counter(_question_type(str(row.get("question", ""))) for row in rows)
    answer_counts = _answer_type_counts(rows)
    answer_total = sum(len(row.get("answers") or []) for row in rows)
    span_counts = [len(row.get("oracle_spans") or []) for row in rows]
    evidence_counts = [
        sum(len(answer.get("evidence") or []) for answer in row.get("answers") or [])
        for row in rows
    ]

    lines: list[str] = []
    lines.append("# QASPER Enriched Subset Profile")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| cases | {len(rows)} |")
    lines.append(f"| answers | {answer_total} |")
    lines.append(f"| max oracle spans per case | {max(span_counts, default=0)} |")
    lines.append(
        f"| max evidence paragraphs per case | {max(evidence_counts, default=0)} |"
    )
    lines.append("")
    lines.append("## Question Types")
    lines.append("")
    lines.append("| type | count |")
    lines.append("|---|---:|")
    for key, count in sorted(qtypes.items()):
        lines.append(f"| {key} | {count} |")
    lines.append("")
    lines.append("## Answer Metadata")
    lines.append("")
    lines.append("| field | answer count |")
    lines.append("|---|---:|")
    for key in [
        "extractive",
        "free_form",
        "yes_no",
        "unanswerable",
        "with_evidence",
        "with_highlighted_evidence",
    ]:
        lines.append(f"| {key} | {answer_counts[key]} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = _load_rows(IN_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(_render_profile(rows), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
