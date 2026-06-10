"""Prepare an enriched QASPER subset matching the existing Day 16 subset.

This script does not call MCP or redownload papers. It reads the accepted case
ids from data/eval/qasper_subset.jsonl and writes enriched rows in that order.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.eval.qasper_loader import EnrichedEvalCase, load_qasper_enriched_cases

QASPER_PATH = Path("data/eval/qasper-source/qasper-train-v0.3.json")
SUBSET_PATH = Path("data/eval/qasper_subset.jsonl")
OUT_PATH = Path("data/eval/qasper_subset_enriched.jsonl")


def _load_subset_case_ids(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"FAIL: {path} not found. Run scripts/day16_prepare_eval.py first.")
    case_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_ids.append(str(row["case_id"]))
    return case_ids


def _index_enriched_cases(cases: list[EnrichedEvalCase]) -> dict[str, EnrichedEvalCase]:
    return {case.case_id: case for case in cases}


def _serialize_enriched_case(case: EnrichedEvalCase) -> dict[str, Any]:
    row = asdict(case)
    row["oracle_spans"] = list(case.oracle_spans)
    for answer in row["answers"]:
        answer["extractive_spans"] = list(answer["extractive_spans"])
        answer["evidence"] = list(answer["evidence"])
        answer["highlighted_evidence"] = list(answer["highlighted_evidence"])
    return row


def main() -> None:
    if not QASPER_PATH.exists():
        sys.exit(
            f"FAIL: {QASPER_PATH} not found. "
            "Download QASPER per Day 16 plan first."
        )

    case_ids = _load_subset_case_ids(SUBSET_PATH)
    enriched_by_id = _index_enriched_cases(load_qasper_enriched_cases(QASPER_PATH))
    missing = [case_id for case_id in case_ids if case_id not in enriched_by_id]
    if missing:
        preview = ", ".join(missing[:5])
        sys.exit(f"FAIL: {len(missing)} subset case ids missing from enriched source: {preview}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for case_id in case_ids:
            row = _serialize_enriched_case(enriched_by_id[case_id])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT_PATH} with {len(case_ids)} cases")


if __name__ == "__main__":
    main()
