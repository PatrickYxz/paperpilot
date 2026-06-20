"""Print deterministic query plans for selected eval cases."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.eval.query_planner import plan_queries


DEFAULT_CASES_PATH = Path("data/eval/qasper_subset_enriched.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = _load_rows(args.cases_path)
    if args.case_ids:
        wanted = set(args.case_ids)
        rows = [row for row in rows if row.get("case_id") in wanted]
        missing = [case_id for case_id in args.case_ids if not any(row.get("case_id") == case_id for row in rows)]
        if missing:
            raise SystemExit(f"Missing cases: {missing}")
    if args.limit is not None:
        rows = rows[:args.limit]

    for row in rows:
        plan = plan_queries(
            str(row.get("question") or ""),
            title=str(row.get("paper_title") or ""),
            abstract=str(row.get("abstract") or ""),
        )
        print(json.dumps({
            "case_id": row.get("case_id"),
            "arxiv_id": row.get("arxiv_id"),
            "question": row.get("question"),
            "plan": plan.to_dict(),
        }, ensure_ascii=False))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
