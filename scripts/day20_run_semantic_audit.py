"""Run semantic audits over QASPER baseline result rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Protocol

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from paperpilot.eval.semantic_audit import SemanticAuditRecord
from paperpilot.eval.semantic_judge import SemanticJudge

load_dotenv()

DEFAULT_ENRICHED_PATH = Path("data/eval/qasper_subset_enriched.jsonl")
DEFAULT_EVAL_DIR = Path("data/eval")


class Judge(Protocol):
    def audit(
        self,
        enriched_case: dict[str, Any],
        result_row: dict[str, Any],
    ) -> SemanticAuditRecord:
        ...


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        sys.exit(f"FAIL: {path} not found.")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _existing_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ids.add(str(json.loads(line)["case_id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def _index_by_case_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["case_id"]): row for row in rows}


def run_semantic_audit(
    *,
    enriched_path: Path,
    results_path: Path,
    out_path: Path,
    judge: Judge,
    baseline: str,
    limit: int | None,
    resume: bool,
) -> int:
    """Join enriched cases with result rows and append semantic audit records."""
    enriched_by_id = _index_by_case_id(_load_jsonl(enriched_path))
    result_rows = [
        row for row in _load_jsonl(results_path) if row.get("baseline") == baseline
    ]
    done = _existing_case_ids(out_path) if resume else set()
    todo = [row for row in result_rows if str(row["case_id"]) not in done]
    if limit is not None:
        todo = todo[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with out_path.open("a", encoding="utf-8") as f:
        for idx, result_row in enumerate(todo, start=1):
            case_id = str(result_row["case_id"])
            if case_id not in enriched_by_id:
                raise ValueError(f"result case_id missing from enriched subset: {case_id}")
            record = judge.audit(enriched_by_id[case_id], result_row)
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            completed += 1
            print(f"[{idx}/{len(todo)}] audited {case_id}: {record.semantic_label}")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="paperpilot")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enriched-path", type=Path, default=DEFAULT_ENRICHED_PATH)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    results_path = args.results_path or DEFAULT_EVAL_DIR / f"results_{args.baseline}.jsonl"
    out_path = args.out_path or DEFAULT_EVAL_DIR / f"semantic_audit_{args.baseline}.jsonl"
    judge = SemanticJudge()
    completed = run_semantic_audit(
        enriched_path=args.enriched_path,
        results_path=results_path,
        out_path=out_path,
        judge=judge,
        baseline=args.baseline,
        limit=args.limit,
        resume=args.resume,
    )
    print(f"Done. Wrote {completed} audit rows to {out_path}")


if __name__ == "__main__":
    main()
