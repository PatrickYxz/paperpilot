"""Rerun selected PaperPilot eval cases and write fresh traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import paperpilot.eval.baselines as baselines
from paperpilot.eval.qasper_loader import EvalCase
from paperpilot.eval.scorer import is_pass


DEFAULT_SUBSET_PATH = Path("data/eval/qasper_subset.jsonl")
DEFAULT_OUT_PATH = Path("data/eval/paperpilot_rerun_cases_20260616.jsonl")
DEFAULT_TRACE_DIR = Path("data/traces_rerun_20260616")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-path", type=Path, default=DEFAULT_SUBSET_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--case-id", action="append", dest="case_ids", required=True)
    args = parser.parse_args()

    rows = _load_rows(args.subset_path)
    missing = [case_id for case_id in args.case_ids if case_id not in rows]
    if missing:
        raise SystemExit(f"Missing cases: {missing}")

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text("", encoding="utf-8")

    baselines._TRACE_DIR = args.trace_dir

    for idx, case_id in enumerate(args.case_ids, start=1):
        case = _to_eval_case(rows[case_id])
        print(f"[{idx}/{len(args.case_ids)}] {case.case_id} :: {case.question[:90]}", flush=True)
        ans = baselines.run_paperpilot(case)
        record = _record(case, ans)
        with args.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        status = "PASS" if record["passed"] else ("ERR" if record["error"] else "FAIL")
        print(
            f"    -> {status} ({record['elapsed_s']}s) "
            f"trace={record['trace_path']} predicted={record['predicted'][:160]}",
            flush=True,
        )

    print(f"Wrote {args.out_path}")
    print(f"Trace dir: {args.trace_dir}")


def _load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["case_id"])] = row
    return rows


def _to_eval_case(row: dict) -> EvalCase:
    return EvalCase(
        case_id=row["case_id"],
        arxiv_id=row["arxiv_id"],
        paper_title=row["paper_title"],
        abstract=row["abstract"],
        full_text=row["full_text"],
        question=row["question"],
        oracle_spans=tuple(row["oracle_spans"]),
    )


def _record(case: EvalCase, ans: dict) -> dict:
    record = {
        "case_id": case.case_id,
        "baseline": "paperpilot_rerun_cases",
        "question": case.question,
        "oracle_spans": list(case.oracle_spans),
        "predicted": ans.get("predicted", ""),
        "passed": is_pass(ans.get("predicted", ""), case.oracle_spans),
        "elapsed_s": ans.get("elapsed_s"),
        "trace_path": ans.get("trace_path"),
        "tool_calls": ans.get("tool_calls"),
        "error": ans.get("error"),
    }
    for extra_key in [
        "predicted_raw",
        "answer_quality",
        "answer_repaired",
        "repair_answer_quality",
        "repair_error",
        "evidence_selection",
        "evidence_rewritten",
        "evidence_selection_error",
    ]:
        if extra_key in ans:
            record[extra_key] = ans[extra_key]
    return record


if __name__ == "__main__":
    main()
