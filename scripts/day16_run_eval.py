"""Day 16 run: execute one baseline, or all baselines, over qasper_subset.jsonl."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from paperpilot.eval.baselines import (
    run_abstract_only,
    run_full_text_dump,
    run_paperpilot,
)
from paperpilot.eval.qasper_loader import EvalCase
from paperpilot.eval.scorer import is_pass

SUBSET_PATH = Path("data/eval/qasper_subset.jsonl")
OUT_DIR = Path("data/eval")

BASELINE_FNS: dict[str, Callable[[EvalCase], dict]] = {
    "abstract_only": run_abstract_only,
    "full_text": run_full_text_dump,
    "paperpilot": run_paperpilot,
    "paperpilot_query_plan_v1": lambda case: run_paperpilot(
        case,
        use_query_plan=True,
        trace_id=f"{case.case_id}__query_plan_v1",
    ),
}


def _load_cases() -> list[EvalCase]:
    if not SUBSET_PATH.exists():
        sys.exit(f"FAIL: {SUBSET_PATH} not found. Run day16_prepare_eval.py first.")

    cases: list[EvalCase] = []
    for line in SUBSET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        cases.append(EvalCase(
            case_id=data["case_id"],
            arxiv_id=data["arxiv_id"],
            paper_title=data["paper_title"],
            abstract=data["abstract"],
            full_text=data["full_text"],
            question=data["question"],
            oracle_spans=tuple(data["oracle_spans"]),
        ))
    return cases


def _existing_case_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()

    ids: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ids.add(json.loads(line)["case_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def run_one_baseline(baseline: str, cases: list[EvalCase], limit: int | None) -> None:
    fn = BASELINE_FNS[baseline]
    out_path = OUT_DIR / f"results_{baseline}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_case_ids(out_path)
    print(f"\n=== Baseline: {baseline} ===")
    print(f"  output: {out_path}")
    print(f"  total cases: {len(cases)}, already done: {len(done)}")

    todo = [case for case in cases if case.case_id not in done]
    if limit is not None:
        todo = todo[:limit]

    for i, case in enumerate(todo, 1):
        print(f"\n  [{i}/{len(todo)}] {case.case_id} :: {case.question[:60]}...")
        ans = fn(case)
        passed = is_pass(ans["predicted"], case.oracle_spans)
        record = {
            "case_id": case.case_id,
            "baseline": baseline,
            "question": case.question,
            "oracle_spans": list(case.oracle_spans),
            "predicted": ans["predicted"],
            "passed": passed,
            "elapsed_s": ans["elapsed_s"],
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
            "query_plan_used",
            "query_plan_version",
            "query_plan",
        ]:
            if extra_key in ans:
                record[extra_key] = ans[extra_key]
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        status = "PASS" if passed else ("ERR" if ans.get("error") else "FAIL")
        print(f"    -> {status} ({ans['elapsed_s']}s) predicted: {ans['predicted'][:120]}...")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        choices=["abstract_only", "full_text", "paperpilot", "paperpilot_query_plan_v1", "all"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = _load_cases()
    print(f"Loaded {len(cases)} cases from {SUBSET_PATH}")

    baselines = list(BASELINE_FNS) if args.baseline == "all" else [args.baseline]
    for baseline in baselines:
        run_one_baseline(baseline, cases, args.limit)
    print("\nDone.")


if __name__ == "__main__":
    main()
