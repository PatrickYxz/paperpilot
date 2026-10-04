"""Compare two smoke_report.jsonl files and flag regressions.

Usage:
    python -m paperpilot.smoke.compare <baseline.jsonl> <current.jsonl> [--json]

Exit codes: 0 = no regression and all current scenarios passed;
1 = regression or current failure; 2 = input problems. Older reports without
citations_count/answer_chars fields are tolerated and rendered as "-".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_report_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"report not found: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        scenario_id = row.get("scenario_id")
        if not scenario_id:
            raise ValueError(f"{path.name}:{line_no}: row without scenario_id")
        if scenario_id in rows:
            raise ValueError(f"{path.name}:{line_no}: duplicate scenario_id {scenario_id}")
        rows[scenario_id] = row
    if not rows:
        raise ValueError(f"{path.name}: no report rows")
    return rows


def _failed_checks(row: dict[str, Any]) -> list[str]:
    return [
        check["name"]
        for check in row.get("checks", [])
        if not check.get("passed", False)
    ]


def compare_reports(
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    regressions = 0
    improvements = 0
    for scenario_id in sorted(baseline.keys() | current.keys()):
        base = baseline.get(scenario_id)
        cur = current.get(scenario_id)
        if base is None:
            verdict = "NEW"
        elif cur is None:
            verdict = "REMOVED"
        else:
            base_failed = _failed_checks(base)
            cur_failed = _failed_checks(cur)
            if not cur_failed and not base_failed:
                verdict = "SAME"
            elif base_failed and not cur_failed:
                verdict = "IMPROVED"
                improvements += 1
            elif cur_failed and not base_failed:
                verdict = "REGRESSION"
                regressions += 1
            elif set(cur_failed) - set(base_failed):
                verdict = "REGRESSION"
                regressions += 1
            else:
                verdict = "SAME"
        entries.append(
            {
                "scenario_id": scenario_id,
                "verdict": verdict,
                "baseline_passed": (base or {}).get("passed"),
                "current_passed": (cur or {}).get("passed"),
                "baseline_failed_checks": _failed_checks(base) if base else None,
                "current_failed_checks": _failed_checks(cur) if cur else None,
                "baseline_elapsed_s": (base or {}).get("elapsed_s"),
                "current_elapsed_s": (cur or {}).get("elapsed_s"),
                "baseline_citations": (base or {}).get("citations_count"),
                "current_citations": (cur or {}).get("citations_count"),
            }
        )
    failed_now = [e["scenario_id"] for e in entries if e["current_passed"] is False]
    return {
        "entries": entries,
        "summary": {
            "scenarios": len(entries),
            "regressions": regressions,
            "improvements": improvements,
            "failed_now": failed_now,
            "ok": not failed_now and regressions == 0,
        },
    }


def render_text(result: dict[str, Any]) -> str:
    lines: list[str] = []
    header = f"{'scenario':36} {'baseline':9} {'current':9} {'verdict':11} {'Δs':>7} {'citations':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for entry in result["entries"]:
        base_p = _fmt_bool(entry["baseline_passed"])
        cur_p = _fmt_bool(entry["current_passed"])
        elapsed = entry.get("current_elapsed_s")
        base_elapsed = entry.get("baseline_elapsed_s")
        if elapsed is not None and base_elapsed is not None:
            delta = f"{elapsed - base_elapsed:+.1f}"
        else:
            delta = "-"
        citations = entry.get("current_citations")
        citations_text = "-" if citations is None else str(citations)
        lines.append(
            f"{entry['scenario_id'][:36]:36} {base_p:9} {cur_p:9} "
            f"{entry['verdict']:11} {delta:>7} {citations_text:>9}"
        )
        newly_failed = set(entry["current_failed_checks"] or []) - set(
            entry["baseline_failed_checks"] or []
        )
        for name in sorted(newly_failed):
            lines.append(f"    newly failed: {name}")
        for name in sorted(
            set(entry["baseline_failed_checks"] or [])
            - set(entry["current_failed_checks"] or [])
        ):
            lines.append(f"    fixed: {name}")
    summary = result["summary"]
    lines.append("")
    lines.append(
        f"scenarios: {summary['scenarios']}  regressions: {summary['regressions']}  "
        f"improvements: {summary['improvements']}  failed-now: {len(summary['failed_now'])}"
    )
    if summary["failed_now"]:
        lines.append(f"failed scenarios: {summary['failed_now']}")
    return "\n".join(lines)


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "PASS" if value else "FAIL"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m paperpilot.smoke.compare",
        description=__doc__,
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable result")
    args = parser.parse_args(argv)

    try:
        baseline = load_report_rows(args.baseline)
        current = load_report_rows(args.current)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = compare_reports(baseline, current)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_text(result))
    return 0 if result["summary"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
