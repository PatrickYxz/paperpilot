"""Command-line entry point for the real-model smoke harness."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from paperpilot.smoke.adapter import ConversationApi
from paperpilot.smoke.judge import run_judge_check
from paperpilot.smoke.runner import build_report_row, run_scenario
from paperpilot.smoke.stats import summarize_repeats
from paperpilot.smoke.runtime import build_isolated_app
from paperpilot.smoke.scenarios import DEFAULT_CASES_PATH, ScenarioError, load_scenarios


def _require_credentials() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit(
            "refusing to run: DEEPSEEK_API_KEY is not set. Real-model runs are a "
            "manual gate (see README); use --dry-run to inspect the plan."
        )
    for optional in ("DASHSCOPE_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"):
        if not os.environ.get(optional):
            print(f"warning: {optional} is not set; some MCP paths may fail", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m paperpilot.smoke",
        description=(
            "Real-model E2E smoke harness: isolated real instance, scenario "
            "table, programmatic checks, JSONL report for cross-run comparison."
        ),
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--only", action="append", default=None, help="scenario id (repeatable)")
    parser.add_argument("--all", action="store_true", help="run inactive scenarios too")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=600.0, help="per-scenario seconds")
    parser.add_argument("--context-management", action="store_true")
    parser.add_argument("--model", default=None, help="override PAPERPILOT_RESEARCH_MODEL_NAME")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="LLM-as-judge scoring for scenarios with expected_points",
    )
    parser.add_argument("--judge-model", default=None, help="model for --judge")
    parser.add_argument(
        "--export-training",
        action="store_true",
        help="after the run, export SFT/DPO/negative-label assets from outcomes",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="run the feature ablation matrix on selected scenarios",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run each scenario N times; report Pass^k / Pass@k reliability",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        scenarios = load_scenarios(args.cases)
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.only:
        by_id = {s.id: s for s in scenarios}
        missing = [i for i in args.only if i not in by_id]
        if missing:
            print(f"error: unknown scenario ids: {missing}", file=sys.stderr)
            return 2
        selected = [by_id[i] for i in args.only]
    else:
        selected = [s for s in scenarios if s.active or args.all]

    if not selected:
        print("no scenarios selected (all inactive; use --all or --only)", file=sys.stderr)
        return 2

    base_dir = args.base_dir or Path(
        f"/tmp/paperpilot-smoke-{os.getpid()}-{int(time.time())}"
    )

    print(f"scenarios: {[s.id for s in selected]}")
    print(f"base dir: {base_dir}")
    print(f"context management: {args.context_management}")
    print(f"model: {args.model or os.environ.get('PAPERPILOT_RESEARCH_MODEL_NAME', '<env-default>')}")
    print(f"judge: {args.judge} ({args.judge_model or '<research model>'})")
    if args.dry_run:
        print("dry run: no instance started, no credentials required")
        return 0

    _require_credentials()
    base_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_id = {
        s.paper_external_id: s.paper_metadata
        for s in selected
        if s.paper_metadata
    }
    client = build_isolated_app(
        base_dir,
        context_management=args.context_management,
        model=args.model,
        metadata_by_external_id=metadata_by_id,
    )

    report_path = base_dir / "smoke_report.jsonl"
    rows: list[dict[str, Any]] = []
    with client:
        api = ConversationApi(client)
        if args.ablate:
            return _run_ablate_mode(args, api, selected, base_dir)
        for index, scenario in enumerate(selected, start=1):
            print(
                f"[{index}/{len(selected)}] {scenario.id} :: "
                f"{scenario.question[:80]}"
                + (f" (x{args.repeat})" if args.repeat > 1 else "")
            )
            repeat_passed: list[bool] = []
            outcome = checks = None
            elapsed = 0.0
            for _rep in range(max(1, args.repeat)):
                outcome, checks, elapsed = run_scenario(
                    api,
                    scenario,
                    context_management=args.context_management,
                    timeout_s=args.timeout,
                )
                if args.judge and scenario.expected_points:
                    checks.append(
                        run_judge_check(
                            scenario,
                            outcome,
                            model_name=(
                                args.judge_model
                                or os.environ.get(
                                    "PAPERPILOT_RESEARCH_MODEL_NAME",
                                    "deepseek-v4-flash",
                                )
                            ),
                        )
                    )
                rep_passed = all(check.passed for check in checks)
                repeat_passed.append(rep_passed)
                raw_path = base_dir / (
                    f"outcome-{scenario.id}"
                    + (f"-r{len(repeat_passed)}" if args.repeat > 1 else "")
                    + ".json"
                )
                raw_path.write_text(
                    json.dumps(
                        {
                            "scenario_id": scenario.id,
                            "question": scenario.question,
                            "passed": rep_passed,
                            "attribution": getattr(outcome, "attribution", None),
                            "turns": [
                                {
                                    "task": turn.task,
                                    "assistant_message": turn.assistant_message,
                                    "citations": turn.citations,
                                    "events": turn.events,
                                    "artifacts": turn.artifacts,
                                    "error": turn.error,
                                }
                                for turn in outcome.turns
                            ],
                            "error": outcome.error,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                if not rep_passed:
                    print(f"       rep {len(repeat_passed)} failed")
                    for check in checks:
                        if not check.passed:
                            print(f"       FAIL {check.name}: {check.detail}")
                    attribution = getattr(outcome, "attribution", None)
                    if attribution:
                        print(
                            f"       attribution[{attribution['category']}]: "
                            f"{attribution['summary']}"
                        )
            row = build_report_row(
                scenario,
                checks,
                elapsed,
                context_management=args.context_management,
                outcome=outcome,
            )
            if args.repeat > 1:
                row["repeats"] = summarize_repeats(repeat_passed)
                print(
                    f"    reliability: Pass^{args.repeat}="
                    f"{row['repeats']['pass_k_consecutive']} "
                    f"Pass@{args.repeat}={row['repeats']['pass_at_k']} "
                    f"rate={row['repeats']['single_pass_rate']:.2f} "
                    f"(95% CI {row['repeats']['wilson95_low']:.2f}-"
                    f"{row['repeats']['wilson95_high']:.2f})"
                )
            rows.append(row)
            verdict = "PASS" if row["passed"] else "FAIL"
            print(f"    -> {verdict} ({row['elapsed_s']}s) {row['status']}")


    if args.export_training:
        from paperpilot.smoke.training_export import (
            export_training_assets,
            load_run_artifacts,
            write_training_assets,
        )

        outcomes, report_rows = load_run_artifacts(base_dir)
        assets = export_training_assets(outcomes, report_rows)
        written = write_training_assets(assets, base_dir / "training_assets")
        print(f"training assets ({assets.summary()}): {[p.name for p in written]}")

    with report_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    failed = [row["scenario_id"] for row in rows if not row["passed"]]
    print(f"report: {report_path}")
    if failed:
        print(f"FAILED scenarios: {failed}")
        return 1
    print(f"all {len(rows)} scenarios passed")
    return 0


def _count_tokens(outcome) -> int:
    total = 0
    for turn in outcome.turns:
        for event in turn.events:
            if event.get("type") != "model_usage":
                continue
            payload = event.get("payload") or event.get("payload_json") or {}
            if isinstance(payload, str):
                import json as _json

                try:
                    payload = _json.loads(payload)
                    payload = payload or {}
                except (TypeError, ValueError):
                    payload = {}
            total += int(payload.get("input_tokens", 0) or 0)
            total += int(payload.get("output_tokens", 0) or 0)
    return total


def _run_ablate_mode(args, api, selected, base_dir) -> int:
    from paperpilot.smoke.ablation import (
        render_ablation_table,
        run_ablation_matrix,
    )

    by_id = {scenario.id: scenario for scenario in selected}

    def make_runner(scenario):
        def _run_one():
            outcome, checks, _elapsed = run_scenario(
                api,
                scenario,
                context_management=args.context_management,
                timeout_s=args.timeout,
            )
            return all(check.passed for check in checks), _count_tokens(outcome)

        return _run_one

    results = run_ablation_matrix(
        [scenario.id for scenario in selected],
        lambda scenario_id: make_runner(by_id[scenario_id]),
    )
    print()
    print(render_ablation_table(results))
    ablation_path = base_dir / "ablation_report.json"
    ablation_path.write_text(json.dumps(results, indent=2))
    print(f"ablation report: {ablation_path}")
    return 0 if results and results[0]["passes"] > 0 else 1
