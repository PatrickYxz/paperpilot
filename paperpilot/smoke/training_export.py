"""Export smoke run artifacts as chapter-7 style training assets.

Mapping (book table 7-4) implemented here:
- successful scenario answers  -> SFT demonstration messages
- same-scenario success/failure answers -> DPO preference pairs
  (chosen = passing answer, rejected = failing answer)
- failing runs with attribution -> negative-label records for process
  supervision (first-error event id + category travel with the sample)

Inputs are the two files a smoke run already produces per scenario:
outcome-{id}.json (full answers) and smoke_report.jsonl rows
(passed/attribution). This only materializes the bridge — consuming the
assets with a training pipeline is a later, harness-first decision.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExportedTrainingAssets:
    sft_samples: list[dict[str, Any]]
    dpo_pairs: list[dict[str, Any]]
    negative_labels: list[dict[str, Any]]

    def summary(self) -> str:
        return (
            f"sft={len(self.sft_samples)} dpo={len(self.dpo_pairs)} "
            f"negative_labels={len(self.negative_labels)}"
        )


def export_training_assets(
    outcomes: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
) -> ExportedTrainingAssets:
    """outcomes: parsed outcome-{id}.json payloads keyed by scenario;
    report_rows: smoke_report.jsonl rows carrying passed/attribution."""
    rows_by_scenario = {row.get("scenario_id"): row for row in report_rows}
    answers: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        scenario_id = outcome.get("scenario_id", "unknown")
        turns = outcome.get("turns") or []
        if not turns:
            continue
        first = turns[0]
        answer = ((first.get("assistant_message") or {}).get("content") or "").strip()
        question = outcome.get("question") or (first.get("task") or {}).get("question") or ""
        report = rows_by_scenario.get(scenario_id, {})
        normalized_id = re.sub(r"-r\d+$", "", scenario_id)
        answers[f"{normalized_id}#{len(answers)}"] = {
            "scenario_id": normalized_id,
            "question": question,
            "answer": answer,
            "passed": bool(outcome.get("passed", report.get("passed"))),
            "attribution": outcome.get("attribution") or report.get("attribution"),
        }

    sft: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for run_key, run in answers.items():
        scenario_id = run["scenario_id"]
        if run["passed"] and run["answer"]:
            sft.append(
                {
                    "source": "paperpilot-smoke",
                    "scenario_id": scenario_id,
                    "messages": [
                        {"role": "user", "content": run["question"]},
                        {"role": "assistant", "content": run["answer"]},
                    ],
                }
            )
        else:
            negatives.append(
                {
                    "source": "paperpilot-smoke",
                    "scenario_id": scenario_id,
                    "attribution": run["attribution"],
                    "failed_answer": run["answer"][:2000],
                }
            )

    dpo: list[dict[str, Any]] = []
    repeats_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for run in answers.values():
        repeats_by_scenario.setdefault(run["scenario_id"], []).append(run)
    for scenario_id, runs in repeats_by_scenario.items():
        chosen = next((r for r in runs if r["passed"] and r["answer"]), None)
        rejected = next((r for r in runs if not r["passed"] and r["answer"]), None)
        if chosen and rejected and chosen["answer"] != rejected["answer"]:
            dpo.append(
                {
                    "source": "paperpilot-smoke",
                    "scenario_id": scenario_id,
                    "prompt": chosen["question"],
                    "chosen": chosen["answer"],
                    "rejected": rejected["answer"],
                }
            )

    return ExportedTrainingAssets(
        sft_samples=sft, dpo_pairs=dpo, negative_labels=negatives
    )


def load_run_artifacts(base_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load (outcomes, report rows) written by a previous smoke run."""
    outcomes: list[dict[str, Any]] = []
    for path in sorted(base_dir.glob("outcome-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        scenario_id = re.sub(r"-r\d+$", "", path.stem.removeprefix("outcome-"))
        outcomes.append({**data, "scenario_id": scenario_id})
    rows: list[dict[str, Any]] = []
    report_path = base_dir / "smoke_report.jsonl"
    if report_path.is_file():
        for line in report_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return outcomes, rows


def write_training_assets(
    assets: ExportedTrainingAssets, out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, rows in (
        ("sft_messages.jsonl", assets.sft_samples),
        ("dpo_pairs.jsonl", assets.dpo_pairs),
        ("negative_labels.jsonl", assets.negative_labels),
    ):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written.append(path)
    return written
