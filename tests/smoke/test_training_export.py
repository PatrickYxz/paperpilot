"""Training asset export tests (book table 7-4 mapping)."""
from __future__ import annotations

import json

from paperpilot.smoke.training_export import (
    export_training_assets,
    load_run_artifacts,
    write_training_assets,
)


def _outcome(scenario_id, answer, passed, question="What is X?"):
    return {
        "scenario_id": scenario_id,
        "question": question,
        "passed": passed,
        "attribution": None if passed else {"category": "check_failure"},
        "turns": [{"assistant_message": {"content": answer}, "task": {}}],
    }


def test_sft_only_from_passing_answers():
    assets = export_training_assets(
        [_outcome("s1", "good answer", True)],
        [{"scenario_id": "s1", "passed": True}],
    )
    assert len(assets.sft_samples) == 1
    assert assets.sft_samples[0]["messages"][1]["content"] == "good answer"
    assert assets.negative_labels == [] and assets.dpo_pairs == []


def test_negative_label_carries_attribution():
    assets = export_training_assets(
        [_outcome("s2", "wrong", False)],
        [{"scenario_id": "s2", "passed": False, "attribution": {"category": "check_failure"}}],
    )
    assert assets.sft_samples == []
    assert assets.negative_labels[0]["attribution"]["category"] == "check_failure"


def test_dpo_pair_from_same_scenario_pass_and_fail():
    outcomes = [
        _outcome("s3-r1", "failed answer", False),
        _outcome("s3-r2", "correct answer", True),
    ]
    assets = export_training_assets(outcomes, [])
    assert len(assets.dpo_pairs) == 1
    pair = assets.dpo_pairs[0]
    assert pair["chosen"] == "correct answer" and pair["rejected"] == "failed answer"
    assert pair["scenario_id"] == "s3"


def test_roundtrip_files(tmp_path):
    base = tmp_path
    (base / "outcome-s1.json").write_text(json.dumps(
        _outcome("s1", "answer text", True)), encoding="utf-8")
    (base / "smoke_report.jsonl").write_text(
        json.dumps({"scenario_id": "s1", "passed": True}) + "\n", encoding="utf-8")
    outcomes, rows = load_run_artifacts(base)
    assets = export_training_assets(outcomes, rows)
    written = write_training_assets(assets, base / "out")
    sft_lines = (base / "out" / "sft_messages.jsonl").read_text().splitlines()
    assert len(written) == 3 and len(sft_lines) == 1
