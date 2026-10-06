"""Ablation matrix runner tests with a scripted run function."""
from __future__ import annotations

import os

from paperpilot.smoke.ablation import (
    ablation_configs,
    render_ablation_table,
    run_ablation_matrix,
)


def test_configs_cover_baseline_plus_four_features():
    labels = [label for label, _ in ablation_configs()]
    assert labels == [
        "baseline",
        "memory_tool_off",
        "profile_off",
        "computation_off",
        "dense_retrieval",
    ]


def test_matrix_runs_all_configs_and_restores_env():
    calls: list[tuple[str, dict]] = []

    def run_one(scenario_id):
        def _execute():
            snapshot = {
                f: os.environ.get(f)
                for f in (
                    "PAPERPILOT_MEMORY_TOOL_ENABLED",
                    "PAPERPILOT_RETRIEVAL_MODE",
                )
            }
            calls.append((scenario_id, snapshot))
            return (scenario_id != "hard", 100)

        return _execute

    before = dict(os.environ)
    results = run_ablation_matrix(["easy", "hard"], run_one)
    assert dict(os.environ) == before, "env must be restored"

    assert len(results) == 5
    baseline = results[0]
    assert baseline["passes"] == 1 and baseline["pass_rate"] == 0.5
    # off configs captured the override
    memory_off_calls = calls[2:4]
    assert all(
        snap["PAPERPILOT_MEMORY_TOOL_ENABLED"] == "off"
        for _, snap in memory_off_calls
    )
    dense_calls = calls[-2:]
    assert all(snap["PAPERPILOT_RETRIEVAL_MODE"] == "dense" for _, snap in dense_calls)
    # deltas attached
    assert all("pass_rate_delta" in row for row in results[1:])


def test_render_table_includes_all_rows():
    configs = ablation_configs()
    results = [
        {"config": label, "scenarios": 2, "passes": 1, "pass_rate": 0.5,
         "total_tokens": 1000, "pass_rate_delta": -0.5, "token_delta_ratio": 1.2}
        for label, _ in configs
    ]
    table = render_ablation_table(results)
    assert "baseline" in table and "dense_retrieval" in table
