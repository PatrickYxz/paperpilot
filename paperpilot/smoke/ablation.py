"""Ablation matrix runner (book ch.6): per-feature contribution report.

Runs the selected scenarios once per configuration — baseline (current
env) plus each feature toggled off — and reports pass rate and token cost
per configuration with deltas against the baseline.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable

from paperpilot import feature_flags

RunOne = Callable[[], tuple[bool, int]]
# Returns (passed, total_tokens) for one scenario run under current flags.


def ablation_configs() -> list[tuple[str, dict[str, str]]]:
    """(label, env overrides) pairs: baseline plus one-off per feature."""
    configs: list[tuple[str, dict[str, str]]] = [("baseline", {})]
    for flag, off_label in (
        (feature_flags.MEMORY_TOOL_FLAG, "memory_tool_off"),
        (feature_flags.PROFILE_INJECTION_FLAG, "profile_off"),
        (feature_flags.COMPUTATION_TOOL_FLAG, "computation_off"),
        (feature_flags.RETRIEVAL_MODE_FLAG, "dense_retrieval"),
    ):
        value = "dense" if flag == feature_flags.RETRIEVAL_MODE_FLAG else "off"
        configs.append((off_label, {flag: value}))
    return configs


def run_ablation_matrix(
    scenario_ids: Iterable[str],
    run_one: Callable[[str], RunOne],
) -> list[dict[str, Any]]:
    """Run the matrix; ``run_one(scenario_id)`` builds a fresh runner that
    executes under the CURRENT process env, returning (passed, tokens)."""
    scenarios = list(scenario_ids)
    saved_env: dict[str, str | None] = {
        flag: os.environ.get(flag) for flag in feature_flags.ALL_FLAGS
    }
    results: list[dict[str, Any]] = []
    try:
        for label, overrides in ablation_configs():
            for flag, value in overrides.items():
                os.environ[flag] = value
            for flag in feature_flags.ALL_FLAGS:
                if flag not in overrides:
                    os.environ.pop(flag, None)

            passes = 0
            tokens = 0
            for scenario_id in scenarios:
                passed, run_tokens = run_one(scenario_id)()
                passes += int(passed)
                tokens += run_tokens
            results.append(
                {
                    "config": label,
                    "scenarios": len(scenarios),
                    "passes": passes,
                    "pass_rate": passes / len(scenarios) if scenarios else 0.0,
                    "total_tokens": tokens,
                }
            )
    finally:
        for flag, value in saved_env.items():
            if value is None:
                os.environ.pop(flag, None)
            else:
                os.environ[flag] = value
    _attach_deltas(results)
    return results


def _attach_deltas(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    baseline = results[0]
    for row in results[1:]:
        row["pass_rate_delta"] = round(
            row["pass_rate"] - baseline["pass_rate"], 4
        )
        base_tokens = baseline["total_tokens"] or 1
        row["token_delta_ratio"] = round(
            row["total_tokens"] / base_tokens, 3
        )


def render_ablation_table(results: list[dict[str, Any]]) -> str:
    header = (
        f"{'config':<22} {'pass':>7} {'rate':>7} {'tokens':>9} "
        f"{'rate Δ':>8} {'tok×':>6}"
    )
    lines = [header, "-" * len(header)]
    for row in results:
        lines.append(
            f"{row['config']:<22} "
            f"{row['passes']:>3}/{row['trials']:<3} "
            f"{row['pass_rate']:>7.2f} "
            f"{row['total_tokens']:>9} "
            f"{row.get('pass_rate_delta', 0):>+8.2f} "
            f"{row.get('token_delta_ratio', 1.0):>6.2f}"
        )
    return "\n".join(lines)
