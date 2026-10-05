"""Scenario execution and report-row construction."""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paperpilot.smoke.adapter import ConversationApi
from paperpilot.smoke.checks import (
    CheckResult,
    ScenarioOutcome,
    TurnObservation,
    evaluate_checks,
    extract_citations,
)
from paperpilot.smoke.scenarios import Scenario

ROOT = Path(__file__).resolve().parents[2]


def run_scenario(
    api: ConversationApi,
    scenario: Scenario,
    *,
    context_management: bool,
    timeout_s: float,
) -> tuple[ScenarioOutcome, list[CheckResult], float]:
    started = time.monotonic()
    outcome = ScenarioOutcome(scenario=scenario)
    questions = [scenario.question, *scenario.follow_ups]
    head_message_id: str | None = None
    try:
        for seed_question in scenario.memory_seed:
            api.open_conversation_as_current_user(scenario)
            api.ask(seed_question, scenario.depth)
            seed_task, _seed_events, _seed_artifacts = (
                api.poll_until_terminal(timeout_s=timeout_s)
            )
            if (seed_task or {}).get("status") != "completed":
                raise RuntimeError(
                    f"memory seed turn failed: {seed_question[:60]!r}"
                )
        api.open_conversation_as_current_user(scenario) if scenario.memory_seed else api.open_conversation(scenario)
        for question in questions:
            api.ask(question, scenario.depth, head_message_id)
            task, events, artifacts = api.poll_until_terminal(timeout_s=timeout_s)
            assistant = api.assistant_message()
            turn = TurnObservation(
                task=task,
                events=events,
                artifacts=artifacts,
                assistant_message=assistant,
                citations=extract_citations(assistant),
            )
            outcome.turns.append(turn)
            head_message_id = (assistant or {}).get("id") or head_message_id
            if (task or {}).get("status") != "completed":
                break  # 终态未完成时不再追问，turns_completed 会暴露缺口
    except Exception as exc:  # noqa: BLE001
        outcome.error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    return outcome, evaluate_checks(outcome, context_management=context_management), elapsed


def build_report_row(
    scenario: Scenario,
    checks: list[CheckResult],
    elapsed: float,
    *,
    context_management: bool,
    outcome: ScenarioOutcome | None = None,
) -> dict[str, Any]:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "scenario_id": scenario.id,
        "paper_external_id": scenario.paper_external_id,
        "depth": scenario.depth,
        "question": scenario.question,
        "tags": list(scenario.tags),
        "model": os.environ.get("PAPERPILOT_RESEARCH_MODEL_NAME", "<env-default>"),
        "context_management": context_management,
        "status": (checks[0].detail if checks else "<no-checks>"),
        "elapsed_s": round(elapsed, 1),
        "checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks
        ],
        "passed": all(c.passed for c in checks) if checks else False,
    }
    if outcome is not None:
        first_turn = outcome.turns[0] if outcome.turns else None
        row["citations_count"] = len(first_turn.citations) if first_turn else 0
        row["turns"] = [
            {
                "status": (turn.task or {}).get("status"),
                "answer_chars": len(
                    ((turn.assistant_message or {}).get("content") or "").strip()
                ),
                "citations_count": len(turn.citations),
            }
            for turn in outcome.turns
        ]
    return row


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    return result.stdout.strip() or "unknown"
