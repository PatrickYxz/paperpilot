"""Programmatic assertion checklist over one scenario's raw observations."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from paperpilot.smoke.scenarios import Scenario

_API_KEY_SHAPE = re.compile(r"sk-[A-Za-z0-9_-]{16,}")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class TurnObservation:
    """Raw observations from one turn (one question → one task → one answer)."""

    task: dict[str, Any] | None = None
    assistant_message: dict[str, Any] | None = None
    citations: list[Any] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


@dataclass
class ScenarioOutcome:
    """Raw observations from one scenario run; checks depend only on this.

    ``turns[0]`` is the primary question; ``follow_ups`` extend the list.
    The flat properties expose the last turn for single-turn convenience.
    """

    scenario: Scenario
    turns: list[TurnObservation] = field(default_factory=list)
    error: str = ""

    @property
    def task(self) -> dict[str, Any] | None:
        return self.turns[-1].task if self.turns else None

    @property
    def assistant_message(self) -> dict[str, Any] | None:
        return self.turns[-1].assistant_message if self.turns else None

    @property
    def citations(self) -> list[Any]:
        return self.turns[-1].citations if self.turns else []

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.turns[-1].events if self.turns else []

    @property
    def artifacts(self) -> list[dict[str, Any]]:
        return self.turns[-1].artifacts if self.turns else []


def extract_citations(message: dict[str, Any] | None) -> list[Any]:
    """Extract the citation list from the assistant message.

    Single fix point for message-payload shape changes. The verified path is
    ``metadata.citations`` (confirmed by the 2026-10-04 real run; items look
    like ``{"evidence_id": ..., "label": "[1]"}``). The bounded defensive
    search remains as fallback for older/alternative payload shapes.
    """
    if not isinstance(message, dict):
        return []
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("citations")
        if isinstance(value, list):
            return value

    def _find(node: Any, depth: int) -> list[Any] | None:
        if depth > 4:
            return None
        if isinstance(node, dict):
            value = node.get("citations")
            if isinstance(value, list):
                return value
            for child in node.values():
                found = _find(child, depth + 1)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = _find(child, depth + 1)
                if found is not None:
                    return found
        return None

    return _find(message, 0) or []


def extract_evidence_ids(message: dict[str, Any] | None) -> list[str]:
    """Extract the evidence-pool IDs from the assistant message.

    Single fix point for payload shape changes. The verified path is
    ``metadata.research_result.evidence_items[].id`` — note pool entries key
    their ID as ``id`` while citations key it as ``evidence_id`` (confirmed by
    the 2026-10-04 real run).
    """
    if not isinstance(message, dict):
        return []
    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        return []
    research = metadata.get("research_result")
    if not isinstance(research, dict):
        return []
    items = research.get("evidence_items")
    if not isinstance(items, list):
        return []
    return [
        item["id"]
        for item in items
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def evaluate_checks(
    outcome: ScenarioOutcome,
    *,
    context_management: bool,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    turns = outcome.turns or [TurnObservation()]
    expected_turns = 1 + len(outcome.scenario.follow_ups)

    completed = sum(
        1 for turn in turns if (turn.task or {}).get("status") == "completed"
    )
    checks.append(
        CheckResult(
            "task_completed",
            bool(turns) and completed == len(turns),
            f"completed turns: {completed}/{len(turns)}",
        )
    )
    checks.append(
        CheckResult(
            "turns_completed",
            len(outcome.turns) == expected_turns,
            f"observed turns: {len(outcome.turns)} (expect {expected_turns})",
        )
    )

    answers = [
        ((turn.assistant_message or {}).get("content") or "").strip() for turn in turns
    ]
    checks.append(
        CheckResult(
            "assistant_answer_present",
            bool(turns) and all(answers),
            f"answer lengths: {[len(a) for a in answers]}",
        )
    )

    first = turns[0]
    first_answer = (
        ((first.assistant_message or {}).get("content") or "").strip() if turns else ""
    )
    if outcome.scenario.expected_points:
        lowered = first_answer.lower()
        missing_points = [
            point
            for point in outcome.scenario.expected_points
            if point.lower() not in lowered
        ]
        checks.append(
            CheckResult(
                "expected_points_covered",
                not missing_points,
                f"missing points: {missing_points or 'none'}",
            )
        )

    citation_count = len(first.citations)
    expect = outcome.scenario.expect_citations_gte
    checks.append(
        CheckResult(
            "citations_gte",
            citation_count >= expect,
            f"citations: {citation_count} (expect >= {expect})",
        )
    )

    unanchored: list[str] = []
    anchored_total = 0
    for index, turn in enumerate(turns, start=1):
        citation_ids: list[str] = []
        for citation in turn.citations:
            if isinstance(citation, dict) and isinstance(citation.get("evidence_id"), str):
                citation_ids.append(citation["evidence_id"])
            else:
                unanchored.append(f"<malformed citation> (turn {index})")
        if not citation_ids:
            continue
        pool_ids = set(extract_evidence_ids(turn.assistant_message))
        if not pool_ids:
            unanchored.append(f"evidence pool missing (turn {index})")
            continue
        for citation_id in citation_ids:
            if citation_id in pool_ids:
                anchored_total += 1
            else:
                unanchored.append(f"{citation_id} (turn {index})")
    checks.append(
        CheckResult(
            "citations_anchored",
            not unanchored,
            f"unanchored: {unanchored or 'none'}; anchored: {anchored_total}",
        )
    )

    if context_management:
        artifact_count = sum(len(turn.artifacts) for turn in turns)
        checks.append(
            CheckResult(
                "context_artifacts_present",
                artifact_count >= 1,
                f"context artifacts: {artifact_count}",
            )
        )

    all_events = [event for turn in turns for event in turn.events]
    retry_events = [
        event
        for event in all_events
        if "retry" in str(event.get("type", "")).lower()
    ]
    checks.append(
        CheckResult(
            "no_retry_events",
            not retry_events,
            f"retry events: {len(retry_events)}",
        )
    )

    secret_markers = _secret_markers(outcome)
    checks.append(
        CheckResult(
            "no_secret_leak",
            not secret_markers,
            f"leaked markers: {secret_markers or 'none'}",
        )
    )

    turn_errors = [turn.error for turn in outcome.turns if turn.error]
    harness_error = outcome.error or ("; ".join(turn_errors) if turn_errors else "")
    if harness_error:
        checks.append(CheckResult("no_harness_error", False, harness_error))

    return checks


def _secret_markers(outcome: ScenarioOutcome) -> list[str]:
    """Scan persisted observations for credential fragments.

    Heuristic, biased toward false positives — but matching key shape
    (``sk-`` followed by enough key characters), not the bare substring,
    because business IDs like ``task-1`` would otherwise flag every run.
    """
    markers: list[str] = []
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    payloads = json.dumps(
        {
            "turns": [
                {
                    "task": turn.task,
                    "events": turn.events,
                    "artifacts": turn.artifacts,
                }
                for turn in outcome.turns
            ],
        },
        ensure_ascii=False,
        default=str,
    )
    if _API_KEY_SHAPE.search(payloads):
        markers.append("sk-<key-shape>")
    if api_key and api_key in payloads:
        markers.append("DEEPSEEK_API_KEY-value")
    return markers
