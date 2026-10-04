"""Scenario data model and JSONL case loading.

One line per scenario, pure business language (which paper, what question,
what depth, what expectation) — zero API detail, so evolving entry points
never touch this file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_DEPTHS = ("quick", "standard", "deep")
DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"


@dataclass(frozen=True)
class Scenario:
    """A single real-business scenario: what to ask and what to expect.

    ``follow_ups`` are asked sequentially in the same conversation after the
    primary question completes (exercises cross-turn context management).
    """

    id: str
    question: str
    depth: str
    paper_external_id: str
    expect_citations_gte: int = 1
    paper_metadata: dict[str, Any] | None = None
    expected_points: tuple[str, ...] = ()
    follow_ups: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    expect_refusal: bool = False
    trap_terms: tuple[str, ...] = ()
    active: bool = False
    notes: str = ""


class ScenarioError(ValueError):
    """The scenario file is invalid."""


def load_scenarios(path: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    if not path.is_file():
        raise ScenarioError(f"cases file not found: {path}")
    for line_no, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScenarioError(f"{path.name}:{line_no}: invalid JSON: {exc}") from exc
        missing = [k for k in ("id", "question", "depth", "paper_external_id") if not row.get(k)]
        if missing:
            raise ScenarioError(f"{path.name}:{line_no}: missing fields: {missing}")
        if row["depth"] not in VALID_DEPTHS:
            raise ScenarioError(
                f"{path.name}:{line_no}: depth must be one of {VALID_DEPTHS}"
            )
        if row["id"] in seen:
            raise ScenarioError(f"{path.name}:{line_no}: duplicate id: {row['id']}")
        seen.add(row["id"])
        raw_points = row.get("expected_points") or []
        if not isinstance(raw_points, list) or not all(
            isinstance(point, str) and point.strip() for point in raw_points
        ):
            raise ScenarioError(
                f"{path.name}:{line_no}: expected_points must be a list of non-empty strings"
            )
        raw_follow_ups = row.get("follow_ups") or []
        if not isinstance(raw_follow_ups, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_follow_ups
        ):
            raise ScenarioError(
                f"{path.name}:{line_no}: follow_ups must be a list of non-empty strings"
            )
        raw_tags = row.get("tags") or []
        if not isinstance(raw_tags, list) or not all(
            isinstance(tag, str) and tag.strip() for tag in raw_tags
        ):
            raise ScenarioError(
                f"{path.name}:{line_no}: tags must be a list of non-empty strings"
            )
        raw_traps = row.get("trap_terms") or []
        if not isinstance(raw_traps, list) or not all(
            isinstance(term, str) and term.strip() for term in raw_traps
        ):
            raise ScenarioError(
                f"{path.name}:{line_no}: trap_terms must be a list of non-empty strings"
            )
        scenarios.append(
            Scenario(
                id=row["id"],
                question=row["question"],
                depth=row["depth"],
                paper_external_id=row["paper_external_id"],
                expect_citations_gte=int(row.get("expect_citations_gte", 1)),
                paper_metadata=row.get("paper_metadata"),
                expected_points=tuple(point.strip() for point in raw_points),
                follow_ups=tuple(item.strip() for item in raw_follow_ups),
                tags=tuple(tag.strip() for tag in raw_tags),
                expect_refusal=bool(row.get("expect_refusal", False)),
                trap_terms=tuple(term.strip() for term in raw_traps),
                active=bool(row.get("active", False)),
                notes=row.get("notes", ""),
            )
        )
    if not scenarios:
        raise ScenarioError(f"{path.name}: no scenarios found")
    return scenarios
