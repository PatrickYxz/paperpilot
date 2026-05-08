"""Pass/fail scoring and failure clustering for Day 16 eval."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


def _normalize(s: str) -> str:
    """Lowercase, strip outer punctuation/whitespace, and fold internal space."""
    return re.sub(r"\s+", " ", s.strip(" .,;:'\"\n\t")).lower()


def is_pass(predicted: str, oracle_spans: list[str] | tuple[str, ...]) -> bool:
    """Return true when any non-empty oracle span appears in the prediction."""
    pred = _normalize(predicted)
    for span in oracle_spans:
        if not span or not span.strip():
            continue
        norm_span = _normalize(span)
        if norm_span and norm_span in pred:
            return True
    return False


def _classify(record: dict) -> str:
    """Bucket a single failed record by its trace-derived tool calls."""
    err = record.get("error") or ""
    if "max_iter" in err or "GuardrailStop" in err:
        return "iter_exhausted"

    calls: list[str] = record.get("tool_calls") or []
    if "load_skill" not in calls:
        return "no_load_skill"
    if "mcp__arxiv__download_paper" not in calls:
        return "no_download"
    if "mcp__colbert__search" not in calls:
        return "no_colbert_search"

    n_search = sum(1 for c in calls if c == "mcp__colbert__search")
    if n_search < 3:
        return "colbert_searched_low"
    return "synthesis_miss"


def cluster_failures(records: Iterable[dict]) -> dict[str, int]:
    """Count failure buckets across records, skipping passed records."""
    failed = (r for r in records if not r.get("passed"))
    return dict(Counter(_classify(r) for r in failed))
