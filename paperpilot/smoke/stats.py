"""Reliability statistics for repeated scenario runs (book ch.6)."""
from __future__ import annotations

import math
from typing import Sequence

Z_95 = 1.96


def wilson_interval(passes: int, total: int) -> tuple[float, float]:
    """Wilson score interval (95%) for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = passes / total
    denominator = 1 + Z_95**2 / total
    center = (p + Z_95**2 / (2 * total)) / denominator
    margin = (
        Z_95
        * math.sqrt(p * (1 - p) / total + Z_95**2 / (4 * total**2))
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_repeats(results: Sequence[bool]) -> dict:
    """Pass^k (all consecutive passes) and Pass@k (at least one) plus a
    Wilson interval on the single-run rate — the two reliability views
    the book distinguishes, plus the noise band for switch decisions."""
    k = len(results)
    if k == 0:
        return {"k": 0}
    passes = sum(1 for r in results if r)
    low, high = wilson_interval(passes, k)
    return {
        "k": k,
        "results": list(results),
        "pass_k_consecutive": all(results),
        "pass_at_k": any(results),
        "single_pass_rate": passes / k,
        "wilson95_low": round(low, 4),
        "wilson95_high": round(high, 4),
    }
