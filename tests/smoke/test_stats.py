"""Repeat reliability statistics tests."""
from __future__ import annotations

from paperpilot.smoke.stats import summarize_repeats, wilson_interval


def test_wilson_known_values():
    low, high = wilson_interval(70, 100)
    assert abs(low - 0.604) < 0.01 and abs(high - 0.782) < 0.01


def test_wilson_extremes():
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high < 0.35


def test_summarize_distinction_pass_at_k_vs_consecutive():
    # 3/4 pass: Pass@4 True, Pass^4 False
    s = summarize_repeats([True, True, False, True])
    assert s["pass_at_k"] is True and s["pass_k_consecutive"] is False
    assert s["single_pass_rate"] == 0.75
    s2 = summarize_repeats([True] * 4)
    assert s2["pass_k_consecutive"] is True
    s3 = summarize_repeats([False] * 4)
    assert s3["pass_at_k"] is False
