"""Feature flag parsing tests."""
from __future__ import annotations

from paperpilot.feature_flags import (
    flag_enabled,
    retrieval_mode,
)


def test_defaults(monkeypatch):
    for name in (
        "PAPERPILOT_MEMORY_TOOL_ENABLED",
        "PAPERPILOT_RETRIEVAL_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    assert flag_enabled("PAPERPILOT_MEMORY_TOOL_ENABLED") is True
    assert retrieval_mode() == "hybrid"


def test_off_values(monkeypatch):
    for value in ("0", "false", "off", "disabled", "OFF"):
        monkeypatch.setenv("PAPERPILOT_MEMORY_TOOL_ENABLED", value)
        assert flag_enabled("PAPERPILOT_MEMORY_TOOL_ENABLED") is False


def test_retrieval_mode_falls_back(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_RETRIEVAL_MODE", "bogus")
    assert retrieval_mode() == "hybrid"
    monkeypatch.setenv("PAPERPILOT_RETRIEVAL_MODE", "dense")
    assert retrieval_mode() == "dense"
