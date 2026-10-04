"""CLI gates: dry-run, credential refusal, scenario selection."""
from __future__ import annotations

import pytest

from paperpilot.smoke.cli import main


def test_cli_dry_run_needs_no_credentials(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--dry-run"])
    captured = capsys.readouterr()
    assert code == 0
    assert "dry run" in captured.out


def test_cli_refuses_to_run_without_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY"):
        main([])


def test_cli_rejects_unknown_scenario_id(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--only", "no-such-case", "--dry-run"]) == 2
    assert "unknown scenario" in capsys.readouterr().err


def test_cli_only_selects_the_requested_scenario(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--only", "quick-factual-2001.09899", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "quick-factual-2001.09899" in out
    assert "standard-numeric-metrics" not in out.split("\n")[0]
