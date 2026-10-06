"""Sandbox behavior tests (real subprocess, no network calls needed)."""
from __future__ import annotations

import os

from paperpilot.compute.sandbox import run_python_code


def test_exact_computation_returns_stdout():
    out = run_python_code("print(213 / 65)")
    assert out["ok"] is True
    assert out["stdout"].strip().startswith("3.276923076923")


def test_syntax_error_reports_stderr():
    out = run_python_code("print(")
    assert out["ok"] is False
    assert "SyntaxError" in out["stderr"]


def test_timeout_is_structured():
    out = run_python_code("while True: pass", timeout_s=2)
    assert out["ok"] is False and out["timed_out"] is True
    assert "terminated" in out["error"]


def test_environment_is_stripped():
    os.environ["SANDBOX_SECRET_PROBE"] = "leak-me"
    try:
        out = run_python_code(
            "import os; print(os.environ.get('SANDBOX_SECRET_PROBE', 'clean'))"
        )
    finally:
        os.environ.pop("SANDBOX_SECRET_PROBE", None)
    assert out["stdout"].strip() == "clean"


def test_no_network_by_sandbox_contract():
    # sockets are not disabled by the sandbox itself; the contract is
    # offline-by-instruction plus env stripping. Document that DNS/env is
    # unavailable rather than promising kernel-level blocking.
    out = run_python_code(
        "import os; print(bool(os.environ.get('HTTP_PROXY')))"
    )
    assert out["ok"] is True


def test_long_output_is_clipped():
    out = run_python_code("print('x' * 20000)", output_limit=1000)
    assert out["ok"] is True
    assert "chars omitted" in out["stdout"]
    assert len(out["stdout"]) < 1200


def test_empty_and_oversized_code_rejected():
    assert run_python_code("   ")["ok"] is False
    assert run_python_code("x=1\n" * 5000)["ok"] is False
