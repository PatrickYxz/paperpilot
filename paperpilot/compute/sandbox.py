"""Process-level Python sandbox: offline, ephemeral cwd, no inherited env.

The sandbox exists so the agent can compute exact numbers (percentages,
ratios, table comparisons) instead of doing error-prone mental arithmetic.
The isolation posture matches the book's local-process guidance: strip the
environment (API keys never reach child processes), run in a throwaway
working directory, cap time and output, and report everything as
structured results the agent can react to.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

EXECUTION_TIMEOUT_SECONDS = 10
OUTPUT_LIMIT_CHARS = 4000
CODE_LENGTH_LIMIT = 8000
_SANDBOX_ENV = {"PYTHONIOENCODING": "utf-8"}


def run_python_code(
    code: str,
    *,
    timeout_s: int = EXECUTION_TIMEOUT_SECONDS,
    output_limit: int = OUTPUT_LIMIT_CHARS,
) -> dict:
    """Execute one Python snippet and return a structured result."""
    if not code.strip():
        return _result(ok=False, error="empty code")
    if len(code) > CODE_LENGTH_LIMIT:
        return _result(ok=False, error="code exceeds length limit")

    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="paperpilot-compute-") as cwd:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=cwd,
                env=_SANDBOX_ENV,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        stdout = _clip((exc.stdout or "") if isinstance(exc.stdout, str) else "", output_limit)
        return _result(
            ok=False,
            timed_out=True,
            stdout=stdout,
            error=f"execution exceeded {timeout_s}s and was terminated",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    stdout = _clip(completed.stdout or "", output_limit)
    stderr = _clip(completed.stderr or "", output_limit)
    return _result(
        ok=completed.returncode == 0,
        stdout=stdout,
        stderr=stderr,
        returncode=completed.returncode,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    omitted = len(text) - limit
    return f"{head}\n... [{omitted} chars omitted] ...\n{tail}"


def _result(
    *,
    ok: bool,
    stdout: str = "",
    stderr: str = "",
    error: str = "",
    timed_out: bool = False,
    returncode: int | None = None,
    duration_ms: int = 0,
) -> dict:
    payload: dict = {
        "ok": ok,
        "stdout": stdout,
        "duration_ms": duration_ms,
    }
    if stderr:
        payload["stderr"] = stderr
    if error:
        payload["error"] = error
    if timed_out:
        payload["timed_out"] = True
    if returncode is not None:
        payload["returncode"] = returncode
    return payload
