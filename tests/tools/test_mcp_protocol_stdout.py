"""Regression tests for the MCP stdio protocol boundary."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mcp_module_imports_do_not_write_to_stdout() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "default",
            "-c",
            (
                "import paperpilot.mcp_servers.arxiv; "
                "import paperpilot.mcp_servers.vlm.page_renderer"
            ),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
