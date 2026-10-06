"""The run_computation agent tool: exact numbers via sandboxed Python."""
from __future__ import annotations

import json
from typing import Callable

from langchain_core.tools import BaseTool, tool

from paperpilot.compute.sandbox import run_python_code

_TOOL_STAGE = "research"


def build_computation_tool(
    context,
    *,
    emit_tool_call: Callable[..., None],
    clip: Callable[[str], str],
    required_id: Callable[[object, str], str],
) -> BaseTool:
    """Build the sandboxed computation tool for one research run."""

    @tool("run_computation")
    def run_computation(code: str) -> str:
        """Run Python code in an offline sandbox for exact computation.

        Use whenever the answer needs precise numbers — percentages, ratios,
        parameter-count comparisons, table aggregation, unit conversion —
        anything error-prone to do mentally. Workflow: retrieve the relevant
        evidence first, copy the exact numbers from it into code, then
        compute and report the result.

        The sandbox has pandas, numpy, and sympy; there is NO network, no
        file writes, a 10-second limit, and output is truncated — print only
        the final values you need. Numbers must come from retrieved
        evidence; never invent inputs or use this to guess paper facts.

        Returns JSON with ok, stdout, duration_ms (and stderr/error on
        failure). Fix the code and retry on failure.
        """
        cleaned = required_id(code, "code")
        emit_tool_call(
            context,
            stage=_TOOL_STAGE,
            name="run_computation",
            arguments={"code": clip(cleaned)},
        )
        return json.dumps(
            run_python_code(cleaned), ensure_ascii=False
        )

    return run_computation
