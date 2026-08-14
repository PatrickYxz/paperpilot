"""Bounded evidence-research node."""
from __future__ import annotations

from langgraph.runtime import Runtime

from ..research_agent import run_research_agent
from ..state import DeepReadingState
from .binding import _validate_runtime_binding
from .context import DeepReadingContext


def research_evidence(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
    """Run bounded research and checkpoint the complete JSON result."""
    context = runtime.context
    _validate_runtime_binding(state, context)
    result = run_research_agent(state, context)
    return {"research_result": result.model_dump(mode="json")}
