"""Bounded evidence-research node."""
from __future__ import annotations

from langgraph.runtime import Runtime

from ..research_agent import run_research_agent
from ..schemas import ResearchResult
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
    outcome = run_research_agent(state, context)
    if isinstance(outcome, ResearchResult):
        return {"research_result": outcome.model_dump(mode="json")}
    update: DeepReadingState = {
        "research_result": outcome.result.model_dump(mode="json"),
        "user_memory_context": outcome.user_memory_context or None,
    }
    if context.context_management.enabled:
        update["research_trace"] = outcome.trace.to_dict()
        update["research_context_delta"] = (
            None
            if outcome.context_delta is None
            else {
                "constraints": [
                    item.model_dump(mode="json")
                    for item in outcome.context_delta.constraints
                ],
                "decisions": [
                    item.model_dump(mode="json")
                    for item in outcome.context_delta.decisions
                ],
                "supersedes": [
                    item.model_dump(mode="json")
                    for item in outcome.context_delta.supersedes
                ],
            }
        )
    return update
