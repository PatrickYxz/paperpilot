"""Failure attribution over task event streams (book ch.6).

Attributes a failed scenario to the FIRST error signal in its event
stream — later retries and cascading failures are consequences, not
root causes. Categories:

- contract_failure: the research decision violated the harness contract
  (captured decision dump included when the instrumented event fired)
- budget_exhausted: retry/model/tool budget ran out before a valid decision
- tool_loop: identical tool calls repeated with no progress
- infrastructure: environment faults (DB locks, provider transport)
- check_failure: the task completed but an answer-level check failed
- unknown: no recognizable signal
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

CATEGORY_CONTRACT = "contract_failure"
CATEGORY_BUDGET = "budget_exhausted"
CATEGORY_TOOL_LOOP = "tool_loop"
CATEGORY_INFRA = "infrastructure"
CATEGORY_CHECK = "check_failure"
CATEGORY_UNKNOWN = "unknown"

_ERROR_TYPE_CATEGORY = {
    "ResearchContractError": CATEGORY_CONTRACT,
    "AgentBudgetExceededError": CATEGORY_BUDGET,
    "OperationalError": CATEGORY_INFRA,
}


@dataclass(frozen=True)
class Attribution:
    category: str
    summary: str
    first_error_event_id: int | None = None
    first_error_type: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


def attribute_failure(
    events: Sequence[dict[str, Any]],
    failed_checks: Sequence[str] = (),
) -> Attribution:
    """Attribute one failed run from its ordered event stream."""
    for event in events:
        kind = event.get("type", "")

        if kind == "research_contract_failure":
            payload = _payload(event)
            return Attribution(
                category=CATEGORY_CONTRACT,
                summary=(
                    f"decision violated contract at event "
                    f"{event.get('id')}: {payload.get('error', '')[:160]}"
                ),
                first_error_event_id=event.get("id"),
                first_error_type=kind,
                evidence={
                    "error": payload.get("error", ""),
                    "selected_evidence_ids": (
                        payload.get("decision", {}).get("selected_evidence_ids")
                    ),
                },
            )

        if kind == "tool_repetition_warning":
            payload = _payload(event)
            return Attribution(
                category=CATEGORY_TOOL_LOOP,
                summary=(
                    f"tool repeated without progress at event "
                    f"{event.get('id')}: {payload.get('name', '?')} "
                    f"x{payload.get('repeats')}"
                ),
                first_error_event_id=event.get("id"),
                first_error_type=kind,
                evidence={"fingerprint": payload.get("fingerprint", "")},
            )

        if kind == "failed":
            payload = _payload(event)
            error_type = str(payload.get("error_type", ""))
            category = _ERROR_TYPE_CATEGORY.get(error_type, CATEGORY_UNKNOWN)
            return Attribution(
                category=category,
                summary=(
                    f"terminal failure at event {event.get('id')}: "
                    f"{error_type or kind}"
                ),
                first_error_event_id=event.get("id"),
                first_error_type=error_type or kind,
                evidence={"error_code": payload.get("error_code")},
            )

    if failed_checks:
        return Attribution(
            category=CATEGORY_CHECK,
            summary=(
                "task completed but answer-level checks failed: "
                f"{', '.join(failed_checks)}"
            ),
            evidence={"failed_checks": list(failed_checks)},
        )
    return Attribution(category=CATEGORY_UNKNOWN, summary="no error signal found")


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload_json") or event.get("payload") or {}
    if isinstance(payload, str):
        import json

        try:
            return json.loads(payload)
        except (TypeError, ValueError):
            return {}
    return payload if isinstance(payload, dict) else {}
