"""Persistent circuit-breaker and deterministic safe-degradation contracts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from paperpilot.deep_reading.context_management.breaker import (
    BreakerState,
    CompressionCircuitBreaker,
)
from paperpilot.deep_reading.context_management.compaction import (
    CompressionCoordinator,
)
from paperpilot.deep_reading.context_management.models import ActiveProjection, ContextView
from paperpilot.deep_reading.context_management.views import ContextViewBuilder
from paperpilot.deep_reading.nodes.prepare_context import _persist_capacity_capsule
from paperpilot.web.store.records import CompressionStateRecord


class _Port:
    def __init__(self) -> None:
        self.state = CompressionStateRecord(
            conversation_id="conversation-1",
            compressor_version="compressor-v1",
            state="CLOSED",
            consecutive_failures=0,
            last_failure_type=None,
            last_input_digest=None,
            opened_at=None,
            updated_at="2026-09-01T00:00:00+00:00",
        )
        self.outcomes: list[object] = []

    def get_compression_state(self, conversation_id, compressor_version):
        assert conversation_id == "conversation-1"
        assert compressor_version == "compressor-v1"
        return self.state

    def claim_compression_probe(self, conversation_id, compressor_version):
        if self.state.state != "OPEN":
            return None
        self.state = CompressionStateRecord(
            **{
                **self.state.__dict__,
                "state": "HALF_OPEN",
            }
        )
        return self.state

    def record_compression_outcome(self, *, outcome):
        self.outcomes.append(outcome)
        if outcome.success:
            self.state = CompressionStateRecord(
                **{
                    **self.state.__dict__,
                    "state": "CLOSED",
                    "consecutive_failures": 0,
                    "last_failure_type": None,
                    "last_input_digest": outcome.input_digest,
                }
            )
        else:
            failures = self.state.consecutive_failures + 1
            self.state = CompressionStateRecord(
                **{
                    **self.state.__dict__,
                    "state": (
                        "OPEN"
                        if failures >= outcome.failure_threshold
                        else self.state.state
                    ),
                    "consecutive_failures": failures,
                    "last_failure_type": outcome.failure_type,
                    "last_input_digest": outcome.input_digest,
                    "opened_at": (
                        "2026-09-01T00:00:00+00:00"
                        if failures >= outcome.failure_threshold
                        else None
                    ),
                }
            )
        return self.state


def test_breaker_opens_on_third_failure_and_success_clears_it() -> None:
    port = _Port()
    now = [datetime(2026, 9, 1, tzinfo=timezone.utc)]
    breaker = CompressionCircuitBreaker(
        state_port=port,
        conversation_id="conversation-1",
        compressor_version="compressor-v1",
        clock=lambda: now[0],
    )

    assert breaker.before_attempt(input_digest="digest-a") is True
    for _ in range(3):
        breaker.record_failure(failure_type="transient", input_digest="digest-a")
    assert breaker.state() == BreakerState.OPEN
    assert breaker.before_attempt(input_digest="digest-a") is False

    now[0] += timedelta(seconds=300)
    assert breaker.before_attempt(input_digest="digest-a") is True
    assert breaker.state() == BreakerState.HALF_OPEN
    assert breaker.before_attempt(input_digest="digest-a") is False
    breaker.record_success(input_digest="digest-a")
    assert breaker.state() == BreakerState.CLOSED


def test_deterministic_open_state_requires_a_changed_input_digest() -> None:
    port = _Port()
    breaker = CompressionCircuitBreaker(
        state_port=port,
        conversation_id="conversation-1",
        compressor_version="compressor-v1",
        clock=lambda: datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    for _ in range(3):
        breaker.record_failure(failure_type="schema_invalid", input_digest="digest-a")

    assert breaker.before_attempt(input_digest="digest-a") is False
    assert breaker.before_attempt(input_digest="digest-b") is True


def test_open_coordinator_does_not_call_compressor() -> None:
    port = _Port()
    breaker = CompressionCircuitBreaker(
        state_port=port,
        conversation_id="conversation-1",
        compressor_version="compressor-v1",
        clock=lambda: datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    for _ in range(3):
        breaker.record_failure(failure_type="transient", input_digest="digest-a")
    calls: list[str] = []
    coordinator = CompressionCoordinator(
        token_counter=type("Counter", (), {"count_text": lambda _self, _value: 90})(),
        usable_input_budget=100,
        full_compaction_enabled=True,
        session_compressor=lambda _view: calls.append("a"),
        breaker=breaker,
    )
    view = ContextView(
        messages=[{"role": "user", "content": "goal"}],
        active_projection=ActiveProjection(
            current_goal="goal",
            active_constraints=[],
            active_decisions=[],
            active_paper_ids=[],
            open_todos=[],
            failed_verifications=[],
            unresolved_questions=[],
        ),
        input_tokens=90,
    )

    decision = coordinator.prepare(view, task_id="t", conversation_id="conversation-1")

    assert calls == []
    assert decision.reason == "compression_circuit_open"


def test_capacity_exhaustion_persists_bounded_recovery_capsule_artifact() -> None:
    requests: list[object] = []

    class ArtifactStore:
        def put(self, request):
            requests.append(request)
            return type(
                "Artifact",
                (),
                {
                    "artifact_id": "capacity-artifact-1",
                    "sha256": "a" * 64,
                    "token_estimate": 50,
                    "preview": request.preview,
                },
            )()

    context = type(
        "Context",
        (),
        {
            "conversation_id": "conversation-1",
            "task_id": "task-1",
            "context_management_runtime": type(
                "Runtime", (), {"artifact_store": ArtifactStore()}
            )(),
        },
    )()
    view = ContextView(
        messages=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer must not be copied"},
            {"role": "user", "content": "current goal"},
        ],
        active_projection=ActiveProjection(
            current_goal="current goal",
            active_constraints=["preserve exact constraint"],
            active_paper_ids=["paper-1"],
        ),
        authority_evidence_ids=["evidence-1"],
        authority_archive_ids=["archive-1"],
    )

    artifact_ref = _persist_capacity_capsule(context, view)

    assert artifact_ref is not None
    assert artifact_ref.artifact_id == "capacity-artifact-1"
    assert len(requests) == 1
    request = requests[0]
    assert request.kind == "continuation_capsule"
    assert request.future_retention == "PROTECTED"
    assert request.payload["current_message"] == {
        "role": "user",
        "content": "current goal",
    }
    assert "old answer must not be copied" not in str(request.payload)
