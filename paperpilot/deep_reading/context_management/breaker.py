"""Persisted compression circuit breaker with one HALF_OPEN probe."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Callable

from paperpilot.web.store.records import CompressionOutcomeRecord


class BreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


_DETERMINISTIC_FAILURES = frozenset(
    {
        "schema_invalid",
        "invalid_reference",
        "protected_context_lost",
        "hash_mismatch",
        "insufficient_reduction",
        "protected_context_oversized",
    }
)


class CompressionCircuitBreaker:
    """Coordinate compression attempts across worker restarts and processes."""

    def __init__(
        self,
        *,
        state_port,
        conversation_id: str,
        compressor_version: str,
        cooldown_seconds: int = 300,
        failure_threshold: int = 3,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if cooldown_seconds < 0 or failure_threshold < 1:
            raise ValueError("invalid breaker cooldown or failure threshold")
        self._state_port = state_port
        self._conversation_id = conversation_id
        self._compressor_version = compressor_version
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._failure_threshold = failure_threshold
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = Lock()

    def state(self) -> BreakerState:
        return BreakerState(self._get_state().state)

    def before_attempt(self, *, input_digest: str) -> bool:
        with self._lock:
            current = self._get_state()
            state = BreakerState(current.state)
            if state is BreakerState.CLOSED:
                return True
            if state is BreakerState.HALF_OPEN:
                return False
            if not self._recovery_allowed(current, input_digest):
                return False
            claim = getattr(self._state_port, "claim_compression_probe", None)
            if callable(claim):
                result = claim(self._conversation_id, self._compressor_version)
                return result is not None
            return True

    def record_success(self, *, input_digest: str) -> None:
        self._record(
            success=True,
            failure_type=None,
            input_digest=input_digest,
        )

    def record_failure(self, *, failure_type: str, input_digest: str) -> None:
        self._record(
            success=False,
            failure_type=failure_type,
            input_digest=input_digest,
        )

    def _get_state(self):
        return self._state_port.get_compression_state(
            self._conversation_id,
            self._compressor_version,
        )

    def _recovery_allowed(self, state, input_digest: str) -> bool:
        if state.last_failure_type in _DETERMINISTIC_FAILURES:
            return state.last_input_digest != input_digest
        if state.last_failure_type == "transient":
            opened_at = _parse_datetime(state.opened_at)
            return opened_at is not None and self._clock() >= opened_at + self._cooldown
        return state.last_input_digest != input_digest

    def _record(self, *, success: bool, failure_type: str | None, input_digest: str) -> None:
        outcome = CompressionOutcomeRecord(
            conversation_id=self._conversation_id,
            compressor_version=self._compressor_version,
            success=success,
            failure_type=failure_type,
            input_digest=input_digest,
            stage="full",
            reason="compression_succeeded" if success else "compression_failed",
            before_tokens=0,
            after_tokens=0,
            reclaimed_tokens=0,
            protected_item_count=0,
            archive_ref_count=0,
            artifact_ref_count=0,
            failure_threshold=self._failure_threshold,
        )
        self._state_port.record_compression_outcome(outcome=outcome)


def _parse_datetime(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        return None
    return result if result.tzinfo is not None else result.replace(tzinfo=timezone.utc)
