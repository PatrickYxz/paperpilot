from __future__ import annotations

from paperpilot.agent.errors import classify_exception, is_step_retryable
from paperpilot.tools.mcp_client import MCPToolTimeout, MCPTransportError


class ProviderError(RuntimeError):
    def __init__(self, status_code: int, message: str = "provider failed") -> None:
        super().__init__(message)
        self.status_code = status_code


def test_transport_and_timeout_failures_are_normalized():
    transport = classify_exception(MCPTransportError("closed"))
    timeout = classify_exception(MCPToolTimeout("slow"))

    assert transport.failure_class == "transport"
    assert transport.transient is True
    assert timeout.failure_class == "timeout"
    assert timeout.transient is True


def test_value_error_is_validation_and_not_transient():
    failure = classify_exception(ValueError("invalid arguments"))

    assert failure.failure_class == "validation"
    assert failure.transient is False


def test_provider_status_codes_have_deterministic_classification():
    assert classify_exception(ProviderError(401)).failure_class == "authentication"
    assert classify_exception(ProviderError(403)).failure_class == "authentication"
    assert classify_exception(ProviderError(429)).failure_class == "rate_limit"
    assert classify_exception(ProviderError(503)).failure_class == "transport"
    assert classify_exception(ProviderError(503)).transient is True


def test_failure_keeps_full_message_and_exposes_bounded_persistence_message():
    message = "x" * 2_500
    exc = RuntimeError(message)

    failure = classify_exception(exc)

    assert failure.message.endswith(message)
    assert len(failure.message) > len(message)
    assert len(failure.persisted_message) == 2_000
    assert str(exc) == message


def test_only_retryable_tool_classes_can_retry_transient_failures():
    transient = classify_exception(MCPTransportError("closed"))

    assert is_step_retryable(transient, "read_only") is True
    assert is_step_retryable(transient, "idempotent_write") is True
    assert is_step_retryable(transient, "non_retryable") is False
    assert is_step_retryable(classify_exception(ValueError("bad")), "read_only") is False
