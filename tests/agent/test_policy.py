import pytest

from paperpilot.agent.policy import RunPolicy


def test_standard_policy_round_trips_without_environment_lookup():
    policy = RunPolicy.for_depth("standard")
    restored = RunPolicy.from_dict(policy.to_dict())

    assert restored == policy
    assert restored.max_iterations == 8
    assert restored.token_budget == 50_000
    assert restored.max_retries == 2
    assert "ask_user" not in restored.allowed_builtin_tools


def test_depth_budgets_and_subagent_permissions_are_explicit():
    assert RunPolicy.for_depth("quick").max_iterations == 4
    assert RunPolicy.for_depth("quick").token_budget == 20_000
    assert RunPolicy.for_depth("quick").allow_subagents is False
    assert RunPolicy.for_depth("deep").max_iterations == 12
    assert RunPolicy.for_depth("deep").token_budget == 120_000
    assert RunPolicy.for_depth("deep").allow_subagents is True


def test_retry_delay_is_bounded_and_deterministic_without_jitter():
    policy = RunPolicy.for_depth("deep")

    assert policy.retry_delay_seconds(1, jitter=0.0) == 2.0
    assert policy.retry_delay_seconds(2, jitter=0.0) == 8.0
    assert policy.retry_delay_seconds(3, jitter=0.0) == 8.0


def test_unknown_depth_and_non_positive_budgets_are_rejected():
    with pytest.raises(ValueError, match="Unknown run depth"):
        RunPolicy.for_depth("extended")
    with pytest.raises(ValueError, match="max_iterations"):
        RunPolicy(max_iterations=0, token_budget=1)
