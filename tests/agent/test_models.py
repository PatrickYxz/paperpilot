from dataclasses import FrozenInstanceError

import pytest

from paperpilot.agent.models import AgentRun, RUN_TERMINAL_STATUSES
from paperpilot.agent.policy import RunPolicy


def test_agent_run_terminal_state_is_immutable():
    run = AgentRun(
        id="run-1",
        task_id="task-1",
        status="completed",
        attempt=1,
        current_step=2,
        cancel_requested_at=None,
        retry_at=None,
        failure_class=None,
        failure_message=None,
        policy=RunPolicy.for_depth("standard"),
        owner_id=None,
        lease_expires_at=None,
        schema_version=1,
        created_at="2026-07-19T00:00:00+00:00",
        updated_at="2026-07-19T00:01:00+00:00",
        started_at="2026-07-19T00:00:01+00:00",
        finished_at="2026-07-19T00:01:00+00:00",
    )

    assert run.is_terminal is True
    assert "completed" in RUN_TERMINAL_STATUSES
    with pytest.raises(FrozenInstanceError):
        run.status = "running"
