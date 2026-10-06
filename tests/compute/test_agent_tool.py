"""run_computation agent tool tests through real harness helpers."""
from __future__ import annotations

from unittest.mock import MagicMock

from paperpilot.compute.agent_tool import build_computation_tool
from paperpilot.deep_reading.research_agent import _clip, _emit_tool_call, _required_id


def _tool():
    context = MagicMock()
    context.user_id = "user-1"
    return build_computation_tool(
        context,
        emit_tool_call=_emit_tool_call,
        clip=_clip,
        required_id=_required_id,
    )


def test_tool_runs_code_and_returns_json():
    import json

    out = json.loads(_tool().invoke({"code": "print(2 + 2)"}))
    assert out["ok"] is True and out["stdout"].strip() == "4"


def test_tool_description_covers_when_to_use_and_boundaries():
    description = _tool().description
    assert "precise numbers" in description
    assert "NO network" in description
    assert "never invent" in description


def test_tool_failure_is_structured_not_raised():
    import json

    out = json.loads(_tool().invoke({"code": "raise ValueError('boom')"}))
    assert out["ok"] is False and "boom" in out["stderr"]
