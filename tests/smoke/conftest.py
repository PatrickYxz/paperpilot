"""Shared fixtures for the smoke-harness test suite.

Everything here is offline: scripted fake clients, no network, no credentials.
"""
from __future__ import annotations

from typing import Any

import pytest

from paperpilot.smoke.checks import ScenarioOutcome, TurnObservation
from paperpilot.smoke.scenarios import Scenario


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else "unused"

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload


def _default_turn_spec(
    *,
    final_status: str = "completed",
    events: list[dict] | None = None,
    artifacts: list[dict] | None = None,
    assistant: dict | None = None,
) -> dict[str, Any]:
    return {
        "final_status": final_status,
        "events": events or [],
        "artifacts": artifacts or [],
        "assistant": assistant,
    }


class _FakeClient:
    """Scripted responses in ConversationApi call order; records every call.

    Multi-turn: pass one spec dict per turn (final_status/events/artifacts/
    assistant). Legacy single-turn kwargs build a one-turn spec. Assistant
    messages appear in the message list only after their turn is polled to
    terminal, mirroring the real pipeline.
    """

    def __init__(
        self,
        *,
        turns: list[dict[str, Any]] | None = None,
        register_status: int = 201,
        final_status: str = "completed",
        events: list[dict] | None = None,
        artifacts: list[dict] | None = None,
        assistant: dict | None = None,
    ) -> None:
        if turns is None:
            turns = [
                _default_turn_spec(
                    final_status=final_status,
                    events=events,
                    artifacts=artifacts,
                    assistant=assistant,
                )
            ]
        self.turn_specs = turns
        self.register_status = register_status
        self.calls: list[tuple[str, str, Any]] = []
        self._posted = 0
        self._messages: list[dict[str, Any]] = []

    def _spec_for(self, turn_no: int) -> dict[str, Any]:
        if turn_no <= len(self.turn_specs):
            return self.turn_specs[turn_no - 1]
        return self.turn_specs[-1]

    def post(self, path: str, json: dict | None = None) -> _FakeResponse:
        self.calls.append(("POST", path, json))
        if path == "/api/auth/register":
            return _FakeResponse(self.register_status, {"id": "user-1"})
        if path == "/api/conversations":
            return _FakeResponse(201, {"id": "conv-1", "primary_paper_id": "p1"})
        if path == "/api/conversations/conv-1/messages":
            self._posted += 1
            turn_no = self._posted
            self._messages.append(
                {"id": f"m-user-{turn_no}", "role": "user", "content": (json or {}).get("content", "")}
            )
            return _FakeResponse(
                202,
                {
                    "user_message": {"id": f"m-user-{turn_no}", "role": "user"},
                    "task": {"id": f"task-{turn_no}", "status": "queued"},
                    "stable_head_message_id": None,
                },
            )
        return _FakeResponse(404, {"detail": "no route"})

    def get(self, path: str, params: dict | None = None) -> _FakeResponse:
        self.calls.append(("GET", path, params))
        if "/tasks/task-" in path and "/updates" in path:
            turn_no = int(path.split("/tasks/task-")[1].split("/")[0])
            spec = self._spec_for(turn_no)
            assistant = spec.get("assistant")
            if assistant is not None and not any(
                item["id"] == f"m-a-{turn_no}" for item in self._messages
            ):
                self._messages.append({**assistant, "id": f"m-a-{turn_no}"})
            return _FakeResponse(
                200,
                {
                    "task": {"id": f"task-{turn_no}", "status": spec["final_status"]},
                    "events": {"items": list(spec.get("events", [])), "next_after_id": 7},
                    "artifacts": {"items": list(spec.get("artifacts", [])), "next_after_id": 3},
                },
            )
        if path == "/api/conversations/conv-1/messages":
            return _FakeResponse(200, {"items": list(self._messages), "unstable_turn": None})
        return _FakeResponse(404, {"detail": "no route"})


def turn_spec(**kwargs: Any) -> dict[str, Any]:
    """Build one scripted turn spec for ``fake_client(turns=[...])``."""
    return _default_turn_spec(**kwargs)


@pytest.fixture
def fake_client() -> Any:
    """Factory for scripted fake clients (single-turn legacy or per-turn specs)."""

    def _make(**kwargs: Any) -> _FakeClient:
        return _FakeClient(**kwargs)

    return _make


@pytest.fixture
def make_scenario() -> Any:
    """Factory for Scenario with sensible defaults."""

    def _make(**overrides: Any) -> Scenario:
        base = dict(
            id="s1",
            question="What dataset is used?",
            depth="quick",
            paper_external_id="2001.09899v1",
            expect_citations_gte=2,
        )
        base.update(overrides)
        return Scenario(**base)

    return _make


@pytest.fixture
def passing_outcome(make_scenario: Any) -> Any:
    """Factory for a healthy single-turn ScenarioOutcome (CoNLL-2003 answer).

    Turn-level kwargs (task/assistant_message/citations/events/artifacts) map
    onto the single turn; scenario/error map onto the outcome.
    """

    def _make(**overrides: Any) -> ScenarioOutcome:
        turn = TurnObservation(
            task={"status": "completed"},
            assistant_message={
                "content": "the dataset is CoNLL-2003",
                "metadata": {
                    "citations": [{"evidence_id": f"e{i}"} for i in range(3)],
                    "research_result": {
                        "evidence_items": [{"id": f"e{i}"} for i in range(3)],
                        "used_papers": [],
                        "limitations": [],
                    },
                },
            },
            citations=[{"evidence_id": f"e{i}"} for i in range(3)],
            events=[{"type": "task_started"}],
            artifacts=[{"id": 1}],
        )
        turn_keys = {"task", "assistant_message", "citations", "events", "artifacts"}
        for key in turn_keys & overrides.keys():
            setattr(turn, key, overrides.pop(key))
        return ScenarioOutcome(
            scenario=overrides.pop("scenario", None) or make_scenario(),
            turns=[turn],
            error=overrides.pop("error", ""),
        )

    return _make
