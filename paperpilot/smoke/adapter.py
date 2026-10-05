"""HTTP adapter: the only place that knows the shape of the public API.

When entry points change, fix this class (and its tests); scenarios, checks,
and reports stay untouched. The client only needs ``.post`` / ``.get``
(TestClient in production, scripted fakes in tests).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from paperpilot.smoke.scenarios import Scenario

TERMINAL_TASK_STATUSES = ("completed", "failed")


class ConversationApi:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.conversation_id: str = ""
        self.task_id: str = ""

    def open_conversation(self, scenario: Scenario) -> None:
        username = f"smoke-{os.getpid()}-{int(time.time())}"
        resp = self._client.post(
            "/api/auth/register",
            json={"username": username, "password": "smoke-password-1"},
        )
        self._raise_for_status(resp, "register")
        self.open_conversation_as_current_user(scenario)

    def open_conversation_as_current_user(self, scenario: Scenario) -> None:
        """Open another conversation under the already-authenticated user."""
        resp = self._client.post(
            "/api/conversations",
            json={
                "paper": {
                    "source": "arxiv",
                    "external_id": scenario.paper_external_id,
                }
            },
        )
        self._raise_for_status(resp, "create conversation")
        self.conversation_id = resp.json()["id"]

    def ask(self, content: str, depth: str, head_message_id: str | None = None) -> str:
        """Submit one question; chain ``head_message_id`` from the previous
        turn's assistant message for follow-ups (first turn uses None)."""
        resp = self._client.post(
            f"/api/conversations/{self.conversation_id}/messages",
            json={
                "content": content,
                "depth": depth,
                "expected_head_message_id": head_message_id,
            },
        )
        self._raise_for_status(resp, "create message")
        body = resp.json()
        self.task_id = body["task"]["id"]
        return self.task_id

    def poll_until_terminal(
        self,
        *,
        timeout_s: float,
        interval_s: float = 2.0,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Poll the task until terminal; return (task, all events, all artifacts)."""
        events: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        after_event_id = 0
        after_artifact_id = 0
        deadline = time.monotonic() + timeout_s
        task: dict[str, Any] = {}
        while True:
            resp = self._client.get(
                f"/api/conversations/{self.conversation_id}"
                f"/tasks/{self.task_id}/updates",
                params={
                    "after_event_id": after_event_id,
                    "after_artifact_id": after_artifact_id,
                    "limit": 100,
                },
            )
            self._raise_for_status(resp, "task updates")
            body = resp.json()
            task = body["task"]
            batch = body.get("events", {})
            events.extend(batch.get("items", []))
            after_event_id = batch.get("next_after_id", after_event_id)
            artifact_batch = body.get("artifacts", {})
            artifacts.extend(artifact_batch.get("items", []))
            after_artifact_id = artifact_batch.get("next_after_id", after_artifact_id)
            if task.get("status") in TERMINAL_TASK_STATUSES:
                return task, events, artifacts
            if time.monotonic() > deadline:
                return task, events, artifacts

    def assistant_message(self) -> dict[str, Any] | None:
        """Return the latest assistant message (message list grows per turn)."""
        resp = self._client.get(
            f"/api/conversations/{self.conversation_id}/messages"
        )
        self._raise_for_status(resp, "list messages")
        latest: dict[str, Any] | None = None
        for item in resp.json().get("items", []):
            if item.get("role") == "assistant":
                latest = item
        return latest

    @staticmethod
    def _raise_for_status(resp: Any, action: str) -> None:
        status = getattr(resp, "status_code", 0)
        if status >= 400:
            detail = ""
            try:
                detail = json.dumps(resp.json(), ensure_ascii=False)[:300]
            except Exception:  # noqa: BLE001
                detail = (resp.text or "")[:300]
            raise RuntimeError(f"{action} failed: HTTP {status} {detail}")
