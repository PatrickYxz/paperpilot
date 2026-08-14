"""HTTP contracts for the PaperPilot conversation workspace."""
from __future__ import annotations

import json
import os
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

import paperpilot.deep_reading.graph as graph_module
from paperpilot.deep_reading.runner import DeepReadingRunner
from paperpilot.deep_reading.runner import DeepReadingCheckpoint
from paperpilot.papers import PaperCandidate
from paperpilot.tools.types import Tool
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.app import create_app
from paperpilot.web.auth import SESSION_COOKIE_NAME
from paperpilot.web.checkpoint import SqliteCheckpointRuntime
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.task_executor import (
    TaskExecutorAtCapacityError,
    TaskSubmissionReservation,
)
from paperpilot.web.task_store import TaskStore


PRIMARY = PaperCandidate(
    external_id="2401.12345v2",
    title="A Test Paper",
    authors=["Ada Lovelace"],
    abstract="abstract",
    source_url="https://arxiv.org/abs/2401.12345v2",
)


class FakeCheckpointRuntime:
    def __init__(self) -> None:
        self.saver = object()
        self.health_calls = 0
        self.close_calls = 0
        self.health_error: Exception | None = None

    def check_health(self) -> None:
        self.health_calls += 1
        if self.health_error is not None:
            raise self.health_error

    def close(self) -> None:
        self.close_calls += 1


class FakeDeepReadingRunner:
    def __init__(self) -> None:
        self.checkpoints: dict[tuple[str, str], DeepReadingCheckpoint | None] = {}
        self.read_calls: list[tuple[str, str]] = []
        self.model_calls = 0

    def read_checkpoint(
        self,
        conversation_id: str,
        checkpoint_id: str,
    ) -> DeepReadingCheckpoint | None:
        self.read_calls.append((conversation_id, checkpoint_id))
        return self.checkpoints.get((conversation_id, checkpoint_id))

    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        del allow_running
        self.model_calls += 1
        raise AssertionError(f"Web API must not run the model directly: {task_id}")


class RecordingExecutor:
    def __init__(self) -> None:
        self.is_shutdown = False
        self.submissions: list[str] = []
        self.releases = 0

    def reserve(self) -> TaskSubmissionReservation:
        def release() -> None:
            self.releases += 1

        return TaskSubmissionReservation(self._submit, release)

    def _submit(self, task_id: str) -> object:
        self.submissions.append(task_id)
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def submit(self, task_id: str) -> object:
        return self.reserve().submit(task_id)

    def shutdown(self) -> None:
        self.is_shutdown = True


class RejectingExecutor(RecordingExecutor):
    def reserve(self) -> TaskSubmissionReservation:
        raise TaskExecutorAtCapacityError("full")


class FailingExecutor(RecordingExecutor):
    def _submit(self, task_id: str) -> object:
        raise ConnectionError("broker unavailable")


class ClaimingFailingExecutor(RecordingExecutor):
    def __init__(self, store: TaskStore) -> None:
        super().__init__()
        self.store = store

    def _submit(self, task_id: str) -> object:
        assert self.store.claim_task(task_id) is not None
        raise ConnectionError("publish result was ambiguous")


class WorkerFailingThenAmbiguousExecutor(RecordingExecutor):
    def __init__(self, store: TaskStore) -> None:
        super().__init__()
        self.store = store

    def _submit(self, task_id: str) -> object:
        assert self.store.claim_task(task_id) is not None
        failed = self.store.fail_conversation_task(
            task_id=task_id,
            message="Worker failed after claiming the task.",
            stage="deep_reading",
            payload={"error_type": "WorkerFailure"},
        )
        assert failed is not None and failed.status == "failed"
        raise ConnectionError("publish result was ambiguous")


class _LifecycleMCPClient:
    def __init__(self, tool_calls: list[tuple[str, dict[str, object]]]) -> None:
        self.tool_calls = tool_calls
        self.start_count = 0
        self.list_tools_count = 0
        self.close_count = 0

    def start(self) -> None:
        self.start_count += 1

    def list_tools(self) -> list[Tool]:
        self.list_tools_count += 1

        def download(arguments: dict[str, object]) -> str:
            self.tool_calls.append(("download", dict(arguments)))
            return json.dumps(
                {
                    "paper_id": arguments["arxiv_id"],
                    "text": "bounded trusted paper text",
                }
            )

        def build(arguments: dict[str, object]) -> str:
            self.tool_calls.append(("build", dict(arguments)))
            documents = arguments["documents"]
            assert isinstance(documents, list) and len(documents) == 1
            return json.dumps(
                {"fresh_papers": [documents[0]["paper_id"]]}
            )

        return [
            Tool(
                name="mcp__arxiv__download_paper",
                description="bounded fake download",
                input_schema={"type": "object"},
                handler=download,
            ),
            Tool(
                name="mcp__colbert__build_index",
                description="bounded fake build",
                input_schema={"type": "object"},
                handler=build,
            ),
        ]

    def close(self) -> None:
        self.close_count += 1


class _LifecycleModel:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = calls

    def write(self, state: dict[str, Any], task_id: str) -> dict[str, object]:
        contents = [message.content for message in state["messages"]]
        self.calls.append({"task_id": task_id, "contents": contents})
        question = contents[-1]
        return {
            "content": f"bounded answer: {question}",
            "citations": [],
            "result_quality": "partial",
        }


class _LifecycleModelFactory:
    def __init__(self, calls: list[dict[str, object]]) -> None:
        self.calls = calls
        self.factory_calls = 0

    def __call__(self) -> _LifecycleModel:
        self.factory_calls += 1
        return _LifecycleModel(self.calls)


@dataclass
class AppHarness:
    store: TaskStore
    runner: FakeDeepReadingRunner
    checkpoint: FakeCheckpointRuntime
    executor: RecordingExecutor
    client: TestClient
    paper_search_calls: list[tuple[str, int]]


def _paper_search(calls: list[tuple[str, int]]):
    def search(query: str, limit: int) -> list[PaperCandidate]:
        calls.append((query, limit))
        return [PRIMARY]

    return search


def _harness(
    tmp_path,
    *,
    executor: RecordingExecutor | None = None,
    runtime_config: WebRuntimeConfig | None = None,
) -> AppHarness:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    actual_executor = executor or RecordingExecutor()
    calls: list[tuple[str, int]] = []
    app = create_app(
        store,
        task_executor=actual_executor,
        runtime_config=runtime_config or WebRuntimeConfig(),
        checkpoint_runtime=checkpoint,
        deep_reading_runner=runner,
        paper_search=_paper_search(calls),
    )
    return AppHarness(
        store=store,
        runner=runner,
        checkpoint=checkpoint,
        executor=actual_executor,
        client=TestClient(app),
        paper_search_calls=calls,
    )


def _register(client: TestClient, username: str = "alice") -> dict[str, str]:
    response = client.post(
        "/api/auth/register",
        json={"username": username, "password": "secret123"},
    )
    assert response.status_code == 201
    return response.json()


def _create_conversation(client: TestClient, *, title: str | None = None) -> dict:
    body: dict[str, Any] = {
        "paper": {"source": "arxiv", "external_id": PRIMARY.external_id},
        "title": title,
    }
    response = client.post("/api/conversations", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def _complete_turn(
    store: TaskStore,
    *,
    user_id: str,
    conversation_id: str,
    expected_head_message_id: str | None,
    checkpoint_id: str,
    content: str,
):
    turn = store.create_conversation_turn(
        user_id=user_id,
        conversation_id=conversation_id,
        content=f"Question for {content}",
        depth="standard",
        expected_head_message_id=expected_head_message_id,
    )
    published = store.publish_conversation_result(
        task_id=turn.task.id,
        content=content,
        metadata={"answer_draft": {"content": content}},
        used_papers=[],
    )
    detail = store.get_conversation_detail(conversation_id, user_id=user_id)
    assert detail is not None
    finalized = store.finalize_conversation_task(
        task_id=turn.task.id,
        assistant_message_id=published.message.id,
        final_checkpoint_id=checkpoint_id,
        result_quality="complete",
        active_paper_ids=[detail.conversation.primary_paper_id],
    )
    return turn, published, finalized


def _checkpoint(
    *,
    checkpoint_id: str,
    task_id: str,
    message_id: str,
    active_paper_ids: list[str],
    schema_version: int = 1,
    graph_version: str = "conversation-v1",
    is_complete: bool = True,
) -> DeepReadingCheckpoint:
    return DeepReadingCheckpoint(
        checkpoint_id=checkpoint_id,
        state={
            "schema_version": schema_version,
            "graph_version": graph_version,
            "current_task_id": task_id,
            "published_message_id": message_id,
            "active_paper_ids": active_paper_ids,
        },
        is_complete=is_complete,
    )


__all__ = [
    name for name in globals() if not name.startswith("__")
]
