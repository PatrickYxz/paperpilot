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


def test_rollback_openapi_exposes_only_owned_message_contract(tmp_path):
    harness = _harness(tmp_path)

    schema = harness.client.get("/openapi.json").json()
    rollback_path = schema["paths"]["/api/conversations/{conversation_id}/rollback"]
    request_ref = rollback_path["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"]
    assert request_ref == "#/components/schemas/RollbackRequest"
    request_schema = schema["components"]["schemas"]["RollbackRequest"]
    assert set(request_schema["properties"]) == {
        "message_id",
        "expected_head_message_id",
    }
    assert set(request_schema["required"]) == {
        "message_id",
        "expected_head_message_id",
    }
    assert request_schema["additionalProperties"] is False
    assert not any("checkpoint" in path for path in schema["paths"])


def test_rollback_cannot_reach_another_users_checkpoint_by_supplying_ids(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    executor = RecordingExecutor()
    calls: list[tuple[str, int]] = []
    app = create_app(
        store,
        task_executor=executor,
        checkpoint_runtime=checkpoint,
        deep_reading_runner=runner,
        paper_search=_paper_search(calls),
    )
    alice_client = TestClient(app)
    bob_client = TestClient(app)
    alice = _register(alice_client, "alice")
    bob = _register(bob_client, "bob")
    alice_conversation = _create_conversation(alice_client)
    bob_conversation = _create_conversation(bob_client)
    alice_turn, alice_first, _ = _complete_turn(
        store,
        user_id=alice["id"],
        conversation_id=alice_conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="alice-cp-1",
        content="Alice first answer",
    )
    _, alice_second, _ = _complete_turn(
        store,
        user_id=alice["id"],
        conversation_id=alice_conversation["id"],
        expected_head_message_id=alice_first.message.id,
        checkpoint_id="alice-cp-2",
        content="Alice second answer",
    )
    _, bob_first, _ = _complete_turn(
        store,
        user_id=bob["id"],
        conversation_id=bob_conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="bob-cp-1",
        content="Bob first answer",
    )
    _, bob_second, _ = _complete_turn(
        store,
        user_id=bob["id"],
        conversation_id=bob_conversation["id"],
        expected_head_message_id=bob_first.message.id,
        checkpoint_id="bob-cp-2",
        content="Bob second answer",
    )
    runner.checkpoints[(alice_conversation["id"], "alice-cp-1")] = _checkpoint(
        checkpoint_id="alice-cp-1",
        task_id=alice_turn.task.id,
        message_id=alice_first.message.id,
        active_paper_ids=[alice_conversation["primary_paper_id"]],
    )

    responses = [
        bob_client.post(
            f"/api/conversations/{alice_conversation['id']}/rollback",
            json={
                "message_id": alice_first.message.id,
                "expected_head_message_id": alice_second.message.id,
            },
        ),
        bob_client.post(
            f"/api/conversations/{bob_conversation['id']}/rollback",
            json={
                "message_id": alice_first.message.id,
                "expected_head_message_id": bob_second.message.id,
            },
        ),
        alice_client.post(
            f"/api/conversations/{alice_conversation['id']}/rollback",
            json={
                "message_id": alice_first.message.id,
                "expected_head_message_id": alice_second.message.id,
                "checkpoint_id": "alice-cp-1",
            },
        ),
    ]

    assert [response.status_code for response in responses] == [404, 404, 422]
    assert [response.json() for response in responses[:2]] == [
        {"detail": "conversation not found"},
        {"detail": "message not found"},
    ]
    assert runner.read_calls == []
    assert runner.model_calls == 0


def test_conversation_lifecycle_survives_restart_rollback_and_branch_switch(
    tmp_path,
    monkeypatch,
):
    for variable in (
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")

    def fake_research(_state: dict[str, Any], runtime: Any) -> dict[str, object]:
        return {
            "research_result": {
                "evidence_items": [],
                "used_papers": [],
                "limitations": [f"bounded fake research for {runtime.context.task_id}"],
            }
        }

    def fake_writer(state: dict[str, Any], runtime: Any) -> dict[str, object]:
        return {
            "answer_draft": runtime.context.model.write(
                state,
                runtime.context.task_id,
            )
        }

    monkeypatch.setattr(graph_module, "research_evidence", fake_research)
    monkeypatch.setattr(graph_module, "write_answer", fake_writer)

    business_path = tmp_path / "business.sqlite3"
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    model_calls: list[dict[str, object]] = []
    model_factory = _LifecycleModelFactory(model_calls)
    tool_calls: list[tuple[str, dict[str, object]]] = []
    mcp_clients: list[_LifecycleMCPClient] = []
    paper_search_calls: list[tuple[str, int]] = []

    def fake_paper_search(query: str, limit: int) -> list[PaperCandidate]:
        paper_search_calls.append((query, limit))
        return [PRIMARY]

    def new_worker_mcp() -> MCPRuntime:
        client = _LifecycleMCPClient(tool_calls)
        mcp_clients.append(client)
        return MCPRuntime(lambda: client)

    def new_web_reader(
        store: TaskStore,
        checkpoint: SqliteCheckpointRuntime,
    ) -> tuple[DeepReadingRunner, MCPRuntime]:
        def forbidden_client():
            raise AssertionError("Web checkpoint reader started MCP")

        def forbidden_model():
            raise AssertionError("Web checkpoint reader started the model")

        reader_mcp = MCPRuntime(forbidden_client)
        return (
            DeepReadingRunner(
                task_store=store,
                checkpointer=checkpoint.saver,
                mcp_runtime=reader_mcp,
                model_factory=forbidden_model,
                paper_search=lambda _query, _limit: (_ for _ in ()).throw(
                    AssertionError("Web checkpoint reader searched papers")
                ),
            ),
            reader_mcp,
        )

    def run_worker_turn(
        store: TaskStore,
        runner: DeepReadingRunner,
        task_id: str,
    ) -> None:
        assert runner.run(task_id) is True
        completed = store.get_task(task_id)
        assert completed is not None and completed.status == "completed"

    web_store_1 = TaskStore(business_path)
    web_checkpoint_1 = SqliteCheckpointRuntime.open(checkpoint_path)
    web_reader_1, web_reader_mcp_1 = new_web_reader(
        web_store_1,
        web_checkpoint_1,
    )
    executor_1 = RecordingExecutor()
    worker_store_1 = TaskStore(business_path)
    worker_checkpoint_1 = SqliteCheckpointRuntime.open(checkpoint_path)
    worker_mcp_1 = new_worker_mcp()
    worker_runner_1 = DeepReadingRunner(
        task_store=worker_store_1,
        checkpointer=worker_checkpoint_1.saver,
        mcp_runtime=worker_mcp_1,
        model_factory=model_factory,
        paper_search=fake_paper_search,
    )
    try:
        app_1 = create_app(
            web_store_1,
            task_executor=executor_1,
            checkpoint_runtime=web_checkpoint_1,
            deep_reading_runner=web_reader_1,
            paper_search=fake_paper_search,
        )
        with TestClient(app_1) as client_1:
            alice = _register(client_1)
            search = client_1.get(
                "/api/papers/search",
                params={"q": "bounded paper search", "limit": 5},
            )
            assert search.status_code == 200
            assert search.json()["items"][0]["external_id"] == PRIMARY.external_id
            conversation = _create_conversation(client_1)
            first_submit = client_1.post(
                f"/api/conversations/{conversation['id']}/messages",
                json={
                    "content": "What is the first finding?",
                    "depth": "standard",
                    "expected_head_message_id": None,
                },
            )
            assert first_submit.status_code == 202
            first_task_id = first_submit.json()["task"]["id"]
            run_worker_turn(worker_store_1, worker_runner_1, first_task_id)
            first_task = worker_store_1.get_task(first_task_id, user_id=alice["id"])
            assert first_task is not None and first_task.final_checkpoint_id
            first_checkpoint_id = first_task.final_checkpoint_id
            first_messages = client_1.get(
                f"/api/conversations/{conversation['id']}/messages"
            )
            assert first_messages.status_code == 200
            first_assistant = first_messages.json()["items"][-1]
            assert first_assistant["content"] == (
                "bounded answer: What is the first finding?"
            )
            session_token = client_1.cookies.get(SESSION_COOKIE_NAME)
            assert session_token
            assert worker_runner_1.read_checkpoint(
                conversation["id"], first_checkpoint_id
            ) is not None
    finally:
        worker_mcp_1.close()
        worker_checkpoint_1.close()
        worker_store_1.close()
        web_reader_mcp_1.close()
        web_checkpoint_1.close()
        web_store_1.close()

    web_store_2 = TaskStore(business_path)
    web_checkpoint_2 = SqliteCheckpointRuntime.open(checkpoint_path)
    web_reader_2, web_reader_mcp_2 = new_web_reader(
        web_store_2,
        web_checkpoint_2,
    )
    executor_2 = RecordingExecutor()
    worker_store_2 = TaskStore(business_path)
    worker_checkpoint_2 = SqliteCheckpointRuntime.open(checkpoint_path)
    worker_mcp_2 = new_worker_mcp()
    worker_runner_2 = DeepReadingRunner(
        task_store=worker_store_2,
        checkpointer=worker_checkpoint_2.saver,
        mcp_runtime=worker_mcp_2,
        model_factory=model_factory,
        paper_search=fake_paper_search,
    )
    try:
        app_2 = create_app(
            web_store_2,
            task_executor=executor_2,
            checkpoint_runtime=web_checkpoint_2,
            deep_reading_runner=web_reader_2,
            paper_search=fake_paper_search,
        )
        with TestClient(app_2) as client_2:
            client_2.cookies.set(SESSION_COOKIE_NAME, session_token)
            me = client_2.get("/api/auth/me")
            assert me.status_code == 200 and me.json()["id"] == alice["id"]

            second_submit = client_2.post(
                f"/api/conversations/{conversation['id']}/messages",
                json={
                    "content": "How does the second point follow?",
                    "depth": "deep",
                    "expected_head_message_id": first_assistant["id"],
                },
            )
            assert second_submit.status_code == 202
            second_task_id = second_submit.json()["task"]["id"]
            run_worker_turn(worker_store_2, worker_runner_2, second_task_id)
            second_task = worker_store_2.get_task(
                second_task_id,
                user_id=alice["id"],
            )
            assert second_task is not None and second_task.final_checkpoint_id
            second_checkpoint_id = second_task.final_checkpoint_id
            assert second_task.base_checkpoint_id == first_checkpoint_id
            second_messages = client_2.get(
                f"/api/conversations/{conversation['id']}/messages"
            ).json()["items"]
            second_assistant = second_messages[-1]
            assert model_calls[1]["contents"] == [
                "What is the first finding?",
                "bounded answer: What is the first finding?",
                "How does the second point follow?",
            ]

            calls_before_rollback = (
                len(model_calls),
                model_factory.factory_calls,
                len(tool_calls),
            )
            rollback = client_2.post(
                f"/api/conversations/{conversation['id']}/rollback",
                json={
                    "message_id": first_assistant["id"],
                    "expected_head_message_id": second_assistant["id"],
                },
            )
            assert rollback.status_code == 200
            assert rollback.json()["head_message_id"] == first_assistant["id"]
            assert (
                len(model_calls),
                model_factory.factory_calls,
                len(tool_calls),
            ) == calls_before_rollback

            third_submit = client_2.post(
                f"/api/conversations/{conversation['id']}/messages",
                json={
                    "content": "Give an alternate third direction.",
                    "depth": "quick",
                    "expected_head_message_id": first_assistant["id"],
                },
            )
            assert third_submit.status_code == 202
            third_task_id = third_submit.json()["task"]["id"]
            run_worker_turn(worker_store_2, worker_runner_2, third_task_id)
            third_task = worker_store_2.get_task(third_task_id, user_id=alice["id"])
            assert third_task is not None and third_task.final_checkpoint_id
            assert third_task.base_checkpoint_id == first_checkpoint_id
            assert third_task.final_checkpoint_id != second_checkpoint_id
            assert model_calls[2]["contents"] == [
                "What is the first finding?",
                "bounded answer: What is the first finding?",
                "Give an alternate third direction.",
            ]
            assert worker_runner_2.read_checkpoint(
                conversation["id"], second_checkpoint_id
            ) is not None

            third_messages = client_2.get(
                f"/api/conversations/{conversation['id']}/messages"
            ).json()["items"]
            third_assistant = third_messages[-1]
            alternatives = client_2.get(
                f"/api/conversations/{conversation['id']}/messages/"
                f"{first_assistant['id']}/alternatives"
            )
            assert alternatives.status_code == 200
            assert {
                item["assistant_message"]["id"]
                for item in alternatives.json()["items"]
            } == {second_assistant["id"], third_assistant["id"]}

            calls_before_switch = (
                len(model_calls),
                model_factory.factory_calls,
                len(tool_calls),
            )
            switch_back = client_2.post(
                f"/api/conversations/{conversation['id']}/rollback",
                json={
                    "message_id": second_assistant["id"],
                    "expected_head_message_id": third_assistant["id"],
                },
            )
            assert switch_back.status_code == 200
            assert switch_back.json()["head_message_id"] == second_assistant["id"]
            assert (
                len(model_calls),
                model_factory.factory_calls,
                len(tool_calls),
            ) == calls_before_switch

            assert model_factory.factory_calls == 3
            assert len(model_calls) == 3
            assert [name for name, _arguments in tool_calls] == [
                "download",
                "build",
                "download",
                "build",
                "download",
                "build",
            ]
            assert paper_search_calls == [
                ("bounded paper search", 5),
                (PRIMARY.external_id, 1),
            ]
            assert executor_1.submissions == [first_task_id]
            assert executor_2.submissions == [
                second_task_id,
                third_task_id,
            ]
            assert all(
                variable not in os.environ
                for variable in (
                    "DEEPSEEK_API_KEY",
                    "ANTHROPIC_API_KEY",
                    "OPENAI_API_KEY",
                )
            )
    finally:
        worker_mcp_2.close()
        worker_checkpoint_2.close()
        worker_store_2.close()
        web_reader_mcp_2.close()
        web_checkpoint_2.close()
        web_store_2.close()

    assert len(mcp_clients) == 2
    assert [client.start_count for client in mcp_clients] == [1, 1]
    assert [client.list_tools_count for client in mcp_clients] == [1, 2]
    assert [client.close_count for client in mcp_clients] == [1, 1]


def test_paper_search_supports_natural_language_and_arxiv_url_without_llm(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)

    natural = harness.client.get(
        "/api/papers/search", params={"q": "retrieval augmented generation", "limit": 7}
    )
    exact = harness.client.get(
        "/api/papers/search",
        params={"q": "https://arxiv.org/abs/2401.12345v2", "limit": 10},
    )

    expected = {
        "items": [
            {
                "source": "arxiv",
                "external_id": "2401.12345v2",
                "title": "A Test Paper",
                "authors": ["Ada Lovelace"],
                "abstract": "abstract",
                "source_url": "https://arxiv.org/abs/2401.12345v2",
            }
        ]
    }
    assert natural.status_code == 200
    assert natural.json() == expected
    assert exact.status_code == 200
    assert exact.json() == expected
    assert harness.paper_search_calls == [
        ("retrieval augmented generation", 7),
        ("https://arxiv.org/abs/2401.12345v2", 10),
    ]
    assert harness.runner.model_calls == 0


@pytest.mark.parametrize(
    "path",
    [
        "/api/papers/search?q=test",
        "/api/conversations",
        "/api/conversations/conv_missing",
        "/api/conversations/conv_missing/messages",
    ],
)
def test_conversation_read_apis_require_authentication(tmp_path, path):
    harness = _harness(tmp_path)

    response = harness.client.get(path)

    assert response.status_code == 401
    assert response.json() == {"detail": "authentication required"}


def test_conversation_create_list_detail_title_and_soft_archive(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)

    created = _create_conversation(harness.client, title="Initial title")
    listed = harness.client.get("/api/conversations")
    detail = harness.client.get(f"/api/conversations/{created['id']}")
    renamed = harness.client.patch(
        f"/api/conversations/{created['id']}", json={"title": "Renamed"}
    )
    archived = harness.client.patch(
        f"/api/conversations/{created['id']}", json={"archived": True}
    )

    assert created["title"] == "Initial title"
    assert created["head_message_id"] is None
    assert created["head_checkpoint_id"] is None
    assert listed.status_code == 200
    assert listed.json()["items"] == [created]
    assert detail.status_code == 200
    assert detail.json()["conversation"] == created
    assert detail.json()["primary_paper"]["external_id"] == PRIMARY.external_id
    assert detail.json()["active_papers"] == [detail.json()["primary_paper"]]
    assert detail.json()["active_task"] is None
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"
    assert archived.status_code == 200
    assert archived.json()["archived_at"] is not None
    assert harness.client.get("/api/conversations").json() == {"items": []}
    archived_list = harness.client.get(
        "/api/conversations", params={"include_archived": True}
    )
    assert [item["id"] for item in archived_list.json()["items"]] == [created["id"]]
    assert harness.runner.model_calls == 0


def test_conversation_create_resolves_server_side_paper_metadata(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)

    created = _create_conversation(harness.client)

    assert created["title"] == PRIMARY.title
    assert harness.paper_search_calls == [(PRIMARY.external_id, 1)]


def test_conversation_owner_isolation_uses_uniform_404(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    executor = RecordingExecutor()
    calls: list[tuple[str, int]] = []
    app = create_app(
        store,
        task_executor=executor,
        checkpoint_runtime=checkpoint,
        deep_reading_runner=runner,
        paper_search=_paper_search(calls),
    )
    alice = TestClient(app)
    bob = TestClient(app)
    _register(alice, "alice")
    _register(bob, "bob")
    created = _create_conversation(alice)

    responses = [
        bob.get(f"/api/conversations/{created['id']}"),
        bob.patch(f"/api/conversations/{created['id']}", json={"title": "stolen"}),
        bob.get(f"/api/conversations/{created['id']}/messages"),
        bob.get(
            f"/api/conversations/{created['id']}/messages/msg_missing/alternatives"
        ),
        bob.post(
            f"/api/conversations/{created['id']}/messages",
            json={
                "content": "stolen",
                "depth": "standard",
                "expected_head_message_id": None,
            },
        ),
        bob.post(
            f"/api/conversations/{created['id']}/rollback",
            json={"message_id": "msg_missing", "expected_head_message_id": None},
        ),
    ]

    assert [response.status_code for response in responses] == [404] * 6


def test_conversation_task_updates_require_matching_owner_conversation_and_task(
    tmp_path,
):
    harness = _harness(tmp_path)
    alice = _register(harness.client, "alice-task-updates-api")
    bob = TestClient(harness.client.app)
    _register(bob, "bob-task-updates-api")
    conversation = _create_conversation(harness.client)
    other_conversation = _create_conversation(harness.client)
    submitted = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Scope this progress update.",
            "depth": "standard",
            "expected_head_message_id": None,
        },
    )
    assert submitted.status_code == 202
    task_id = submitted.json()["task"]["id"]

    owner_response = harness.client.get(
        f"/api/conversations/{conversation['id']}/tasks/{task_id}/updates"
    )
    forbidden_responses = [
        bob.get(f"/api/conversations/{conversation['id']}/tasks/{task_id}/updates"),
        harness.client.get(f"/api/conversations/conv_missing/tasks/{task_id}/updates"),
        harness.client.get(
            f"/api/conversations/{conversation['id']}/tasks/task_missing/updates"
        ),
        harness.client.get(
            f"/api/conversations/{other_conversation['id']}/tasks/{task_id}/updates"
        ),
    ]

    assert owner_response.status_code == 200
    assert owner_response.json()["task"]["id"] == task_id
    assert [response.status_code for response in forbidden_responses] == [404] * 4
    schema = harness.client.get("/openapi.json").json()
    assert (
        "/api/conversations/{conversation_id}/tasks/{task_id}/updates"
        in schema["paths"]
    )


def test_message_submission_reserves_then_persists_then_submits_real(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)
    conversation = _create_conversation(harness.client)

    response = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Compare its method.",
            "depth": "deep",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["user_message"]["content"] == "Compare its method."
    assert payload["task"]["status"] == "pending"
    assert payload["stable_head_message_id"] is None
    assert harness.executor.submissions == [payload["task"]["id"]]
    stored = harness.store.get_task(payload["task"]["id"])
    assert stored is not None and stored.conversation_id == conversation["id"]
    assert harness.runner.model_calls == 0


def test_message_submission_maps_busy_and_stale_head_to_409(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)
    conversation = _create_conversation(harness.client)
    body = {
        "content": "First",
        "depth": "standard",
        "expected_head_message_id": None,
    }

    first = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages", json=body
    )
    busy = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages", json=body
    )
    task_id = first.json()["task"]["id"]
    assert harness.store.fail_pending_task(task_id) is not None
    stale = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={**body, "content": "Stale", "expected_head_message_id": "msg_old"},
    )

    assert first.status_code == 202
    assert busy.status_code == 409
    assert busy.json() == {"detail": "conversation already has an active task"}
    assert stale.status_code == 409
    assert stale.json() == {"detail": "conversation head has changed"}


def test_message_capacity_rejection_writes_nothing(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = RejectingExecutor()
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    calls: list[tuple[str, int]] = []
    client = TestClient(
        create_app(
            store,
            task_executor=executor,
            checkpoint_runtime=checkpoint,
            deep_reading_runner=runner,
            paper_search=_paper_search(calls),
        )
    )
    user = _register(client)
    conversation = _create_conversation(client)

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Do not persist",
            "depth": "quick",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    detail = store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None
    assert detail.active_task is None
    assert store.list_active_messages(conversation["id"], user_id=user["id"]) == []
    assert store.get_unstable_turn(conversation["id"], user_id=user["id"]) is None


def test_message_capacity_rejection_uses_configured_retry_after(tmp_path):
    harness = _harness(
        tmp_path,
        executor=RejectingExecutor(),
        runtime_config=WebRuntimeConfig(overload_retry_after_seconds=5),
    )
    _register(harness.client)
    conversation = _create_conversation(harness.client)

    response = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Retry after configured delay",
            "depth": "quick",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"


def test_submit_failure_marks_pending_task_failed_and_preserves_stable_head(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    executor = FailingExecutor()
    calls: list[tuple[str, int]] = []
    client = TestClient(
        create_app(
            store,
            task_executor=executor,
            checkpoint_runtime=checkpoint,
            deep_reading_runner=runner,
            paper_search=_paper_search(calls),
        )
    )
    user = _register(client)
    conversation = _create_conversation(client)

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Queue failure",
            "depth": "quick",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 503
    detail = store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None and detail.conversation.head_message_id is None
    unstable = store.get_unstable_turn(conversation["id"], user_id=user["id"])
    assert unstable is not None and unstable.task.status == "failed"
    events = store.list_events_page(
        unstable.task.id, user_id=user["id"], after_id=0, limit=100
    )
    assert events is not None
    assert [event.type for event in events.items] == ["queued", "failed"]
    assert events.items[-1].stage == "queue"
    visible = client.get(f"/api/conversations/{conversation['id']}/messages")
    assert visible.status_code == 200
    assert visible.json()["unstable_turn"]["task"]["status"] == "failed"


def test_ambiguous_submit_failure_does_not_fail_worker_claimed_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = ClaimingFailingExecutor(store)
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    calls: list[tuple[str, int]] = []
    client = TestClient(
        create_app(
            store,
            task_executor=executor,
            checkpoint_runtime=checkpoint,
            deep_reading_runner=runner,
            paper_search=_paper_search(calls),
        )
    )
    user = _register(client)
    conversation = _create_conversation(client)

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Ambiguous publish",
            "depth": "standard",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 202
    unstable = store.get_unstable_turn(conversation["id"], user_id=user["id"])
    assert unstable is not None
    task = unstable.task
    assert task.status == "running"
    events = store.list_events_page(task.id, user_id=user["id"], after_id=0, limit=100)
    assert events is not None
    assert [event.type for event in events.items] == ["queued"]


def test_ambiguous_submit_after_worker_failure_still_returns_accepted(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = WorkerFailingThenAmbiguousExecutor(store)
    runner = FakeDeepReadingRunner()
    checkpoint = FakeCheckpointRuntime()
    calls: list[tuple[str, int]] = []
    client = TestClient(
        create_app(
            store,
            task_executor=executor,
            checkpoint_runtime=checkpoint,
            deep_reading_runner=runner,
            paper_search=_paper_search(calls),
        )
    )
    user = _register(client)
    conversation = _create_conversation(client)

    response = client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={
            "content": "Worker fails quickly",
            "depth": "standard",
            "expected_head_message_id": None,
        },
    )

    assert response.status_code == 202
    task_id = response.json()["task"]["id"]
    task = store.get_task(task_id, user_id=user["id"])
    assert task is not None and task.status == "failed"
    events = store.list_events_page(
        task_id,
        user_id=user["id"],
        after_id=0,
        limit=100,
    )
    assert events is not None
    assert [event.type for event in events.items] == ["queued", "failed"]
    assert events.items[-1].stage == "deep_reading"
    assert events.items[-1].message == "Worker failed after claiming the task."


def test_messages_expose_active_path_and_unstable_turn(tmp_path):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    _, published, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="Stable answer",
    )
    pending = harness.store.create_conversation_turn(
        user_id=user["id"],
        conversation_id=conversation["id"],
        content="Pending follow-up",
        depth="quick",
        expected_head_message_id=published.message.id,
    )

    response = harness.client.get(f"/api/conversations/{conversation['id']}/messages")

    assert response.status_code == 200
    payload = response.json()
    assert [item["role"] for item in payload["items"]] == ["user", "assistant"]
    assert payload["items"][-1]["id"] == published.message.id
    assert payload["unstable_turn"]["user_message"]["id"] == pending.user_message.id
    assert payload["unstable_turn"]["task"]["id"] == pending.task.id
    assert payload["unstable_turn"]["task"]["status"] == "pending"


def test_message_alternatives_return_direct_completed_user_assistant_pairs(tmp_path):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    detail = harness.store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None
    _, branch_point, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="Branch point",
    )
    _, first_branch, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=branch_point.message.id,
        checkpoint_id="cp-2",
        content="First branch",
    )
    harness.store.switch_conversation_head(
        conversation["id"],
        user_id=user["id"],
        expected_head_message_id=first_branch.message.id,
        target_message_id=branch_point.message.id,
        target_checkpoint_id="cp-1",
        active_paper_ids=[detail.conversation.primary_paper_id],
    )
    _, second_branch, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=branch_point.message.id,
        checkpoint_id="cp-3",
        content="Second branch",
    )

    response = harness.client.get(
        f"/api/conversations/{conversation['id']}/messages/"
        f"{branch_point.message.id}/alternatives"
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert {item["assistant_message"]["id"] for item in items} == {
        first_branch.message.id,
        second_branch.message.id,
    }
    assert all(item["user_message"]["role"] == "user" for item in items)


def test_rollback_switches_head_from_valid_checkpoint_without_model_call(tmp_path):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    detail = harness.store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None
    first_turn, first, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="First answer",
    )
    _, second, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=first.message.id,
        checkpoint_id="cp-2",
        content="Second answer",
    )
    harness.runner.checkpoints[(conversation["id"], "cp-1")] = _checkpoint(
        checkpoint_id="cp-1",
        task_id=first_turn.task.id,
        message_id=first.message.id,
        active_paper_ids=[detail.conversation.primary_paper_id],
    )

    response = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={
            "message_id": first.message.id,
            "expected_head_message_id": second.message.id,
        },
    )

    assert response.status_code == 200
    assert response.json()["head_message_id"] == first.message.id
    assert response.json()["head_checkpoint_id"] == "cp-1"
    assert harness.runner.read_calls == [(conversation["id"], "cp-1")]
    assert harness.runner.model_calls == 0


@pytest.mark.parametrize(
    ("mutation", "expected_detail"),
    [
        ("missing", "checkpoint is unavailable"),
        ("schema", "checkpoint schema is unsupported"),
        ("graph", "checkpoint graph is unsupported"),
        ("incomplete", "checkpoint is not complete"),
        ("published", "checkpoint does not match target message"),
        ("task", "checkpoint does not match target task"),
        ("active", "checkpoint active paper ids are invalid"),
    ],
)
def test_rollback_rejects_missing_or_invalid_checkpoint_without_moving_head(
    tmp_path,
    mutation,
    expected_detail,
):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    detail = harness.store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert detail is not None
    first_turn, first, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="First answer",
    )
    _, second, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=first.message.id,
        checkpoint_id="cp-2",
        content="Second answer",
    )
    checkpoint: DeepReadingCheckpoint | None = _checkpoint(
        checkpoint_id="cp-1",
        task_id=first_turn.task.id,
        message_id=first.message.id,
        active_paper_ids=[detail.conversation.primary_paper_id],
    )
    if mutation == "missing":
        checkpoint = None
    elif mutation == "schema":
        checkpoint.state["schema_version"] = 999
    elif mutation == "graph":
        checkpoint.state["graph_version"] = "conversation-v999"
    elif mutation == "incomplete":
        checkpoint = _checkpoint(
            checkpoint_id="cp-1",
            task_id=first_turn.task.id,
            message_id=first.message.id,
            active_paper_ids=[detail.conversation.primary_paper_id],
            is_complete=False,
        )
    elif mutation == "published":
        checkpoint.state["published_message_id"] = "msg_other"
    elif mutation == "task":
        checkpoint.state["current_task_id"] = "task_other"
    elif mutation == "active":
        checkpoint.state["active_paper_ids"] = [detail.conversation.primary_paper_id] * 2
    harness.runner.checkpoints[(conversation["id"], "cp-1")] = checkpoint

    response = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={
            "message_id": first.message.id,
            "expected_head_message_id": second.message.id,
        },
    )

    assert response.status_code == 409
    assert response.json() == {"detail": expected_detail}
    after = harness.store.get_conversation_detail(conversation["id"], user_id=user["id"])
    assert after is not None
    assert after.conversation.head_message_id == second.message.id
    assert after.conversation.head_checkpoint_id == "cp-2"
    assert harness.runner.model_calls == 0


def test_rollback_rejects_active_task_and_stale_head_before_checkpoint_read(tmp_path):
    harness = _harness(tmp_path)
    user = _register(harness.client)
    conversation = _create_conversation(harness.client)
    _, target, _ = _complete_turn(
        harness.store,
        user_id=user["id"],
        conversation_id=conversation["id"],
        expected_head_message_id=None,
        checkpoint_id="cp-1",
        content="Target answer",
    )
    active = harness.store.create_conversation_turn(
        user_id=user["id"],
        conversation_id=conversation["id"],
        content="Active",
        depth="standard",
        expected_head_message_id=target.message.id,
    )

    busy = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={
            "message_id": target.message.id,
            "expected_head_message_id": target.message.id,
        },
    )
    assert harness.store.fail_pending_task(active.task.id) is not None
    stale = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={
            "message_id": target.message.id,
            "expected_head_message_id": "msg_stale",
        },
    )

    assert busy.status_code == 409
    assert busy.json() == {"detail": "conversation already has an active task"}
    assert stale.status_code == 409
    assert stale.json() == {"detail": "conversation head has changed"}
    assert harness.runner.read_calls == []


def test_expected_head_fields_are_required_but_nullable(tmp_path):
    harness = _harness(tmp_path)
    _register(harness.client)
    conversation = _create_conversation(harness.client)

    missing_message_head = harness.client.post(
        f"/api/conversations/{conversation['id']}/messages",
        json={"content": "Question", "depth": "standard"},
    )
    missing_rollback_head = harness.client.post(
        f"/api/conversations/{conversation['id']}/rollback",
        json={"message_id": "msg_missing"},
    )

    assert missing_message_head.status_code == 422
    assert missing_rollback_head.status_code == 422
