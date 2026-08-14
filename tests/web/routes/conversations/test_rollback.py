"""Conversation route tests grouped by HTTP responsibility."""
from __future__ import annotations

from tests.web.routes.conversations.conftest import *

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
