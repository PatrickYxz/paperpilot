"""Served contracts for the Conversation workspace UI."""
from __future__ import annotations

import re

from fastapi.testclient import TestClient

from paperpilot.web.app import create_app
from paperpilot.web.task_executor import SynchronousTaskExecutor
from paperpilot.web.task_store import TaskStore
from paperpilot.web.workflow import WorkflowRunner


def _client(tmp_path) -> TestClient:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    return TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=SynchronousTaskExecutor(runner),
        )
    )


def _register(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "secret123"},
    )
    assert response.status_code == 201


def test_conversation_workspace_and_legacy_workbench_assets_are_served(tmp_path):
    client = _client(tmp_path)

    html = client.get("/")
    conversation_css = client.get("/static/conversations.css")
    conversation_js = client.get("/static/conversations.js")
    legacy_css = client.get("/static/styles.css")
    legacy_js = client.get("/static/app.js")

    assert html.status_code == 200
    for element_id in (
        "conversationView",
        "conversationList",
        "paperSearch",
        "paperCandidates",
        "messageList",
        "messageComposer",
        "branchSelector",
    ):
        assert f'id="{element_id}"' in html.text
    for legacy_id in ("taskForm", "taskList", "evalSnapshot"):
        assert f'id="{legacy_id}"' in html.text
    assert 'href="/static/styles.css"' in html.text
    assert 'href="/static/conversations.css"' in html.text
    assert 'src="/static/app.js"' in html.text
    assert 'src="/static/conversations.js"' in html.text
    assert conversation_css.status_code == 200
    assert conversation_js.status_code == 200
    assert legacy_css.status_code == 200
    assert legacy_js.status_code == 200


def test_conversation_client_exposes_state_machine_and_auth_lifecycle(tmp_path):
    client = _client(tmp_path)

    conversation_js = client.get("/static/conversations.js").text
    legacy_js = client.get("/static/app.js").text

    for function_name in (
        "searchPapers",
        "createConversation",
        "loadConversations",
        "loadConversation",
        "submitMessage",
        "pollConversationTask",
        "loadAlternatives",
        "rollbackToMessage",
    ):
        assert f"function {function_name}" in conversation_js
    for state_field in (
        "selectedConversationId",
        "selectedConversation",
        "messages",
        "activeTaskId",
        "eventAfterId",
        "artifactAfterId",
        "pollTimer",
        "requestVersion",
    ):
        assert state_field in conversation_js
    assert "expected_head_message_id" in conversation_js
    assert "encodeURIComponent" in conversation_js
    assert "会话已更新，请重试" in conversation_js
    assert "paperpilot:authenticated" in legacy_js
    assert "paperpilot:unauthenticated" in legacy_js
    assert "paperpilot:authenticated" in conversation_js
    assert "paperpilot:unauthenticated" in conversation_js


def test_served_hidden_rule_keeps_auth_and_workspace_views_exclusive(tmp_path):
    css = _client(tmp_path).get("/static/conversations.css")

    assert css.status_code == 200
    assert re.search(
        r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important\s*;?[^}]*\}",
        css.text,
        flags=re.DOTALL,
    )


def test_stale_alternative_failure_is_guarded_by_conversation_selection(tmp_path):
    script = _client(tmp_path).get("/static/conversations.js").text
    function_body = script.split("async function loadAlternatives", 1)[1].split(
        "async function rollbackToMessage", 1
    )[0]
    catch_index = function_body.find("catch (error)")

    assert catch_index >= 0
    assert (
        function_body.find(
            "if (!isCurrentConversation(conversationId, version))", catch_index
        )
        > catch_index
    )
    control_body = script.split("const alternativesButton", 1)[1].split(
        "controls.append", 1
    )[0]
    assert "loadAlternatives(message.id);" in control_body


def test_search_create_and_list_requests_ignore_prior_auth_session(tmp_path):
    script = _client(tmp_path).get("/static/conversations.js").text

    assert "let authenticationVersion = 0;" in script
    assert script.count("authenticationVersion += 1;") >= 2
    for function_name, next_function_name in (
        ("searchPapers", "createConversation"),
        ("createConversation", "renderConversationList"),
        ("loadConversations", "renderConversationHeader"),
    ):
        function_body = script.split(
            f"async function {function_name}", 1
        )[1].split(next_function_name, 1)[0]
        assert "const authRequestVersion = authenticationVersion;" in function_body
        assert "isCurrentAuthentication(authRequestVersion)" in function_body


def test_rollback_success_notice_is_guarded_after_conversation_reload(tmp_path):
    script = _client(tmp_path).get("/static/conversations.js").text
    function_body = script.split("async function rollbackToMessage", 1)[1].split(
        "function resetConversationWorkspace", 1
    )[0]
    reload_index = function_body.find("await loadConversation(conversationId);")
    success_index = function_body.find(
        'setConversationMessage("Conversation rolled back.", "success")'
    )

    assert reload_index >= 0
    assert success_index > reload_index
    guard_body = function_body[reload_index:success_index]
    assert "const reloadVersion = conversationState.requestVersion;" in guard_body
    assert "isCurrentConversation(conversationId, reloadVersion)" in guard_body


def test_conversation_task_updates_route_is_the_conversation_polling_boundary(tmp_path):
    script = _client(tmp_path).get("/static/conversations.js").text
    polling_body = script.split("async function pollConversationTask", 1)[1].split(
        "async function reloadAfterConflict", 1
    )[0]

    assert "`/api/conversations/${encodeURIComponent(conversationId)}`" in polling_body
    assert "`/tasks/${encodeURIComponent(taskId)}/updates?${params.toString()}`" in polling_body
    assert "`/api/tasks/${encodeURIComponent(taskId)}/updates?${params.toString()}`" not in polling_body
