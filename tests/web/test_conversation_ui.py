"""Served contracts for the single Conversation workspace UI."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from paperpilot.web.app import create_app
from paperpilot.web.task_executor import SynchronousTaskExecutor
from paperpilot.web.task_store import TaskStore


STATIC_DIR = Path(__file__).parents[2] / "paperpilot" / "web" / "static"


class _UnusedRunner:
    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        raise AssertionError(f"unexpected task execution: {task_id}, {allow_running}")

    def fail_retry_exhausted(self, *_args, **_kwargs) -> None:
        raise AssertionError("unexpected retry exhaustion")


def _client(tmp_path) -> TestClient:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    return TestClient(
        create_app(
            store,
            task_executor=SynchronousTaskExecutor(_UnusedRunner()),
        )
    )


def test_only_single_conversation_workspace_assets_are_served(tmp_path):
    client = _client(tmp_path)

    html_response = client.get("/")
    css_response = client.get("/static/styles.css")
    javascript_response = client.get("/static/app.js")

    assert sorted(path.name for path in STATIC_DIR.iterdir()) == [
        "app.js",
        "index.html",
        "styles.css",
    ]
    assert html_response.status_code == 200
    assert css_response.status_code == 200
    assert javascript_response.status_code == 200

    html = html_response.text
    javascript = javascript_response.text
    removed_workbench_title = "Legacy " + "Workbench"
    removed_tab_id = "legacy" + "Tab"
    removed_task_api = "/api/" + "tasks"
    removed_eval_api = "/api/" + "eval"
    removed_script = "conversations" + ".js"
    removed_stylesheet = "conversations" + ".css"
    for element_id in (
        "authForm",
        "currentUser",
        "logoutButton",
        "workspace",
        "conversationView",
        "conversationList",
        "paperSearch",
        "paperCandidates",
        "messageList",
        "messageComposer",
        "branchSelector",
    ):
        assert f'id="{element_id}"' in html
    for removed_id in ("taskForm", "taskList", "evalSnapshot"):
        assert f'id="{removed_id}"' not in html
    assert removed_workbench_title not in html
    assert removed_tab_id not in html + javascript
    assert removed_task_api not in javascript
    assert removed_eval_api not in javascript
    assert removed_script not in html
    assert removed_stylesheet not in html
    assert 'href="/static/styles.css"' in html
    assert 'src="/static/app.js" defer' in html


def test_single_client_exposes_state_auth_and_conversation_lifecycle(tmp_path):
    javascript = _client(tmp_path).get("/static/app.js").text

    assert re.search(
        r"const state\s*=\s*\{\s*"
        r"currentUser:\s*null,\s*"
        r"conversations:\s*\[\],\s*"
        r"selectedConversationId:\s*null,\s*"
        r"selectedConversation:\s*null,\s*"
        r"activeTaskPoll:\s*null,?\s*\}",
        javascript,
    )
    for function_name in (
        "requestJson",
        "loadCurrentUser",
        "submitAuth",
        "setAuthenticatedUser",
        "clearAuthenticatedUser",
        "searchPapers",
        "createConversation",
        "loadConversations",
        "loadConversation",
        "submitMessage",
        "pollConversationTask",
        "loadAlternatives",
        "rollbackToMessage",
        "resetConversationWorkspace",
        "escapeHtml",
    ):
        assert f"function {function_name}" in javascript
    assert "let authenticationVersion = 0;" in javascript
    assert javascript.count("authenticationVersion += 1;") >= 2
    assert "let conversationListVersion = 0;" in javascript
    assert "let requestVersion = 0;" in javascript
    assert "expected_head_message_id" in javascript
    assert "encodeURIComponent" in javascript
    assert "会话已更新，请重试" in javascript
    assert "paperpilot:authenticated" not in javascript
    assert "paperpilot:unauthenticated" not in javascript


def test_auth_and_paper_search_endpoints_remain_in_single_client(tmp_path):
    javascript = _client(tmp_path).get("/static/app.js").text

    for endpoint in (
        "/api/auth/me",
        "/api/auth/${mode}",
        "/api/auth/logout",
        "/api/papers/search?${params.toString()}",
        "/api/conversations",
        "/api/conversations?limit=100",
    ):
        assert endpoint in javascript
    assert 'submitAuth("login")' in javascript
    assert 'submitAuth("register")' in javascript


def test_served_hidden_rule_keeps_auth_and_workspace_views_exclusive(tmp_path):
    css = _client(tmp_path).get("/static/styles.css")

    assert css.status_code == 200
    assert re.search(
        r"\A\s*\[hidden\]\s*\{[^}]*display:\s*none\s*!important\s*;?[^}]*\}",
        css.text,
        flags=re.DOTALL,
    )


def test_stale_alternative_failure_is_guarded_by_conversation_selection(tmp_path):
    script = _client(tmp_path).get("/static/app.js").text
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
    script = _client(tmp_path).get("/static/app.js").text

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
    script = _client(tmp_path).get("/static/app.js").text
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
    assert "const reloadVersion = requestVersion;" in guard_body
    assert "isCurrentConversation(conversationId, reloadVersion)" in guard_body


def test_assistant_messages_and_alternatives_render_full_content_as_text(tmp_path):
    script = _client(tmp_path).get("/static/app.js").text
    make_element_body = script.split("function makeElement", 1)[1].split(
        "function setAuthMessage", 1
    )[0]
    render_message_body = script.split("function renderMessage", 1)[1].split(
        "function renderMessages", 1
    )[0]
    alternatives_body = script.split("async function loadAlternatives", 1)[1].split(
        "async function rollbackToMessage", 1
    )[0]

    assert 'makeElement("div", "message-body", message.content)' in render_message_body
    assert "alternative.assistant_message.content" in alternatives_body
    for truncation_operation in (".slice(", ".substring(", ".substr("):
        assert truncation_operation not in render_message_body
        assert truncation_operation not in alternatives_body
    assert "element.textContent = text;" in make_element_body
    for unsafe_html_sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
    ):
        assert unsafe_html_sink not in script


def test_conversation_task_updates_route_is_the_polling_boundary(tmp_path):
    script = _client(tmp_path).get("/static/app.js").text
    polling_body = script.split("async function pollConversationTask", 1)[1].split(
        "async function reloadAfterConflict", 1
    )[0]
    removed_generic_updates_route = (
        "`/api/" + "tasks/${encodeURIComponent(taskId)}/updates?${params.toString()}`"
    )

    assert "`/api/conversations/${encodeURIComponent(conversationId)}`" in polling_body
    assert (
        "`/tasks/${encodeURIComponent(taskId)}/updates?${params.toString()}`"
        in polling_body
    )
    assert removed_generic_updates_route not in polling_body
