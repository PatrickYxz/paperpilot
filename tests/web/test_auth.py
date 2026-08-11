"""Authentication API tests for the Web workbench."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from paperpilot.web.auth import AuthService, UsernameAlreadyExistsError
from paperpilot.web.app import create_app
from paperpilot.web.task_executor import SynchronousTaskExecutor
from paperpilot.web.task_store import TaskStore


class _UnusedRunner:
    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        raise AssertionError(f"unexpected task execution: {task_id}, {allow_running}")

    def fail_retry_exhausted(self, *_args, **_kwargs) -> None:
        raise AssertionError("unexpected retry exhaustion")


def _client(tmp_path) -> TestClient:
    store = TaskStore(tmp_path / "tasks.sqlite3")
    executor = SynchronousTaskExecutor(_UnusedRunner())
    return TestClient(create_app(store, task_executor=executor))


def test_register_sets_session_and_me_returns_user(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "secret123"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"].startswith("user_")
    assert payload["username"] == "alice"
    assert "password_hash" not in payload

    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json() == payload


def test_register_rejects_duplicate_username(tmp_path):
    client = _client(tmp_path)
    client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "secret123"},
    )

    response = client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "other123"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "username already exists"


def test_concurrent_registration_maps_unique_constraint_to_duplicate_error(
    tmp_path,
    monkeypatch,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    auth = AuthService(store)
    original_lookup = store.get_user_by_username
    both_prechecks = threading.Barrier(2)

    def racing_lookup(username: str):
        result = original_lookup(username)
        both_prechecks.wait(timeout=2)
        return result

    monkeypatch.setattr(store, "get_user_by_username", racing_lookup)

    def register(password: str) -> str:
        try:
            auth.register(username="alice", password=password)
        except UsernameAlreadyExistsError:
            return "duplicate"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(register, ["secret123", "other123"]))

    assert sorted(outcomes) == ["created", "duplicate"]


def test_login_rejects_bad_password(tmp_path):
    client = _client(tmp_path)
    client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "secret123"},
    )
    client.post("/api/auth/logout")

    response = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "wrong123"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid username or password"


def test_logout_clears_session(tmp_path):
    client = _client(tmp_path)
    client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "secret123"},
    )

    logout = client.post("/api/auth/logout")
    me = client.get("/api/auth/me")

    assert logout.status_code == 204
    assert me.status_code == 401
    assert me.json()["detail"] == "authentication required"
