"""SessionStore tests."""
from __future__ import annotations

import json

import pytest

from paperpilot.session_store import SessionStore


def test_load_missing_session_returns_empty_messages(tmp_path):
    store = SessionStore(tmp_path)

    assert store.load("missing") == []


def test_save_and_load_session_messages(tmp_path):
    store = SessionStore(tmp_path)
    messages = [{"role": "user", "content": "hello"}]

    store.save("demo", messages)

    assert store.load("demo") == messages
    payload = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["session_name"] == "demo"
    assert payload["metadata"]["message_count"] == 1
    assert payload["created_at"]
    assert payload["updated_at"]


def test_reset_writes_empty_session(tmp_path):
    store = SessionStore(tmp_path)
    store.save("demo", [{"role": "user", "content": "hello"}])

    store.reset("demo")

    assert store.load("demo") == []
    payload = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["message_count"] == 0


def test_list_sessions_reports_name_update_and_count(tmp_path):
    store = SessionStore(tmp_path)
    store.save("b-session", [{"role": "user", "content": "b"}])
    store.save("a-session", [])

    sessions = store.list_sessions()

    assert [session.name for session in sessions] == ["a-session", "b-session"]
    assert sessions[0].message_count == 0
    assert sessions[1].message_count == 1
    assert sessions[0].updated_at


@pytest.mark.parametrize("name", ["", ".", "..", "../x", "a/b", "a\\b", "x y"])
def test_rejects_invalid_session_names(tmp_path, name):
    store = SessionStore(tmp_path)

    with pytest.raises(ValueError, match="invalid session name"):
        store.save(name, [])


def test_load_corrupt_session_raises_clear_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    store = SessionStore(tmp_path)

    with pytest.raises(ValueError, match="invalid session file"):
        store.load("broken")

