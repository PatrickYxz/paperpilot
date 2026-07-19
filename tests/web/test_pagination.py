"""Task pagination cursor tests."""
from __future__ import annotations

import pytest

from paperpilot.web.pagination import (
    InvalidTaskCursor,
    TaskCursor,
    decode_task_cursor,
    encode_task_cursor,
)


def test_task_cursor_round_trip_binds_user_and_status():
    encoded = encode_task_cursor(
        user_id="user_1",
        status="running",
        created_at="2026-07-14T08:00:00+00:00",
        task_id="task_2",
    )
    assert decode_task_cursor(
        encoded,
        expected_user_id="user_1",
        expected_status="running",
    ) == TaskCursor(
        created_at="2026-07-14T08:00:00+00:00",
        task_id="task_2",
    )


@pytest.mark.parametrize("value", ["not-base64", "e30"])
def test_task_cursor_rejects_malformed_payload(value):
    with pytest.raises(InvalidTaskCursor, match="invalid task cursor"):
        decode_task_cursor(
            value,
            expected_user_id="user_1",
            expected_status=None,
        )


def test_task_cursor_rejects_different_request_context():
    encoded = encode_task_cursor(
        user_id="user_1",
        status="pending",
        created_at="2026-07-14T08:00:00+00:00",
        task_id="task_2",
    )
    with pytest.raises(InvalidTaskCursor):
        decode_task_cursor(
            encoded,
            expected_user_id="user_2",
            expected_status="pending",
        )
    with pytest.raises(InvalidTaskCursor):
        decode_task_cursor(
            encoded,
            expected_user_id="user_1",
            expected_status="completed",
        )
