"""Request observability middleware tests."""
from __future__ import annotations

import asyncio
import io
import json
import logging

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.observability import (
    ACCESS_LOGGER_NAME,
    JsonLogFormatter,
    RequestObservabilityMiddleware,
    TextLogFormatter,
    configure_paperpilot_logging,
)


class RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class CloseTrackingStreamHandler(logging.StreamHandler):
    def __init__(self) -> None:
        super().__init__(io.StringIO())
        self.was_closed = False

    def close(self) -> None:
        self.was_closed = True
        super().close()


def _observed_app():
    app = FastAPI()
    logger = logging.getLogger("paperpilot.tests.access")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = RecordHandler()
    logger.addHandler(handler)
    config = WebRuntimeConfig(log_level="DEBUG")
    app.add_middleware(
        RequestObservabilityMiddleware,
        config=config,
        logger=logger,
    )

    @app.get("/api/jobs/{task_id}")
    def get_task(task_id: str, request: Request):
        request.state.user_id = "user_1"
        return {"id": task_id}

    @app.get("/known-error")
    def known_error():
        raise HTTPException(status_code=409, detail="known")

    @app.get("/explode")
    def explode():
        raise RuntimeError("internal database detail")

    return app, handler


def test_request_id_is_generated_and_dynamic_route_is_not_logged():
    app, handler = _observed_app()
    response = TestClient(app).get(
        "/api/jobs/task_secret?question=private",
        headers={"cookie": "paperpilot_session=secret"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"]
    record = handler.records[-1]
    assert record.route == "/api/jobs/{task_id}"
    assert record.user_id == "user_1"
    assert record.status_code == 200
    serialized = repr(record.__dict__)
    assert "task_secret" not in serialized
    assert "question=private" not in serialized
    assert "paperpilot_session" not in serialized


def test_valid_request_id_is_preserved_once():
    app, _ = _observed_app()
    response = TestClient(app).get(
        "/api/jobs/task_1",
        headers={"X-Request-ID": "upstream.request-1:attempt_2"},
    )

    assert response.headers.get_list("x-request-id") == [
        "upstream.request-1:attempt_2"
    ]


@pytest.mark.parametrize(
    "raw_request_id",
    [b"", b"contains space", b"line\nbreak", b"x" * 129, "请求".encode()],
)
def test_invalid_request_id_is_replaced(raw_request_id):
    from paperpilot.web.observability import _request_id_from_scope

    generated = _request_id_from_scope(
        {"type": "http", "headers": [(b"x-request-id", raw_request_id)]}
    )

    assert len(generated) == 36


def test_known_http_error_keeps_shape_and_request_id():
    app, _ = _observed_app()
    response = TestClient(app).get("/known-error")

    assert response.status_code == 409
    assert response.json() == {"detail": "known"}
    assert response.headers["x-request-id"]


def test_unmatched_route_uses_low_cardinality_fallback():
    app, handler = _observed_app()
    response = TestClient(app).get("/missing/raw/value")

    assert response.status_code == 404
    assert response.headers["x-request-id"]
    assert handler.records[-1].route == "<unmatched>"


def test_unexpected_error_returns_safe_body_and_request_id():
    app, handler = _observed_app()
    response = TestClient(app).get("/explode")

    assert response.status_code == 500
    assert response.json() == {
        "detail": "internal server error",
        "request_id": response.headers["x-request-id"],
    }
    assert "internal database detail" not in response.text
    assert handler.records[-1].exception_type == "RuntimeError"


def test_unexpected_error_access_record_does_not_render_exception_details():
    app, handler = _observed_app()
    TestClient(app).get("/explode")

    record = handler.records[-1]
    assert record.exc_info is None
    assert record.exception_type == "RuntimeError"
    for rendered in (
        JsonLogFormatter().format(record),
        TextLogFormatter().format(record),
    ):
        assert "RuntimeError" in rendered
        assert "internal database detail" not in rendered


def test_exception_after_response_start_does_not_send_second_start():
    messages = []

    async def partial_app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"x-request-id", b"downstream")],
            }
        )
        await send({"type": "http.response.body", "body": b"part", "more_body": True})
        raise RuntimeError("late failure")

    logger = logging.getLogger("paperpilot.tests.partial")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = RecordHandler()
    logger.addHandler(handler)
    middleware = RequestObservabilityMiddleware(
        partial_app,
        WebRuntimeConfig(log_level="DEBUG"),
        logger=logger,
    )

    async def invoke():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            messages.append(message)

        await middleware(
            {"type": "http", "method": "GET", "path": "/partial", "headers": []},
            receive,
            send,
        )

    with pytest.raises(RuntimeError, match="late failure"):
        asyncio.run(invoke())
    assert [message["type"] for message in messages].count("http.response.start") == 1
    start = next(message for message in messages if message["type"] == "http.response.start")
    request_ids = [
        value for key, value in start["headers"] if key == b"x-request-id"
    ]
    assert len(request_ids) == 1
    assert request_ids != [b"downstream"]
    assert handler.records[-1].status_code == 200
    assert handler.records[-1].exception_type == "RuntimeError"


def test_logger_configuration_is_idempotent_and_does_not_touch_root():
    root_handlers = list(logging.getLogger().handlers)
    stream = io.StringIO()
    config = WebRuntimeConfig(log_level="INFO", log_format="json")

    configure_paperpilot_logging(config, stream=stream)
    configure_paperpilot_logging(config, stream=stream)

    logger = logging.getLogger(ACCESS_LOGGER_NAME)
    owned = [handler for handler in logger.handlers if getattr(handler, "_paperpilot_owned", False)]
    assert len(owned) == 1
    assert logging.getLogger().handlers == root_handlers


def test_logger_configuration_removes_and_closes_duplicate_owned_handlers():
    from paperpilot.web.observability import RUNTIME_LOGGER_NAME

    root_handlers = list(logging.getLogger().handlers)
    logger_names = (ACCESS_LOGGER_NAME, RUNTIME_LOGGER_NAME)
    original_handlers = {
        name: list(logging.getLogger(name).handlers) for name in logger_names
    }
    unrelated_handlers: dict[str, logging.Handler] = {}
    seeded_owned_handlers: dict[str, list[CloseTrackingStreamHandler]] = {}

    try:
        for name in logger_names:
            logger = logging.getLogger(name)
            unrelated = RecordHandler()
            owned = [CloseTrackingStreamHandler(), CloseTrackingStreamHandler()]
            for handler in owned:
                handler._paperpilot_owned = True  # type: ignore[attr-defined]
            logger.handlers = [unrelated, *owned]
            unrelated_handlers[name] = unrelated
            seeded_owned_handlers[name] = owned

        configure_paperpilot_logging(WebRuntimeConfig(log_format="json"))

        for name in logger_names:
            logger = logging.getLogger(name)
            owned = [
                handler
                for handler in logger.handlers
                if getattr(handler, "_paperpilot_owned", False)
            ]
            assert len(owned) == 1
            assert unrelated_handlers[name] in logger.handlers
            removed = [
                handler
                for handler in seeded_owned_handlers[name]
                if handler not in logger.handlers
            ]
            assert len(removed) == 1
            assert removed[0].was_closed
        assert logging.getLogger().handlers == root_handlers
    finally:
        for name in logger_names:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if handler not in original_handlers[name]:
                    handler.close()
            logger.handlers = original_handlers[name]


def test_json_formatter_emits_machine_readable_structured_fields():
    record = logging.LogRecord(
        name=ACCESS_LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="HTTP request",
        args=(),
        exc_info=None,
    )
    record.event = "http.request.completed"
    record.request_id = "request_1"
    record.route = "/api/jobs/{task_id}"
    record.status_code = 200

    payload = json.loads(JsonLogFormatter().format(record))

    assert payload["message"] == "HTTP request"
    assert payload["event"] == "http.request.completed"
    assert payload["request_id"] == "request_1"
    assert payload["route"] == "/api/jobs/{task_id}"
    assert payload["status_code"] == 200

    text_line = TextLogFormatter().format(record)
    assert "event=\"http.request.completed\"" in text_line
    assert "request_id=\"request_1\"" in text_line


def test_live_success_is_debug_ready_failure_is_warning_and_slow_is_warning(monkeypatch):
    app = FastAPI()
    logger = logging.getLogger("paperpilot.tests.levels")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = RecordHandler()
    logger.addHandler(handler)
    config = WebRuntimeConfig(log_level="DEBUG", slow_request_ms=10)
    app.add_middleware(RequestObservabilityMiddleware, config=config, logger=logger)

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        raise HTTPException(status_code=503, detail="not ready")

    @app.get("/slow")
    def slow():
        return {"ok": True}

    client = TestClient(app)
    client.get("/health/live")
    client.get("/health/ready")
    times = iter([1.0, 1.020])
    monkeypatch.setattr("paperpilot.web.observability._perf_counter", lambda: next(times))
    client.get("/slow")

    by_route = {record.route: record for record in handler.records[-3:]}
    assert by_route["/health/live"].levelno == logging.DEBUG
    assert by_route["/health/ready"].levelno == logging.WARNING
    assert by_route["/slow"].levelno == logging.WARNING
