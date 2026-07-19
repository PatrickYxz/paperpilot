# PaperPilot Runtime Protection Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded task-admission layer, startup-validated runtime configuration, safe structured request logging, request IDs, and liveness/readiness endpoints to PaperPilot's FastAPI backend.

**Architecture:** A frozen `WebRuntimeConfig` owns startup settings. A pure ASGI middleware supplies request context and structured access logs, while task creation reserves executor capacity before one atomic SQLite task/event transaction. Thread execution uses process-local bounded permits; synchronous and Celery executors keep the same reservation protocol without a local queue limit.

**Tech Stack:** Python 3.12, FastAPI 0.138.0, Starlette 1.3.1, SQLite WAL, standard-library logging and concurrency primitives, pytest, HTTPX/TestClient.

## Global Constraints

- Keep `thread` as the default executor; Celery remains opt-in with `PAPERPILOT_TASK_EXECUTOR=celery`.
- Default thread capacity is exactly `2` workers plus `4` queued tasks, for `6` unfinished tasks per Web process.
- Reject thread saturation before writing task, event, or artifact rows.
- Return `503` plus `Retry-After: 1` for capacity saturation; return `503` without `Retry-After` after executor shutdown begins.
- Preserve the existing `201` response when Celery publication raises after a worker has already claimed the task as `running` or `completed`.
- Keep task and initial `queued` event creation in one SQLite transaction.
- Do not change task success schemas, authentication requirements, or user isolation.
- Do not add Python dependencies, migrate SQLite, add generic HTTP rate limiting, add global request timeouts, or add Prometheus/OpenTelemetry in this phase.
- Do not make Redis a readiness dependency and do not probe MCP/ColBERT from health endpoints.
- Do not log query strings, request/response bodies, cookies, authorization data, passwords, task questions, artifact content, or raw dynamic task paths.
- Preserve all approved uncommitted work currently in the checkout.
- Do not stage, commit, push, reset, stash, clean, or delete files unless the user explicitly requests it.
- Use a red/green TDD cycle for every behavior and run all commands through `./.venv/bin/python`.

## File Responsibility Map

- Create `paperpilot/web/config.py`: parse and validate Web runtime environment settings once.
- Create `paperpilot/web/observability.py`: request IDs, pure ASGI middleware, structured formatters, and owned logger setup.
- Modify `paperpilot/web/task_executor.py`: reservation protocol, bounded thread permits, shutdown state, and config-driven construction.
- Modify `paperpilot/web/task_store.py`: atomic queued-task creation and a lightweight SQLite health probe.
- Modify `paperpilot/web/app.py`: wire config, middleware, request user context, health routes, and pre-write task admission.
- Create `tests/web/test_config.py`: runtime configuration behavior.
- Create `tests/web/test_observability.py`: ASGI request context, logging, error response, and response-start behavior.
- Modify `tests/web/test_task_executor.py`: bounded capacity, one-shot reservation, release, shutdown, and builder behavior.
- Modify `tests/web/test_task_store.py`: atomic task/event transaction and database health probe.
- Modify `tests/web/test_web_app.py`: end-to-end task admission, health endpoints, request IDs, and preserved Celery ambiguity behavior.
- Create `scripts/benchmark_web_admission.py`: reproducible overload and capacity-recovery check without fixed latency thresholds.
- Modify `tests/web/test_runtime_config.py`: README/runtime artifact checks and admission benchmark smoke.
- Modify `README.md`: runtime settings, health semantics, overload behavior, multi-worker capacity, and benchmark command.

---

### Task 1: Centralized Runtime Configuration

**Files:**
- Create: `paperpilot/web/config.py`
- Create: `tests/web/test_config.py`

**Interfaces:**
- Consumes: environment variables through an injected `Mapping[str, str]` or `os.environ`.
- Produces: `WebRuntimeConfig.from_env(environ=None) -> WebRuntimeConfig`.
- Produces fields: `task_executor`, `thread_workers`, `thread_queue_capacity`, `overload_retry_after_seconds`, `log_level`, `log_format`, `slow_request_ms`, and `environment`.

- [ ] **Step 1: Write failing configuration tests**

```python
"""Web runtime configuration tests."""
from __future__ import annotations

import pytest

from paperpilot.web.config import WebRuntimeConfig


def test_runtime_config_defaults_are_local_and_bounded():
    config = WebRuntimeConfig.from_env({})

    assert config.task_executor == "thread"
    assert config.thread_workers == 2
    assert config.thread_queue_capacity == 4
    assert config.overload_retry_after_seconds == 1
    assert config.log_level == "INFO"
    assert config.log_format == "json"
    assert config.slow_request_ms == 1000
    assert config.environment == "development"


def test_runtime_config_accepts_explicit_overrides():
    config = WebRuntimeConfig.from_env(
        {
            "PAPERPILOT_TASK_EXECUTOR": "CELERY",
            "PAPERPILOT_THREAD_WORKERS": "3",
            "PAPERPILOT_THREAD_QUEUE_CAPACITY": "0",
            "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS": "5",
            "PAPERPILOT_LOG_LEVEL": "debug",
            "PAPERPILOT_LOG_FORMAT": "text",
            "PAPERPILOT_SLOW_REQUEST_MS": "250",
            "PAPERPILOT_ENV": "test",
        }
    )

    assert config.task_executor == "celery"
    assert config.thread_workers == 3
    assert config.thread_queue_capacity == 0
    assert config.overload_retry_after_seconds == 5
    assert config.log_level == "DEBUG"
    assert config.log_format == "text"
    assert config.slow_request_ms == 250
    assert config.environment == "test"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("PAPERPILOT_TASK_EXECUTOR", "other", "PAPERPILOT_TASK_EXECUTOR"),
        ("PAPERPILOT_THREAD_WORKERS", "0", "PAPERPILOT_THREAD_WORKERS"),
        ("PAPERPILOT_THREAD_WORKERS", "many", "PAPERPILOT_THREAD_WORKERS"),
        ("PAPERPILOT_THREAD_QUEUE_CAPACITY", "-1", "PAPERPILOT_THREAD_QUEUE_CAPACITY"),
        ("PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS", "0", "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS"),
        ("PAPERPILOT_LOG_LEVEL", "verbose", "PAPERPILOT_LOG_LEVEL"),
        ("PAPERPILOT_LOG_FORMAT", "yaml", "PAPERPILOT_LOG_FORMAT"),
        ("PAPERPILOT_SLOW_REQUEST_MS", "0", "PAPERPILOT_SLOW_REQUEST_MS"),
        ("PAPERPILOT_ENV", "   ", "PAPERPILOT_ENV"),
    ],
)
def test_runtime_config_rejects_invalid_values(name, value, message):
    with pytest.raises(ValueError, match=message):
        WebRuntimeConfig.from_env({name: value})
```

- [ ] **Step 2: Verify RED**

Run: `./.venv/bin/python -m pytest tests/web/test_config.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'paperpilot.web.config'`.

- [ ] **Step 3: Implement immutable parsing and validation**

```python
"""Validated runtime configuration for the PaperPilot Web backend."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping, cast

TaskExecutorBackend = Literal["thread", "celery"]
LogFormat = Literal["json", "text"]
VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}


@dataclass(frozen=True)
class WebRuntimeConfig:
    task_executor: TaskExecutorBackend = "thread"
    thread_workers: int = 2
    thread_queue_capacity: int = 4
    overload_retry_after_seconds: int = 1
    log_level: str = "INFO"
    log_format: LogFormat = "json"
    slow_request_ms: int = 1000
    environment: str = "development"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "WebRuntimeConfig":
        values = os.environ if environ is None else environ
        task_executor = values.get("PAPERPILOT_TASK_EXECUTOR", "thread").strip().lower()
        if task_executor not in {"thread", "celery"}:
            raise ValueError(
                "PAPERPILOT_TASK_EXECUTOR must be 'thread' or 'celery'"
            )

        log_level = values.get("PAPERPILOT_LOG_LEVEL", "INFO").strip().upper()
        if log_level not in VALID_LOG_LEVELS:
            raise ValueError(
                "PAPERPILOT_LOG_LEVEL must be one of "
                + ", ".join(sorted(VALID_LOG_LEVELS))
            )

        log_format = values.get("PAPERPILOT_LOG_FORMAT", "json").strip().lower()
        if log_format not in {"json", "text"}:
            raise ValueError("PAPERPILOT_LOG_FORMAT must be 'json' or 'text'")

        environment = values.get("PAPERPILOT_ENV", "development").strip()
        if not environment:
            raise ValueError("PAPERPILOT_ENV must not be empty")

        return cls(
            task_executor=cast(TaskExecutorBackend, task_executor),
            thread_workers=_read_int(
                values, "PAPERPILOT_THREAD_WORKERS", default=2, minimum=1
            ),
            thread_queue_capacity=_read_int(
                values,
                "PAPERPILOT_THREAD_QUEUE_CAPACITY",
                default=4,
                minimum=0,
            ),
            overload_retry_after_seconds=_read_int(
                values,
                "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS",
                default=1,
                minimum=1,
            ),
            log_level=log_level,
            log_format=cast(LogFormat, log_format),
            slow_request_ms=_read_int(
                values, "PAPERPILOT_SLOW_REQUEST_MS", default=1000, minimum=1
            ),
            environment=environment,
        )


def _read_int(
    environ: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    raw = environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value
```

- [ ] **Step 4: Verify GREEN**

Run: `./.venv/bin/python -m pytest tests/web/test_config.py -q`

Expected: all configuration tests pass.

---

### Task 2: Request IDs And Structured Observability Middleware

**Files:**
- Create: `paperpilot/web/observability.py`
- Create: `tests/web/test_observability.py`

**Interfaces:**
- Consumes: `WebRuntimeConfig` from Task 1 and a Starlette ASGI application.
- Produces: `RequestObservabilityMiddleware(app, config, logger=None)`.
- Produces: `configure_paperpilot_logging(config)`, `ACCESS_LOGGER_NAME`, and `RUNTIME_LOGGER_NAME`.
- Response contract: exactly one canonical `X-Request-ID`; safe JSON 500 before response start; re-raise after response start.

- [ ] **Step 1: Write failing request-context and exception tests**

```python
"""Request observability middleware tests."""
from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.observability import RequestObservabilityMiddleware


class RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


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

    @app.get("/api/tasks/{task_id}")
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
        "/api/tasks/task_secret?question=private",
        headers={"cookie": "paperpilot_session=secret"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"]
    record = handler.records[-1]
    assert record.route == "/api/tasks/{task_id}"
    assert record.user_id == "user_1"
    assert record.status_code == 200
    serialized = repr(record.__dict__)
    assert "task_secret" not in serialized
    assert "question=private" not in serialized
    assert "paperpilot_session" not in serialized


def test_valid_request_id_is_preserved_once():
    app, _ = _observed_app()
    response = TestClient(app).get(
        "/api/tasks/task_1",
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
```

- [ ] **Step 2: Add failing log-level and idempotent logger tests**

```python
import io
import json

from paperpilot.web.observability import (
    ACCESS_LOGGER_NAME,
    JsonLogFormatter,
    TextLogFormatter,
    configure_paperpilot_logging,
)


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
    record.route = "/api/tasks/{task_id}"
    record.status_code = 200

    payload = json.loads(JsonLogFormatter().format(record))

    assert payload["message"] == "HTTP request"
    assert payload["event"] == "http.request.completed"
    assert payload["request_id"] == "request_1"
    assert payload["route"] == "/api/tasks/{task_id}"
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
    monkeypatch.setattr("paperpilot.web.observability.time.perf_counter", lambda: next(times))
    client.get("/slow")

    by_route = {record.route: record for record in handler.records[-3:]}
    assert by_route["/health/live"].levelno == logging.DEBUG
    assert by_route["/health/ready"].levelno == logging.WARNING
    assert by_route["/slow"].levelno == logging.WARNING
```

- [ ] **Step 3: Verify RED**

Run: `./.venv/bin/python -m pytest tests/web/test_observability.py -q`

Expected: collection fails because `paperpilot.web.observability` does not exist.

- [ ] **Step 4: Implement the pure ASGI middleware and owned formatters**

```python
"""Low-overhead request context and structured Web logging."""
from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import TextIO

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from paperpilot.web.config import WebRuntimeConfig

ACCESS_LOGGER_NAME = "paperpilot.web.access"
RUNTIME_LOGGER_NAME = "paperpilot.web.runtime"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_REQUEST_ID_HEADER = b"x-request-id"
_STRUCTURED_FIELDS = (
    "event",
    "request_id",
    "method",
    "route",
    "status_code",
    "duration_ms",
    "user_id",
    "environment",
    "exception_type",
    "reason",
    "executor",
    "workers",
    "queue_capacity",
)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _STRUCTURED_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class TextLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        parts = [record.levelname, record.name, record.getMessage()]
        for field in _STRUCTURED_FIELDS:
            if hasattr(record, field):
                value = json.dumps(getattr(record, field), ensure_ascii=False)
                parts.append(f"{field}={value}")
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))
        return " ".join(parts)


def configure_paperpilot_logging(
    config: WebRuntimeConfig,
    *,
    stream: TextIO | None = None,
) -> None:
    formatter: logging.Formatter
    if config.log_format == "json":
        formatter = JsonLogFormatter()
    else:
        formatter = TextLogFormatter()
    level = getattr(logging, config.log_level)
    target_stream = stream or sys.stderr

    for name in (ACCESS_LOGGER_NAME, RUNTIME_LOGGER_NAME):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False
        handler = next(
            (
                candidate
                for candidate in logger.handlers
                if getattr(candidate, "_paperpilot_owned", False)
            ),
            None,
        )
        if handler is None:
            handler = logging.StreamHandler(target_stream)
            handler._paperpilot_owned = True  # type: ignore[attr-defined]
            logger.addHandler(handler)
        else:
            assert isinstance(handler, logging.StreamHandler)
            handler.setStream(target_stream)
        handler.setLevel(level)
        handler.setFormatter(formatter)


class RequestObservabilityMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        config: WebRuntimeConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.app = app
        self.config = config
        self.logger = logger or logging.getLogger(ACCESS_LOGGER_NAME)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope)
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        started_at = time.perf_counter()
        response_started = False
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != _REQUEST_ID_HEADER
                ]
                headers.append((_REQUEST_ID_HEADER, request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as exc:
            self._log_request(
                scope,
                state,
                request_id=request_id,
                status_code=status_code if response_started else 500,
                started_at=started_at,
                exception=exc,
            )
            if response_started:
                raise
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "internal server error",
                    "request_id": request_id,
                },
            )
            await response(scope, receive, send_with_request_id)
            return

        self._log_request(
            scope,
            state,
            request_id=request_id,
            status_code=status_code,
            started_at=started_at,
            exception=None,
        )

    def _log_request(
        self,
        scope: Scope,
        state: dict,
        *,
        request_id: str,
        status_code: int,
        started_at: float,
        exception: Exception | None,
    ) -> None:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        route = _route_template(scope)
        if exception is not None:
            level = logging.ERROR
            event = "http.request.failed"
        elif route == "/health/live" and status_code < 400:
            level = logging.DEBUG
            event = "http.request.completed"
        elif route == "/health/ready" and status_code >= 500:
            level = logging.WARNING
            event = "http.request.completed"
        elif status_code >= 500:
            level = logging.ERROR
            event = "http.request.completed"
        elif duration_ms >= self.config.slow_request_ms:
            level = logging.WARNING
            event = "http.request.completed"
        else:
            level = logging.INFO
            event = "http.request.completed"

        exc_info = None
        if exception is not None:
            exc_info = (type(exception), exception, exception.__traceback__)
        self.logger.log(
            level,
            "HTTP request",
            extra={
                "event": event,
                "request_id": request_id,
                "method": scope.get("method", "<unknown>"),
                "route": route,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "user_id": state.get("user_id"),
                "environment": self.config.environment,
                "exception_type": type(exception).__name__ if exception else None,
            },
            exc_info=exc_info,
        )


def _request_id_from_scope(scope: Scope) -> str:
    values = [
        value
        for key, value in scope.get("headers", [])
        if key.lower() == _REQUEST_ID_HEADER
    ]
    if len(values) == 1:
        try:
            candidate = values[0].decode("ascii")
        except UnicodeDecodeError:
            candidate = ""
        if _REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
    return str(uuid.uuid4())


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return "<unmatched>"
```

- [ ] **Step 5: Verify GREEN and formatter safety**

Run: `./.venv/bin/python -m pytest tests/web/test_observability.py -q`

Expected: all middleware, request ID, response-start, log-level, and idempotence tests pass.

---

### Task 3: Bounded Executor Reservation Protocol

**Files:**
- Modify: `paperpilot/web/task_executor.py`
- Modify: `tests/web/test_task_executor.py`

**Interfaces:**
- Consumes: `WebRuntimeConfig` from Task 1.
- Produces: `TaskExecutorAtCapacityError`, `TaskExecutorShuttingDownError`, and `TaskSubmissionReservation`.
- Produces: `TaskExecutorLike.is_shutdown`, `TaskExecutorLike.reserve()`, and `TaskExecutorLike.shutdown()`.
- Keeps concrete `submit()` convenience methods for existing callers; each convenience method delegates through `reserve()`.

- [ ] **Step 1: Write failing bounded-capacity tests**

```python
import threading
import time

from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.task_executor import (
    TaskExecutorAtCapacityError,
    TaskExecutorShuttingDownError,
    TaskSubmissionReservation,
)


class BlockingRunner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def run_simulated(self, task_id: str) -> None:
        self.started.set()
        assert self.release.wait(timeout=2)

    def run_real(self, task_id: str) -> None:
        self.run_simulated(task_id)


def _reserve_eventually(executor):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            return executor.reserve()
        except TaskExecutorAtCapacityError:
            time.sleep(0.01)
    raise AssertionError("executor capacity was not restored")


def test_thread_executor_bounds_running_plus_queued_work():
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=2)
    reservations = [executor.reserve() for _ in range(3)]
    futures = [
        reservation.submit(f"task_{index}", "simulated")
        for index, reservation in enumerate(reservations)
    ]
    assert runner.started.wait(timeout=1)

    with pytest.raises(TaskExecutorAtCapacityError):
        executor.reserve()

    runner.release.set()
    for future in futures:
        future.result(timeout=2)
    recovered = _reserve_eventually(executor)
    recovered.release()
    executor.shutdown()


def test_unsubmitted_reservation_release_restores_capacity_once():
    executor = TaskExecutor(FakeRunner(), max_workers=1, queue_capacity=0)
    reservation = executor.reserve()
    reservation.release()
    reservation.release()

    replacement = executor.reserve()
    replacement.release()
    executor.shutdown()


def test_failed_future_releases_capacity():
    class FailingRunner(FakeRunner):
        def run_simulated(self, task_id: str) -> None:
            raise RuntimeError("workflow failed")

    executor = TaskExecutor(FailingRunner(), max_workers=1, queue_capacity=0)
    future = executor.submit("task_failed", "simulated")

    with pytest.raises(RuntimeError, match="workflow failed"):
        future.result(timeout=1)
    replacement = _reserve_eventually(executor)
    replacement.release()
    executor.shutdown()


def test_cancelled_queued_future_releases_its_queue_slot():
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=1)
    running = executor.submit("task_running", "simulated")
    assert runner.started.wait(timeout=1)
    queued = executor.submit("task_queued", "simulated")

    assert queued.cancel() is True
    replacement = _reserve_eventually(executor)
    replacement.release()
    runner.release.set()
    running.result(timeout=2)
    executor.shutdown()


def test_reservation_submit_failure_releases_exactly_once():
    release_count = 0

    def fail(task_id, execution_mode):
        raise ConnectionError("submit failed")

    def release_once():
        nonlocal release_count
        release_count += 1

    reservation = TaskSubmissionReservation(fail, release_once)
    with pytest.raises(ConnectionError, match="submit failed"):
        reservation.submit("task_1", "real")
    reservation.release()

    assert release_count == 1
    with pytest.raises(RuntimeError, match="already used"):
        reservation.submit("task_1", "real")


def test_shutdown_rejects_new_reservations():
    executor = TaskExecutor(FakeRunner(), max_workers=1, queue_capacity=0)
    executor.shutdown()

    assert executor.is_shutdown is True
    with pytest.raises(TaskExecutorShuttingDownError):
        executor.reserve()
```

- [ ] **Step 2: Add failing no-capacity and config-driven builder tests**

```python
def test_synchronous_and_celery_reservations_reject_after_shutdown():
    sync = SynchronousTaskExecutor(FakeRunner())
    celery = CeleryTaskExecutor(FakeTaskSender())
    sync.shutdown()
    celery.shutdown()

    for executor in (sync, celery):
        assert executor.is_shutdown is True
        with pytest.raises(TaskExecutorShuttingDownError):
            executor.reserve()


def test_build_task_executor_uses_validated_thread_capacity():
    config = WebRuntimeConfig(
        task_executor="thread",
        thread_workers=3,
        thread_queue_capacity=7,
    )

    executor = build_task_executor(FakeRunner(), config=config)

    assert isinstance(executor, TaskExecutor)
    assert executor.max_workers == 3
    assert executor.queue_capacity == 7
    executor.shutdown()


def test_build_task_executor_uses_config_instead_of_rereading_environment(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_TASK_EXECUTOR", "celery")
    config = WebRuntimeConfig(task_executor="thread")

    executor = build_task_executor(FakeRunner(), config=config)

    assert isinstance(executor, TaskExecutor)
    executor.shutdown()
```

- [ ] **Step 3: Verify RED**

Run: `./.venv/bin/python -m pytest tests/web/test_task_executor.py -q`

Expected: tests fail because reservation classes, queue capacity, `is_shutdown`, and config injection are absent.

- [ ] **Step 4: Implement the one-shot reservation state machine**

```python
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from typing import Callable, Literal, Protocol, cast

from paperpilot.web.config import WebRuntimeConfig


class TaskExecutorAtCapacityError(RuntimeError):
    """Raised when all running and queued thread slots are reserved."""


class TaskExecutorShuttingDownError(RuntimeError):
    """Raised after executor shutdown begins."""


Submitter = Callable[[str, ExecutionMode], object]
ReleaseCallback = Callable[[], None]


class TaskSubmissionReservation:
    def __init__(
        self,
        submitter: Submitter,
        release_capacity: ReleaseCallback | None = None,
    ) -> None:
        self._submitter = submitter
        self._release_capacity = release_capacity
        self._lock = Lock()
        self._state = "reserved"

    def submit(self, task_id: str, execution_mode: ExecutionMode) -> object:
        with self._lock:
            if self._state != "reserved":
                raise RuntimeError("task submission reservation was already used")
            self._state = "submitting"
        try:
            result = self._submitter(task_id, execution_mode)
        except BaseException:
            self._release_after_submit_failure()
            raise

        with self._lock:
            self._state = "submitted"
        if self._release_capacity is not None:
            future = cast(Future[None], result)
            future.add_done_callback(self._release_after_future)
        return result

    def release(self) -> None:
        callback = None
        with self._lock:
            if self._state == "reserved":
                self._state = "released"
                callback = self._release_capacity
        if callback is not None:
            callback()

    def _release_after_submit_failure(self) -> None:
        callback = None
        with self._lock:
            if self._state == "submitting":
                self._state = "released"
                callback = self._release_capacity
        if callback is not None:
            callback()

    def _release_after_future(self, future: Future[None]) -> None:
        callback = None
        with self._lock:
            if self._state == "submitted":
                self._state = "released"
                callback = self._release_capacity
        if callback is not None:
            callback()


class TaskExecutorLike(Protocol):
    @property
    def is_shutdown(self) -> bool: ...

    def reserve(self) -> TaskSubmissionReservation: ...

    def shutdown(self) -> None: ...
```

- [ ] **Step 5: Implement bounded thread, synchronous, and Celery executors**

```python
class TaskExecutor:
    def __init__(
        self,
        runner: WorkflowRunnerLike,
        *,
        max_workers: int = 2,
        queue_capacity: int = 4,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if queue_capacity < 0:
            raise ValueError("queue_capacity must be at least 0")
        self.runner = runner
        self.max_workers = max_workers
        self.queue_capacity = queue_capacity
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._capacity = BoundedSemaphore(max_workers + queue_capacity)
        self._state_lock = Lock()
        self._shutdown = False

    @property
    def is_shutdown(self) -> bool:
        with self._state_lock:
            return self._shutdown

    def reserve(self) -> TaskSubmissionReservation:
        with self._state_lock:
            if self._shutdown:
                raise TaskExecutorShuttingDownError("task executor is shutting down")
            if not self._capacity.acquire(blocking=False):
                raise TaskExecutorAtCapacityError("task executor is at capacity")
        return TaskSubmissionReservation(self._submit_reserved, self._capacity.release)

    def submit(self, task_id: str, execution_mode: ExecutionMode) -> object:
        return self.reserve().submit(task_id, execution_mode)

    def shutdown(self) -> None:
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._pool.shutdown(wait=True)

    def _submit_reserved(
        self,
        task_id: str,
        execution_mode: ExecutionMode,
    ) -> Future[None]:
        return self._pool.submit(self._run, task_id, execution_mode)


class SynchronousTaskExecutor:
    def __init__(self, runner: WorkflowRunnerLike) -> None:
        self.runner = runner
        self.submissions: list[tuple[str, ExecutionMode]] = []
        self._state_lock = Lock()
        self._shutdown = False

    @property
    def is_shutdown(self) -> bool:
        with self._state_lock:
            return self._shutdown

    def reserve(self) -> TaskSubmissionReservation:
        with self._state_lock:
            if self._shutdown:
                raise TaskExecutorShuttingDownError("task executor is shutting down")
        return TaskSubmissionReservation(self._submit_reserved)

    def submit(self, task_id: str, execution_mode: ExecutionMode) -> object:
        return self.reserve().submit(task_id, execution_mode)

    def shutdown(self) -> None:
        with self._state_lock:
            self._shutdown = True

    def _submit_reserved(self, task_id: str, execution_mode: ExecutionMode) -> Future[None]:
        self.submissions.append((task_id, execution_mode))
        future: Future[None] = Future()
        try:
            if execution_mode == "real":
                self.runner.run_real(task_id)
            elif execution_mode == "simulated":
                self.runner.run_simulated(task_id)
            else:
                raise ValueError(f"invalid execution mode: {execution_mode!r}")
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(None)
        return future


class CeleryTaskExecutor:
    def __init__(self, sender: TaskSenderLike | None = None) -> None:
        self.sender = sender or celery_app
        self._state_lock = Lock()
        self._shutdown = False

    @property
    def is_shutdown(self) -> bool:
        with self._state_lock:
            return self._shutdown

    def reserve(self) -> TaskSubmissionReservation:
        with self._state_lock:
            if self._shutdown:
                raise TaskExecutorShuttingDownError("task executor is shutting down")
        return TaskSubmissionReservation(self._submit_reserved)

    def submit(self, task_id: str, execution_mode: ExecutionMode) -> object:
        return self.reserve().submit(task_id, execution_mode)

    def shutdown(self) -> None:
        with self._state_lock:
            self._shutdown = True

    def _submit_reserved(self, task_id: str, execution_mode: ExecutionMode) -> object:
        return self.sender.send_task(
            EXECUTE_RESEARCH_TASK_NAME,
            args=[task_id, execution_mode],
            task_id=task_id,
        )


def build_task_executor(
    runner: WorkflowRunnerLike,
    *,
    config: WebRuntimeConfig | None = None,
) -> TaskExecutorLike:
    runtime = config or WebRuntimeConfig.from_env()
    if runtime.task_executor == "thread":
        return TaskExecutor(
            runner,
            max_workers=runtime.thread_workers,
            queue_capacity=runtime.thread_queue_capacity,
        )
    return CeleryTaskExecutor()
```

Retain the existing `TaskExecutor._run()` implementation unchanged.

- [ ] **Step 6: Verify GREEN and existing executor behavior**

Run: `./.venv/bin/python -m pytest tests/web/test_task_executor.py -q`

Expected: capacity, recovery, shutdown, synchronous execution, Celery publication, and builder tests all pass.

---

### Task 4: Atomic Queued Task Creation And SQLite Health Probe

**Files:**
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/web/test_task_store.py`

**Interfaces:**
- Produces: `TaskStore.create_queued_task(question, depth, user_id, execution_mode) -> ResearchTask`.
- Produces: `TaskStore.check_health() -> None`, raising the underlying database error on failure.
- Keeps: `TaskStore.create_task()` for existing internal and test callers.

- [ ] **Step 1: Write failing atomicity tests**

```python
import sqlite3


def test_create_queued_task_writes_task_and_event_together(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = store.create_user(
        username="atomic-user",
        password_hash="hash",
        password_salt="salt",
    )

    task = store.create_queued_task(
        question="Atomic queue",
        depth="quick",
        user_id=user.id,
        execution_mode="real",
    )

    events = store.list_events_page(
        task.id,
        user_id=user.id,
        after_id=0,
        limit=100,
    )
    assert events is not None
    assert [event.type for event in events.items] == ["queued"]
    assert events.items[0].stage == "queue"
    assert events.items[0].payload == {
        "depth": "quick",
        "execution_mode": "real",
        "simulated": False,
    }


def test_create_queued_task_rolls_back_task_when_event_insert_fails(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = store.create_user(
        username="rollback-user",
        password_hash="hash",
        password_salt="salt",
    )
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            CREATE TRIGGER reject_queued_event
            BEFORE INSERT ON task_events
            BEGIN
                SELECT RAISE(ABORT, 'event insert rejected');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="event insert rejected"):
        store.create_queued_task(
            question="Must roll back",
            depth="standard",
            user_id=user.id,
            execution_mode="simulated",
        )

    page = store.list_tasks_page(user_id=user.id, limit=100)
    assert page.items == []
```

- [ ] **Step 2: Write the failing health probe test**

```python
def test_check_health_executes_a_lightweight_select(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    statements = []
    original_connect = store._connect

    def traced_connect():
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(store, "_connect", traced_connect)

    assert store.check_health() is None
    assert [statement.strip().upper() for statement in statements] == ["SELECT 1"]
```

- [ ] **Step 3: Verify RED**

Run: `./.venv/bin/python -m pytest tests/web/test_task_store.py -q`

Expected: new tests fail because `create_queued_task()` and `check_health()` are absent.

- [ ] **Step 4: Extract insert helpers and implement one transaction**

```python
def create_task(
    self,
    *,
    question: str,
    depth: str = "standard",
    user_id: str | None = None,
) -> ResearchTask:
    task = _new_task(question=question, depth=depth, user_id=user_id)
    with self._connect() as conn:
        _insert_task(conn, task)
    return task


def create_queued_task(
    self,
    *,
    question: str,
    depth: str,
    user_id: str,
    execution_mode: str,
) -> ResearchTask:
    if execution_mode not in {"simulated", "real"}:
        raise ValueError(f"invalid execution mode: {execution_mode!r}")
    task = _new_task(question=question, depth=depth, user_id=user_id)
    payload = {
        "depth": task.depth,
        "execution_mode": execution_mode,
        "simulated": execution_mode == "simulated",
    }
    with self._connect() as conn:
        _insert_task(conn, task)
        conn.execute(
            """
            INSERT INTO task_events (
                task_id, type, stage, message, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                task.id,
                "queued",
                "queue",
                f"Task queued for {execution_mode} workflow.",
                json.dumps(payload, ensure_ascii=False),
                _utc_now(),
            ),
        )
    return task


def check_health(self) -> None:
    with self._connect() as conn:
        row = conn.execute("SELECT 1").fetchone()
    if row is None or int(row[0]) != 1:
        raise RuntimeError("SQLite health probe returned an invalid result")
```

Add module-level helpers immediately before `_task_from_row`:

```python
def _new_task(
    *,
    question: str,
    depth: str,
    user_id: str | None,
) -> ResearchTask:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question is required")
    if depth not in VALID_DEPTHS:
        raise ValueError(f"invalid depth: {depth!r}")
    now = _utc_now()
    return ResearchTask(
        id=f"task_{uuid.uuid4().hex}",
        question=cleaned_question,
        depth=depth,
        status="pending",
        created_at=now,
        updated_at=now,
        user_id=user_id,
    )


def _insert_task(conn: sqlite3.Connection, task: ResearchTask) -> None:
    conn.execute(
        """
        INSERT INTO research_tasks (
            id, question, depth, status, created_at, updated_at, user_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task.id,
            task.question,
            task.depth,
            task.status,
            task.created_at,
            task.updated_at,
            task.user_id,
        ),
    )
```

- [ ] **Step 5: Verify GREEN and store regressions**

Run: `./.venv/bin/python -m pytest tests/web/test_task_store.py -q`

Expected: atomicity, health, pagination, ownership, and concurrent-write store tests pass.

---

### Task 5: FastAPI Admission, Health, And Request Context Integration

**Files:**
- Modify: `paperpilot/web/app.py`
- Modify: `tests/web/test_web_app.py`

**Interfaces:**
- Consumes: `WebRuntimeConfig`, `RequestObservabilityMiddleware`, executor reservations, and `TaskStore.create_queued_task()`.
- Produces: `create_app(..., runtime_config=None)` with the same existing dependency-injection arguments.
- Produces public hidden routes: `GET /health/live` and `GET /health/ready`.
- Preserves all existing task/auth response schemas and Celery ambiguous-publication behavior.

- [ ] **Step 1: Migrate fake executors to the reservation protocol**

In `tests/web/test_web_app.py`, replace direct-only fakes with explicit reservation fakes:

```python
from paperpilot.web.task_executor import (
    TaskExecutor,
    TaskExecutorAtCapacityError,
    TaskExecutorShuttingDownError,
    TaskSubmissionReservation,
)


class FailingExecutor:
    is_shutdown = False

    def reserve(self):
        def fail(task_id: str, execution_mode: str):
            raise ConnectionError("broker unavailable")

        return TaskSubmissionReservation(fail)

    def shutdown(self) -> None:
        self.is_shutdown = True


class ClaimingFailingExecutor:
    def __init__(self, store: TaskStore) -> None:
        self.store = store
        self.is_shutdown = False

    def reserve(self):
        def claim_then_fail(task_id: str, execution_mode: str):
            assert self.store.claim_task(task_id) is not None
            raise ConnectionError("publish result was ambiguous")

        return TaskSubmissionReservation(claim_then_fail)

    def shutdown(self) -> None:
        self.is_shutdown = True


class RejectingExecutor:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.is_shutdown = reason == "shutdown"

    def reserve(self):
        if self.reason == "capacity":
            raise TaskExecutorAtCapacityError("full")
        raise TaskExecutorShuttingDownError("stopping")

    def shutdown(self) -> None:
        self.is_shutdown = True
```

- [ ] **Step 2: Write failing admission-before-database tests**

```python
from paperpilot.web.config import WebRuntimeConfig


def test_authentication_error_has_request_id(tmp_path):
    client = _client(tmp_path)

    response = client.get("/api/tasks")

    assert response.status_code == 401
    assert response.json() == {"detail": "authentication required"}
    assert response.headers["x-request-id"]


def test_create_app_fails_fast_for_invalid_runtime_environment(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("PAPERPILOT_THREAD_WORKERS", "0")
    store = TaskStore(tmp_path / "tasks.sqlite3")

    with pytest.raises(ValueError, match="PAPERPILOT_THREAD_WORKERS"):
        create_app(store)


@pytest.mark.parametrize(
    ("reason", "detail", "retry_after"),
    [
        ("capacity", "task executor is at capacity", "1"),
        ("shutdown", "task executor is shutting down", None),
    ],
)
def test_task_admission_rejection_writes_nothing(
    tmp_path,
    reason,
    detail,
    retry_after,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=RejectingExecutor(reason),
            runtime_config=WebRuntimeConfig(),
        )
    )
    user = _register(client)

    response = client.post(
        "/api/tasks",
        json={"question": "Do not persist me", "depth": "quick"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": detail}
    assert response.headers.get("retry-after") == retry_after
    assert response.headers["x-request-id"]
    assert store.list_tasks_page(user_id=user["id"], limit=100).items == []
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_artifacts").fetchone()[0] == 0
```

- [ ] **Step 3: Write failing capacity-recovery and database-failure tests**

```python
import threading
import time


class BlockingRunner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def run_simulated(self, task_id: str) -> None:
        self.started.set()
        assert self.release.wait(timeout=2)
        self.finished.set()

    def run_real(self, task_id: str) -> None:
        self.run_simulated(task_id)


def test_thread_capacity_recovers_after_background_work_finishes(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=0)
    app = create_app(
        store,
        workflow_runner=runner,
        task_executor=executor,
        runtime_config=WebRuntimeConfig(thread_workers=1, thread_queue_capacity=0),
    )
    with TestClient(app) as client:
        _register(client)
        first = client.post("/api/tasks", json={"question": "first", "depth": "quick"})
        assert first.status_code == 201
        assert runner.started.wait(timeout=1)

        overloaded = client.post(
            "/api/tasks", json={"question": "second", "depth": "quick"}
        )
        assert overloaded.status_code == 503

        runner.release.set()
        assert runner.finished.wait(timeout=1)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            recovered = client.post(
                "/api/tasks", json={"question": "third", "depth": "quick"}
            )
            if recovered.status_code == 201:
                break
            assert recovered.status_code == 503
            time.sleep(0.01)
        else:
            raise AssertionError("task capacity was not restored")
        assert recovered.status_code == 201


class FailingQueuedTaskStore(TaskStore):
    def create_queued_task(self, **kwargs):
        raise sqlite3.OperationalError("write unavailable")


class ReleaseSpyExecutor:
    def __init__(self) -> None:
        self.is_shutdown = False
        self.released = 0
        self.submitted = 0

    def reserve(self):
        def release():
            self.released += 1

        def submit(task_id, execution_mode):
            self.submitted += 1

        return TaskSubmissionReservation(submit, release)

    def shutdown(self) -> None:
        self.is_shutdown = True


def test_database_failure_releases_reservation_without_submit(tmp_path):
    store = FailingQueuedTaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = ReleaseSpyExecutor()
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    )
    _register(client)

    response = client.post(
        "/api/tasks",
        json={"question": "database failure", "depth": "quick"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "internal server error"
    assert executor.released == 1
    assert executor.submitted == 0
```

- [ ] **Step 4: Write failing liveness and readiness tests**

```python
class FailingHealthStore(TaskStore):
    def check_health(self) -> None:
        raise sqlite3.OperationalError("database unavailable")


def test_health_routes_are_public_hidden_and_split_dependencies(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = SynchronousTaskExecutor(runner)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    )

    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready"}
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health/live" not in paths
    assert "/health/ready" not in paths


def test_ready_fails_for_database_but_live_remains_ok(tmp_path):
    store = FailingHealthStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = SynchronousTaskExecutor(runner)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    )

    assert client.get("/health/live").status_code == 200
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"database": "failed", "executor": "ok"},
    }


def test_ready_fails_after_executor_shutdown(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = WorkflowRunner(store, delay_seconds=0)
    executor = SynchronousTaskExecutor(runner)
    executor.shutdown()
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(),
        )
    )

    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"] == {
        "database": "ok",
        "executor": "failed",
    }


def test_ready_stays_ok_while_thread_executor_is_saturated(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=0)
    reservation = executor.reserve()
    future = reservation.submit("health_probe_task", "simulated")
    assert runner.started.wait(timeout=1)
    client = TestClient(
        create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=WebRuntimeConfig(
                thread_workers=1,
                thread_queue_capacity=0,
            ),
        )
    )

    try:
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
    finally:
        runner.release.set()
        future.result(timeout=2)
        executor.shutdown()
```

- [ ] **Step 5: Verify RED**

Run: `./.venv/bin/python -m pytest tests/web/test_web_app.py tests/web/test_auth.py -q`

Expected: failures show absent runtime config injection, health routes, middleware, and reservation-based task creation.

- [ ] **Step 6: Wire config, middleware, request user context, and health routes**

Update imports and app-factory setup:

```python
import logging

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.observability import (
    RUNTIME_LOGGER_NAME,
    RequestObservabilityMiddleware,
    configure_paperpilot_logging,
)
from paperpilot.web.task_executor import (
    TaskExecutorAtCapacityError,
    TaskExecutorLike,
    TaskExecutorShuttingDownError,
    build_task_executor,
)


def create_app(
    task_store: TaskStore | None = None,
    *,
    simulation_delay_seconds: float = 0.4,
    workflow_runner: WorkflowRunner | None = None,
    task_executor: TaskExecutorLike | None = None,
    runtime_config: WebRuntimeConfig | None = None,
) -> FastAPI:
    config = runtime_config or WebRuntimeConfig.from_env()
    store = task_store or TaskStore()
    runner = workflow_runner or WorkflowRunner(
        store,
        delay_seconds=simulation_delay_seconds,
    )
    executor = task_executor or build_task_executor(runner, config=config)
    auth = AuthService(store)
    configure_paperpilot_logging(config)
    runtime_logger = logging.getLogger(RUNTIME_LOGGER_NAME)
    app = FastAPI(title="PaperPilot Web Workbench")
    app.add_middleware(RequestObservabilityMiddleware, config=config)
    app.state.runtime_config = config
    app.state.task_store = store
    app.state.task_executor = executor
```

Change the dependency so middleware can read the authenticated user after the sync dependency returns:

```python
    def require_user(
        request: Request,
        session_token: str | None = Cookie(
            default=None,
            alias=SESSION_COOKIE_NAME,
        ),
    ) -> WebUser:
        user = auth.get_user_for_token(session_token)
        if user is None:
            raise HTTPException(status_code=401, detail="authentication required")
        request.state.user_id = user.id
        return user
```

Add health routes before the authenticated task routes:

```python
    @app.get("/health/live", include_in_schema=False)
    def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", include_in_schema=False)
    def health_ready() -> Response:
        checks = {"database": "ok", "executor": "ok"}
        try:
            store.check_health()
        except Exception:
            checks["database"] = "failed"
        if executor.is_shutdown:
            checks["executor"] = "failed"
        if "failed" in checks.values():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "checks": checks},
            )
        return JSONResponse(status_code=200, content={"status": "ready"})
```

- [ ] **Step 7: Replace task creation with reserve, atomic write, and submit**

```python
    @app.post("/api/tasks", response_model=TaskResponse, status_code=201)
    def create_task(
        payload: CreateTaskRequest,
        request: Request,
        user: WebUser = Depends(require_user),
    ) -> dict[str, str]:
        try:
            reservation = executor.reserve()
        except TaskExecutorAtCapacityError as exc:
            runtime_logger.warning(
                "Task admission rejected",
                extra={
                    "event": "task.admission_rejected",
                    "request_id": request.state.request_id,
                    "user_id": user.id,
                    "environment": config.environment,
                    "reason": "capacity",
                    "executor": config.task_executor,
                    "workers": config.thread_workers,
                    "queue_capacity": config.thread_queue_capacity,
                },
            )
            raise HTTPException(
                status_code=503,
                detail="task executor is at capacity",
                headers={
                    "Retry-After": str(config.overload_retry_after_seconds)
                },
            ) from exc
        except TaskExecutorShuttingDownError as exc:
            runtime_logger.warning(
                "Task admission rejected",
                extra={
                    "event": "task.admission_rejected",
                    "request_id": request.state.request_id,
                    "user_id": user.id,
                    "environment": config.environment,
                    "reason": "shutdown",
                    "executor": config.task_executor,
                    "workers": config.thread_workers,
                    "queue_capacity": config.thread_queue_capacity,
                },
            )
            raise HTTPException(
                status_code=503,
                detail="task executor is shutting down",
            ) from exc

        try:
            task = store.create_queued_task(
                question=payload.question,
                depth=payload.depth,
                user_id=user.id,
                execution_mode=payload.execution_mode,
            )
        except Exception:
            reservation.release()
            raise

        response_payload = task.to_dict()
        try:
            reservation.submit(task.id, payload.execution_mode)
        except Exception as exc:
            try:
                failed_task = store.fail_pending_task(task.id)
                if failed_task is None:
                    current_task = store.get_task(task.id)
                    if current_task is not None and current_task.status in {
                        "running",
                        "completed",
                    }:
                        return response_payload
                else:
                    store.add_event(
                        task_id=task.id,
                        type="failed",
                        stage="queue",
                        message="Task queue submission failed.",
                        payload={
                            "execution_mode": payload.execution_mode,
                            "error_type": type(exc).__name__,
                        },
                    )
            except Exception:
                runtime_logger.exception(
                    "Task submission cleanup failed",
                    extra={
                        "event": "task.submission_cleanup_failed",
                        "request_id": request.state.request_id,
                        "user_id": user.id,
                        "environment": config.environment,
                        "exception_type": type(exc).__name__,
                    },
                )
            raise HTTPException(
                status_code=503,
                detail="task queue unavailable",
            ) from exc
        return response_payload
```

The reservation owns permit cleanup after `submit()` begins; do not call `reservation.release()` in the submit-exception branch.

- [ ] **Step 8: Preserve and extend existing submission-failure assertions**

Keep the existing tests that assert:

```python
assert response.status_code == 503
assert task.status == "failed"
assert [event.type for event in events] == ["queued", "failed"]
```

Also keep the ambiguous-publication assertions unchanged:

```python
assert response.status_code == 201
assert task.status == "running"
assert [event.type for event in event_page.items] == ["queued"]
```

- [ ] **Step 9: Verify GREEN and the entire Web slice**

Run: `./.venv/bin/python -m pytest tests/web/test_config.py tests/web/test_observability.py tests/web/test_task_executor.py tests/web/test_task_store.py tests/web/test_auth.py tests/web/test_web_app.py tests/web/test_workflow.py -q`

Expected: all selected Web tests pass, including admission rejection with zero writes, recovery, health semantics, request IDs, user isolation, and Celery ambiguous publication.

---

### Task 6: Reproducible Admission Benchmark And Operations Documentation

**Files:**
- Create: `scripts/benchmark_web_admission.py`
- Modify: `tests/web/test_runtime_config.py`
- Modify: `README.md`

**Interfaces:**
- Produces command: `./.venv/bin/python scripts/benchmark_web_admission.py --workers 2 --queue-capacity 4 --requests 12`.
- Produces one JSON object with configured capacity, accepted/rejected counts, database row counts, latency summaries, and capacity recovery.
- Does not enforce a machine-specific latency threshold.

- [ ] **Step 1: Write the failing benchmark smoke test**

```python
import json
import subprocess
import sys


def test_admission_benchmark_proves_bound_and_recovery():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_web_admission.py",
            "--workers",
            "1",
            "--queue-capacity",
            "0",
            "--requests",
            "2",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["configured_capacity"] == 1
    assert payload["accepted"] == 1
    assert payload["rejected"] == 1
    assert payload["task_rows_before_recovery"] == 1
    assert payload["event_rows_before_recovery"] == 1
    assert payload["capacity_recovered"] is True
    assert payload["unexpected_statuses"] == []
```

- [ ] **Step 2: Verify RED**

Run: `./.venv/bin/python -m pytest tests/web/test_runtime_config.py::test_admission_benchmark_proves_bound_and_recovery -q`

Expected: the test fails because `scripts/benchmark_web_admission.py` does not exist.

- [ ] **Step 3: Implement the deterministic benchmark**

The script must contain these concrete building blocks:

```python
"""Exercise bounded task admission without machine-specific thresholds."""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi.testclient import TestClient

from paperpilot.web.app import create_app
from paperpilot.web.auth import SESSION_COOKIE_NAME
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.task_executor import TaskExecutor
from paperpilot.web.task_store import TaskStore


class BlockingRunner:
    def __init__(self) -> None:
        self.release = threading.Event()

    def run_simulated(self, task_id: str) -> None:
        if not self.release.wait(timeout=30):
            raise TimeoutError("benchmark runner was not released")

    def run_real(self, task_id: str) -> None:
        self.run_simulated(task_id)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


def _row_counts(db_path: Path) -> tuple[int, int]:
    with sqlite3.connect(db_path) as conn:
        tasks = int(conn.execute("SELECT COUNT(*) FROM research_tasks").fetchone()[0])
        events = int(conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0])
    return tasks, events


def run_benchmark(*, workers: int, queue_capacity: int, requests: int) -> dict:
    capacity = workers + queue_capacity
    if requests <= capacity:
        raise ValueError("requests must be greater than configured capacity")

    with tempfile.TemporaryDirectory(prefix="paperpilot-admission-") as directory:
        db_path = Path(directory) / "tasks.sqlite3"
        store = TaskStore(db_path)
        runner = BlockingRunner()
        executor = TaskExecutor(
            runner,
            max_workers=workers,
            queue_capacity=queue_capacity,
        )
        config = WebRuntimeConfig(
            thread_workers=workers,
            thread_queue_capacity=queue_capacity,
            log_level="ERROR",
        )
        app = create_app(
            store,
            workflow_runner=runner,
            task_executor=executor,
            runtime_config=config,
        )
        latencies = []
        statuses = []
        with TestClient(app) as client:
            try:
                register = client.post(
                    "/api/auth/register",
                    json={"username": "benchmark", "password": "secret123"},
                )
                register.raise_for_status()
                session_token = client.cookies.get(SESSION_COOKIE_NAME)
                if session_token is None:
                    raise RuntimeError("registration did not create a session cookie")
                client.cookies.clear()

                def submit(index: int) -> tuple[int, float]:
                    started = time.perf_counter()
                    response = client.post(
                        "/api/tasks",
                        headers={
                            "Cookie": f"{SESSION_COOKIE_NAME}={session_token}"
                        },
                        json={
                            "question": f"benchmark task {index}",
                            "depth": "quick",
                            "execution_mode": "simulated",
                        },
                    )
                    return response.status_code, (time.perf_counter() - started) * 1000

                with ThreadPoolExecutor(max_workers=requests) as pool:
                    futures = [pool.submit(submit, index) for index in range(requests)]
                    for future in as_completed(futures):
                        status, latency = future.result()
                        statuses.append(status)
                        latencies.append(latency)

                accepted = statuses.count(201)
                rejected = statuses.count(503)
                task_rows, event_rows = _row_counts(db_path)
                runner.release.set()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        probe = executor.reserve()
                    except Exception:
                        time.sleep(0.01)
                    else:
                        probe.release()
                        break
                else:
                    raise RuntimeError("executor capacity did not recover")

                recovery = client.post(
                    "/api/tasks",
                    headers={"Cookie": f"{SESSION_COOKIE_NAME}={session_token}"},
                    json={"question": "recovery", "depth": "quick"},
                )
            finally:
                runner.release.set()

        unexpected = sorted(status for status in statuses if status not in {201, 503})
        result = {
            "configured_capacity": capacity,
            "requests": requests,
            "accepted": accepted,
            "rejected": rejected,
            "unexpected_statuses": unexpected,
            "task_rows_before_recovery": task_rows,
            "event_rows_before_recovery": event_rows,
            "latency_ms_median": round(statistics.median(latencies), 3),
            "latency_ms_p95": round(_percentile(latencies, 0.95), 3),
            "capacity_recovered": recovery.status_code == 201,
        }
        if accepted != capacity:
            raise RuntimeError(f"accepted {accepted}, expected {capacity}")
        if rejected != requests - capacity:
            raise RuntimeError(f"rejected {rejected}, expected {requests - capacity}")
        if task_rows != accepted or event_rows != accepted:
            raise RuntimeError("rejected requests wrote database rows")
        if unexpected or not result["capacity_recovered"]:
            raise RuntimeError("admission benchmark invariants failed")
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument("--requests", type=int, default=12)
    args = parser.parse_args()
    result = run_benchmark(
        workers=args.workers,
        queue_capacity=args.queue_capacity,
        requests=args.requests,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
```

The inner `finally` executes before `TestClient.__exit__`, so the blocking runner is always released before the app shutdown hook waits for executor completion.

- [ ] **Step 4: Verify benchmark GREEN**

Run: `./.venv/bin/python -m pytest tests/web/test_runtime_config.py::test_admission_benchmark_proves_bound_and_recovery -q`

Expected: the smoke test passes and validates counts rather than absolute milliseconds.

- [ ] **Step 5: Document runtime protection and exact commands**

Add a `Web Runtime Protection` section to `README.md` containing this environment block:

```bash
export PAPERPILOT_TASK_EXECUTOR=thread
export PAPERPILOT_THREAD_WORKERS=2
export PAPERPILOT_THREAD_QUEUE_CAPACITY=4
export PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS=1
export PAPERPILOT_LOG_LEVEL=INFO
export PAPERPILOT_LOG_FORMAT=json
export PAPERPILOT_SLOW_REQUEST_MS=1000
export PAPERPILOT_ENV=development
```

Document all of the following exact facts:

- Thread capacity is `workers + queue capacity` per Uvicorn process.
- With two Uvicorn workers and defaults, the theoretical process-local total is 12, but load distribution is not guaranteed to be even.
- Saturated task creation returns 503 with `Retry-After` and creates no database rows.
- `/health/live` has no dependency probe; `/health/ready` checks `SELECT 1` and executor shutdown only.
- Redis, MCP, and temporary executor saturation do not fail readiness.
- `X-Request-ID` is accepted only for `[A-Za-z0-9._:-]` values up to 128 characters; otherwise PaperPilot generates one.
- Disable duplicate Uvicorn access logs with `--no-access-log` when PaperPilot structured access logging is active.
- Prometheus/OpenTelemetry and cross-process rate limiting remain separate deployment-topology work.

Update the Uvicorn command to include:

```bash
./.venv/bin/uvicorn paperpilot.web.app:app \
  --host 127.0.0.1 --port 8000 --workers 2 --no-access-log
```

Add the benchmark command:

```bash
./.venv/bin/python scripts/benchmark_web_admission.py \
  --workers 2 --queue-capacity 4 --requests 12
```

- [ ] **Step 6: Extend README artifact assertions and verify**

Add assertions to `tests/web/test_runtime_config.py`:

```python
def test_readme_documents_runtime_protection_contract():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for value in (
        "PAPERPILOT_THREAD_WORKERS",
        "PAPERPILOT_THREAD_QUEUE_CAPACITY",
        "PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS",
        "PAPERPILOT_LOG_FORMAT",
        "/health/live",
        "/health/ready",
        "Retry-After",
        "X-Request-ID",
        "--no-access-log",
        "benchmark_web_admission.py",
    ):
        assert value in text
```

Run: `./.venv/bin/python -m pytest tests/web/test_runtime_config.py -q`

Expected: Celery runtime docs, runtime protection docs, and admission benchmark smoke all pass.

---

### Task 7: Full Verification And Review Gate

**Files:**
- Review all files listed in the File Responsibility Map.
- Do not modify unrelated files discovered in the dirty worktree.

**Interfaces:**
- Consumes every deliverable from Tasks 1-6.
- Produces fresh evidence for focused behavior, full regression, importability, formatting, runtime health, and overload recovery.

- [ ] **Step 1: Run focused runtime-protection tests**

Run:

```bash
./.venv/bin/python -m pytest \
  tests/web/test_config.py \
  tests/web/test_observability.py \
  tests/web/test_task_executor.py \
  tests/web/test_task_store.py \
  tests/web/test_auth.py \
  tests/web/test_web_app.py \
  tests/web/test_workflow.py \
  tests/web/test_runtime_config.py -q
```

Expected: all selected tests pass with no hangs or leaked executor threads.

- [ ] **Step 2: Run the complete default suite**

Run: `./.venv/bin/python -m pytest`

Expected: every non-slow test selected by `pytest.ini` passes. Record the fresh pass/deselection counts; do not reuse earlier totals.

- [ ] **Step 3: Verify compilation and whitespace**

Run:

```bash
./.venv/bin/python -m compileall -q paperpilot scripts/benchmark_web_admission.py
git diff --check
rg -n "[[:blank:]]+$" \
  paperpilot/web/config.py \
  paperpilot/web/observability.py \
  scripts/benchmark_web_admission.py \
  tests/web/test_config.py \
  tests/web/test_observability.py
```

Expected: compileall and `git diff --check` exit 0; the trailing-whitespace scan returns no matches.

- [ ] **Step 4: Run the default admission benchmark**

Run:

```bash
./.venv/bin/python scripts/benchmark_web_admission.py \
  --workers 2 --queue-capacity 4 --requests 12
```

Expected JSON invariants:

```json
{
  "configured_capacity": 6,
  "accepted": 6,
  "rejected": 6,
  "task_rows_before_recovery": 6,
  "event_rows_before_recovery": 6,
  "capacity_recovered": true,
  "unexpected_statuses": []
}
```

Latency fields are reported but have no fixed pass threshold.

- [ ] **Step 5: Run an actual Uvicorn health smoke**

Start a temporary server in a persistent terminal session:

```bash
PAPERPILOT_TASK_DB_PATH=/tmp/paperpilot-runtime-protection.sqlite3 \
PAPERPILOT_LOG_FORMAT=text \
./.venv/bin/uvicorn paperpilot.web.app:app \
  --host 127.0.0.1 --port 8765 --workers 1 --no-access-log
```

From another command session, run:

```bash
curl -i http://127.0.0.1:8765/health/live
curl -i http://127.0.0.1:8765/health/ready
```

Expected: both return 200, each response has exactly one `X-Request-ID`, and the server emits one PaperPilot access log per request. Stop the server cleanly before proceeding.

- [ ] **Step 6: Perform an independent code-review pass**

Use `superpowers:requesting-code-review` against the complete diff. Reviewers must explicitly inspect:

- permit over-release, submit/shutdown races, and thread leaks;
- database writes before admission and transaction rollback;
- Celery ambiguous-publication preservation;
- response-start exception behavior and duplicate headers;
- route cardinality and sensitive log fields;
- readiness dependency boundaries;
- tests that can hang when a blocking runner is not released.

Address every confirmed finding with a new red/green cycle, then rerun Steps 1-5.

- [ ] **Step 7: Report outcome without changing Git state**

The final report must include:

- implemented behavior and why each part exists;
- exact files created or modified;
- focused and full-suite counts from fresh output;
- compileall, whitespace, benchmark, and Uvicorn smoke results;
- residual boundaries: per-process capacity, SQLite single writer, and Celery DB/broker dual-write window;
- confirmation that no files were staged, committed, pushed, reset, stashed, cleaned, or deleted.

## Plan Self-Review

### Specification Coverage

- Central configuration and startup failure: Task 1.
- Pure ASGI request ID, logs, safe 500, response-start boundary: Task 2.
- Bounded thread admission, permit ownership, shutdown race: Task 3.
- Atomic task plus queued event and SQLite `SELECT 1`: Task 4.
- Pre-write API rejection, retry semantics, health routes, `request.state.user_id`, Celery ambiguity preservation: Task 5.
- Reproducible behavior benchmark and operations documentation: Task 6.
- Focused/full verification, real server smoke, and independent review: Task 7.
- Explicit non-goals and residual risks: Global Constraints, README work, and final report requirements.

### Type And Name Consistency

- `WebRuntimeConfig` is defined in Task 1 and consumed unchanged by Tasks 2, 3, 5, and 6.
- `TaskSubmissionReservation.submit()` returns `object`, preserving `Future` and Celery result compatibility.
- `TaskExecutorLike` exposes `is_shutdown`, `reserve()`, and `shutdown()` in both tests and production code.
- `TaskStore.create_queued_task()` accepts the exact fields used by `app.create_task()`.
- Logger names and middleware constructor names match between Tasks 2 and 5.
- Health response keys match the API tests and design document.

### Execution Safety

- Every blocking runner has a bounded wait and an explicit release path.
- No task is persisted before a reservation succeeds.
- The database-failure branch releases an unsubmitted reservation.
- The submit-failure branch relies on reservation-owned cleanup and never releases twice.
- No plan step mutates Git staging, commits, remotes, or unrelated dirty files.
