"""Low-overhead request context and structured Web logging."""
from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from time import perf_counter as _perf_counter
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
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class TextLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        parts = [record.levelname, record.name, record.getMessage()]
        for field in _STRUCTURED_FIELDS:
            if hasattr(record, field):
                value = json.dumps(getattr(record, field), ensure_ascii=False)
                parts.append(f"{field}={value}")
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
        owned_handlers = [
            candidate
            for candidate in logger.handlers
            if getattr(candidate, "_paperpilot_owned", False)
        ]
        if owned_handlers:
            handler = owned_handlers[0]
            for duplicate in owned_handlers[1:]:
                logger.removeHandler(duplicate)
                duplicate.close()
        else:
            handler = logging.StreamHandler(target_stream)
            handler._paperpilot_owned = True  # type: ignore[attr-defined]
            logger.addHandler(handler)
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
        started_at = _perf_counter()
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
        duration_ms = round((_perf_counter() - started_at) * 1000, 3)
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
