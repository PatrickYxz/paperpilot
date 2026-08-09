"""FastAPI app for the local PaperPilot Web workbench."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from paperpilot.deep_reading.runner import DeepReadingRunner
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.auth import (
    SESSION_COOKIE_NAME,
    AuthService,
    InvalidCredentialsError,
    UsernameAlreadyExistsError,
)
from paperpilot.web.checkpoint import SqliteCheckpointRuntime
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.conversation_routes import (
    PaperSearch,
    create_conversation_router,
    default_web_paper_search,
)
from paperpilot.web.eval_summary import (
    build_eval_snapshot,
    list_calibration_candidates,
)
from paperpilot.web.observability import (
    RUNTIME_LOGGER_NAME,
    RequestObservabilityMiddleware,
    configure_paperpilot_logging,
)
from paperpilot.web.pagination import (
    InvalidTaskCursor,
    decode_task_cursor,
    encode_task_cursor,
)
from paperpilot.web.task_executor import (
    TaskExecutorAtCapacityError,
    TaskExecutorLike,
    TaskExecutorShuttingDownError,
    build_task_executor,
)
from paperpilot.web.task_store import (
    TaskArtifactBatch,
    TaskEventBatch,
    TaskStore,
    WebUser,
)
from paperpilot.web.workflow import WorkflowRunner

STATIC_DIR = Path(__file__).parent / "static"


class CreateTaskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    depth: Literal["quick", "standard", "deep"] = "standard"
    execution_mode: Literal["simulated", "real"] = "simulated"


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)


class UserResponse(BaseModel):
    id: str
    username: str
    created_at: str


class TaskResponse(BaseModel):
    id: str
    question: str
    depth: str
    status: str
    created_at: str
    updated_at: str


class TaskEventResponse(BaseModel):
    id: int
    task_id: str
    type: str
    stage: str | None
    message: str
    payload: dict
    created_at: str


class TaskArtifactResponse(BaseModel):
    id: int
    task_id: str
    kind: str
    title: str
    content: str
    payload: dict
    created_at: str


class TaskPageResponse(BaseModel):
    items: list[TaskResponse]
    next_cursor: str | None
    has_more: bool


class TaskEventPageResponse(BaseModel):
    items: list[TaskEventResponse]
    next_after_id: int
    has_more: bool


class TaskArtifactPageResponse(BaseModel):
    items: list[TaskArtifactResponse]
    next_after_id: int
    has_more: bool


class TaskUpdatesResponse(BaseModel):
    task: TaskResponse
    events: TaskEventPageResponse
    artifacts: TaskArtifactPageResponse


def _event_page_dict(batch: TaskEventBatch) -> dict:
    return {
        "items": [event.to_dict() for event in batch.items],
        "next_after_id": batch.next_after_id,
        "has_more": batch.has_more,
    }


def _artifact_page_dict(batch: TaskArtifactBatch) -> dict:
    return {
        "items": [artifact.to_dict() for artifact in batch.items],
        "next_after_id": batch.next_after_id,
        "has_more": batch.has_more,
    }


@dataclass
class _OwnedAppResources:
    """Resources created by create_app and eligible for rollback cleanup."""

    store: TaskStore | None = None
    checkpoint: SqliteCheckpointRuntime | None = None
    mcp: MCPRuntime | None = None
    executor: TaskExecutorLike | None = None


def _best_effort_construction_cleanup(resources: _OwnedAppResources) -> None:
    """Close every constructed owned resource without masking build errors."""
    closers = (
        (resources.executor, "shutdown"),
        (resources.checkpoint, "close"),
        (resources.mcp, "close"),
        (resources.store, "close"),
    )
    for resource, method_name in closers:
        if resource is None:
            continue
        try:
            getattr(resource, method_name)()
        except BaseException:
            # create_app's original construction exception remains primary.
            pass


def create_app(
    task_store: TaskStore | None = None,
    *,
    simulation_delay_seconds: float = 0.4,
    workflow_runner: WorkflowRunner | None = None,
    task_executor: TaskExecutorLike | None = None,
    runtime_config: WebRuntimeConfig | None = None,
    checkpoint_runtime: SqliteCheckpointRuntime | None = None,
    mcp_runtime: MCPRuntime | None = None,
    deep_reading_runner: DeepReadingRunner | None = None,
    paper_search: PaperSearch | None = None,
) -> FastAPI:
    owned_resources = _OwnedAppResources()
    try:
        return _create_app(
            task_store,
            simulation_delay_seconds=simulation_delay_seconds,
            workflow_runner=workflow_runner,
            task_executor=task_executor,
            runtime_config=runtime_config,
            checkpoint_runtime=checkpoint_runtime,
            mcp_runtime=mcp_runtime,
            deep_reading_runner=deep_reading_runner,
            paper_search=paper_search,
            owned_resources=owned_resources,
        )
    except BaseException:
        _best_effort_construction_cleanup(owned_resources)
        raise


def _create_app(
    task_store: TaskStore | None = None,
    *,
    simulation_delay_seconds: float = 0.4,
    workflow_runner: WorkflowRunner | None = None,
    task_executor: TaskExecutorLike | None = None,
    runtime_config: WebRuntimeConfig | None = None,
    checkpoint_runtime: SqliteCheckpointRuntime | None = None,
    mcp_runtime: MCPRuntime | None = None,
    deep_reading_runner: DeepReadingRunner | None = None,
    paper_search: PaperSearch | None = None,
    owned_resources: _OwnedAppResources,
) -> FastAPI:
    config = runtime_config or WebRuntimeConfig.from_env()
    owns_task_store = task_store is None
    store = task_store if task_store is not None else TaskStore()
    if owns_task_store:
        owned_resources.store = store
    owns_checkpoint_runtime = False
    owns_mcp_runtime = False
    checkpoint = checkpoint_runtime
    mcp = mcp_runtime
    deep_runner = deep_reading_runner
    if checkpoint is None:
        checkpoint_path = (
            store.db_path.parent / "checkpoints.sqlite3"
            if task_store is not None
            else config.checkpoint_db_path
        )
        checkpoint = SqliteCheckpointRuntime.open(checkpoint_path)
        owns_checkpoint_runtime = True
        owned_resources.checkpoint = checkpoint
    if deep_runner is None:
        if mcp is None:
            mcp = MCPRuntime()
            owns_mcp_runtime = True
            owned_resources.mcp = mcp
        deep_runner = DeepReadingRunner(
            task_store=store,
            checkpointer=checkpoint.saver,
            mcp_runtime=mcp,
            summary_token_threshold=config.summary_token_threshold,
            summary_recent_turns=config.summary_recent_turns,
            research_recursion_limit=config.research_recursion_limit,
            research_model_call_limit=config.research_model_call_limit,
            research_tool_call_limit=config.research_tool_call_limit,
            research_max_output_tokens=config.research_max_output_tokens,
            research_model_retries=config.research_model_retries,
        )
    runner = workflow_runner or WorkflowRunner(
        store,
        delay_seconds=simulation_delay_seconds,
        deep_reading_runner=deep_runner,
    )
    if task_executor is None:
        executor = build_task_executor(runner, config=config)
        owned_resources.executor = executor
    else:
        executor = task_executor
    auth = AuthService(store)
    configure_paperpilot_logging(config)
    runtime_logger = logging.getLogger(RUNTIME_LOGGER_NAME)
    app = FastAPI(title="PaperPilot Web Workbench")
    app.add_middleware(RequestObservabilityMiddleware, config=config)
    app.state.runtime_config = config
    app.state.task_store = store
    app.state.task_executor = executor
    app.state.checkpoint_runtime = checkpoint
    app.state.deep_reading_runner = deep_runner

    def shutdown_resources() -> None:
        try:
            executor.shutdown()
        finally:
            try:
                if owns_checkpoint_runtime and checkpoint is not None:
                    checkpoint.close()
            finally:
                try:
                    if owns_mcp_runtime and mcp is not None:
                        mcp.close()
                finally:
                    if owns_task_store:
                        store.close()

    if hasattr(app, "add_event_handler"):
        app.add_event_handler("shutdown", shutdown_resources)
    else:
        app.router.add_event_handler("shutdown", shutdown_resources)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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

    app.include_router(
        create_conversation_router(
            store=store,
            executor=executor,
            deep_reading_runner=deep_runner,
            require_user=require_user,
            paper_search=paper_search or default_web_paper_search,
        )
    )

    def set_session_cookie(response: Response, token: str) -> None:
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            httponly=True,
            samesite="lax",
            path="/",
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health/live", include_in_schema=False)
    def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", include_in_schema=False)
    def health_ready() -> Response:
        checks = {"database": "ok", "checkpoint": "ok", "executor": "ok"}
        try:
            store.check_health()
        except Exception:
            checks["database"] = "failed"
        try:
            checkpoint.check_health()
        except Exception:
            checks["checkpoint"] = "failed"
        if executor.is_shutdown:
            checks["executor"] = "failed"
        if "failed" in checks.values():
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "checks": checks},
            )
        return JSONResponse(status_code=200, content={"status": "ready"})

    @app.post("/api/auth/register", response_model=UserResponse, status_code=201)
    def register(payload: AuthRequest, response: Response) -> dict[str, str]:
        try:
            session = auth.register(
                username=payload.username,
                password=payload.password,
            )
        except UsernameAlreadyExistsError as exc:
            raise HTTPException(
                status_code=409,
                detail="username already exists",
            ) from exc
        set_session_cookie(response, session.token)
        return session.user.to_public_dict()

    @app.post("/api/auth/login", response_model=UserResponse)
    def login(payload: AuthRequest, response: Response) -> dict[str, str]:
        try:
            session = auth.login(
                username=payload.username,
                password=payload.password,
            )
        except InvalidCredentialsError as exc:
            raise HTTPException(
                status_code=401,
                detail="invalid username or password",
            ) from exc
        set_session_cookie(response, session.token)
        return session.user.to_public_dict()

    @app.post("/api/auth/logout", status_code=204)
    def logout(
        response: Response,
        session_token: str | None = Cookie(
            default=None,
            alias=SESSION_COOKIE_NAME,
        ),
    ) -> None:
        auth.logout(session_token)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        return None

    @app.get("/api/auth/me", response_model=UserResponse)
    def get_current_user(user: WebUser = Depends(require_user)) -> dict[str, str]:
        return user.to_public_dict()

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
            except Exception as cleanup_exc:
                runtime_logger.exception(
                    "Task submission cleanup failed",
                    extra={
                        "event": "task.submission_cleanup_failed",
                        "request_id": request.state.request_id,
                        "user_id": user.id,
                        "environment": config.environment,
                        "exception_type": type(cleanup_exc).__name__,
                    },
                )
            raise HTTPException(
                status_code=503,
                detail="task queue unavailable",
            ) from exc
        return response_payload

    @app.get("/api/tasks", response_model=TaskPageResponse)
    def list_tasks(
        status: Literal["pending", "running", "completed", "failed"] | None = Query(
            default=None
        ),
        limit: int = Query(default=50, ge=1, le=100),
        cursor: str | None = Query(default=None),
        user: WebUser = Depends(require_user),
    ) -> dict:
        before_created_at = None
        before_id = None
        if cursor is not None:
            try:
                position = decode_task_cursor(
                    cursor,
                    expected_user_id=user.id,
                    expected_status=status,
                )
            except InvalidTaskCursor as exc:
                raise HTTPException(
                    status_code=422,
                    detail="invalid task cursor",
                ) from exc
            before_created_at = position.created_at
            before_id = position.task_id

        page = store.list_tasks_page(
            user_id=user.id,
            status=status,
            limit=limit,
            before_created_at=before_created_at,
            before_id=before_id,
        )
        next_cursor = None
        if page.has_more:
            final_task = page.items[-1]
            next_cursor = encode_task_cursor(
                user_id=user.id,
                status=status,
                created_at=final_task.created_at,
                task_id=final_task.id,
            )
        return {
            "items": [task.to_dict() for task in page.items],
            "next_cursor": next_cursor,
            "has_more": page.has_more,
        }

    @app.get("/api/tasks/{task_id}", response_model=TaskResponse)
    def get_task(
        task_id: str,
        user: WebUser = Depends(require_user),
    ) -> dict[str, str]:
        task = store.get_task(task_id, user_id=user.id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task.to_dict()

    @app.get(
        "/api/tasks/{task_id}/events",
        response_model=TaskEventPageResponse,
    )
    def list_task_events(
        task_id: str,
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
        user: WebUser = Depends(require_user),
    ) -> dict:
        page = store.list_events_page(
            task_id,
            user_id=user.id,
            after_id=after_id,
            limit=limit,
        )
        if page is None:
            raise HTTPException(status_code=404, detail="task not found")
        return _event_page_dict(page)

    @app.get(
        "/api/tasks/{task_id}/artifacts",
        response_model=TaskArtifactPageResponse,
    )
    def list_task_artifacts(
        task_id: str,
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
        user: WebUser = Depends(require_user),
    ) -> dict:
        page = store.list_artifacts_page(
            task_id,
            user_id=user.id,
            after_id=after_id,
            limit=limit,
        )
        if page is None:
            raise HTTPException(status_code=404, detail="task not found")
        return _artifact_page_dict(page)

    @app.get(
        "/api/tasks/{task_id}/updates",
        response_model=TaskUpdatesResponse,
    )
    def get_task_updates(
        task_id: str,
        after_event_id: int = Query(default=0, ge=0),
        after_artifact_id: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=100),
        user: WebUser = Depends(require_user),
    ) -> dict:
        updates = store.get_task_updates(
            task_id,
            user_id=user.id,
            after_event_id=after_event_id,
            after_artifact_id=after_artifact_id,
            limit=limit,
        )
        if updates is None:
            raise HTTPException(status_code=404, detail="task not found")
        return {
            "task": updates.task.to_dict(),
            "events": _event_page_dict(updates.events),
            "artifacts": _artifact_page_dict(updates.artifacts),
        }

    @app.get("/api/eval/summary")
    def get_eval_summary() -> dict:
        return build_eval_snapshot()

    @app.get("/api/eval/calibration-candidates")
    def get_calibration_candidates(
        category: str | None = Query(default=None),
        review_decision: str | None = Query(default=None),
    ) -> dict:
        return list_calibration_candidates(
            category=category,
            review_decision=review_decision,
        )

    return app


app = create_app()
