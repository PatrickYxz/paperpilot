"""FastAPI composition for the local PaperPilot Web workbench."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import Cookie, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from paperpilot.deep_reading.runner import DeepReadingRunner
from paperpilot.tools.mcp_runtime import MCPRuntime
from paperpilot.web.auth import SESSION_COOKIE_NAME, AuthService
from paperpilot.web.checkpoint import SqliteCheckpointRuntime
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.observability import (
    RequestObservabilityMiddleware,
    configure_paperpilot_logging,
)
from paperpilot.web.routes.auth import build_auth_router
from paperpilot.web.routes.conversations import build_conversation_router
from paperpilot.web.routes.papers import (
    PaperSearch,
    build_paper_router,
    default_web_paper_search,
)
from paperpilot.web.routes.task_updates import build_task_updates_router
from paperpilot.web.task_executor import (
    TaskExecutorLike,
    build_task_executor,
)
from paperpilot.web.task_store import TaskStore, WebUser
from paperpilot.web.workflow import WorkflowRunner


STATIC_DIR = Path(__file__).parent / "static"


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
            pass


def create_app(
    task_store: TaskStore | None = None,
    *,
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
            task_store=task_store,
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
    *,
    task_store: TaskStore | None,
    task_executor: TaskExecutorLike | None,
    runtime_config: WebRuntimeConfig | None,
    checkpoint_runtime: SqliteCheckpointRuntime | None,
    mcp_runtime: MCPRuntime | None,
    deep_reading_runner: DeepReadingRunner | None,
    paper_search: PaperSearch | None,
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

    runner = WorkflowRunner(store, deep_reading_runner=deep_runner)
    if task_executor is None:
        executor = build_task_executor(runner, config=config)
        owned_resources.executor = executor
    else:
        executor = task_executor
    auth = AuthService(store)
    search = paper_search or default_web_paper_search
    configure_paperpilot_logging(config)

    def shutdown_resources() -> None:
        try:
            executor.shutdown()
        finally:
            try:
                if owns_checkpoint_runtime:
                    checkpoint.close()
            finally:
                try:
                    if owns_mcp_runtime and mcp is not None:
                        mcp.close()
                finally:
                    if owns_task_store:
                        store.close()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        shutdown_resources()

    app = FastAPI(title="PaperPilot Web Workbench", lifespan=lifespan)
    app.add_middleware(RequestObservabilityMiddleware, config=config)
    app.state.runtime_config = config
    app.state.task_store = store
    app.state.task_executor = executor
    app.state.checkpoint_runtime = checkpoint
    app.state.deep_reading_runner = deep_runner

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

    app.include_router(build_auth_router(auth=auth, require_user=require_user))
    app.include_router(
        build_paper_router(require_user=require_user, paper_search=search)
    )
    app.include_router(
        build_conversation_router(
            store=store,
            executor=executor,
            require_user=require_user,
            deep_reading_runner=deep_runner,
            paper_search=search,
        )
    )
    app.include_router(
        build_task_updates_router(store=store, require_user=require_user)
    )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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

    return app


app = create_app()
