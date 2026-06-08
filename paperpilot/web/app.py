"""FastAPI app for the local PaperPilot Web workbench."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from paperpilot.web.task_store import TaskStore
from paperpilot.web.workflow import WorkflowRunner

STATIC_DIR = Path(__file__).parent / "static"


class CreateTaskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    depth: Literal["quick", "standard", "deep"] = "standard"
    execution_mode: Literal["simulated", "real"] = "simulated"


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


def create_app(
    task_store: TaskStore | None = None,
    *,
    simulation_delay_seconds: float = 0.4,
    workflow_runner: WorkflowRunner | None = None,
) -> FastAPI:
    store = task_store or TaskStore()
    runner = workflow_runner or WorkflowRunner(
        store,
        delay_seconds=simulation_delay_seconds,
    )
    app = FastAPI(title="PaperPilot Web Workbench")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/tasks", response_model=TaskResponse, status_code=201)
    def create_task(
        payload: CreateTaskRequest,
        background_tasks: BackgroundTasks,
    ) -> dict[str, str]:
        task = store.create_task(
            question=payload.question,
            depth=payload.depth,
        )
        store.add_event(
            task_id=task.id,
            type="queued",
            stage="queue",
            message=f"Task queued for {payload.execution_mode} workflow.",
            payload={
                "depth": task.depth,
                "execution_mode": payload.execution_mode,
                "simulated": payload.execution_mode == "simulated",
            },
        )
        if payload.execution_mode == "real":
            background_tasks.add_task(runner.run_real, task.id)
        else:
            background_tasks.add_task(runner.run_simulated, task.id)
        return task.to_dict()

    @app.get("/api/tasks", response_model=list[TaskResponse])
    def list_tasks(
        status: Literal["pending", "running", "completed", "failed"] | None = Query(
            default=None
        ),
    ) -> list[dict[str, str]]:
        return [task.to_dict() for task in store.list_tasks(status=status)]

    @app.get("/api/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str) -> dict[str, str]:
        task = store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task.to_dict()

    @app.get("/api/tasks/{task_id}/events", response_model=list[TaskEventResponse])
    def list_task_events(task_id: str) -> list[dict[str, int | str | dict | None]]:
        events = store.list_events(task_id)
        if events is None:
            raise HTTPException(status_code=404, detail="task not found")
        return [event.to_dict() for event in events]

    @app.get(
        "/api/tasks/{task_id}/artifacts",
        response_model=list[TaskArtifactResponse],
    )
    def list_task_artifacts(task_id: str) -> list[dict[str, int | str | dict]]:
        artifacts = store.list_artifacts(task_id)
        if artifacts is None:
            raise HTTPException(status_code=404, detail="task not found")
        return [artifact.to_dict() for artifact in artifacts]

    return app


app = create_app()
