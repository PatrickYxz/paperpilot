"""Workflow runner boundary for Web research tasks."""
from __future__ import annotations

import time
from inspect import Parameter, signature
from typing import Callable, Protocol

from paperpilot.web.event_mapper import map_paperpilot_event
from paperpilot.web.task_store import TaskStore

RealRunner = Callable[..., list[dict]]


class DeepReadingRunnerLike(Protocol):
    def run(self, task_id: str) -> None: ...


class WorkflowRunner:
    """Owns task state transitions, events, and artifacts."""

    def __init__(
        self,
        store: TaskStore,
        *,
        delay_seconds: float = 0.4,
        real_runner: RealRunner | None = None,
        deep_reading_runner: DeepReadingRunnerLike | None = None,
    ) -> None:
        self.store = store
        self.delay_seconds = delay_seconds
        self.real_runner = real_runner or _default_real_runner
        self.deep_reading_runner = deep_reading_runner

    def run_simulated(self, task_id: str) -> None:
        """Run a placeholder workflow that exercises the task boundary."""
        try:
            task = self.store.get_task(task_id)
            if task is None:
                raise ValueError(f"task not found: {task_id}")

            self._sleep()
            self.store.update_status(task_id, "running")
            self.store.add_event(
                task_id=task_id,
                type="started",
                stage="start",
                message="Simulated workflow started.",
                payload={"simulated": True, "step": 1},
            )
            self._sleep()
            self.store.add_event(
                task_id=task_id,
                type="progress",
                stage="prepare",
                message="Preparing research workflow state.",
                payload={"simulated": True, "step": 2},
            )
            self._sleep()
            self.store.add_event(
                task_id=task_id,
                type="progress",
                stage="deep_read_placeholder",
                message="Placeholder deep-read stage skipped in A3.0.",
                payload={"simulated": True, "step": 3},
            )
            self.store.add_artifact(
                task_id=task_id,
                kind="result",
                title="Simulated research result",
                content=(
                    "This is a simulated research result. Real PaperPilot "
                    "deep-read execution will be connected in A3.1."
                ),
                payload={
                    "simulated": True,
                    "source": "WorkflowRunner.run_simulated",
                    "depth": task.depth,
                },
            )
            self.store.update_status(task_id, "completed")
            self.store.add_event(
                task_id=task_id,
                type="completed",
                stage="complete",
                message="Simulated workflow completed.",
                payload={"simulated": True, "step": 4, "artifact_kind": "result"},
            )
        except Exception as exc:
            self.store.update_status(task_id, "failed")
            self.store.add_event(
                task_id=task_id,
                type="failed",
                stage="failure",
                message=f"Simulated workflow failed: {exc}",
                payload={"simulated": True},
            )

    def run_real(self, task_id: str) -> None:
        """Route conversation work to LangGraph, or run the legacy one-shot agent."""
        task = self.store.get_task(task_id)
        if task is not None and task.conversation_id is not None:
            if self.deep_reading_runner is None:
                raise RuntimeError(
                    "conversation task requires a configured deep-reading runner"
                )
            # DeepReadingRunner deliberately lets infrastructure failures escape so
            # Celery can redeliver them. Keep this call outside the legacy broad
            # exception boundary below.
            self.deep_reading_runner.run(task_id)
            return

        try:
            if task is None:
                raise ValueError(f"task not found: {task_id}")

            self.store.update_status(task_id, "running")
            self.store.add_event(
                task_id=task_id,
                type="started",
                stage="real_start",
                message="Real PaperPilot execution started.",
                payload={"execution_mode": "real", "depth": task.depth},
            )
            messages = _call_real_runner(
                self.real_runner,
                task.question,
                on_event=lambda kind, payload: self._record_agent_event(
                    task_id,
                    kind,
                    payload,
                ),
            )
            final_text = _extract_final_text(messages)
            self.store.add_artifact(
                task_id=task_id,
                kind="result",
                title="PaperPilot result",
                content=final_text,
                payload={
                    "execution_mode": "real",
                    "source": "WorkflowRunner.run_real",
                    "message_count": len(messages),
                    "depth": task.depth,
                },
            )
            self.store.update_status(task_id, "completed")
            self.store.add_event(
                task_id=task_id,
                type="completed",
                stage="real_complete",
                message="Real PaperPilot execution completed.",
                payload={
                    "execution_mode": "real",
                    "artifact_kind": "result",
                    "message_count": len(messages),
                },
            )
        except Exception as exc:
            self.store.update_status(task_id, "failed")
            self.store.add_event(
                task_id=task_id,
                type="failed",
                stage="failure",
                message=f"Real PaperPilot execution failed: {exc}",
                payload={"execution_mode": "real"},
            )

    def _sleep(self) -> None:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

    def _record_agent_event(
        self,
        task_id: str,
        kind: str,
        payload: dict | None,
    ) -> None:
        event = map_paperpilot_event(kind, payload)
        self.store.add_event(
            task_id=task_id,
            type=str(event["type"]),
            stage=str(event["stage"]),
            message=str(event["message"]),
            payload=dict(event["payload"]),
        )


def _default_real_runner(query: str, *, on_event=None) -> list[dict]:
    from paperpilot.conversation import run

    return run(query, on_event=on_event)


def _call_real_runner(
    runner: RealRunner,
    query: str,
    *,
    on_event: Callable[[str, dict], None],
) -> list[dict]:
    if _accepts_on_event(runner):
        return runner(query, on_event=on_event)
    return runner(query)


def _accepts_on_event(runner: RealRunner) -> bool:
    try:
        params = signature(runner).parameters
    except (TypeError, ValueError):
        return False
    if "on_event" in params:
        return True
    return any(param.kind == Parameter.VAR_KEYWORD for param in params.values())


def _extract_final_text(messages: list[dict]) -> str:
    if not messages:
        raise ValueError("real runner returned no messages")
    content = messages[-1].get("content")
    if isinstance(content, str):
        text = content.strip()
        if text:
            return text
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")).strip())
            else:
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(str(text).strip())
        joined = "\n\n".join(part for part in parts if part)
        if joined:
            return joined
    raise ValueError("real runner returned no final text")
