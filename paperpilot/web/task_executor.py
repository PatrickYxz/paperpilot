"""Background execution boundary for Web research tasks."""
from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from typing import Callable, Literal, Protocol, cast

from paperpilot.web.celery_app import (
    EXECUTE_RESEARCH_TASK_NAME,
    celery_app,
)
from paperpilot.web.config import WebRuntimeConfig

ExecutionMode = Literal["simulated", "real"]
_LOGGER = logging.getLogger("paperpilot.web.runtime")


class WorkflowRunnerLike(Protocol):
    def run_simulated(self, task_id: str) -> None: ...

    def run_real(self, task_id: str) -> None: ...

    def claim_conversation_task(self, task_id: str) -> str: ...

    def fail_conversation_execution(
        self,
        task_id: str,
        *,
        backend: str,
        attempts: int,
        max_retries: int,
        exc: Exception,
    ) -> None: ...


class TaskExecutorAtCapacityError(RuntimeError):
    """Raised when all running and queued thread slots are reserved."""


class TaskExecutorShuttingDownError(RuntimeError):
    """Raised after executor shutdown begins."""


Submitter = Callable[[str, ExecutionMode], object]
ReleaseCallback = Callable[[], None]


class TaskSubmissionReservation:
    """One-shot ownership token for an admitted task submission."""

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

        if self._release_capacity is not None and not isinstance(result, Future):
            self._release_after_submit_failure()
            raise TypeError(
                "capacity-tracked task submitter must return a Future"
            )

        with self._lock:
            self._state = "submitted"
        if self._release_capacity is not None:
            future = cast(Future[None], result)
            try:
                future.add_done_callback(self._release_after_future)
            except BaseException:
                self._release_once("submitted")
                raise
        return result

    def release(self) -> None:
        self._release_once("reserved")

    def _release_after_submit_failure(self) -> None:
        self._release_once("submitting")

    def _release_after_future(self, future: Future[None]) -> None:
        try:
            if future.done() and not future.cancelled():
                exc = future.exception()
                if exc is not None:
                    _LOGGER.error(
                        "Background task execution failed",
                        exc_info=(type(exc), exc, exc.__traceback__),
                        extra={
                            "event": "task.execution_failed",
                            "executor": "thread",
                            "reason": "future_exception",
                            "exception_type": type(exc).__name__,
                        },
                    )
        except BaseException:
            _LOGGER.exception(
                "Background task Future inspection failed",
                extra={
                    "event": "task.future_inspection_failed",
                    "executor": "thread",
                    "reason": "callback_exception",
                },
            )
        finally:
            self._release_once("submitted")

    def _release_once(self, expected_state: str) -> None:
        callback = None
        with self._lock:
            if self._state == expected_state:
                self._state = "released"
                callback = self._release_capacity
        if callback is not None:
            callback()


class TaskExecutorLike(Protocol):
    @property
    def is_shutdown(self) -> bool: ...

    def reserve(self) -> TaskSubmissionReservation: ...

    def submit(self, task_id: str, execution_mode: ExecutionMode) -> object: ...

    def shutdown(self) -> None: ...


class TaskSenderLike(Protocol):
    def send_task(
        self,
        name: str,
        *,
        args: list[str],
        task_id: str,
    ) -> object: ...


class TaskExecutor:
    """Submit task workflow runs to a small thread pool."""

    def __init__(
        self,
        runner: WorkflowRunnerLike,
        *,
        max_workers: int = 2,
        queue_capacity: int = 4,
        max_retries: int = 3,
        retry_backoff_seconds: int = 1,
        retry_backoff_max_seconds: int = 30,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if queue_capacity < 0:
            raise ValueError("queue_capacity must be at least 0")
        if max_retries < 0:
            raise ValueError("max_retries must be at least 0")
        if retry_backoff_seconds < 1:
            raise ValueError("retry_backoff_seconds must be at least 1")
        if retry_backoff_max_seconds < 1:
            raise ValueError("retry_backoff_max_seconds must be at least 1")
        if retry_backoff_seconds > retry_backoff_max_seconds:
            raise ValueError(
                "retry_backoff_seconds must not exceed retry_backoff_max_seconds"
            )
        self.runner = runner
        self.max_workers = max_workers
        self.queue_capacity = queue_capacity
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_backoff_max_seconds = retry_backoff_max_seconds
        self._sleeper = sleeper
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

    def submit(self, task_id: str, execution_mode: ExecutionMode) -> Future[None]:
        return cast(Future[None], self.reserve().submit(task_id, execution_mode))

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

    def _run(self, task_id: str, execution_mode: ExecutionMode) -> None:
        if execution_mode == "real":
            claim = getattr(self.runner, "claim_conversation_task", None)
            if claim is not None:
                claim_result = claim(task_id)
                if claim_result == "not_claimed":
                    return
                if claim_result not in {"claimed", "not_conversation"}:
                    raise RuntimeError(
                        f"invalid Conversation claim result: {claim_result!r}"
                    )
            self._run_real_with_retries(task_id)
            return
        if execution_mode == "simulated":
            self.runner.run_simulated(task_id)
            return
        raise ValueError(f"invalid execution mode: {execution_mode!r}")

    def _run_real_with_retries(self, task_id: str) -> None:
        for attempt_index in range(self.max_retries + 1):
            try:
                self.runner.run_real(task_id)
                return
            except Exception as exc:
                if attempt_index < self.max_retries:
                    countdown = task_retry_countdown_seconds(
                        attempt_index,
                        initial_seconds=self.retry_backoff_seconds,
                        max_seconds=self.retry_backoff_max_seconds,
                    )
                    _LOGGER.warning(
                        "Conversation execution retry %d/%d in %d seconds",
                        attempt_index + 1,
                        self.max_retries,
                        countdown,
                        exc_info=True,
                        extra={
                            "event": "task.execution_retry",
                            "executor": "thread",
                            "reason": "workflow_exception",
                            "exception_type": type(exc).__name__,
                        },
                    )
                    self._sleeper(countdown)
                    continue

                try:
                    self.runner.fail_conversation_execution(
                        task_id,
                        backend="thread",
                        attempts=attempt_index + 1,
                        max_retries=self.max_retries,
                        exc=exc,
                    )
                except Exception:
                    _LOGGER.exception(
                        "Conversation retry exhaustion finalization failed",
                        extra={
                            "event": "task.execution_failure_finalize_failed",
                            "executor": "thread",
                            "reason": "store_exception",
                        },
                    )
                raise


class SynchronousTaskExecutor:
    """Executor used by tests to run submitted work deterministically."""

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

    def submit(self, task_id: str, execution_mode: ExecutionMode) -> Future[None]:
        return cast(Future[None], self.reserve().submit(task_id, execution_mode))

    def shutdown(self) -> None:
        with self._state_lock:
            self._shutdown = True

    def _submit_reserved(
        self,
        task_id: str,
        execution_mode: ExecutionMode,
    ) -> Future[None]:
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
    """Submit lightweight task references to a Celery broker."""

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
            max_retries=runtime.task_max_retries,
            retry_backoff_seconds=runtime.task_retry_backoff_seconds,
            retry_backoff_max_seconds=runtime.task_retry_backoff_max_seconds,
        )
    return CeleryTaskExecutor()


def task_retry_countdown_seconds(
    retry_index: int,
    *,
    initial_seconds: int,
    max_seconds: int,
) -> int:
    """Return the bounded delay before the zero-based retry attempt."""
    if retry_index < 0:
        raise ValueError("retry_index must be at least 0")
    return min(initial_seconds * (2**retry_index), max_seconds)
