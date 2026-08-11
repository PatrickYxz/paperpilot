"""Task executor tests."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest

from paperpilot.web import task_executor as task_executor_module
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.task_executor import (
    CeleryTaskExecutor,
    SynchronousTaskExecutor,
    TaskExecutor,
    TaskExecutorAtCapacityError,
    TaskExecutorShuttingDownError,
    TaskSubmissionReservation,
    build_task_executor,
)
class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.exhaustions: list[dict[str, object]] = []

    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        self.calls.append((task_id, allow_running))
        return True

    def fail_retry_exhausted(self, task_id: str, **kwargs) -> None:
        self.exhaustions.append({"task_id": task_id, **kwargs})


class FakeTaskSender:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def send_task(self, name: str, *, args: list[str], task_id: str):
        call = {"name": name, "args": args, "task_id": task_id}
        self.calls.append(call)
        return call


class BlockingRunner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        del task_id, allow_running
        self.started.set()
        assert self.release.wait(timeout=2)
        return True

    def fail_retry_exhausted(self, *_args, **_kwargs) -> None:
        raise AssertionError("blocking runner must not exhaust retries")


def _reserve_eventually(executor):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            return executor.reserve()
        except TaskExecutorAtCapacityError:
            time.sleep(0.01)
    raise AssertionError("executor capacity was not restored")


def test_mode_free_reservation_submits_only_task_id() -> None:
    submissions: list[str] = []
    reservation = TaskSubmissionReservation(
        lambda task_id: submissions.append(task_id)
    )

    reservation.submit("task_1")

    assert submissions == ["task_1"]
    with pytest.raises(TypeError):
        TaskSubmissionReservation(lambda task_id: task_id).submit(
            "task_2",
            "real",
        )


def test_mode_free_thread_executor_submits_only_task_id() -> None:
    calls: list[tuple[str, bool]] = []

    class Runner:
        def run(self, task_id: str, *, allow_running: bool = False) -> bool:
            calls.append((task_id, allow_running))
            return True

        def fail_retry_exhausted(self, *_args, **_kwargs) -> None:
            raise AssertionError("successful execution must not be failed")

    executor = TaskExecutor(Runner(), max_workers=1)
    try:
        assert executor.submit("task_2").result(timeout=1) is None
        assert calls == [("task_2", False)]
        with pytest.raises(TypeError):
            executor.submit("task_3", "real")
    finally:
        executor.shutdown()


def test_synchronous_executor_runs_conversation_task():
    runner = FakeRunner()
    executor = SynchronousTaskExecutor(runner)

    future = executor.submit("task_sync")

    assert future.result(timeout=1) is None
    assert runner.calls == [("task_sync", False)]


def test_threaded_executor_submits_work_to_runner():
    runner = FakeRunner()
    executor = TaskExecutor(runner, max_workers=2)

    future = executor.submit("task_1")

    assert future.result(timeout=1) is None
    assert runner.calls == [("task_1", False)]
    assert executor.max_workers == 2
    executor.shutdown()


def test_thread_retries_recover_running_task_and_backoff():
    class FlakyRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()

        def run(self, task_id: str, *, allow_running: bool = False) -> bool:
            super().run(task_id, allow_running=allow_running)
            if len(self.calls) <= 3:
                raise ConnectionError("temporary checkpoint outage")
            return True

    runner = FlakyRunner()
    sleeps: list[float] = []
    executor = TaskExecutor(
        runner,
        max_workers=1,
        max_retries=3,
        sleeper=sleeps.append,
    )
    try:
        future = executor.submit("task_retry")
        assert future.result(timeout=1) is None
    finally:
        executor.shutdown()

    assert runner.calls == [
        ("task_retry", False),
        ("task_retry", True),
        ("task_retry", True),
        ("task_retry", True),
    ]
    assert sleeps == [1, 2, 4]


def test_retry_countdown_uses_exponential_backoff_with_cap():
    assert [
        task_executor_module.task_retry_countdown_seconds(
            retry_index,
            initial_seconds=3,
            max_seconds=10,
        )
        for retry_index in range(5)
    ] == [3, 6, 10, 10, 10]


@pytest.mark.parametrize("failures_before_success", [1, 2])
def test_thread_executor_retries_real_work_until_second_or_third_attempt(
    failures_before_success,
):
    class FlakyRunner(FakeRunner):
        def run(self, task_id: str, *, allow_running: bool = False) -> bool:
            super().run(task_id, allow_running=allow_running)
            if len(self.calls) <= failures_before_success:
                raise ConnectionError("temporary checkpoint outage")
            return True

    runner = FlakyRunner()
    sleeps: list[float] = []
    executor = TaskExecutor(
        runner,
        max_workers=1,
        max_retries=3,
        retry_backoff_seconds=2,
        retry_backoff_max_seconds=30,
        sleeper=sleeps.append,
    )
    try:
        future = executor.submit("task_retry")

        assert future.result(timeout=1) is None
        assert runner.calls == [("task_retry", False)] + [
            ("task_retry", True)
        ] * failures_before_success
        assert sleeps == [2, 4][:failures_before_success]
    finally:
        executor.shutdown()


def test_thread_executor_exhaustion_finalizes_once_without_leaking():
    class FailingRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__()

        def run(self, task_id: str, *, allow_running: bool = False) -> bool:
            super().run(task_id, allow_running=allow_running)
            raise ConnectionError("secret-token paper-content")

    runner = FailingRunner()
    sleeps: list[float] = []
    executor = TaskExecutor(runner, max_workers=1, sleeper=sleeps.append)
    try:
        future = executor.submit("task_exhausted")

        with pytest.raises(ConnectionError, match="secret-token paper-content"):
            future.result(timeout=1)
        recovered = _reserve_eventually(executor)
        recovered.release()
    finally:
        executor.shutdown()

    assert runner.calls == [
        ("task_exhausted", False),
        ("task_exhausted", True),
        ("task_exhausted", True),
        ("task_exhausted", True),
    ]
    assert sleeps == [1, 2, 4]
    assert len(runner.exhaustions) == 1
    exhaustion = runner.exhaustions[0]
    assert exhaustion["task_id"] == "task_exhausted"
    assert exhaustion["backend"] == "thread"
    assert exhaustion["attempts"] == 4
    assert exhaustion["max_retries"] == 3
    assert isinstance(exhaustion["exc"], ConnectionError)


def test_thread_executor_bounds_running_plus_queued_work():
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=2)
    try:
        reservations = [executor.reserve() for _ in range(3)]
        futures = [
            reservation.submit(f"task_{index}")
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
    finally:
        runner.release.set()
        executor.shutdown()


def test_unsubmitted_reservation_release_restores_capacity_once():
    executor = TaskExecutor(FakeRunner(), max_workers=1, queue_capacity=0)
    reservation = executor.reserve()
    reservation.release()
    reservation.release()

    replacement = executor.reserve()
    replacement.release()
    executor.shutdown()


def test_concurrent_reservation_release_restores_capacity_once():
    release_count = 0
    release_count_lock = threading.Lock()

    def release_once():
        nonlocal release_count
        with release_count_lock:
            release_count += 1

    reservation = TaskSubmissionReservation(lambda *_: None, release_once)
    release_threads = [
        threading.Thread(target=reservation.release) for _ in range(8)
    ]

    for thread in release_threads:
        thread.start()
    for thread in release_threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in release_threads)
    assert release_count == 1


def test_submitted_reservation_release_does_not_restore_capacity_early():
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=0)
    try:
        reservation = executor.reserve()
        future = reservation.submit("task_running")
        assert runner.started.wait(timeout=1)

        reservation.release()
        with pytest.raises(TaskExecutorAtCapacityError):
            executor.reserve()

        runner.release.set()
        future.result(timeout=2)
        replacement = _reserve_eventually(executor)
        replacement.release()
    finally:
        runner.release.set()
        executor.shutdown()


def test_failed_future_releases_capacity():
    class FailingRunner(FakeRunner):
        def run(self, task_id: str, *, allow_running: bool = False) -> bool:
            del task_id, allow_running
            raise RuntimeError("workflow failed")

    executor = TaskExecutor(
        FailingRunner(),
        max_workers=1,
        queue_capacity=0,
        max_retries=0,
    )
    future = executor.submit("task_failed")

    with pytest.raises(RuntimeError, match="workflow failed"):
        future.result(timeout=1)
    replacement = _reserve_eventually(executor)
    replacement.release()
    executor.shutdown()


def test_cancelled_queued_future_releases_its_queue_slot():
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=1)
    try:
        running = executor.submit("task_running")
        assert runner.started.wait(timeout=1)
        queued = executor.submit("task_queued")

        assert queued.cancel() is True
        replacement = _reserve_eventually(executor)
        replacement.release()
        runner.release.set()
        running.result(timeout=2)
    finally:
        runner.release.set()
        executor.shutdown()


def test_reservation_submit_failure_releases_exactly_once():
    release_count = 0

    def fail(task_id):
        del task_id
        raise ConnectionError("submit failed")

    def release_once():
        nonlocal release_count
        release_count += 1

    reservation = TaskSubmissionReservation(fail, release_once)
    with pytest.raises(ConnectionError, match="submit failed"):
        reservation.submit("task_1")
    reservation.release()

    assert release_count == 1
    with pytest.raises(RuntimeError, match="already used"):
        reservation.submit("task_1")


def test_capacity_reservation_rejects_non_future_and_releases_once():
    release_count = 0

    def release_once():
        nonlocal release_count
        release_count += 1

    reservation = TaskSubmissionReservation(lambda *_: object(), release_once)

    with pytest.raises(
        TypeError,
        match="capacity-tracked task submitter must return a Future",
    ):
        reservation.submit("task_1")
    reservation.release()

    assert release_count == 1


def test_callback_registration_failure_releases_once():
    class RegistrationFailureFuture(Future[None]):
        def add_done_callback(self, fn):
            raise RuntimeError("callback registration failed")

    release_count = 0

    def release_once():
        nonlocal release_count
        release_count += 1

    reservation = TaskSubmissionReservation(
        lambda *_: RegistrationFailureFuture(),
        release_once,
    )

    with pytest.raises(RuntimeError, match="callback registration failed"):
        reservation.submit("task_1")
    reservation.release()

    assert release_count == 1


def test_callback_invocation_then_registration_failure_does_not_release_twice():
    class CallbackThenFailureFuture(Future[None]):
        def add_done_callback(self, fn):
            fn(self)
            raise RuntimeError("callback failed after invocation")

    release_count = 0

    def release_once():
        nonlocal release_count
        release_count += 1

    reservation = TaskSubmissionReservation(
        lambda *_: CallbackThenFailureFuture(),
        release_once,
    )

    with pytest.raises(RuntimeError, match="callback failed after invocation"):
        reservation.submit("task_1")
    reservation.release()

    assert release_count == 1


def test_already_completed_future_releases_deterministically():
    future: Future[None] = Future()
    future.set_result(None)
    release_count = 0

    def release_once():
        nonlocal release_count
        release_count += 1

    reservation = TaskSubmissionReservation(lambda *_: future, release_once)

    result = reservation.submit("task_1")

    assert result is future
    assert release_count == 1
    reservation.release()
    assert release_count == 1


def test_release_during_submit_transition_does_not_release_early():
    future: Future[None] = Future()
    submitter_started = threading.Event()
    allow_submitter_return = threading.Event()
    release_finished = threading.Event()
    capacity_released = threading.Event()
    release_count = 0
    submission_results: list[object] = []
    submission_errors: list[BaseException] = []

    def submitter(task_id):
        del task_id
        submitter_started.set()
        assert allow_submitter_return.wait(timeout=1)
        return future

    def release_once():
        nonlocal release_count
        release_count += 1
        capacity_released.set()

    reservation = TaskSubmissionReservation(submitter, release_once)

    def submit_reservation():
        try:
            submission_results.append(reservation.submit("task_1"))
        except BaseException as exc:
            submission_errors.append(exc)

    def release_reservation():
        reservation.release()
        release_finished.set()

    submit_thread = threading.Thread(target=submit_reservation)
    release_thread = threading.Thread(target=release_reservation)
    submit_thread.start()
    try:
        assert submitter_started.wait(timeout=1)
        release_thread.start()
        assert release_finished.wait(timeout=1)
        assert release_count == 0

        allow_submitter_return.set()
        submit_thread.join(timeout=1)
        assert not submit_thread.is_alive()
        assert submission_errors == []
        assert submission_results == [future]
        assert release_count == 0

        future.set_result(None)
        assert capacity_released.wait(timeout=1)
        assert release_count == 1
    finally:
        allow_submitter_return.set()
        if not future.done():
            future.set_result(None)
        submit_thread.join(timeout=1)
        release_thread.join(timeout=1)


def test_submit_accepted_before_shutdown_completes_and_releases():
    runner = BlockingRunner()
    executor = TaskExecutor(runner, max_workers=1, queue_capacity=0)
    shutdown_started = threading.Event()
    shutdown_finished = threading.Event()
    shutdown_thread = None

    def shutdown_executor():
        shutdown_started.set()
        executor.shutdown()
        shutdown_finished.set()

    try:
        future = executor.submit("task_running")
        assert runner.started.wait(timeout=1)
        shutdown_thread = threading.Thread(target=shutdown_executor)
        shutdown_thread.start()
        assert shutdown_started.wait(timeout=1)
        assert not shutdown_finished.wait(timeout=0.05)
    finally:
        runner.release.set()
        if shutdown_thread is not None:
            shutdown_thread.join(timeout=1)
        executor.shutdown()

    assert shutdown_thread is not None
    assert not shutdown_thread.is_alive()
    assert shutdown_finished.is_set()
    assert future.result(timeout=1) is None


def test_shutdown_before_reserved_submit_releases_submit_failure_once():
    executor = TaskExecutor(FakeRunner(), max_workers=1, queue_capacity=0)
    reservation = executor.reserve()
    executor.shutdown()

    with pytest.raises(RuntimeError, match="cannot schedule new futures"):
        reservation.submit("task_1")

    reservation.release()


def test_reserve_before_shutdown_leaves_reservation_releasable_once():
    executor = TaskExecutor(FakeRunner(), max_workers=1, queue_capacity=0)
    reservation = executor.reserve()

    executor.shutdown()
    reservation.release()
    reservation.release()

    assert executor.is_shutdown is True


def test_shutdown_rejects_new_reservations():
    executor = TaskExecutor(FakeRunner(), max_workers=1, queue_capacity=0)
    executor.shutdown()

    assert executor.is_shutdown is True
    with pytest.raises(TaskExecutorShuttingDownError):
        executor.reserve()


def test_celery_executor_sends_json_safe_task_reference():
    sender = FakeTaskSender()
    executor = CeleryTaskExecutor(sender)

    result = executor.submit("task_1")

    assert sender.calls == [
        {
            "name": "paperpilot.web.execute_research_task",
            "args": ["task_1"],
            "task_id": "task_1",
        }
    ]
    assert result == sender.calls[0]
    assert executor.shutdown() is None


def test_synchronous_and_celery_reservations_reject_after_shutdown():
    sync = SynchronousTaskExecutor(FakeRunner())
    celery = CeleryTaskExecutor(FakeTaskSender())
    sync.shutdown()
    celery.shutdown()

    for executor in (sync, celery):
        assert executor.is_shutdown is True
        with pytest.raises(TaskExecutorShuttingDownError):
            executor.reserve()


def test_build_task_executor_selects_celery_from_environment(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_TASK_EXECUTOR", "celery")

    executor = build_task_executor(FakeRunner())

    assert isinstance(executor, CeleryTaskExecutor)


def test_build_task_executor_keeps_thread_as_default(monkeypatch):
    monkeypatch.delenv("PAPERPILOT_TASK_EXECUTOR", raising=False)

    executor = build_task_executor(FakeRunner())

    assert isinstance(executor, TaskExecutor)
    executor.shutdown()


def test_build_task_executor_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("PAPERPILOT_TASK_EXECUTOR", "unknown")

    with pytest.raises(ValueError, match="PAPERPILOT_TASK_EXECUTOR"):
        build_task_executor(FakeRunner())


def test_build_task_executor_uses_validated_thread_capacity():
    config = WebRuntimeConfig(
        task_executor="thread",
        thread_workers=3,
        thread_queue_capacity=7,
        task_max_retries=5,
        task_retry_backoff_seconds=3,
        task_retry_backoff_max_seconds=12,
    )

    executor = build_task_executor(FakeRunner(), config=config)

    assert isinstance(executor, TaskExecutor)
    assert executor.max_workers == 3
    assert executor.queue_capacity == 7
    assert executor.max_retries == 5
    assert executor.retry_backoff_seconds == 3
    assert executor.retry_backoff_max_seconds == 12
    executor.shutdown()


def test_build_task_executor_uses_config_instead_of_rereading_environment(
    monkeypatch,
):
    monkeypatch.setenv("PAPERPILOT_TASK_EXECUTOR", "celery")
    config = WebRuntimeConfig(task_executor="thread")

    executor = build_task_executor(FakeRunner(), config=config)

    assert isinstance(executor, TaskExecutor)
    executor.shutdown()
