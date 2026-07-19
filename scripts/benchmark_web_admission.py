"""Exercise bounded task admission without machine-specific thresholds."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from paperpilot.web.auth import SESSION_COOKIE_NAME
from paperpilot.web.config import WebRuntimeConfig
from paperpilot.web.task_executor import TaskExecutor
from paperpilot.web.task_store import TaskStore


class BlockingRunner:
    """Keep accepted tasks pending until the benchmark has counted database rows."""

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
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if queue_capacity < 0:
        raise ValueError("queue capacity must be at least 0")

    capacity = workers + queue_capacity
    if requests <= capacity:
        raise ValueError("requests must be greater than configured capacity")

    with tempfile.TemporaryDirectory(prefix="paperpilot-admission-") as directory:
        db_path = Path(directory) / "tasks.sqlite3"
        previous_db_path = os.environ.get("PAPERPILOT_TASK_DB_PATH")
        os.environ["PAPERPILOT_TASK_DB_PATH"] = str(db_path)
        try:
            from paperpilot.web.app import create_app
        finally:
            if previous_db_path is None:
                os.environ.pop("PAPERPILOT_TASK_DB_PATH", None)
            else:
                os.environ["PAPERPILOT_TASK_DB_PATH"] = previous_db_path

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
        latencies: list[float] = []
        statuses: list[int] = []
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
                        headers={"Cookie": f"{SESSION_COOKIE_NAME}={session_token}"},
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
                # This must run before TestClient closes and waits for executor shutdown.
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
    parser = argparse.ArgumentParser(
        description="Verify bounded Web task admission and capacity recovery."
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument("--requests", type=int, default=12)
    args = parser.parse_args()
    try:
        result = run_benchmark(
            workers=args.workers,
            queue_capacity=args.queue_capacity,
            requests=args.requests,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
