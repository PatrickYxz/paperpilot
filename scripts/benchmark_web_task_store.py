"""Benchmark bounded Web task reads and concurrent SQLite writes."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.web.task_store import TaskStore


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[index], 6)


def _run(db_path: Path, args: argparse.Namespace) -> dict:
    store = TaskStore(db_path)
    user = store.create_user(
        username=f"benchmark-{uuid.uuid4().hex}",
        password_hash="benchmark",
        password_salt="benchmark",
    )

    started = time.perf_counter()
    for index in range(args.tasks):
        store.create_task(question=f"seed task {index}", user_id=user.id)
    seed_seconds = time.perf_counter() - started

    started = time.perf_counter()
    page = store.list_tasks_page(user_id=user.id, limit=args.page_size)
    page_ms = (time.perf_counter() - started) * 1000
    page_json = json.dumps(
        [task.to_dict() for task in page.items],
        ensure_ascii=False,
    ).encode()

    def create_one(index: int) -> float:
        write_started = time.perf_counter()
        store.create_task(question=f"concurrent task {index}", user_id=user.id)
        return (time.perf_counter() - write_started) * 1000

    latencies: list[float] = []
    errors: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(create_one, index) for index in range(args.writes)]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    concurrent_wall_seconds = time.perf_counter() - started

    with store._connect() as conn:
        indexes = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'index' ORDER BY name"
            ).fetchall()
        ]
        unfiltered_plan = [
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM research_tasks "
                "WHERE user_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (user.id, args.page_size + 1),
            ).fetchall()
        ]
        filtered_plan = [
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM research_tasks "
                "WHERE user_id = ? AND status = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (user.id, "pending", args.page_size + 1),
            ).fetchall()
        ]
        journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0])
        foreign_keys = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
        busy_timeout = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
        synchronous = int(conn.execute("PRAGMA synchronous").fetchone()[0])

    return {
        "seed_tasks": args.tasks,
        "seed_seconds": round(seed_seconds, 6),
        "page_rows": len(page.items),
        "page_has_more": page.has_more,
        "page_ms": round(page_ms, 6),
        "page_json_bytes": len(page_json),
        "concurrent_writes": len(latencies),
        "concurrent_errors": errors,
        "concurrent_wall_seconds": round(concurrent_wall_seconds, 6),
        "write_p50_ms": _percentile(latencies, 0.50) if latencies else None,
        "write_p95_ms": _percentile(latencies, 0.95) if latencies else None,
        "write_max_ms": round(max(latencies), 6) if latencies else None,
        "journal_mode": journal_mode,
        "busy_timeout_ms": busy_timeout,
        "foreign_keys": foreign_keys,
        "synchronous": synchronous,
        "indexes": indexes,
        "query_plans": {
            "unfiltered": unfiltered_plan,
            "pending": filtered_plan,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=2000)
    parser.add_argument("--writes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--db-path", type=Path)
    args = parser.parse_args()

    for option in ("tasks", "writes", "workers", "page_size"):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.page_size > 100:
        parser.error("--page-size must not exceed 100")

    if args.db_path is not None:
        result = _run(args.db_path, args)
    else:
        with TemporaryDirectory() as tmp:
            result = _run(Path(tmp) / "benchmark.sqlite3", args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
