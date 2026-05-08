"""Eval-only tracer that persists agent loop events to JSONL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def make_jsonl_tracer(run_id: str, out_dir: Path) -> Callable[[str, dict], None]:
    """Return a tracer that appends each ``(kind, payload)`` as one JSONL line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    out_path.write_text("", encoding="utf-8")

    def tracer(kind: str, payload: dict[str, Any]) -> None:
        record = {"kind": kind, "payload": _json_safe(payload)}
        line = json.dumps(record, default=str, ensure_ascii=False)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return tracer


def _json_safe(value: Any) -> Any:
    """Convert common SDK objects to JSON-friendly values while preserving text."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "text"):
        return getattr(value, "text")
    return str(value)
