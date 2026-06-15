"""Tests for paperpilot.eval.jsonl_tracer."""
import json
from pathlib import Path

from paperpilot.eval.jsonl_tracer import make_jsonl_tracer


def test_tracer_writes_two_events(tmp_path: Path) -> None:
    tracer = make_jsonl_tracer("case-001", tmp_path)
    tracer("tool_call", {"name": "load_skill", "arguments": {"name": "deep-read-paper"}})
    tracer("tool_result", {"name": "load_skill", "content": "skill body"})

    out = tmp_path / "case-001.jsonl"
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    assert e1["kind"] == "tool_call"
    assert e1["payload"]["name"] == "load_skill"
    e2 = json.loads(lines[1])
    assert e2["kind"] == "tool_result"


def test_tracer_creates_nested_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    tracer = make_jsonl_tracer("x", nested)
    tracer("guardrail_stop", {"reason": "max_iter"})
    assert (nested / "x.jsonl").exists()


def test_tracer_handles_non_json_payload(tmp_path: Path) -> None:
    """Anthropic SDK content blocks may not be directly JSON serializable."""

    class Block:
        text = "hello world"

    tracer = make_jsonl_tracer("y", tmp_path)
    tracer("tool_result", {"name": "x", "content": [Block()]})
    out = tmp_path / "y.jsonl"
    line = json.loads(out.read_text(encoding="utf-8").strip())
    assert "hello world" in str(line["payload"])


def test_tracer_preserves_long_tool_result_content(tmp_path: Path) -> None:
    long_content = "x" * 1200
    tracer = make_jsonl_tracer("long", tmp_path)
    tracer("tool_result", {"name": "mcp__colbert__search", "content": long_content})

    out = tmp_path / "long.jsonl"
    line = json.loads(out.read_text(encoding="utf-8").strip())
    assert line["payload"]["content"] == long_content
