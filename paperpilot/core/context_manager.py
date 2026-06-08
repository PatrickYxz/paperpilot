"""Context-window preflight checks for agent LLM calls."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from paperpilot.core.adapter import Tool


DEFAULT_CONTEXT_WINDOW_TOKENS = 120_000
DEFAULT_EXPECTED_OUTPUT_TOKENS = 4_096


@dataclass(frozen=True)
class ContextState:
    estimated_tokens: int
    window_tokens: int
    soft_limit: int
    hard_limit: int
    critical_limit: int
    expected_output_tokens: int

    @property
    def over_soft_limit(self) -> bool:
        return self.estimated_tokens >= self.soft_limit

    @property
    def over_hard_limit(self) -> bool:
        return self.estimated_tokens >= self.hard_limit

    @property
    def over_critical_limit(self) -> bool:
        return self.estimated_tokens >= self.critical_limit

    @property
    def needs_compact(self) -> bool:
        return self.over_soft_limit


@dataclass(frozen=True)
class ContextManager:
    window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS
    soft_ratio: float = 0.70
    hard_ratio: float = 0.85
    critical_ratio: float = 0.95

    @classmethod
    def from_env(cls) -> "ContextManager":
        return cls(
            window_tokens=_env_int(
                "CONTEXT_WINDOW_TOKENS",
                DEFAULT_CONTEXT_WINDOW_TOKENS,
            ),
            expected_output_tokens=_env_int(
                "EXPECTED_OUTPUT_TOKENS",
                DEFAULT_EXPECTED_OUTPUT_TOKENS,
            ),
            soft_ratio=_env_float("CONTEXT_SOFT_RATIO", 0.70),
            hard_ratio=_env_float("CONTEXT_HARD_RATIO", 0.85),
            critical_ratio=_env_float("CONTEXT_CRITICAL_RATIO", 0.95),
        )

    def inspect(
        self,
        *,
        system: str,
        tools: list[Tool],
        messages: list[dict],
    ) -> ContextState:
        chars = len(system)
        chars += sum(_message_chars(message) for message in messages)
        chars += sum(_tool_chars(tool) for tool in tools)
        estimated = _chars_to_tokens(chars) + self.expected_output_tokens
        return ContextState(
            estimated_tokens=estimated,
            window_tokens=self.window_tokens,
            soft_limit=int(self.window_tokens * self.soft_ratio),
            hard_limit=int(self.window_tokens * self.hard_ratio),
            critical_limit=int(self.window_tokens * self.critical_ratio),
            expected_output_tokens=self.expected_output_tokens,
        )


def _message_chars(message: dict) -> int:
    return len(str(message.get("role", ""))) + _content_chars(message.get("content"))


def _content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_chars(block) for block in content)
    if isinstance(content, dict):
        return sum(len(str(key)) + _content_chars(value) for key, value in content.items())
    return len(str(content))


def _tool_chars(tool: Tool) -> int:
    return (
        len(tool.name)
        + len(tool.description)
        + len(str(tool.input_schema))
    )


def _chars_to_tokens(chars: int) -> int:
    return math.ceil(chars / 4)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
