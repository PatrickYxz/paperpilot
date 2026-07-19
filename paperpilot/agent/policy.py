"""Immutable execution policy for a durable agent run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_DEPTH_BUDGETS = {
    "quick": (4, 20_000, False),
    "standard": (8, 50_000, True),
    "deep": (12, 120_000, True),
}


@dataclass(frozen=True)
class RunPolicy:
    max_iterations: int
    token_budget: int
    max_retries: int = 2
    retry_delays: tuple[float, ...] = (2.0, 8.0)
    allowed_builtin_tools: tuple[str, ...] = (
        "load_skill",
        "research_todo",
        "paper_deep_read",
        "compact_context",
        "search_user_document",
    )
    allowed_mcp_servers: tuple[str, ...] = ("arxiv", "colbert", "graph", "vlm")
    allow_subagents: bool = True
    lease_seconds: int = 300
    lease_renewal_seconds: int = 60
    llm_timeout_seconds: int = 180
    tool_timeout_seconds: int = 180

    def __post_init__(self) -> None:
        positive_fields = (
            "max_iterations",
            "token_budget",
            "max_retries",
            "lease_seconds",
            "lease_renewal_seconds",
            "llm_timeout_seconds",
            "tool_timeout_seconds",
        )
        for field_name in positive_fields:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not self.retry_delays:
            raise ValueError("retry_delays must not be empty")
        if any(delay <= 0 for delay in self.retry_delays):
            raise ValueError("retry_delays must contain only positive values")
        if self.lease_renewal_seconds > self.lease_seconds:
            raise ValueError("lease_renewal_seconds must not exceed lease_seconds")

    @classmethod
    def for_depth(cls, depth: str) -> "RunPolicy":
        try:
            max_iterations, token_budget, allow_subagents = _DEPTH_BUDGETS[depth]
        except KeyError as exc:
            raise ValueError(f"Unknown run depth: {depth}") from exc
        return cls(
            max_iterations=max_iterations,
            token_budget=token_budget,
            allow_subagents=allow_subagents,
        )

    def retry_delay_seconds(self, retry_number: int, jitter: float) -> float:
        if retry_number <= 0:
            raise ValueError("retry_number must be positive")
        if jitter < 0:
            raise ValueError("jitter must not be negative")
        delay = self.retry_delays[min(retry_number - 1, len(self.retry_delays) - 1)]
        return min(delay + jitter, self.retry_delays[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "token_budget": self.token_budget,
            "max_retries": self.max_retries,
            "retry_delays": list(self.retry_delays),
            "allowed_builtin_tools": list(self.allowed_builtin_tools),
            "allowed_mcp_servers": list(self.allowed_mcp_servers),
            "allow_subagents": self.allow_subagents,
            "lease_seconds": self.lease_seconds,
            "lease_renewal_seconds": self.lease_renewal_seconds,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "tool_timeout_seconds": self.tool_timeout_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunPolicy":
        return cls(
            max_iterations=payload["max_iterations"],
            token_budget=payload["token_budget"],
            max_retries=payload.get("max_retries", 2),
            retry_delays=tuple(payload.get("retry_delays", (2.0, 8.0))),
            allowed_builtin_tools=tuple(
                payload.get(
                    "allowed_builtin_tools",
                    cls.__dataclass_fields__["allowed_builtin_tools"].default,
                )
            ),
            allowed_mcp_servers=tuple(
                payload.get(
                    "allowed_mcp_servers",
                    cls.__dataclass_fields__["allowed_mcp_servers"].default,
                )
            ),
            allow_subagents=payload.get("allow_subagents", True),
            lease_seconds=payload.get("lease_seconds", 300),
            lease_renewal_seconds=payload.get("lease_renewal_seconds", 60),
            llm_timeout_seconds=payload.get("llm_timeout_seconds", 180),
            tool_timeout_seconds=payload.get("tool_timeout_seconds", 180),
        )
