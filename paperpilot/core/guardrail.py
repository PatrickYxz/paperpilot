"""Loop 防护栏:防止死循环 / token 超支 / 同一 tool 反复调用。

只做边界,不做决策——LLM 选什么 tool、怎么选,这里不干预。
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field

from paperpilot.core.adapter import ToolCall


class GuardrailStop(Exception):
    """由 guardrail 抛出,通知 loop 停止。"""


@dataclass
class Guardrail:
    max_iterations: int = 8
    budget_tokens: int = 50_000
    max_same_call: int = 3  # 连续 N 次完全相同调用 → 停
    turn_count: int = 0
    tokens_used: int = 0
    _recent: deque = field(default_factory=deque)
    _stop_reason: str | None = None

    def __post_init__(self):
        self._recent = deque(maxlen=self.max_same_call)

    def record_turn(self, usage: dict | None) -> None:
        self.turn_count += 1
        if usage:
            self.tokens_used += usage.get("total_tokens", 0)

    def should_stop(self) -> bool:
        if self.turn_count >= self.max_iterations:
            self._stop_reason = f"max_iterations reached ({self.max_iterations})"
            return True
        if self.tokens_used >= self.budget_tokens:
            self._stop_reason = (
                f"token budget exceeded ({self.tokens_used}/{self.budget_tokens})"
            )
            return True
        return False

    def check_repeated_call(self, tc: ToolCall) -> None:
        key = (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))
        self._recent.append(key)
        if (
            len(self._recent) == self._recent.maxlen
            and len(set(self._recent)) == 1
        ):
            raise GuardrailStop(
                f"repeated tool_call: {tc.name} x{self._recent.maxlen}"
            )

    def stop_reason(self) -> str | None:
        return self._stop_reason
