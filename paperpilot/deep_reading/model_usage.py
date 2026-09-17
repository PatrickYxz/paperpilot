"""Best-effort DeepSeek model usage collection for one graph execution."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from threading import Lock
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


_LOGGER = logging.getLogger("paperpilot.web.runtime")


@dataclass(frozen=True)
class _RunMetadata:
    stage: str
    prompt_version: str
    model: str


@dataclass(frozen=True)
class _UsageObservation:
    stage: str
    prompt_version: str
    model: str
    run_id: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cache_hit_tokens: int | None
    cache_miss_tokens: int | None


@dataclass(frozen=True)
class ModelUsageSummary:
    stage: str
    prompt_version: str
    model: str
    call_count: int
    observed_usage_call_count: int
    missing_usage_call_count: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cache_hit_tokens: int | None
    cache_miss_tokens: int | None
    cache_hit_ratio: float | None
    observed_cache_call_count: int
    missing_cache_call_count: int

    def to_event_payload(self) -> dict[str, object]:
        call_label = "model call" if self.call_count == 1 else "model calls"
        parts = [f"{self.call_count} {call_label}"]
        if self.input_tokens is not None:
            parts.append(f"{self.input_tokens} input tokens")
        if self.cache_hit_ratio is None:
            parts.append("cache metrics unavailable")
        else:
            parts.append(f"{self.cache_hit_ratio:.1%} cache hit")
        cache_is_partial = (
            self.observed_cache_call_count > 0
            and self.missing_cache_call_count > 0
        )
        if self.missing_usage_call_count or cache_is_partial:
            parts.append("partial")
        name = " · ".join(parts)
        return {
            "name": name,
            "stage": self.stage,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "call_count": self.call_count,
            "observed_usage_call_count": self.observed_usage_call_count,
            "missing_usage_call_count": self.missing_usage_call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "cache_hit_ratio": self.cache_hit_ratio,
            "observed_cache_call_count": self.observed_cache_call_count,
            "missing_cache_call_count": self.missing_cache_call_count,
        }


def _token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _start_model_name(
    serialized: Mapping[str, object],
    metadata: Mapping[str, object],
    kwargs: Mapping[str, object],
) -> str:
    invocation_params = kwargs.get("invocation_params")
    invocation_map = (
        invocation_params if isinstance(invocation_params, Mapping) else {}
    )
    serialized_kwargs = serialized.get("kwargs")
    serialized_map = (
        serialized_kwargs if isinstance(serialized_kwargs, Mapping) else {}
    )
    for candidate in (
        metadata.get("ls_model_name"),
        invocation_map.get("model"),
        serialized_map.get("model"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown"


class DeepSeekUsageCallback(BaseCallbackHandler):
    """Collect tagged chat-model token usage without affecting model calls."""

    raise_error = False

    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: dict[UUID, _RunMetadata] = {}
        self._observations: list[_UsageObservation] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del messages, parent_run_id, tags
        try:
            values = metadata or {}
            stage = values.get("paperpilot_stage")
            if not isinstance(stage, str) or not stage:
                return
            prompt_version = values.get("prompt_version")
            if not isinstance(prompt_version, str) or not prompt_version:
                prompt_version = "unknown"
            model = _start_model_name(serialized, values, kwargs)
            with self._lock:
                self._runs[run_id] = _RunMetadata(stage, prompt_version, model)
        except Exception:
            _LOGGER.warning(
                "Failed to initialize model usage observation",
                exc_info=True,
                extra={"event": "model.usage_callback_error"},
            )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del parent_run_id, tags, kwargs
        with self._lock:
            run = self._runs.pop(run_id, None)
        if run is None:
            return
        try:
            output = response.llm_output
            output_map = output if isinstance(output, Mapping) else {}
            raw_usage = output_map.get("token_usage")
            usage = raw_usage if isinstance(raw_usage, Mapping) else {}
            output_model = output_map.get("model_name") or output_map.get("model")
            model = (
                output_model
                if isinstance(output_model, str) and output_model
                else run.model
            )
            observation = _UsageObservation(
                stage=run.stage,
                prompt_version=run.prompt_version,
                model=model,
                run_id=str(run_id),
                input_tokens=_token_count(usage.get("prompt_tokens")),
                output_tokens=_token_count(usage.get("completion_tokens")),
                total_tokens=_token_count(usage.get("total_tokens")),
                cache_hit_tokens=_token_count(
                    usage.get("prompt_cache_hit_tokens")
                ),
                cache_miss_tokens=_token_count(
                    usage.get("prompt_cache_miss_tokens")
                ),
            )
            with self._lock:
                self._observations.append(observation)
            _LOGGER.info(
                "Model usage observed",
                extra={
                    "event": "model.usage",
                    "stage": observation.stage,
                    "prompt_version": observation.prompt_version,
                    "model": observation.model,
                    "run_id": observation.run_id,
                    "input_tokens": observation.input_tokens,
                    "output_tokens": observation.output_tokens,
                    "total_tokens": observation.total_tokens,
                    "cache_hit_tokens": observation.cache_hit_tokens,
                    "cache_miss_tokens": observation.cache_miss_tokens,
                },
            )
        except Exception:
            _LOGGER.warning(
                "Failed to parse model usage",
                exc_info=True,
                extra={"event": "model.usage_callback_error"},
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        del error, parent_run_id, tags, kwargs
        with self._lock:
            self._runs.pop(run_id, None)

    def stage_summaries(self) -> tuple[ModelUsageSummary, ...]:
        with self._lock:
            observations = tuple(self._observations)
        grouped: dict[tuple[str, str, str], list[_UsageObservation]] = defaultdict(
            list
        )
        for item in observations:
            grouped[(item.stage, item.prompt_version, item.model)].append(item)

        summaries: list[ModelUsageSummary] = []
        for (stage, prompt_version, model), items in sorted(grouped.items()):
            complete_usage = [
                item
                for item in items
                if item.input_tokens is not None
                and item.output_tokens is not None
                and item.total_tokens is not None
            ]
            complete_cache = [
                item
                for item in items
                if item.cache_hit_tokens is not None
                and item.cache_miss_tokens is not None
            ]
            hit = (
                sum(cast(int, item.cache_hit_tokens) for item in complete_cache)
                if complete_cache
                else None
            )
            miss = (
                sum(cast(int, item.cache_miss_tokens) for item in complete_cache)
                if complete_cache
                else None
            )
            denominator = (
                hit + miss if hit is not None and miss is not None else 0
            )
            summaries.append(
                ModelUsageSummary(
                    stage=stage,
                    prompt_version=prompt_version,
                    model=model,
                    call_count=len(items),
                    observed_usage_call_count=len(complete_usage),
                    missing_usage_call_count=len(items) - len(complete_usage),
                    input_tokens=(
                        sum(cast(int, item.input_tokens) for item in complete_usage)
                        if complete_usage
                        else None
                    ),
                    output_tokens=(
                        sum(cast(int, item.output_tokens) for item in complete_usage)
                        if complete_usage
                        else None
                    ),
                    total_tokens=(
                        sum(cast(int, item.total_tokens) for item in complete_usage)
                        if complete_usage
                        else None
                    ),
                    cache_hit_tokens=hit,
                    cache_miss_tokens=miss,
                    cache_hit_ratio=(hit / denominator if denominator else None),
                    observed_cache_call_count=len(complete_cache),
                    missing_cache_call_count=len(items) - len(complete_cache),
                )
            )
        return tuple(summaries)
