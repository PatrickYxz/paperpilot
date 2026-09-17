"""DeepSeek model-usage callback and graph propagation tests."""
from __future__ import annotations

import logging
from typing import Any, TypedDict
from uuid import uuid4

from langchain.messages import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langgraph.graph import END, START, StateGraph

import paperpilot.deep_reading.model_usage as model_usage_module
from paperpilot.deep_reading.model_usage import DeepSeekUsageCallback


def _finish_call(
    callback: DeepSeekUsageCallback,
    *,
    stage: str,
    prompt_version: str,
    usage: dict[str, int] | None,
    model: str = "deepseek-chat",
) -> None:
    run_id = uuid4()
    callback.on_chat_model_start(
        {"kwargs": {"model": model}},
        [[]],
        run_id=run_id,
        metadata={
            "paperpilot_stage": stage,
            "prompt_version": prompt_version,
            "ls_model_name": model,
        },
    )
    callback.on_llm_end(
        LLMResult(
            generations=[[]],
            llm_output={"model_name": model, "token_usage": usage or {}},
        ),
        run_id=run_id,
    )


def test_callback_aggregates_deepseek_cache_usage_by_stage() -> None:
    callback = DeepSeekUsageCallback()
    _finish_call(
        callback,
        stage="research",
        prompt_version="research-v2",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        },
    )
    _finish_call(
        callback,
        stage="research",
        prompt_version="research-v2",
        usage={
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "total_tokens": 60,
            "prompt_cache_hit_tokens": 30,
            "prompt_cache_miss_tokens": 20,
        },
    )

    summary = callback.stage_summaries()[0]
    assert summary.stage == "research"
    assert summary.prompt_version == "research-v2"
    assert summary.model == "deepseek-chat"
    assert summary.call_count == 2
    assert summary.observed_usage_call_count == 2
    assert summary.input_tokens == 150
    assert summary.output_tokens == 30
    assert summary.total_tokens == 180
    assert summary.cache_hit_tokens == 110
    assert summary.cache_miss_tokens == 40
    assert summary.cache_hit_ratio == 110 / 150
    assert summary.observed_cache_call_count == 2
    assert summary.missing_cache_call_count == 0


def test_callback_preserves_missing_cache_usage_as_none() -> None:
    callback = DeepSeekUsageCallback()
    _finish_call(
        callback,
        stage="summary",
        prompt_version="summary-v1",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )

    summary = callback.stage_summaries()[0]
    assert summary.cache_hit_tokens is None
    assert summary.cache_miss_tokens is None
    assert summary.cache_hit_ratio is None
    assert summary.observed_cache_call_count == 0
    assert summary.missing_cache_call_count == 1
    assert summary.to_event_payload()["name"].endswith(
        "cache metrics unavailable"
    )


def test_callback_separates_groups_and_ignores_unstaged_calls() -> None:
    callback = DeepSeekUsageCallback()
    _finish_call(
        callback,
        stage="summary",
        prompt_version="summary-v1",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    _finish_call(
        callback,
        stage="write_answer",
        prompt_version="answer-v1",
        usage={"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
    )
    ignored_run = uuid4()
    callback.on_chat_model_start({}, [[]], run_id=ignored_run, metadata={})
    callback.on_llm_end(
        LLMResult(generations=[[]], llm_output={"token_usage": {}}),
        run_id=ignored_run,
    )

    assert [(item.stage, item.prompt_version) for item in callback.stage_summaries()] == [
        ("summary", "summary-v1"),
        ("write_answer", "answer-v1"),
    ]


def test_callback_marks_incomplete_standard_usage_as_partial() -> None:
    callback = DeepSeekUsageCallback()
    _finish_call(
        callback,
        stage="research",
        prompt_version="research-v2",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 20,
        },
    )
    _finish_call(
        callback,
        stage="research",
        prompt_version="research-v2",
        usage={"prompt_tokens": 50},
    )

    summary = callback.stage_summaries()[0]
    assert summary.call_count == 2
    assert summary.observed_usage_call_count == 1
    assert summary.missing_usage_call_count == 1
    assert summary.input_tokens == 100
    assert summary.to_event_payload()["name"].endswith("partial")


def test_callback_discards_error_run_and_rejects_invalid_token_counts() -> None:
    callback = DeepSeekUsageCallback()
    failed_run = uuid4()
    callback.on_chat_model_start(
        {},
        [[]],
        run_id=failed_run,
        metadata={
            "paperpilot_stage": "summary",
            "prompt_version": "summary-v1",
        },
    )
    callback.on_llm_error(ConnectionError("provider failed"), run_id=failed_run)
    _finish_call(
        callback,
        stage="summary",
        prompt_version="summary-v1",
        usage={
            "prompt_tokens": -1,
            "completion_tokens": True,
            "total_tokens": 12,
            "prompt_cache_hit_tokens": -1,
            "prompt_cache_miss_tokens": False,
        },
    )

    summary = callback.stage_summaries()[0]
    assert summary.call_count == 1
    assert summary.observed_usage_call_count == 0
    assert summary.input_tokens is None
    assert summary.cache_hit_tokens is None


def test_event_payload_contains_only_aggregate_fields() -> None:
    callback = DeepSeekUsageCallback()
    _finish_call(
        callback,
        stage="write_answer",
        prompt_version="answer-v1",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )

    payload = callback.stage_summaries()[0].to_event_payload()
    assert set(payload) == {
        "name",
        "stage",
        "prompt_version",
        "model",
        "call_count",
        "observed_usage_call_count",
        "missing_usage_call_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
        "cache_hit_ratio",
        "observed_cache_call_count",
        "missing_cache_call_count",
    }
    rendered = str(payload).lower()
    assert "messages" not in rendered
    assert "prompt" not in rendered.replace("prompt_version", "")
    assert "content" not in rendered


def test_callback_parse_failure_is_best_effort(monkeypatch, caplog) -> None:
    callback = DeepSeekUsageCallback()
    run_id = uuid4()

    def fail_parse(_value: object) -> int | None:
        raise ValueError("parse failed")

    monkeypatch.setattr(model_usage_module, "_token_count", fail_parse)
    callback.on_chat_model_start(
        {},
        [[]],
        run_id=run_id,
        metadata={
            "paperpilot_stage": "summary",
            "prompt_version": "summary-v1",
        },
    )

    with caplog.at_level(logging.WARNING, logger="paperpilot.web.runtime"):
        callback.on_llm_end(
            LLMResult(
                generations=[[]],
                llm_output={"token_usage": {"prompt_tokens": 10}},
            ),
            run_id=run_id,
        )

    assert callback.stage_summaries() == ()
    assert any(
        getattr(record, "event", None) == "model.usage_callback_error"
        for record in caplog.records
    )


class _UsageState(TypedDict):
    done: bool


class _UsageChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "usage-test-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="answer"))],
            llm_output={
                "model_name": "deepseek-chat",
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_cache_hit_tokens": 8,
                    "prompt_cache_miss_tokens": 2,
                },
            },
        )


def test_graph_level_callback_reaches_node_internal_model_call() -> None:
    callback = DeepSeekUsageCallback()
    model = _UsageChatModel()

    def call_model(state: _UsageState) -> dict[str, bool]:
        del state
        model.invoke(
            [HumanMessage(content="question")],
            config={
                "tags": ["paperpilot:model"],
                "metadata": {
                    "paperpilot_stage": "research",
                    "prompt_version": "research-v2",
                },
            },
        )
        return {"done": True}

    builder = StateGraph(_UsageState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    graph = builder.compile()

    graph.invoke({"done": False}, config={"callbacks": [callback]})

    summary = callback.stage_summaries()[0]
    assert summary.stage == "research"
    assert summary.prompt_version == "research-v2"
    assert summary.input_tokens == 10
