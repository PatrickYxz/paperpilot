"""Deterministic context token counting tests."""
from __future__ import annotations

from math import ceil

from langchain.messages import HumanMessage, ToolMessage

from paperpilot.deep_reading.context_management.budget import (
    ModelAwareTokenCounter,
    normalize_messages,
    reclaim_threshold,
    usable_input_budget,
)


def test_fallback_count_uses_conservative_utf8_and_character_bound():
    counter = ModelAwareTokenCounter()
    for value in ("", "abcdef", "中文", "abc中文"):
        expected = max(
            ceil(len(value.encode("utf-8")) / 3),
            ceil(len(value) / 2),
        )
        assert counter.count_text(value) == expected


def test_message_count_includes_roles_tools_results_wrappers_and_schemas():
    counter = ModelAwareTokenCounter()
    messages = [
        HumanMessage(content="find papers", name="researcher"),
        ToolMessage(
            content='{"id":"paper-1","title":"A"}',
            tool_call_id="call-1",
            name="search_related_papers",
        ),
    ]
    schemas = [{"name": "search_related_papers", "parameters": {"type": "object"}}]
    normalized = normalize_messages(messages, tool_schemas=schemas)
    assert '"role":"user"' in normalized
    assert '"role":"tool"' in normalized
    assert '"tool_call_id":"call-1"' in normalized
    assert '"tool_schemas"' in normalized
    assert counter.count_messages(messages, tool_schemas=schemas) == counter.count_text(
        normalized
    )


def test_model_counter_is_preferred_and_fallback_hides_counter_errors():
    class CountingModel:
        def __init__(self, result):
            self.result = result
            self.calls = 0

        def get_num_tokens_from_messages(self, messages):
            self.calls += 1
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    model = CountingModel(17)
    counter = ModelAwareTokenCounter(model=model)
    assert counter.count_messages([HumanMessage(content="ignored")]) == 17
    assert model.calls == 1

    broken = ModelAwareTokenCounter(model=CountingModel(RuntimeError("no tokenizer")))
    messages = [HumanMessage(content="fallback")]
    assert broken.count_messages(messages) == broken.count_text(normalize_messages(messages))


def test_model_counter_adds_tool_schema_tokens_when_provider_only_counts_messages():
    class CountingModel:
        def get_num_tokens_from_messages(self, messages):
            return 17

    counter = ModelAwareTokenCounter(model=CountingModel())
    short_schema = [{"name": "search", "parameters": {"type": "object"}}]
    long_schema = [
        {
            "name": "search",
            "description": "focused academic paper search with explicit scope",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]

    short_count = counter.count_messages(
        [HumanMessage(content="find papers")],
        tool_schemas=short_schema,
    )
    long_count = counter.count_messages(
        [HumanMessage(content="find papers")],
        tool_schemas=long_schema,
    )

    assert short_count > 17
    assert long_count > short_count


def test_model_counter_does_not_trust_silently_ignored_tools_argument():
    class SilentlyIgnoringModel:
        def __init__(self) -> None:
            self.seen_tools = "not-called"

        def get_num_tokens_from_messages(self, messages, tools=None):
            del messages
            self.seen_tools = tools
            return 17

    model = SilentlyIgnoringModel()
    counter = ModelAwareTokenCounter(model=model)
    schemas = [{"name": "search", "parameters": {"type": "object"}}]

    count = counter.count_messages(
        [HumanMessage(content="find papers")],
        tool_schemas=schemas,
    )

    assert count > 17
    assert model.seen_tools is None


def test_budget_and_reclaim_threshold_use_exact_boundaries():
    assert usable_input_budget(
        model_context_window_tokens=1_000,
        configured_max_output_tokens=100,
        context_safety_margin_ratio=0.05,
    ) == 850
    assert reclaim_threshold(850, minimum_tokens=8_000, minimum_ratio=0.10) == 8_000
    assert reclaim_threshold(100_000, minimum_tokens=8_000, minimum_ratio=0.10) == 10_000


def test_truncate_text_never_exceeds_requested_token_budget():
    counter = ModelAwareTokenCounter()
    text = "abc中文" * 100
    truncated = counter.truncate_text(text, 11)
    assert text.startswith(truncated)
    assert counter.count_text(truncated) <= 11
