"""Local request-copy editing for previously consumed tool results."""
from __future__ import annotations

from types import SimpleNamespace

from langchain.messages import HumanMessage, ToolMessage
from langchain.agents.middleware import ModelResponse
from langchain.messages import AIMessage

from paperpilot.deep_reading.context_management.editing import (
    EditingDisposition,
    LocalContextEditingAdapter,
    ResearchContextMiddleware,
)
from paperpilot.deep_reading.context_management.models import (
    ArtifactRef,
    FutureRetention,
    InitialAction,
    ToolResultDisposition,
)


class _Counter:
    def count_messages(self, messages, *, tool_schemas=()):
        del tool_schemas
        return sum(self.count_text(str(message.content)) for message in messages)

    def count_text(self, text):
        return len(text)

    def truncate_text(self, text, max_tokens):
        return text[:max_tokens]


def _disposition(index: int, *, protected=False, consumed=True, entered=True):
    ref = ArtifactRef(
        artifact_id=f"artifact-{index}",
        sha256=f"{index:064x}",
        token_estimate=20,
        preview="preview",
    )
    value = ToolResultDisposition(
        tool_call_id=f"call-{index}",
        content_sha256=f"{index + 10:064x}",
        initial_action=InitialAction.EXTERNALIZE_NOW,
        future_retention=(
            FutureRetention.PROTECTED
            if protected
            else FutureRetention.CLEARABLE_AFTER_USE
        ),
        preview="preview",
        artifact_ref=ref,
        result_id=f"result-{index}",
    )
    return EditingDisposition(
        message_index=index + 1,
        disposition=value,
        entered_successful_model_call=entered,
        downstream_consumed=consumed,
        recoverable=True,
    )


def test_micro_compaction_requires_all_eligibility_conditions_and_keeps_recent_three():
    messages = [HumanMessage(content="question")] + [
        ToolMessage(content=f"tool result {index}" * 10, tool_call_id=f"call-{index}")
        for index in range(4)
    ]
    dispositions = [_disposition(index) for index in range(4)]
    request = SimpleNamespace(messages=messages, input_tokens=600)
    adapter = LocalContextEditingAdapter(
        token_counter=_Counter(),
        usable_input_budget=100,
        trigger_ratio=0.70,
        min_reclaim_tokens=8,
        min_reclaim_ratio=0.10,
        keep_recent_tool_results=3,
    )

    edited = adapter.edit(request, dispositions)
    assert edited.input_tokens < request.input_tokens
    assert messages[1].content != edited.messages[1].content
    assert messages[2].content == edited.messages[2].content
    assert messages[3].content == edited.messages[3].content
    assert messages[4].content == edited.messages[4].content
    assert "artifact-0" in edited.messages[1].content
    assert request.messages == messages


def test_micro_compaction_is_batch_stable_and_does_not_edit_below_threshold():
    messages = [HumanMessage(content="question")] + [
        ToolMessage(content="tool result", tool_call_id=f"call-{index}")
        for index in range(4)
    ]
    dispositions = [_disposition(index) for index in range(4)]
    adapter = LocalContextEditingAdapter(
        token_counter=_Counter(),
        usable_input_budget=100,
        trigger_ratio=0.70,
        min_reclaim_tokens=8,
        min_reclaim_ratio=0.10,
        keep_recent_tool_results=3,
    )
    below = adapter.edit(SimpleNamespace(messages=messages, input_tokens=69), dispositions)
    assert tuple(below.messages) == tuple(messages)
    first = adapter.edit(SimpleNamespace(messages=messages, input_tokens=70), dispositions)
    second = adapter.edit(SimpleNamespace(messages=messages, input_tokens=70), dispositions)
    assert [message.content for message in first.messages] == [
        message.content for message in second.messages
    ]


def test_micro_compaction_skips_unconsumed_unrecoverable_and_protected_messages():
    messages = [HumanMessage(content="question")] + [
        ToolMessage(content=f"tool result {index}" * 10, tool_call_id=f"call-{index}")
        for index in range(4)
    ]
    dispositions = [
        _disposition(0),
        _disposition(1, protected=True),
        _disposition(2, consumed=False),
        _disposition(3, entered=False),
    ]
    adapter = LocalContextEditingAdapter(
        token_counter=_Counter(),
        usable_input_budget=100,
        trigger_ratio=0.70,
        min_reclaim_tokens=8,
        min_reclaim_ratio=0.10,
        keep_recent_tool_results=0,
    )
    edited = adapter.edit(SimpleNamespace(messages=messages, input_tokens=70), dispositions)
    assert "artifact-0" in edited.messages[1].content
    assert edited.messages[2].content == messages[2].content
    assert edited.messages[3].content == messages[3].content
    assert edited.messages[4].content == messages[4].content


def test_micro_compaction_emits_redacted_start_and_completed_events():
    events = []
    messages = [HumanMessage(content="question")] + [
        ToolMessage(content=f"tool result {index}" * 10, tool_call_id=f"call-{index}")
        for index in range(4)
    ]
    adapter = LocalContextEditingAdapter(
        token_counter=_Counter(),
        usable_input_budget=100,
        trigger_ratio=0.70,
        min_reclaim_tokens=8,
        min_reclaim_ratio=0.10,
        keep_recent_tool_results=3,
        event_sink=lambda kind, payload: events.append((kind, payload)),
    )

    edited = adapter.edit(
        SimpleNamespace(messages=messages, input_tokens=600),
        [_disposition(index) for index in range(4)],
    )

    assert edited.input_tokens < 600
    assert [kind for kind, _payload in events] == [
        "micro_compaction_started",
        "micro_compaction_completed",
    ]
    for _kind, payload in events:
        assert payload["stage"] == "micro"
        assert payload["compressor_version"] == "context-editing-v1"
        assert payload["protected_item_count"] == 0
        assert payload["artifact_ref_count"] == 1
        assert "messages" not in payload


def test_middleware_persists_real_tool_message_index_and_marks_only_consumed_results() -> None:
    present = _disposition(7, consumed=False, entered=False)
    present = EditingDisposition(
        message_index=-1,
        disposition=present.disposition,
        entered_successful_model_call=False,
        downstream_consumed=False,
        recoverable=True,
    )
    absent = _disposition(8, consumed=False, entered=False)
    absent = EditingDisposition(
        message_index=-1,
        disposition=absent.disposition,
        entered_successful_model_call=False,
        downstream_consumed=False,
        recoverable=True,
    )
    middleware = ResearchContextMiddleware()
    middleware._dispositions = [present, absent]
    messages = [
        HumanMessage(content="question"),
        ToolMessage(content="large result", tool_call_id="call-7"),
        HumanMessage(content="follow-up"),
    ]
    request = SimpleNamespace(messages=messages)

    indexed = middleware._indexed_dispositions(request)

    assert indexed[0].message_index == 1
    assert indexed[1].message_index == -1
    assert middleware.dispositions[0].message_index == 1
    middleware._mark_successful_input(messages)
    assert middleware.dispositions[0].entered_successful_model_call is True
    assert middleware.dispositions[0].downstream_consumed is True
    assert middleware.dispositions[1].entered_successful_model_call is False
    assert middleware.dispositions[1].downstream_consumed is False


def test_middleware_micro_compacts_result_on_model_call_after_it_was_consumed() -> None:
    disposition = _disposition(1, consumed=False, entered=False).disposition

    class Ingestor:
        def ingest(self, _request):
            return SimpleNamespace(
                disposition=disposition,
                model_content="large tool result " * 20,
            )

    class Request:
        def __init__(self, messages):
            self.messages = list(messages)
            self.tools = []

        def override(self, *, messages):
            return Request(messages)

    adapter = LocalContextEditingAdapter(
        token_counter=_Counter(),
        usable_input_budget=100,
        trigger_ratio=0.70,
        min_reclaim_tokens=8,
        min_reclaim_ratio=0.10,
        keep_recent_tool_results=0,
    )
    middleware = ResearchContextMiddleware(
        ingestor=Ingestor(),
        editing_adapter=adapter,
        conversation_id="conversation-1",
        task_id="task-1",
    )
    tool_result = middleware.wrap_tool_call(
        SimpleNamespace(tool_call={"name": "search_related_papers", "id": "call-1"}),
        lambda _request: ToolMessage(
            content='{"papers": []}',
            tool_call_id="call-1",
            name="search_related_papers",
        ),
    )
    request = Request([HumanMessage(content="question"), tool_result])
    seen_contents: list[str] = []

    def handler(model_request):
        seen_contents.append(str(model_request.messages[-1].content))
        return ModelResponse(result=[AIMessage(content="next")])

    middleware.wrap_model_call(request, handler)
    middleware.wrap_model_call(request, handler)

    assert "large tool result" in seen_contents[0]
    assert "<artifact_ref" in seen_contents[1]
    assert middleware.dispositions[0].message_index == 1
    assert middleware.dispositions[0].downstream_consumed is True
