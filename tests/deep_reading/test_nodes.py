"""State and node behavior for the deep-reading graph."""
from __future__ import annotations

import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Barrier
from types import SimpleNamespace
from typing import Any, Sequence, get_type_hints

import pytest
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.runtime import Runtime
from pydantic import PrivateAttr, ValidationError

import paperpilot.deep_reading.nodes as nodes_module
from paperpilot.deep_reading.nodes import (
    DeepReadingContext,
    initialize_turn,
    needs_summary,
    prepare_primary_paper,
    publish_result,
    research_evidence,
    summarize_history,
    write_answer,
)
from paperpilot.deep_reading.research_agent import ResearchContractError
from paperpilot.deep_reading.schemas import (
    AnswerCitation,
    AnswerDraft,
    ConversationSummary,
    EvidenceItem,
    PaperUse,
    ResearchResult,
)
from paperpilot.core.adapter import Tool
from paperpilot.papers import PaperCandidate
from paperpilot.tools.mcp_client import (
    MCPToolError,
    MCPToolTimeout,
    MCPTransportError,
)
from paperpilot.web.task_store import TaskStore, UsedPaperInput
from paperpilot.deep_reading.state import (
    GRAPH_VERSION,
    SCHEMA_VERSION,
    DeepReadingState,
)


SUMMARY = ConversationSummary(
    confirmed_facts=["Fact"],
    paper_findings=["Finding"],
    comparison_context=["Comparison"],
    open_questions=["Question"],
)


class _StructuredModel:
    def __init__(self, result: object) -> None:
        self.result = result
        self.invocations: list[object] = []

    def invoke(self, model_input: object) -> object:
        self.invocations.append(model_input)
        return self.result


class _FakeModel:
    def __init__(
        self,
        result: ConversationSummary | dict[str, object] = SUMMARY,
    ) -> None:
        self.structured = _StructuredModel(result)
        self.schemas: list[type[ConversationSummary]] = []

    def with_structured_output(
        self,
        schema: type[ConversationSummary],
        *,
        include_raw: bool = False,
    ) -> object:
        self.schemas.append(schema)
        if include_raw:
            return _IncludeRawStructuredModel(self.structured)
        return self.structured


class _IncludeRawStructuredModel:
    def __init__(self, structured: _StructuredModel) -> None:
        self.structured = structured

    def invoke(self, model_input: object) -> object:
        parsed = self.structured.invoke(model_input)
        return {
            "raw": AIMessage(content=""),
            "parsed": parsed,
            "parsing_error": None,
        }


class _ParserBackedChatModel(BaseChatModel):
    """Return one invalid tool call through LangChain's real Pydantic parser."""

    invalid_args: dict[str, object]
    _schema: type | None = PrivateAttr(default=None)
    _invoke_count: int = PrivateAttr(default=0)
    _structured_output_calls: list[tuple[type, bool]] = PrivateAttr(
        default_factory=list
    )

    @property
    def _llm_type(self) -> str:
        return "parser-backed-test-model"

    @property
    def invoke_count(self) -> int:
        return self._invoke_count

    @property
    def structured_output_calls(self) -> list[tuple[type, bool]]:
        return list(self._structured_output_calls)

    def with_structured_output(
        self,
        schema: type,
        *,
        include_raw: bool = False,
        **kwargs: object,
    ) -> object:
        self._structured_output_calls.append((schema, include_raw))
        return super().with_structured_output(
            schema,
            include_raw=include_raw,
            **kwargs,
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        del tool_choice, kwargs
        tool = tools[0]
        assert isinstance(tool, type)
        self._schema = tool
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        self._invoke_count += 1
        assert self._schema is not None
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self._schema.__name__,
                    "args": self.invalid_args,
                    "id": "invalid-structured-call",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class _EnvelopeModel:
    def __init__(self, envelope: object) -> None:
        self.structured = _StructuredModel(envelope)
        self.calls: list[tuple[type, bool]] = []

    def with_structured_output(
        self,
        schema: type,
        *,
        include_raw: bool = False,
    ) -> _StructuredModel:
        self.calls.append((schema, include_raw))
        return self.structured


class _ProviderFailureModel:
    def __init__(self, failure: ConnectionError, *, fail_during: str) -> None:
        self.failure = failure
        self.fail_during = fail_during
        self.calls: list[tuple[type, bool]] = []
        self.invoke_count = 0

    def with_structured_output(
        self,
        schema: type,
        *,
        include_raw: bool = False,
    ) -> object:
        self.calls.append((schema, include_raw))
        if self.fail_during == "construction":
            raise self.failure
        return self

    def invoke(self, _model_input: object) -> object:
        self.invoke_count += 1
        raise self.failure


def _paper(
    *,
    internal_id: str = "paper-primary",
    external_id: str = "2401.12345v1",
    title: str = "Primary paper",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=internal_id,
        source="arxiv",
        external_id=external_id,
        title=title,
        authors=["Ada"],
        abstract="Trusted abstract",
        source_url=f"https://arxiv.org/abs/{external_id}",
    )


class _NodeStore:
    def __init__(self) -> None:
        self.task = SimpleNamespace(
            id="task-current",
            question="Current question",
            user_id="user-1",
            conversation_id="conversation-1",
            base_checkpoint_id="checkpoint-base",
        )
        self.user_message = SimpleNamespace(
            id="message-current",
            conversation_id="conversation-1",
            task_id="task-current",
            parent_message_id=None,
            role="user",
            content="Current question",
            status="complete",
            metadata={},
        )
        self.assistant_message: SimpleNamespace | None = None
        self.detail = SimpleNamespace(
            conversation=SimpleNamespace(
                id="conversation-1",
                user_id="user-1",
            ),
            primary_paper=_paper(),
            active_papers=[
                _paper(),
                _paper(
                    internal_id="paper-active",
                    external_id="2401.54321v2",
                    title="Active paper",
                ),
            ],
        )
        self.detail_calls: list[tuple[str, str]] = []
        self.publish_calls: list[dict[str, object]] = []

    def get_task(self, task_id: str, *, user_id: str) -> object | None:
        if self.task.id != task_id or self.task.user_id != user_id:
            return None
        return self.task

    def get_task_message(self, task_id: str, role: str) -> object | None:
        if task_id != self.task.id:
            return None
        if role == "user":
            return self.user_message
        if role == "assistant":
            return self.assistant_message
        return None

    def get_conversation_detail(
        self,
        conversation_id: str,
        *,
        user_id: str,
    ) -> object:
        self.detail_calls.append((conversation_id, user_id))
        return self.detail

    def publish_conversation_result(self, **kwargs: object) -> object:
        self.publish_calls.append(kwargs)
        if self.assistant_message is None:
            self.assistant_message = SimpleNamespace(
                id="assistant-published",
                conversation_id="conversation-1",
                task_id="task-current",
                parent_message_id="message-current",
                role="assistant",
                content=kwargs["content"],
                status="complete",
                metadata=deepcopy(kwargs["metadata"]),
            )
        return SimpleNamespace(
            message=self.assistant_message,
            artifact=SimpleNamespace(id=7),
            active_paper_ids=["paper-primary", "paper-active", "paper-related"],
        )


def _tool(name: str, handler: Any) -> Tool:
    return Tool(name=name, description=name, input_schema={}, handler=handler)


def _research_result() -> ResearchResult:
    related = PaperCandidate(
        external_id="2401.99999v1",
        title="Related paper",
        authors=["Grace"],
        abstract="Related abstract",
        source_url="https://arxiv.org/abs/2401.99999v1",
    )
    return ResearchResult(
        evidence_items=[
            EvidenceItem(
                id="ev-primary",
                paper_external_id="2401.12345v1",
                paper_title="Primary paper",
                chunk_text="Primary evidence",
                score=0.9,
                supports=["primary claim"],
            ),
            EvidenceItem(
                id="ev-related",
                paper_external_id=related.external_id,
                paper_title=related.title,
                chunk_text="Related evidence",
                score=0.8,
                supports=["comparison claim"],
            ),
        ],
        used_papers=[
            PaperUse(
                paper=related,
                role="comparison",
                evidence_ids=["ev-related"],
            )
        ],
        limitations=["Only one related paper was used."],
    )


def _bound_state(**updates: object) -> dict[str, object]:
    state: dict[str, object] = {
        "current_task_id": "task-current",
        "current_user_message_id": "message-current",
        "primary_paper_id": "paper-primary",
        "messages": [
            HumanMessage(content="Current question", id="message-current")
        ],
    }
    state.update(updates)
    return state


def _node_context(
    *,
    store: _NodeStore,
    model: object,
    tools: dict[str, Tool] | None = None,
    event_sink: Any | None = None,
    recent_turns: int = 2,
) -> DeepReadingContext:
    return DeepReadingContext(
        user_id="user-1",
        conversation_id="conversation-1",
        task_id="task-current",
        current_user_message_id="message-current",
        base_checkpoint_id="checkpoint-base",
        task_store=store,  # type: ignore[arg-type]
        model=model,
        mcp_tools=tools or {},
        paper_search=lambda _query, _limit: [],
        event_sink=event_sink or (lambda _event, _payload: None),
        summary_recent_turns=recent_turns,
    )


def _context(
    model: object,
    *,
    threshold: int = 32_000,
    recent_turns: int = 6,
    recursion_limit: int = 12,
    store: _NodeStore | None = None,
) -> DeepReadingContext:
    return DeepReadingContext(
        user_id="user-1",
        conversation_id="conversation-1",
        task_id="task-current",
        current_user_message_id="message-current",
        base_checkpoint_id="checkpoint-base",
        task_store=store or _NodeStore(),  # type: ignore[arg-type]
        model=model,
        mcp_tools={},
        paper_search=lambda _query, _limit: [],
        event_sink=lambda _event, _payload: None,
        summary_token_threshold=threshold,
        summary_recent_turns=recent_turns,
        research_recursion_limit=recursion_limit,
    )


def test_context_is_frozen_and_rejects_unbounded_configuration() -> None:
    context = _context(_FakeModel())
    default_context = _node_context(store=_NodeStore(), model=_FakeModel())

    assert (
        default_context.research_recursion_limit,
        default_context.research_model_call_limit,
        default_context.research_tool_call_limit,
        default_context.research_max_output_tokens,
        default_context.research_model_retries,
    ) == (24, 8, 12, 4096, 1)

    with pytest.raises(FrozenInstanceError):
        context.task_id = "changed"  # type: ignore[misc]

    for overrides in (
        {"threshold": 0},
        {"recent_turns": 0},
        {"recursion_limit": 0},
    ):
        with pytest.raises(ValueError):
            _context(_FakeModel(), **overrides)


def test_state_has_one_complete_research_result_field() -> None:
    annotations = get_type_hints(DeepReadingState, include_extras=True)

    assert set(annotations) == {
        "schema_version",
        "graph_version",
        "messages",
        "conversation_summary",
        "current_task_id",
        "current_user_message_id",
        "primary_paper_id",
        "active_paper_ids",
        "research_result",
        "answer_draft",
        "published_message_id",
        "error",
    }
    assert annotations["research_result"] == dict[str, object] | None
    assert "evidence_items" not in annotations


def test_initialize_turn_clears_only_per_turn_fields() -> None:
    messages = [HumanMessage(content="Earlier question", id="human-old")]
    old_summary = SUMMARY.model_dump(mode="json")
    state = {
        "schema_version": SCHEMA_VERSION,
        "graph_version": GRAPH_VERSION,
        "messages": messages,
        "conversation_summary": old_summary,
        "current_task_id": "task-old",
        "current_user_message_id": "message-old",
        "primary_paper_id": "paper-primary",
        "active_paper_ids": ["paper-primary", "paper-related"],
        "research_result": {
            "evidence_items": [{"id": "old-evidence"}],
            "used_papers": [],
            "limitations": [],
        },
        "answer_draft": {"content": "old draft"},
        "published_message_id": "assistant-old",
        "error": {"message": "old error"},
    }

    update = initialize_turn(state, Runtime(context=_context(_FakeModel())))

    assert update == {
        "schema_version": 1,
        "graph_version": "conversation-v1",
        "current_task_id": "task-current",
        "current_user_message_id": "message-current",
        "research_result": None,
        "answer_draft": None,
        "published_message_id": None,
        "error": None,
    }
    merged = state | update
    assert merged["messages"] is messages
    assert merged["conversation_summary"] == old_summary
    assert merged["primary_paper_id"] == "paper-primary"
    assert merged["active_paper_ids"] == ["paper-primary", "paper-related"]


def test_needs_summary_uses_exact_character_estimate_boundary() -> None:
    context = _context(_FakeModel(), threshold=2)

    assert not needs_summary(
        {"messages": [HumanMessage(content="12345678901", id="short")]},
        Runtime(context=context),
    )
    assert needs_summary(
        {"messages": [HumanMessage(content="123456789012", id="long")]},
        Runtime(context=context),
    )


def test_summarize_history_below_threshold_does_not_call_model() -> None:
    model = _FakeModel()
    state = _bound_state()

    update = summarize_history(
        state,
        Runtime(context=_context(model, threshold=10)),
    )

    assert update == {}
    assert model.schemas == []
    assert model.structured.invocations == []


def test_summarize_history_writes_json_and_retains_recent_six_turns() -> None:
    model = _FakeModel(SUMMARY.model_dump(mode="json"))
    messages: list[Any] = []
    for index in range(8):
        messages.extend(
            [
                HumanMessage(content=f"question-{index}-long", id=f"h-{index}"),
                AIMessage(content=f"answer-{index}-long", id=f"a-{index}"),
            ]
        )
    state = {
        "current_task_id": "task-current",
        "current_user_message_id": "message-current",
        "messages": messages,
        "conversation_summary": {
            "confirmed_facts": ["Older fact"],
            "paper_findings": [],
            "comparison_context": [],
            "open_questions": [],
        },
    }
    state["messages"].append(
        HumanMessage(content="Current question", id="message-current")
    )

    update = summarize_history(
        state,
        Runtime(context=_context(model, threshold=1, recent_turns=6)),
    )

    assert model.schemas == [ConversationSummary]
    assert update["conversation_summary"] == SUMMARY.model_dump(mode="json")
    message_update = update["messages"]
    assert isinstance(message_update[0], RemoveMessage)
    assert message_update[0].id == REMOVE_ALL_MESSAGES
    assert [message.id for message in message_update[1:]] == [
        "h-3",
        "a-3",
        "h-4",
        "a-4",
        "h-5",
        "a-5",
        "h-6",
        "a-6",
        "h-7",
        "a-7",
        "message-current",
    ]
    assert [message.id for message in add_messages(messages, message_update)] == [
        "h-3",
        "a-3",
        "h-4",
        "a-4",
        "h-5",
        "a-5",
        "h-6",
        "a-6",
        "h-7",
        "a-7",
        "message-current",
    ]


@pytest.mark.parametrize(
    ("node_name", "schema", "invalid_args", "expected_message"),
    [
        (
            "summary",
            ConversationSummary,
            {
                "confirmed_facts": [""],
                "paper_findings": [],
                "comparison_context": [],
                "open_questions": [],
            },
            "model returned an invalid conversation summary",
        ),
        (
            "answer",
            AnswerDraft,
            {
                "content": "",
                "citations": [],
                "result_quality": "partial",
            },
            "model returned an invalid answer draft",
        ),
    ],
)
def test_structured_parser_validation_error_becomes_terminal_contract(
    node_name: str,
    schema: type,
    invalid_args: dict[str, object],
    expected_message: str,
) -> None:
    model = _ParserBackedChatModel(invalid_args=invalid_args)
    if node_name == "summary":
        call = lambda: summarize_history(  # noqa: E731
            _bound_state(),
            Runtime(context=_context(model, threshold=1)),
        )
    else:
        call = lambda: write_answer(  # noqa: E731
            _bound_state(
                research_result=_research_result().model_dump(mode="json"),
            ),
            Runtime(context=_node_context(store=_NodeStore(), model=model)),
        )

    with pytest.raises(ResearchContractError, match=expected_message) as exc_info:
        call()

    assert isinstance(exc_info.value.__cause__, ValidationError)
    assert model.invoke_count == 1
    assert model.structured_output_calls == [(schema, True)]


@pytest.mark.parametrize("node_name", ["summary", "answer"])
@pytest.mark.parametrize(
    "envelope_kind",
    ["missing", "not_mapping", "missing_parsed", "invalid_parsed", "invalid_error"],
)
def test_structured_envelope_defects_become_safe_terminal_contract(
    node_name: str,
    envelope_kind: str,
) -> None:
    schema = ConversationSummary if node_name == "summary" else AnswerDraft
    valid_parsed: object = (
        SUMMARY
        if node_name == "summary"
        else AnswerDraft(content="Answer", citations=[], result_quality="partial")
    )
    if envelope_kind == "missing":
        envelope: object = None
    elif envelope_kind == "not_mapping":
        envelope = []
    elif envelope_kind == "missing_parsed":
        envelope = {"raw": "secret raw output", "parsing_error": None}
    elif envelope_kind == "invalid_parsed":
        envelope = {"raw": "secret raw output", "parsed": {}, "parsing_error": None}
    else:
        envelope = {
            "raw": "secret raw output",
            "parsed": valid_parsed,
            "parsing_error": "secret parser text",
        }
    model = _EnvelopeModel(envelope)

    if node_name == "summary":
        call = lambda: summarize_history(  # noqa: E731
            _bound_state(),
            Runtime(context=_context(model, threshold=1)),
        )
        expected_message = "model returned an invalid conversation summary"
    else:
        call = lambda: write_answer(  # noqa: E731
            _bound_state(
                research_result=_research_result().model_dump(mode="json"),
            ),
            Runtime(context=_node_context(store=_NodeStore(), model=model)),
        )
        expected_message = "model returned an invalid answer draft"

    with pytest.raises(ResearchContractError, match=expected_message) as exc_info:
        call()

    assert "secret" not in str(exc_info.value)
    assert model.calls == [(schema, True)]


@pytest.mark.parametrize("node_name", ["summary", "answer"])
@pytest.mark.parametrize("fail_during", ["construction", "invoke"])
def test_structured_provider_failure_preserves_identity(
    node_name: str,
    fail_during: str,
) -> None:
    failure = ConnectionError(f"provider {fail_during}")
    model = _ProviderFailureModel(failure, fail_during=fail_during)
    if node_name == "summary":
        call = lambda: summarize_history(  # noqa: E731
            _bound_state(),
            Runtime(context=_context(model, threshold=1)),
        )
        schema = ConversationSummary
    else:
        call = lambda: write_answer(  # noqa: E731
            _bound_state(
                research_result=_research_result().model_dump(mode="json"),
            ),
            Runtime(context=_node_context(store=_NodeStore(), model=model)),
        )
        schema = AnswerDraft

    with pytest.raises(ConnectionError) as exc_info:
        call()

    assert exc_info.value is failure
    assert model.calls == [(schema, True)]
    assert model.invoke_count == (1 if fail_during == "invoke" else 0)


def test_research_evidence_writes_complete_json_research_result(monkeypatch) -> None:
    paper = PaperCandidate(
        external_id="2401.12345v1",
        title="Related paper",
        authors=["Ada"],
        abstract="Abstract",
        source_url="https://arxiv.org/abs/2401.12345v1",
    )
    result = ResearchResult(
        evidence_items=[
            EvidenceItem(
                id="ev-1",
                paper_external_id=paper.external_id,
                paper_title=paper.title,
                chunk_text="Retrieved evidence",
                score=0.9,
                supports=["method"],
            )
        ],
        used_papers=[
            PaperUse(
                paper=paper,
                role="comparison",
                evidence_ids=["ev-1"],
            )
        ],
        limitations=["One comparison paper."],
    )
    seen: list[tuple[object, object]] = []

    def fake_run(state: object, context: object) -> ResearchResult:
        seen.append((state, context))
        return result

    monkeypatch.setattr(nodes_module, "run_research_agent", fake_run)
    state = _bound_state()
    context = _context(_FakeModel())

    update = research_evidence(state, Runtime(context=context))

    assert seen == [(state, context)]
    assert update == {"research_result": result.model_dump(mode="json")}
    assert update["research_result"]["used_papers"][0]["paper"] == (
        paper.model_dump(mode="json")
    )


def test_prepare_primary_paper_uses_canonical_identity_and_safe_tool_sequence() -> None:
    store = _NodeStore()
    calls: list[tuple[str, dict[str, object]]] = []
    events: list[tuple[str, dict[str, object]]] = []
    original_detail = deepcopy(store.detail)

    def download(arguments: dict[str, object]) -> str:
        calls.append(("download", arguments))
        return json.dumps(
            {
                "paper_id": "https://arxiv.org/pdf/2401.12345v1.pdf",
                "text": "FULL PAPER TEXT THAT MUST NOT ENTER EVENTS",
                "untrusted": "must not reach build_index",
            }
        )

    def build(arguments: dict[str, object]) -> str:
        calls.append(("build", arguments))
        return json.dumps({"cached_papers": ["2401.12345v1"]})

    tools = {
        "mcp__arxiv__download_paper": _tool(
            "mcp__arxiv__download_paper", download
        ),
        "mcp__colbert__build_index": _tool("mcp__colbert__build_index", build),
    }
    context = _node_context(
        store=store,
        model=_FakeModel(),
        tools=tools,
        event_sink=lambda event, payload: events.append((event, payload)),
    )
    state = _bound_state()

    first = prepare_primary_paper(state, Runtime(context=context))
    second = prepare_primary_paper(state, Runtime(context=context))

    assert first == second == {
        "primary_paper_id": "paper-primary",
        "active_paper_ids": ["paper-primary", "paper-active"],
    }
    assert calls == [
        ("download", {"arxiv_id": "2401.12345v1"}),
        (
            "build",
            {
                "documents": [
                    {
                        "paper_id": "2401.12345v1",
                        "text": "FULL PAPER TEXT THAT MUST NOT ENTER EVENTS",
                    }
                ]
            },
        ),
        ("download", {"arxiv_id": "2401.12345v1"}),
        (
            "build",
            {
                "documents": [
                    {
                        "paper_id": "2401.12345v1",
                        "text": "FULL PAPER TEXT THAT MUST NOT ENTER EVENTS",
                    }
                ]
            },
        ),
    ]
    assert store.detail == original_detail
    assert "FULL PAPER TEXT" not in json.dumps(events)


def test_prepare_primary_paper_rejects_bad_identity_before_build() -> None:
    for external_id, download_result, expected_message in (
        ("../../outside", None, "valid arXiv"),
        (
            "2401.12345v1",
            {"paper_id": "2401.99999v1", "text": "wrong paper"},
            "does not match",
        ),
    ):
        store = _NodeStore()
        store.detail.primary_paper.external_id = external_id
        calls: list[str] = []

        def download(_arguments: dict[str, object]) -> str:
            calls.append("download")
            return json.dumps(download_result)

        def build(_arguments: dict[str, object]) -> str:
            calls.append("build")
            return json.dumps({"fresh_papers": ["2401.12345v1"]})

        context = _node_context(
            store=store,
            model=_FakeModel(),
            tools={
                "mcp__arxiv__download_paper": _tool(
                    "mcp__arxiv__download_paper", download
                ),
                "mcp__colbert__build_index": _tool(
                    "mcp__colbert__build_index", build
                ),
            },
        )

        with pytest.raises(ResearchContractError, match=expected_message):
            prepare_primary_paper(
                _bound_state(),
                Runtime(context=context),
            )

        assert calls == ([] if download_result is None else ["download"])


@pytest.mark.parametrize(
    "failing_tool_name",
    ["mcp__arxiv__download_paper", "mcp__colbert__build_index"],
    ids=["download", "build"],
)
def test_prepare_primary_paper_converts_only_mcp_tool_error(
    failing_tool_name: str,
) -> None:
    failure = MCPToolError("remote-secret full-paper-text")
    calls: list[str] = []
    events: list[tuple[str, dict[str, object]]] = []

    def download(arguments: dict[str, object]) -> str:
        calls.append("mcp__arxiv__download_paper")
        if failing_tool_name == "mcp__arxiv__download_paper":
            raise failure
        return json.dumps(
            {"paper_id": arguments["arxiv_id"], "text": "trusted paper text"}
        )

    def build(arguments: dict[str, object]) -> str:
        calls.append("mcp__colbert__build_index")
        if failing_tool_name == "mcp__colbert__build_index":
            raise failure
        return json.dumps(
            {"fresh_papers": [arguments["documents"][0]["paper_id"]]}
        )

    context = _node_context(
        store=_NodeStore(),
        model=_FakeModel(),
        tools={
            "mcp__arxiv__download_paper": _tool(
                "mcp__arxiv__download_paper", download
            ),
            "mcp__colbert__build_index": _tool(
                "mcp__colbert__build_index", build
            ),
        },
        event_sink=lambda event, payload: events.append((event, payload)),
    )

    with pytest.raises(ResearchContractError) as exc_info:
        prepare_primary_paper(_bound_state(), Runtime(context=context))

    assert exc_info.value.__cause__ is failure
    assert "remote-secret" not in str(exc_info.value)
    assert "full-paper-text" not in str(exc_info.value)
    assert calls[-1] == failing_tool_name
    assert "remote-secret" not in str(events)
    assert "full-paper-text" not in str(events)


@pytest.mark.parametrize(
    "failure",
    [
        MCPToolTimeout("MCP timeout"),
        MCPTransportError("MCP transport closed"),
        ConnectionError("connection reset"),
        TimeoutError("socket timeout"),
        OSError("filesystem unavailable"),
        RuntimeError("unknown infrastructure failure"),
        ValueError("unknown dependency failure"),
    ],
    ids=[
        "mcp-timeout",
        "mcp-transport",
        "connection",
        "timeout",
        "os-error",
        "unknown-runtime",
        "unknown-value",
    ],
)
def test_prepare_primary_paper_preserves_infrastructure_failure_identity(
    failure,
) -> None:
    calls: list[dict[str, object]] = []

    def download(arguments: dict[str, object]) -> str:
        calls.append(arguments)
        raise failure

    context = _node_context(
        store=_NodeStore(),
        model=_FakeModel(),
        tools={
            "mcp__arxiv__download_paper": _tool(
                "mcp__arxiv__download_paper", download
            ),
        },
    )

    with pytest.raises(type(failure)) as exc_info:
        prepare_primary_paper(_bound_state(), Runtime(context=context))

    assert exc_info.value is failure
    assert calls == [{"arxiv_id": "2401.12345v1"}]


def test_write_answer_uses_bounded_trusted_context_and_writes_json_draft() -> None:
    store = _NodeStore()
    draft = AnswerDraft(
        content="Evidence-grounded answer.",
        citations=[AnswerCitation(evidence_id="ev-primary", label="Primary")],
        result_quality="complete",
    )
    model = _FakeModel(draft.model_dump(mode="json"))
    messages: list[Any] = []
    for index in range(4):
        messages.extend(
            [
                HumanMessage(content=f"question-{index}", id=f"human-{index}"),
                AIMessage(content=f"answer-{index}", id=f"assistant-{index}"),
            ]
        )
    messages.append(HumanMessage(content="Current question", id="message-current"))
    state = _bound_state(
        messages=messages,
        conversation_summary=SUMMARY.model_dump(mode="json"),
        research_result=_research_result().model_dump(mode="json"),
    )

    update = write_answer(
        state,
        Runtime(context=_node_context(store=store, model=model, recent_turns=2)),
    )

    assert model.schemas == [AnswerDraft]
    assert update == {"answer_draft": draft.model_dump(mode="json")}
    prompt_messages = model.structured.invocations[0]
    assert [message.id for message in prompt_messages if message.id] == [
        "human-3",
        "assistant-3",
        "message-current",
    ]
    rendered = "\n".join(str(message.content) for message in prompt_messages)
    assert "question-0" not in rendered
    assert "answer-1" not in rendered
    assert "question-3" in rendered
    assert "Current question" in rendered
    assert "Primary evidence" in rendered
    assert "Only one related paper was used." in rendered
    assert "Primary paper" in rendered
    assert "Trusted abstract" in rendered


def test_write_answer_rejects_citation_outside_research_result() -> None:
    store = _NodeStore()
    invalid_draft = AnswerDraft(
        content="Unsupported answer.",
        citations=[AnswerCitation(evidence_id="ev-invented", label="Invented")],
        result_quality="partial",
    )
    model = _FakeModel(invalid_draft)

    with pytest.raises(ResearchContractError, match="unknown evidence"):
        write_answer(
            _bound_state(
                research_result=_research_result().model_dump(mode="json"),
            ),
            Runtime(context=_node_context(store=store, model=model)),
        )


def test_write_answer_converts_invalid_structured_payload_to_terminal_contract() -> None:
    store = _NodeStore()
    model = _FakeModel(
        {
            "content": "",
            "citations": [],
            "result_quality": "partial",
        }
    )

    with pytest.raises(ResearchContractError) as exc_info:
        write_answer(
            _bound_state(
                research_result=_research_result().model_dump(mode="json"),
            ),
            Runtime(context=_node_context(store=store, model=model)),
        )

    assert exc_info.value.error_code == "research_contract_invalid"


def test_publish_result_publishes_only_validated_used_papers() -> None:
    store = _NodeStore()
    result = _research_result()
    draft = AnswerDraft(
        content="Final answer.",
        citations=[
            AnswerCitation(evidence_id="ev-primary", label="Primary evidence"),
            AnswerCitation(evidence_id="ev-related", label="Related evidence"),
        ],
        result_quality="partial",
    )

    update = publish_result(
        _bound_state(
            research_result=result.model_dump(mode="json"),
            answer_draft=draft.model_dump(mode="json"),
        ),
        Runtime(context=_node_context(store=store, model=_FakeModel())),
    )

    assert len(store.publish_calls) == 2
    assert store.publish_calls[0]["used_papers"] == []
    call = store.publish_calls[1]
    assert call["task_id"] == "task-current"
    assert call["content"] == "Final answer."
    assert call["metadata"] == {
        "citations": [
            {"evidence_id": "ev-primary", "label": "Primary evidence"},
            {"evidence_id": "ev-related", "label": "Related evidence"},
        ],
        "result_quality": "partial",
        "limitations": ["Only one related paper was used."],
        "research_result": result.model_dump(mode="json"),
    }
    json.dumps(call["metadata"])
    assert call["used_papers"] == [
        UsedPaperInput(
            paper=result.used_papers[0].paper,
            role="comparison",
        )
    ]
    assert update == {
        "research_result": result.model_dump(mode="json"),
        "answer_draft": draft.model_dump(mode="json"),
        "published_message_id": "assistant-published",
        "active_paper_ids": [
            "paper-primary",
            "paper-active",
            "paper-related",
        ],
        "messages": [
            {
                "role": "assistant",
                "content": "Final answer.",
                "id": "assistant-published",
            }
        ],
    }


def test_publish_result_rejects_dangling_citation_without_store_write() -> None:
    store = _NodeStore()
    result = _research_result()
    invalid_draft = AnswerDraft(
        content="Unsupported answer.",
        citations=[AnswerCitation(evidence_id="ev-invented", label="Invented")],
        result_quality="partial",
    )

    with pytest.raises(ResearchContractError, match="unknown evidence"):
        publish_result(
            _bound_state(
                research_result=result.model_dump(mode="json"),
                answer_draft=invalid_draft.model_dump(mode="json"),
            ),
            Runtime(context=_node_context(store=store, model=_FakeModel())),
        )

    assert store.publish_calls == []


@pytest.mark.parametrize(
    "node_name",
    [
        "summarize_history",
        "prepare_primary_paper",
        "research_evidence",
        "write_answer",
        "publish_result",
    ],
)
def test_external_and_publish_nodes_validate_binding_before_side_effect(
    node_name: str,
    monkeypatch,
) -> None:
    store = _NodeStore()
    store.task.conversation_id = "conversation-other"
    model = _FakeModel(
        AnswerDraft(content="Never called", citations=[], result_quality="partial")
    )
    tool_calls: list[str] = []
    agent_calls: list[str] = []

    def external_tool(_arguments: dict[str, object]) -> str:
        tool_calls.append("called")
        return "{}"

    monkeypatch.setattr(
        nodes_module,
        "run_research_agent",
        lambda _state, _context: agent_calls.append("called"),
    )
    context = _node_context(
        store=store,
        model=model,
        tools={
            "mcp__arxiv__download_paper": _tool(
                "mcp__arxiv__download_paper", external_tool
            ),
            "mcp__colbert__build_index": _tool(
                "mcp__colbert__build_index", external_tool
            ),
        },
    )
    state = _bound_state(
        messages=[
            HumanMessage(
                content="Current question",
                id="message-current",
            )
        ],
        research_result=_research_result().model_dump(mode="json"),
        answer_draft=AnswerDraft(
            content="Draft",
            citations=[],
            result_quality="partial",
        ).model_dump(mode="json"),
    )
    selected = {
        "summarize_history": summarize_history,
        "prepare_primary_paper": prepare_primary_paper,
        "research_evidence": research_evidence,
        "write_answer": write_answer,
        "publish_result": publish_result,
    }[node_name]
    if node_name == "summarize_history":
        context = _context(model, threshold=1, store=store)

    with pytest.raises(ResearchContractError, match="conversation") as exc_info:
        selected(state, Runtime(context=context))

    assert exc_info.value.error_code == "task_binding_invalid"
    assert tool_calls == []
    assert agent_calls == []
    assert model.schemas == []
    assert store.publish_calls == []


@pytest.mark.parametrize(
    "broken_binding",
    [
        "cross_owner",
        "cross_conversation",
        "wrong_base_checkpoint",
        "cross_user_message",
        "wrong_state_task",
        "wrong_state_user_message",
    ],
)
def test_runtime_binding_rejects_mismatched_trusted_entities(
    broken_binding: str,
    monkeypatch,
) -> None:
    store = _NodeStore()
    state = _bound_state()
    context = _node_context(store=store, model=_FakeModel())
    if broken_binding == "cross_owner":
        store.task.user_id = "user-other"
    elif broken_binding == "cross_conversation":
        store.task.conversation_id = "conversation-other"
    elif broken_binding == "wrong_base_checkpoint":
        store.task.base_checkpoint_id = "checkpoint-other"
    elif broken_binding == "cross_user_message":
        store.user_message.id = "message-other"
    elif broken_binding == "wrong_state_task":
        state["current_task_id"] = "task-other"
    else:
        state["current_user_message_id"] = "message-other"
    agent_calls: list[str] = []
    monkeypatch.setattr(
        nodes_module,
        "run_research_agent",
        lambda _state, _context: agent_calls.append("called"),
    )

    with pytest.raises(ResearchContractError) as exc_info:
        research_evidence(state, Runtime(context=context))

    assert exc_info.value.error_code == "task_binding_invalid"
    assert agent_calls == []


def test_publish_result_rejects_malformed_persisted_metadata() -> None:
    store = _NodeStore()
    store.assistant_message = SimpleNamespace(
        id="assistant-persisted",
        conversation_id="conversation-1",
        task_id="task-current",
        parent_message_id="message-current",
        role="assistant",
        content="Persisted answer",
        status="complete",
        metadata={"citations": []},
    )
    result = _research_result()
    draft = AnswerDraft(content="Retry answer", citations=[], result_quality="partial")

    with pytest.raises(
        ResearchContractError,
        match="persisted assistant metadata",
    ) as exc_info:
        publish_result(
            _bound_state(
                research_result=result.model_dump(mode="json"),
                answer_draft=draft.model_dump(mode="json"),
            ),
            Runtime(context=_node_context(store=store, model=_FakeModel())),
        )

    assert exc_info.value.error_code == "final_checkpoint_invalid"
    assert store.publish_calls == []


@pytest.mark.parametrize(
    "bad_messages",
    [
        [],
        [
            HumanMessage(content="Current question", id="message-current"),
            HumanMessage(content="Current question", id="message-current"),
        ],
        [HumanMessage(content="Current question", id="message-other")],
        [HumanMessage(content="Forged question", id="message-current")],
        [AIMessage(content="Current question", id="message-current")],
        [
            HumanMessage(content="Current question", id="message-current"),
            HumanMessage(content="Rogue question", id="message-rogue"),
        ],
    ],
    ids=[
        "missing",
        "duplicate",
        "wrong-id",
        "wrong-content",
        "wrong-type",
        "rogue-human-after-current",
    ],
)
def test_runtime_binding_rejects_untrusted_current_human_before_agent(
    bad_messages: list[object],
    monkeypatch,
) -> None:
    store = _NodeStore()
    calls: list[str] = []
    monkeypatch.setattr(
        nodes_module,
        "run_research_agent",
        lambda _state, _context: calls.append("agent"),
    )

    with pytest.raises(ResearchContractError):
        research_evidence(
            _bound_state(messages=bad_messages),
            Runtime(context=_node_context(store=store, model=_FakeModel())),
        )

    assert calls == []


@pytest.mark.parametrize(
    "broken_binding",
    [
        "conversation_detail_owner",
        "conversation_detail_id",
        "task_question",
    ],
)
def test_runtime_binding_rejects_invalid_conversation_or_task_question(
    broken_binding: str,
    monkeypatch,
) -> None:
    store = _NodeStore()
    if broken_binding == "conversation_detail_owner":
        store.detail.conversation.user_id = "user-other"
    elif broken_binding == "conversation_detail_id":
        store.detail.conversation.id = "conversation-other"
    else:
        store.task.question = "Another question"
    calls: list[str] = []
    monkeypatch.setattr(
        nodes_module,
        "run_research_agent",
        lambda _state, _context: calls.append("agent"),
    )

    with pytest.raises(ResearchContractError):
        research_evidence(
            _bound_state(),
            Runtime(context=_node_context(store=store, model=_FakeModel())),
        )

    assert calls == []


def _real_node_case(tmp_path, *, suffix: str):
    store = TaskStore(tmp_path / f"binding-{suffix}.sqlite3")
    user = store.create_user(
        username=f"binding-{suffix}",
        password_hash="hash",
        password_salt="salt",
    )
    conversation = store.create_conversation(
        user_id=user.id,
        paper=PaperCandidate(
            external_id="2401.92000v1",
            title=f"Primary {suffix}",
            authors=["Ada"],
            abstract="Abstract",
            source_url="https://arxiv.org/abs/2401.92000v1",
        ),
    )
    turn = store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content="Current SQLite question",
        depth="deep",
        expected_head_message_id=None,
    )
    assert store.claim_task(turn.task.id) is not None
    return store, user, conversation, turn


@pytest.mark.parametrize(
    "node_name",
    [
        "summarize_history",
        "prepare_primary_paper",
        "research_evidence",
        "write_answer",
        "publish_result",
    ],
)
@pytest.mark.parametrize(
    "tampering",
    ["cross_owner_conversation", "wrong_human_content", "rogue_human"],
)
def test_real_sqlite_binding_fails_before_node_side_effect(
    node_name: str,
    tampering: str,
    tmp_path,
    monkeypatch,
) -> None:
    store, user, conversation, turn = _real_node_case(
        tmp_path,
        suffix=f"{node_name[:2]}-{tampering[:2]}",
    )
    context_conversation_id = conversation.id
    if tampering == "cross_owner_conversation":
        other_user = store.create_user(
            username=f"other-{node_name}-{tampering}",
            password_hash="hash",
            password_salt="salt",
        )
        other_conversation = store.create_conversation(
            user_id=other_user.id,
            paper=PaperCandidate(
                external_id="2401.92999v1",
                title="Other owner's paper",
                authors=["Grace"],
                abstract="Other abstract",
                source_url="https://arxiv.org/abs/2401.92999v1",
            ),
        )
        context_conversation_id = other_conversation.id

    current_human = HumanMessage(
        content=turn.user_message.content,
        id=turn.user_message.id,
    )
    messages = [current_human]
    if tampering == "wrong_human_content":
        messages = [HumanMessage(content="Forged question", id=turn.user_message.id)]
    elif tampering == "rogue_human":
        messages.append(HumanMessage(content="Rogue question", id="message-rogue"))

    result = _research_result()
    draft = AnswerDraft(
        content="Answer that must not publish",
        citations=[AnswerCitation(evidence_id="ev-primary", label="Primary")],
        result_quality="complete",
    )
    state = {
        "current_task_id": turn.task.id,
        "current_user_message_id": turn.user_message.id,
        "primary_paper_id": conversation.primary_paper_id,
        "messages": messages,
        "research_result": result.model_dump(mode="json"),
        "answer_draft": draft.model_dump(mode="json"),
    }
    model = _FakeModel(
        SUMMARY.model_dump(mode="json")
        if node_name == "summarize_history"
        else draft.model_dump(mode="json")
    )
    side_effects: list[str] = []

    def external_tool(arguments: dict[str, object]) -> str:
        side_effects.append("mcp")
        if "arxiv_id" in arguments:
            return json.dumps(
                {
                    "paper_id": "2401.92000v1",
                    "text": "Paper text",
                }
            )
        return json.dumps({"cached_papers": ["2401.92000v1"]})

    monkeypatch.setattr(
        nodes_module,
        "run_research_agent",
        lambda _state, _context: side_effects.append("agent"),
    )
    context = DeepReadingContext(
        user_id=user.id,
        conversation_id=context_conversation_id,
        task_id=turn.task.id,
        current_user_message_id=turn.user_message.id,
        base_checkpoint_id=None,
        task_store=store,
        model=model,
        mcp_tools={
            "mcp__arxiv__download_paper": _tool(
                "mcp__arxiv__download_paper", external_tool
            ),
            "mcp__colbert__build_index": _tool(
                "mcp__colbert__build_index", external_tool
            ),
        },
        paper_search=lambda _query, _limit: [],
        event_sink=lambda _event, _payload: None,
        summary_token_threshold=1,
    )
    selected = {
        "summarize_history": summarize_history,
        "prepare_primary_paper": prepare_primary_paper,
        "research_evidence": research_evidence,
        "write_answer": write_answer,
        "publish_result": publish_result,
    }[node_name]

    with pytest.raises(ResearchContractError):
        selected(state, Runtime(context=context))

    assert side_effects == []
    assert model.schemas == []
    assert store.get_task_message(turn.task.id, "assistant") is None


def _concurrent_result(suffix: str) -> tuple[ResearchResult, AnswerDraft]:
    paper = PaperCandidate(
        external_id=f"2401.93{suffix.zfill(3)}v1",
        title=f"Concurrent paper {suffix}",
        authors=[f"Author {suffix}"],
        abstract=f"Abstract {suffix}",
        source_url=f"https://arxiv.org/abs/2401.93{suffix.zfill(3)}v1",
    )
    evidence_id = f"ev-concurrent-{suffix}"
    result = ResearchResult(
        evidence_items=[
            EvidenceItem(
                id=evidence_id,
                paper_external_id=paper.external_id,
                paper_title=paper.title,
                chunk_text=f"Evidence {suffix}",
                score=0.7,
                supports=[f"claim {suffix}"],
            )
        ],
        used_papers=[
            PaperUse(
                paper=paper,
                role="comparison",
                evidence_ids=[evidence_id],
            )
        ],
        limitations=[f"limitation-{suffix}"],
    )
    draft = AnswerDraft(
        content=f"concurrent-answer-{suffix}",
        citations=[AnswerCitation(evidence_id=evidence_id, label=f"result-{suffix}")],
        result_quality="complete" if suffix == "1" else "partial",
    )
    return result, draft


def test_two_sqlite_stores_concurrently_publish_only_one_authoritative_result(
    tmp_path,
) -> None:
    prepublish_barrier = Barrier(2)

    class BarrierTaskStore(TaskStore):
        def get_task_message(self, task_id: str, role: str):
            message = super().get_task_message(task_id, role)
            if role == "assistant" and message is None:
                prepublish_barrier.wait(timeout=5)
            return message

    db_path = tmp_path / "concurrent-publication.sqlite3"
    first_store = BarrierTaskStore(db_path)
    user = first_store.create_user(
        username="concurrent-node",
        password_hash="hash",
        password_salt="salt",
    )
    conversation = first_store.create_conversation(
        user_id=user.id,
        paper=PaperCandidate(
            external_id="2401.93000v1",
            title="Concurrent primary",
            authors=["Primary"],
            abstract="Primary abstract",
            source_url="https://arxiv.org/abs/2401.93000v1",
        ),
    )
    turn = first_store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content="Concurrent question",
        depth="deep",
        expected_head_message_id=None,
    )
    assert first_store.claim_task(turn.task.id) is not None
    second_store = BarrierTaskStore(db_path)
    candidates = [_concurrent_result("1"), _concurrent_result("2")]

    def publish(store: TaskStore, candidate: tuple[ResearchResult, AnswerDraft]):
        result, draft = candidate
        state = {
            "current_task_id": turn.task.id,
            "current_user_message_id": turn.user_message.id,
            "primary_paper_id": conversation.primary_paper_id,
            "messages": [
                HumanMessage(
                    content=turn.user_message.content,
                    id=turn.user_message.id,
                )
            ],
            "research_result": result.model_dump(mode="json"),
            "answer_draft": draft.model_dump(mode="json"),
        }
        context = DeepReadingContext(
            user_id=user.id,
            conversation_id=conversation.id,
            task_id=turn.task.id,
            current_user_message_id=turn.user_message.id,
            base_checkpoint_id=None,
            task_store=store,
            model=object(),
            mcp_tools={},
            paper_search=lambda _query, _limit: [],
            event_sink=lambda _event, _payload: None,
        )
        return publish_result(state, Runtime(context=context))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish, first_store, candidates[0]),
            executor.submit(publish, second_store, candidates[1]),
        ]
        outputs = [future.result() for future in futures]

    assert outputs[0]["published_message_id"] == outputs[1]["published_message_id"]
    assert outputs[0]["research_result"] == outputs[1]["research_result"]
    assert outputs[0]["answer_draft"] == outputs[1]["answer_draft"]
    assert outputs[0]["messages"] == outputs[1]["messages"]
    assistant = first_store.get_task_message(turn.task.id, "assistant")
    assert assistant is not None
    winning_suffix = "1" if assistant.content == "concurrent-answer-1" else "2"
    losing_suffix = "2" if winning_suffix == "1" else "1"
    winning_result, winning_draft = candidates[int(winning_suffix) - 1]
    assert outputs[0]["research_result"] == winning_result.model_dump(mode="json")
    assert outputs[0]["answer_draft"] == winning_draft.model_dump(mode="json")
    assert assistant.metadata["result_quality"] == winning_draft.result_quality

    with first_store.engine.connect() as connection:
        associated_external_ids = set(
            connection.exec_driver_sql(
                """
                SELECT papers.external_id
                FROM conversation_papers
                JOIN papers ON papers.id = conversation_papers.paper_id
                WHERE conversation_papers.conversation_id = ?
                """,
                (conversation.id,),
            ).scalars()
        )
    assert winning_result.used_papers[0].paper.external_id in associated_external_ids
    assert (
        candidates[int(losing_suffix) - 1][0].used_papers[0].paper.external_id
        not in associated_external_ids
    )
