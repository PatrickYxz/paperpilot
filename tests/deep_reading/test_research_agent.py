"""Bounded LangChain research-agent contracts without network or model calls."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.agents.structured_output import StructuredOutputError, ToolStrategy
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from pydantic import PrivateAttr

from paperpilot.core.adapter import Tool
from paperpilot.deep_reading.nodes import DeepReadingContext
from paperpilot.deep_reading.research_agent import (
    AgentResearchDecision,
    DeepReadingTaskError,
    ResearchContractError,
    run_research_agent,
)
from paperpilot.papers import PaperCandidate
from paperpilot.tools.mcp_client import (
    MCPToolError,
    MCPToolTimeout,
    MCPTransportError,
)


PRIMARY = PaperCandidate(
    external_id="2401.10001v1",
    title="Primary Paper",
    authors=["Primary Author"],
    abstract="Primary abstract",
    source_url="https://arxiv.org/abs/2401.10001v1",
)
ACTIVE = PaperCandidate(
    external_id="2401.10002v1",
    title="Active Paper",
    authors=["Active Author"],
    abstract="Active abstract",
    source_url="https://arxiv.org/abs/2401.10002v1",
)
RELATED = PaperCandidate(
    external_id="2401.10003v1",
    title="Related Paper",
    authors=["Related Author"],
    abstract="Related abstract",
    source_url="https://arxiv.org/abs/2401.10003v1",
)
UNUSED = PaperCandidate(
    external_id="2401.10004v1",
    title="Unused Search Result",
    authors=["Unused Author"],
    abstract="Unused abstract",
    source_url="https://arxiv.org/abs/2401.10004v1",
)


def _paper_record(candidate: PaperCandidate, internal_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=internal_id,
        source=candidate.source,
        external_id=candidate.external_id,
        title=candidate.title,
        authors=candidate.authors,
        abstract=candidate.abstract,
        source_url=candidate.source_url,
    )


class _Store:
    def __init__(self) -> None:
        self.detail = SimpleNamespace(
            primary_paper=_paper_record(PRIMARY, "paper-primary"),
            active_papers=[
                _paper_record(PRIMARY, "paper-primary"),
                _paper_record(ACTIVE, "paper-active"),
            ],
        )
        self.calls: list[tuple[str, str]] = []

    def get_conversation_detail(
        self,
        conversation_id: str,
        *,
        user_id: str,
    ) -> object:
        self.calls.append((conversation_id, user_id))
        return self.detail


class _Agent:
    def __init__(
        self,
        tools: list[Any],
        behavior: Callable[[Mapping[str, Any], int], object],
    ) -> None:
        self.tools = {item.name: item for item in tools}
        self.behavior = behavior
        self.invocations: list[tuple[object, object]] = []

    def invoke(self, model_input: object, *, config: object) -> object:
        self.invocations.append((model_input, config))
        return self.behavior(self.tools, len(self.invocations))


class _AgentFactory:
    def __init__(
        self,
        behavior: Callable[[Mapping[str, Any], int], object],
    ) -> None:
        self.behavior = behavior
        self.calls: list[dict[str, object]] = []
        self.agent: _Agent | None = None

    def __call__(self, **kwargs: object) -> _Agent:
        self.calls.append(kwargs)
        self.agent = _Agent(list(kwargs["tools"]), self.behavior)  # type: ignore[arg-type]
        return self.agent


class _ScriptedResearchChatModel(BaseChatModel):
    """Drive LangChain's real Agent/ToolNode loop without provider I/O."""

    script: str = "full"
    _invoke_count: int = PrivateAttr(default=0)
    _tool_trace: list[str] = PrivateAttr(default_factory=list)
    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-research-test-model"

    @property
    def invoke_count(self) -> int:
        return self._invoke_count

    @property
    def tool_trace(self) -> list[str]:
        return list(self._tool_trace)

    @property
    def bound_tool_names(self) -> list[str]:
        return list(self._bound_tool_names)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        del tool_choice, kwargs
        names: list[str] = []
        for bound_tool in tools:
            if isinstance(bound_tool, BaseTool):
                names.append(bound_tool.name)
            elif isinstance(bound_tool, type):
                names.append(bound_tool.__name__)
            else:
                names.append(str(bound_tool.get("name", "provider-tool")))
        self._bound_tool_names = names
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        self._invoke_count += 1
        if self.script == "prepare_only":
            name = "prepare_paper"
            arguments: dict[str, object] = {"external_id": PRIMARY.external_id}
        elif self._invoke_count == 1:
            name = "search_related_papers"
            arguments = {"query": "related method", "limit": 1}
        elif self._invoke_count == 2:
            last_tool = _last_tool_message(messages)
            assert last_tool.name == "search_related_papers"
            found = json.loads(str(last_tool.content))
            assert found[0]["external_id"] == RELATED.external_id
            name = "prepare_paper"
            arguments = {"external_id": RELATED.external_id}
        elif self._invoke_count == 3:
            last_tool = _last_tool_message(messages)
            assert last_tool.name == "prepare_paper"
            name = "retrieve_paper_evidence"
            arguments = {
                "question": "How do the methods differ?",
                "external_id": RELATED.external_id,
                "top_k_each": 3,
                "summary_k": 2,
            }
        elif self._invoke_count == 4:
            last_tool = _last_tool_message(messages)
            assert last_tool.name == "retrieve_paper_evidence"
            retrieved = json.loads(str(last_tool.content))
            evidence_id = retrieved["evidence_items"][0]["id"]
            name = AgentResearchDecision.__name__
            arguments = {
                "selected_evidence_ids": [evidence_id],
                "paper_uses": [
                    {
                        "external_id": RELATED.external_id,
                        "role": "comparison",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "limitations": ["Only one related paper was compared."],
            }
        else:  # pragma: no cover - a bounded successful run must stop at call four
            raise AssertionError("real Agent made an unexpected fifth model call")

        self._tool_trace.append(name)
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": arguments,
                    "id": f"scripted-call-{self._invoke_count}",
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _last_tool_message(messages: list[BaseMessage]) -> ToolMessage:
    selected = next(
        (message for message in reversed(messages) if isinstance(message, ToolMessage)),
        None,
    )
    assert selected is not None
    return selected


def _evidence_payload(
    external_id: str,
    evidence_id: str,
    *,
    best_score: float = 5.0,
) -> dict[str, object]:
    return {
        "summary_text": "Planned retrieval completed.",
        "query_errors": [],
        "evidence_pool": {
            "plan_id": "plan-1",
            "question": "How do the methods differ?",
            "query_plan": {},
            "items": [
                {
                    "id": evidence_id,
                    "paper_id": external_id,
                    "chunk_id": "chunk-1",
                    "chunk_text": f"Evidence from {external_id}",
                    "best_score": best_score,
                    "matched_queries": [
                        {
                            "query_id": "q-1",
                            "query": "method comparison",
                            "role": "focused_rewrite",
                            "rank": 1,
                            "score": 0.91,
                            "targets": ["method", "result"],
                        }
                    ],
                }
            ],
            "summary_items": [evidence_id],
            "missing_requirements": [],
            "stats": {
                "query_count": 1,
                "raw_result_count": 1,
                "deduped_count": 1,
            },
        },
    }


def _mcp_tools(
    *,
    download_result: object | None = None,
    build_result: object | None = None,
    retrieval_result: object | None = None,
) -> tuple[dict[str, Tool], list[tuple[str, dict[str, object]]]]:
    calls: list[tuple[str, dict[str, object]]] = []

    def handler(name: str, configured: object | None) -> Callable[[dict], object]:
        def call(arguments: dict) -> object:
            calls.append((name, dict(arguments)))
            if callable(configured):
                return configured(arguments)
            if configured is not None:
                return configured
            if name == "mcp__arxiv__download_paper":
                return json.dumps(
                    {
                        "paper_id": arguments["arxiv_id"],
                        "text": "downloaded paper text",
                    }
                )
            if name == "mcp__colbert__build_index":
                paper_id = arguments["documents"][0]["paper_id"]
                return json.dumps(
                    {"fresh_papers": [paper_id], "existing_papers": []}
                )
            paper_id = arguments["paper_id"]
            return json.dumps(_evidence_payload(paper_id, "ev_1"))

        return call

    names = (
        "mcp__arxiv__download_paper",
        "mcp__colbert__build_index",
        "mcp__colbert__planned_retrieval",
    )
    configured = (download_result, build_result, retrieval_result)
    tools = {
        name: Tool(name, name, {}, handler(name, result))
        for name, result in zip(names, configured, strict=True)
    }
    return tools, calls


def _context(
    behavior: Callable[[Mapping[str, Any], int], object],
    *,
    model: object | None = None,
    mcp_tools: dict[str, Tool] | None = None,
    search: Callable[[str, int], list[PaperCandidate]] | None = None,
    event_sink: Callable[[str, dict[str, object]], None] | None = None,
    store: object | None = None,
) -> tuple[DeepReadingContext, _AgentFactory, list[tuple[str, dict[str, object]]]]:
    tools, mcp_calls = _mcp_tools() if mcp_tools is None else (mcp_tools, [])
    events: list[tuple[str, dict[str, object]]] = []
    factory = _AgentFactory(behavior)
    context = DeepReadingContext(
        user_id="user-1",
        conversation_id="conversation-1",
        task_id="task-1",
        current_user_message_id="message-1",
        base_checkpoint_id="checkpoint-base",
        task_store=store or _Store(),  # type: ignore[arg-type]
        model=object() if model is None else model,
        mcp_tools=tools,
        paper_search=search or (lambda _query, _limit: [RELATED, UNUSED]),
        event_sink=event_sink
        or (lambda kind, payload: events.append((kind, payload))),
    )
    return context, factory, mcp_calls


STATE = {
    "messages": [HumanMessage(content="Compare the methods", id="human-1")],
    "conversation_summary": None,
    "primary_paper_id": "paper-primary",
    "active_paper_ids": ["paper-primary", "paper-active"],
}


def test_agent_uses_exact_tools_budget_and_authoritative_selected_result() -> None:
    selected_ids: list[str] = []

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        found = tools["search_related_papers"].invoke(
            {"query": "related method", "limit": 2}
        )
        assert [item["external_id"] for item in found] == [
            RELATED.external_id,
            UNUSED.external_id,
        ]
        tools["prepare_paper"].invoke({"external_id": RELATED.external_id})
        retrieved = tools["retrieve_paper_evidence"].invoke(
            {
                "question": "How do the methods differ?",
                "external_id": RELATED.external_id,
                "top_k_each": 3,
                "summary_k": 2,
            }
        )
        assert retrieved["evidence_items"][0]["paper_external_id"] == (
            RELATED.external_id
        )
        evidence_id = retrieved["evidence_items"][0]["id"]
        assert evidence_id != "ev_1"
        selected_ids.append(evidence_id)
        return {
            "structured_response": {
                "selected_evidence_ids": [evidence_id],
                "paper_uses": [
                    {
                        "external_id": RELATED.external_id,
                        "role": "comparison",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "limitations": ["Only one related paper was compared."],
            }
        }

    context, factory, mcp_calls = _context(behavior)

    result = run_research_agent(
        STATE, context, create_agent_factory=factory
    )

    assert [item.id for item in result.evidence_items] == selected_ids
    assert [item.paper.external_id for item in result.used_papers] == [
        RELATED.external_id
    ]
    assert UNUSED.external_id not in {
        item.paper.external_id for item in result.used_papers
    }
    assert result.limitations == ["Only one related paper was compared."]
    assert len(factory.calls) == 1
    create_call = factory.calls[0]
    assert create_call["model"] is context.model
    assert [tool.name for tool in create_call["tools"]] == [  # type: ignore[union-attr]
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
    ]
    response_format = create_call["response_format"]
    assert isinstance(response_format, ToolStrategy)
    assert response_format.schema is AgentResearchDecision
    assert response_format.handle_errors is False
    assert factory.agent is not None
    assert factory.agent.invocations[0][1] == {"recursion_limit": 24}
    assert mcp_calls == [
        (
            "mcp__arxiv__download_paper",
            {"arxiv_id": RELATED.external_id},
        ),
        (
            "mcp__colbert__build_index",
            {
                "documents": [
                    {
                        "paper_id": RELATED.external_id,
                        "text": "downloaded paper text",
                    }
                ]
            },
        ),
        (
            "mcp__colbert__planned_retrieval",
            {
                "question": "How do the methods differ?",
                "paper_id": RELATED.external_id,
                "paper_title": RELATED.title,
                "abstract": RELATED.abstract,
                "top_k_each": 3,
                "summary_k": 2,
            },
        ),
    ]


def test_real_agent_completes_three_tool_chain_in_exactly_four_model_calls() -> None:
    model = _ScriptedResearchChatModel()
    context, _factory, mcp_calls = _context(
        lambda _tools, _attempt: None,
        model=model,
        search=lambda _query, _limit: [RELATED],
    )

    result = run_research_agent(STATE, context)

    assert model.invoke_count == 4
    assert model.tool_trace == [
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
        AgentResearchDecision.__name__,
    ]
    assert model.bound_tool_names == [
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
        AgentResearchDecision.__name__,
    ]
    assert [name for name, _arguments in mcp_calls] == [
        "mcp__arxiv__download_paper",
        "mcp__colbert__build_index",
        "mcp__colbert__planned_retrieval",
    ]
    assert len(result.evidence_items) == 1
    assert result.evidence_items[0].chunk_text == f"Evidence from {RELATED.external_id}"
    assert result.used_papers[0].paper == RELATED
    assert result.limitations == ["Only one related paper was compared."]


def test_real_agent_rejects_old_three_model_call_attempt_budget() -> None:
    model = _ScriptedResearchChatModel()
    context, _factory, _mcp_calls = _context(
        lambda _tools, _attempt: None,
        model=model,
        search=lambda _query, _limit: [RELATED],
    )
    old_budget_context = replace(
        context,
        research_model_call_limit=6,
        research_recursion_limit=24,
    )

    with pytest.raises(DeepReadingTaskError) as exc_info:
        run_research_agent(STATE, old_budget_context)

    assert exc_info.value.error_code == "agent_budget_exhausted"
    assert isinstance(exc_info.value.__cause__, ModelCallLimitExceededError)
    assert model.invoke_count == 3
    assert model.tool_trace == [
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
    ]


def test_agent_installs_official_per_attempt_limits_and_model_retry() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        return {
            "structured_response": {
                "selected_evidence_ids": [],
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _calls = _context(behavior)

    run_research_agent(STATE, context, create_agent_factory=factory)

    middleware = factory.calls[0]["middleware"]
    assert isinstance(middleware, list)
    assert len(middleware) == 3
    model_limit, tool_limit, model_retry = middleware
    assert isinstance(model_limit, ModelCallLimitMiddleware)
    assert model_limit.thread_limit is None
    assert model_limit.run_limit == 4
    assert model_limit.exit_behavior == "error"
    assert isinstance(tool_limit, ToolCallLimitMiddleware)
    assert tool_limit.tool_name is None
    assert tool_limit.thread_limit is None
    assert tool_limit.run_limit == 6
    assert tool_limit.exit_behavior == "error"
    assert isinstance(model_retry, ModelRetryMiddleware)
    assert model_retry.max_retries == 1
    assert model_retry.on_failure == "error"


def test_official_model_retry_exhaustion_preserves_provider_exception() -> None:
    provider_error = ConnectionError("provider connection reset")
    attempts = 0
    middleware = ModelRetryMiddleware(
        max_retries=1,
        on_failure="error",
        initial_delay=0,
        jitter=False,
    )

    def failing_handler(_request):
        nonlocal attempts
        attempts += 1
        raise provider_error

    with pytest.raises(ConnectionError) as exc_info:
        middleware.wrap_model_call(object(), failing_handler)

    assert exc_info.value is provider_error
    assert attempts == 2


def test_two_papers_with_pool_local_ev_1_receive_distinct_global_ids() -> None:
    selected_ids: list[str] = []

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        found = tools["search_related_papers"].invoke(
            {"query": "two papers", "limit": 2}
        )
        paper_uses: list[dict[str, object]] = []
        for candidate in found:
            external_id = candidate["external_id"]
            tools["prepare_paper"].invoke({"external_id": external_id})
            retrieved = tools["retrieve_paper_evidence"].invoke(
                {
                    "question": "compare",
                    "external_id": external_id,
                    "top_k_each": 2,
                    "summary_k": 2,
                }
            )
            evidence_id = retrieved["evidence_items"][0]["id"]
            assert retrieved["summary_item_ids"] == [evidence_id]
            selected_ids.append(evidence_id)
            paper_uses.append(
                {
                    "external_id": external_id,
                    "role": "comparison",
                    "evidence_ids": [evidence_id],
                }
            )
        return {
            "structured_response": {
                "selected_evidence_ids": selected_ids,
                "paper_uses": paper_uses,
                "limitations": [],
            }
        }

    context, factory, _calls = _context(behavior)

    result = run_research_agent(STATE, context, create_agent_factory=factory)

    assert len(selected_ids) == 2
    assert len(set(selected_ids)) == 2
    assert "ev_1" not in selected_ids
    assert [item.id for item in result.evidence_items] == selected_ids
    assert {item.paper_external_id for item in result.evidence_items} == {
        RELATED.external_id,
        UNUSED.external_id,
    }


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [(5.0, 0.5), (10.0, 1.0), (30.0, 1.0)],
)
def test_colbert_scores_are_normalized_to_schema_range(
    raw_score: float,
    expected: float,
) -> None:
    def retrieval(arguments: dict) -> str:
        return json.dumps(
            _evidence_payload(
                arguments["paper_id"],
                "ev_1",
                best_score=raw_score,
            )
        )

    mcp_tools, _calls = _mcp_tools(retrieval_result=retrieval)

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
        retrieved = tools["retrieve_paper_evidence"].invoke(
            {
                "question": "question",
                "external_id": PRIMARY.external_id,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        assert retrieved["evidence_items"][0]["score"] == expected
        return {
            "structured_response": {
                "selected_evidence_ids": [retrieved["evidence_items"][0]["id"]],
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _unused = _context(behavior, mcp_tools=mcp_tools)

    result = run_research_agent(STATE, context, create_agent_factory=factory)

    assert result.evidence_items[0].score == expected


@pytest.mark.parametrize("raw_score", [-0.1, float("inf"), float("nan")])
def test_colbert_scores_reject_negative_or_nonfinite_values(raw_score: float) -> None:
    payload = _evidence_payload(
        PRIMARY.external_id,
        "ev_1",
        best_score=raw_score,
    )
    mcp_tools, _calls = _mcp_tools(retrieval_result=json.dumps(payload))

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
        tools["retrieve_paper_evidence"].invoke(
            {
                "question": "question",
                "external_id": PRIMARY.external_id,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        raise AssertionError("invalid score should have raised")

    context, factory, _unused = _context(behavior, mcp_tools=mcp_tools)

    with pytest.raises(ResearchContractError, match="score"):
        run_research_agent(STATE, context, create_agent_factory=factory)


def test_prepare_guard_accepts_trusted_primary_active_and_searched_ids() -> None:
    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
        tools["prepare_paper"].invoke({"external_id": ACTIVE.external_id})
        tools["search_related_papers"].invoke({"query": "related", "limit": 1})
        tools["prepare_paper"].invoke({"external_id": RELATED.external_id})
        return {
            "structured_response": {
                "selected_evidence_ids": [],
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _mcp_calls = _context(
        behavior,
        search=lambda _query, _limit: [RELATED],
    )

    result = run_research_agent(STATE, context, create_agent_factory=factory)

    assert result.evidence_items == []
    assert result.used_papers == []


def test_search_keeps_trusted_metadata_for_an_already_active_paper() -> None:
    richer_active = ACTIVE.model_copy(
        update={"pdf_url": f"https://arxiv.org/pdf/{ACTIVE.external_id}.pdf"}
    )

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        found = tools["search_related_papers"].invoke(
            {"query": "active", "limit": 1}
        )
        assert found[0]["external_id"] == ACTIVE.external_id
        assert found[0]["pdf_url"] is None
        return {
            "structured_response": {
                "selected_evidence_ids": [],
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _calls = _context(
        behavior,
        search=lambda _query, _limit: [richer_active],
    )

    run_research_agent(STATE, context, create_agent_factory=factory)


def test_tool_events_are_bounded_and_do_not_include_downloaded_text() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
        retrieved = tools["retrieve_paper_evidence"].invoke(
            {
                "question": "question",
                "external_id": PRIMARY.external_id,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        return {
            "structured_response": {
                "selected_evidence_ids": [retrieved["evidence_items"][0]["id"]],
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _calls = _context(
        behavior,
        event_sink=lambda kind, payload: events.append((kind, payload)),
    )

    run_research_agent(STATE, context, create_agent_factory=factory)

    assert {kind for kind, _payload in events} == {"tool_call", "tool_result"}
    assert {payload["stage"] for _kind, payload in events} == {
        "prepare",
        "research",
    }
    rendered = json.dumps(events, ensure_ascii=False)
    assert "downloaded paper text" not in rendered
    assert max(len(json.dumps(payload)) for _kind, payload in events) <= 800


def test_prepare_guard_rejects_untrusted_external_id_before_mcp_call() -> None:
    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": "2401.99999v1"})
        raise AssertionError("guard should have raised")

    context, factory, mcp_calls = _context(behavior)

    with pytest.raises(ResearchContractError, match="not allowed"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert mcp_calls == []


def test_model_arxiv_url_is_normalized_before_downloader_call() -> None:
    primary_url = f"https://arxiv.org/abs/{PRIMARY.external_id}"
    mcp_tools, mcp_calls = _mcp_tools(
        download_result=json.dumps(
            {
                "paper_id": primary_url,
                "text": "downloaded paper text",
                "untrusted_extra": "must not reach build_index",
            }
        )
    )

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        prepared = tools["prepare_paper"].invoke({"external_id": primary_url})
        assert prepared["external_id"] == PRIMARY.external_id
        retrieved = tools["retrieve_paper_evidence"].invoke(
            {
                "question": "question",
                "external_id": primary_url,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        return {
            "structured_response": {
                "selected_evidence_ids": [retrieved["evidence_items"][0]["id"]],
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _unused = _context(behavior, mcp_tools=mcp_tools)

    run_research_agent(STATE, context, create_agent_factory=factory)

    assert mcp_calls == [
        (
            "mcp__arxiv__download_paper",
            {"arxiv_id": PRIMARY.external_id},
        ),
        (
            "mcp__colbert__build_index",
            {
                "documents": [
                    {
                        "paper_id": PRIMARY.external_id,
                        "text": "downloaded paper text",
                    }
                ]
            },
        ),
        (
            "mcp__colbert__planned_retrieval",
            {
                "question": "question",
                "paper_id": PRIMARY.external_id,
                "paper_title": PRIMARY.title,
                "abstract": PRIMARY.abstract,
                "top_k_each": 2,
                "summary_k": 2,
            },
        ),
    ]


def test_model_invalid_external_id_never_reaches_downloader() -> None:
    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": "../../outside"})
        raise AssertionError("invalid arXiv ID should have raised")

    context, factory, mcp_calls = _context(behavior)

    with pytest.raises(ResearchContractError, match="valid arXiv"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert mcp_calls == []


def test_invalid_trusted_database_external_id_is_rejected_before_agent() -> None:
    store = _Store()
    store.detail.primary_paper.external_id = "../../outside"

    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        raise AssertionError("agent should not be created for invalid DB metadata")

    context, factory, mcp_calls = _context(behavior, store=store)

    with pytest.raises(ResearchContractError, match="valid arXiv"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert factory.agent is None
    assert mcp_calls == []


def test_trusted_database_source_mismatch_is_rejected_before_agent() -> None:
    store = _Store()
    store.detail.primary_paper.source = "local"

    context, factory, mcp_calls = _context(
        lambda _tools, _attempt: {},
        store=store,
    )

    with pytest.raises(ResearchContractError, match="source"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert factory.agent is None
    assert mcp_calls == []


@pytest.mark.parametrize(
    "bad_candidate",
    [
        RELATED.model_copy(update={"external_id": "../../outside"}),
        {
            **RELATED.model_dump(mode="json"),
            "source": "local",
        },
    ],
)
def test_invalid_search_identity_never_reaches_downloader(
    bad_candidate: object,
) -> None:
    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["search_related_papers"].invoke({"query": "unsafe", "limit": 1})
        raise AssertionError("invalid search identity should have raised")

    context, factory, mcp_calls = _context(
        behavior,
        search=lambda _query, _limit: [bad_candidate],  # type: ignore[list-item]
    )

    with pytest.raises(ResearchContractError, match="arXiv|source"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert mcp_calls == []


def test_retrieval_guard_rejects_paper_that_has_not_been_prepared() -> None:
    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["retrieve_paper_evidence"].invoke(
            {
                "question": "question",
                "external_id": PRIMARY.external_id,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        raise AssertionError("guard should have raised")

    context, factory, mcp_calls = _context(behavior)

    with pytest.raises(ResearchContractError, match="not prepared"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert mcp_calls == []


@pytest.mark.parametrize(
    ("retrieval_result", "message"),
    [
        ("not json", "valid JSON"),
        (json.dumps({"summary_text": "missing"}), "evidence_pool"),
        (
            json.dumps(_evidence_payload("2401.99999v1", "ev-wrong")),
            "paper ID",
        ),
    ],
)
def test_retrieval_rejects_malformed_or_mismatched_mcp_payloads(
    retrieval_result: object,
    message: str,
) -> None:
    mcp_tools, mcp_calls = _mcp_tools(retrieval_result=retrieval_result)

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
        tools["retrieve_paper_evidence"].invoke(
            {
                "question": "question",
                "external_id": PRIMARY.external_id,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        raise AssertionError("payload validation should have raised")

    context, factory, _unused = _context(behavior, mcp_tools=mcp_tools)

    with pytest.raises(ResearchContractError, match=message):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert mcp_calls[-1][0] == "mcp__colbert__planned_retrieval"


def test_prepare_rejects_non_json_and_download_paper_id_mismatch() -> None:
    for bad_download, message in (
        ("not json", "valid JSON"),
        (
            json.dumps({"paper_id": "2401.99999v1", "text": "paper"}),
            "paper ID",
        ),
    ):
        mcp_tools, _calls = _mcp_tools(download_result=bad_download)

        def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
            tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
            raise AssertionError("payload validation should have raised")

        context, factory, _unused = _context(behavior, mcp_tools=mcp_tools)
        with pytest.raises(ResearchContractError, match=message):
            run_research_agent(STATE, context, create_agent_factory=factory)


@pytest.mark.parametrize(
    ("pool_change", "message"),
    [
        (
            lambda pool: pool["items"].append(dict(pool["items"][0])),
            "duplicate evidence ID",
        ),
        (
            lambda pool: pool["summary_items"].append("ev-unknown"),
            "unknown ID",
        ),
    ],
)
def test_evidence_pool_rejects_duplicate_or_dangling_ids(
    pool_change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    retrieval_payload = _evidence_payload(
        PRIMARY.external_id,
        f"ev-{PRIMARY.external_id}",
    )
    pool = retrieval_payload["evidence_pool"]
    assert isinstance(pool, dict)
    pool_change(pool)
    mcp_tools, _calls = _mcp_tools(
        retrieval_result=json.dumps(retrieval_payload)
    )

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
        tools["retrieve_paper_evidence"].invoke(
            {
                "question": "question",
                "external_id": PRIMARY.external_id,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        raise AssertionError("evidence-pool validation should have raised")

    context, factory, _unused = _context(behavior, mcp_tools=mcp_tools)

    with pytest.raises(ResearchContractError, match=message):
        run_research_agent(STATE, context, create_agent_factory=factory)


@pytest.mark.parametrize(
    "decision",
    [
        {
            "selected_evidence_ids": ["ev-hallucinated"],
            "paper_uses": [],
            "limitations": [],
        },
        {
            "selected_evidence_ids": [
                f"ev-{PRIMARY.external_id}",
                f"ev-{PRIMARY.external_id}",
            ],
            "paper_uses": [],
            "limitations": [],
        },
        {
            "selected_evidence_ids": [f"ev-{PRIMARY.external_id}"],
            "paper_uses": [
                {
                    "external_id": ACTIVE.external_id,
                    "role": "comparison",
                    "evidence_ids": [f"ev-{PRIMARY.external_id}"],
                }
            ],
            "limitations": [],
        },
    ],
)
def test_authoritative_ledger_rejects_hallucinated_duplicate_or_mismatched_ids(
    decision: dict[str, object],
) -> None:
    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
        tools["retrieve_paper_evidence"].invoke(
            {
                "question": "question",
                "external_id": PRIMARY.external_id,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        return {"structured_response": decision}

    context, factory, _calls = _context(behavior)

    with pytest.raises(ResearchContractError):
        run_research_agent(STATE, context, create_agent_factory=factory)


def test_invalid_structured_response_is_attempted_at_most_twice() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        return {"structured_response": {"selected_evidence_ids": []}}

    context, factory, _calls = _context(behavior)

    with pytest.raises(DeepReadingTaskError, match="structured response") as exc_info:
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert factory.agent is not None
    assert len(factory.agent.invocations) == 2
    assert exc_info.value.error_code == "agent_budget_exhausted"
    middleware = factory.calls[0]["middleware"]
    model_limit = next(
        item for item in middleware if isinstance(item, ModelCallLimitMiddleware)
    )
    tool_limit = next(
        item for item in middleware if isinstance(item, ToolCallLimitMiddleware)
    )
    assert len(factory.agent.invocations) * model_limit.run_limit <= 8
    assert len(factory.agent.invocations) * tool_limit.run_limit <= 12


def test_second_structured_attempt_can_idempotently_repeat_retrieval() -> None:
    retrieved_ids: list[str] = []

    def behavior(tools: Mapping[str, Any], attempt: int) -> object:
        tools["search_related_papers"].invoke({"query": "related", "limit": 1})
        tools["prepare_paper"].invoke({"external_id": RELATED.external_id})
        retrieved = tools["retrieve_paper_evidence"].invoke(
            {
                "question": "compare",
                "external_id": RELATED.external_id,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        evidence_id = retrieved["evidence_items"][0]["id"]
        retrieved_ids.append(evidence_id)
        if attempt == 1:
            return {"structured_response": {"selected_evidence_ids": []}}
        return {
            "structured_response": {
                "selected_evidence_ids": [evidence_id],
                "paper_uses": [
                    {
                        "external_id": RELATED.external_id,
                        "role": "comparison",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "limitations": [],
            }
        }

    context, factory, mcp_calls = _context(
        behavior,
        search=lambda _query, _limit: [RELATED],
    )

    result = run_research_agent(STATE, context, create_agent_factory=factory)

    assert factory.agent is not None
    assert len(factory.agent.invocations) == 2
    assert retrieved_ids[0] == retrieved_ids[1]
    assert [item.id for item in result.evidence_items] == [retrieved_ids[0]]
    assert sum(
        name == "mcp__colbert__planned_retrieval" for name, _args in mcp_calls
    ) == 2


def test_same_paper_local_ev_1_is_scoped_by_retrieval_request() -> None:
    def retrieval(arguments: dict) -> str:
        payload = _evidence_payload(arguments["paper_id"], "ev_1")
        pool = payload["evidence_pool"]
        assert isinstance(pool, dict)
        pool["items"][0]["chunk_text"] = f"evidence for {arguments['question']}"
        return json.dumps(payload)

    mcp_tools, _calls = _mcp_tools(retrieval_result=retrieval)
    selected_ids: list[str] = []

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
        for question in ("What is the method?", "What are the results?"):
            retrieved = tools["retrieve_paper_evidence"].invoke(
                {
                    "question": question,
                    "external_id": PRIMARY.external_id,
                    "top_k_each": 2,
                    "summary_k": 2,
                }
            )
            evidence_id = retrieved["evidence_items"][0]["id"]
            assert retrieved["summary_item_ids"] == [evidence_id]
            selected_ids.append(evidence_id)
        return {
            "structured_response": {
                "selected_evidence_ids": selected_ids,
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _unused = _context(behavior, mcp_tools=mcp_tools)

    result = run_research_agent(STATE, context, create_agent_factory=factory)

    assert len(set(selected_ids)) == 2
    assert [item.id for item in result.evidence_items] == selected_ids
    assert [item.chunk_text for item in result.evidence_items] == [
        "evidence for What is the method?",
        "evidence for What are the results?",
    ]


def test_repeated_global_evidence_id_rejects_conflicting_content() -> None:
    retrieval_count = 0

    def retrieval(arguments: dict) -> str:
        nonlocal retrieval_count
        retrieval_count += 1
        payload = _evidence_payload(arguments["paper_id"], "ev_1")
        if retrieval_count == 2:
            pool = payload["evidence_pool"]
            assert isinstance(pool, dict)
            pool["items"][0]["chunk_text"] = "conflicting evidence text"
        return json.dumps(payload)

    mcp_tools, _calls = _mcp_tools(retrieval_result=retrieval)

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
        for _index in range(2):
            tools["retrieve_paper_evidence"].invoke(
                {
                    "question": "question",
                    "external_id": PRIMARY.external_id,
                    "top_k_each": 2,
                    "summary_k": 2,
                }
            )
        raise AssertionError("conflicting replay should have raised")

    context, factory, _unused = _context(behavior, mcp_tools=mcp_tools)

    with pytest.raises(ResearchContractError, match="conflicting evidence"):
        run_research_agent(STATE, context, create_agent_factory=factory)


def test_structured_output_error_is_retried_once_at_outer_boundary() -> None:
    def behavior(_tools: Mapping[str, Any], attempt: int) -> object:
        if attempt == 1:
            raise StructuredOutputError("invalid structured tool output")
        return {
            "structured_response": {
                "selected_evidence_ids": [],
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _calls = _context(behavior)

    result = run_research_agent(STATE, context, create_agent_factory=factory)

    assert result.evidence_items == []
    assert factory.agent is not None
    assert len(factory.agent.invocations) == 2


def test_structured_output_error_stops_after_two_agent_invocations() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        raise StructuredOutputError("invalid structured tool output")

    context, factory, _calls = _context(behavior)

    with pytest.raises(DeepReadingTaskError, match="structured response"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert factory.agent is not None
    assert len(factory.agent.invocations) == 2


def test_graph_recursion_exhaustion_becomes_deep_reading_task_error() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        raise GraphRecursionError("recursion limit reached")

    context, factory, _calls = _context(behavior)

    with pytest.raises(DeepReadingTaskError, match="budget exhausted") as exc_info:
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert exc_info.value.error_code == "agent_budget_exhausted"
    assert "recursion limit reached" not in exc_info.value.public_message
    assert factory.agent is not None
    assert len(factory.agent.invocations) == 1


@pytest.mark.parametrize(
    "budget_error",
    [
        ModelCallLimitExceededError(
            thread_count=0,
            run_count=3,
            thread_limit=None,
            run_limit=3,
        ),
        ToolCallLimitExceededError(
            thread_count=0,
            run_count=7,
            thread_limit=None,
            run_limit=6,
        ),
    ],
)
def test_official_call_limit_errors_become_safe_terminal_budget_error(
    budget_error,
) -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        raise budget_error

    context, factory, _calls = _context(behavior)

    with pytest.raises(DeepReadingTaskError) as exc_info:
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert exc_info.value.error_code == "agent_budget_exhausted"
    assert exc_info.value.public_message == (
        "The research agent exhausted its bounded execution budget."
    )
    assert str(budget_error) not in exc_info.value.public_message
    assert factory.agent is not None
    assert len(factory.agent.invocations) == 1


def test_unrelated_contract_error_is_not_retried_or_wrapped() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        raise ResearchContractError("bad MCP contract")

    context, factory, _calls = _context(behavior)

    with pytest.raises(ResearchContractError, match="bad MCP contract"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert factory.agent is not None
    assert len(factory.agent.invocations) == 1


def test_provider_error_is_preserved_for_outer_task_retry() -> None:
    provider_error = ConnectionError("provider connection reset")

    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        raise provider_error

    context, factory, _calls = _context(behavior)

    with pytest.raises(ConnectionError) as exc_info:
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert exc_info.value is provider_error
    assert factory.agent is not None
    assert len(factory.agent.invocations) == 1


def test_transport_error_is_not_retried_or_wrapped() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        raise MCPTransportError("connection lost")

    context, factory, _calls = _context(behavior)

    with pytest.raises(MCPTransportError, match="connection lost"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert factory.agent is not None
    assert len(factory.agent.invocations) == 1


def test_real_tool_node_converts_mcp_tool_error_to_contract_failure() -> None:
    failure = MCPToolError("remote-secret full-paper-text")

    def fail_download(_arguments: dict[str, object]) -> str:
        raise failure

    mcp_tools, calls = _mcp_tools(download_result=fail_download)
    model = _ScriptedResearchChatModel(script="prepare_only")
    context, _factory, _unused = _context(
        lambda _tools, _attempt: None,
        model=model,
        mcp_tools=mcp_tools,
    )

    with pytest.raises(ResearchContractError) as exc_info:
        run_research_agent(STATE, context)

    assert exc_info.value.__cause__ is failure
    assert "remote-secret" not in str(exc_info.value)
    assert "full-paper-text" not in str(exc_info.value)
    assert model.invoke_count == 1
    assert model.tool_trace == ["prepare_paper"]
    assert calls == [
        ("mcp__arxiv__download_paper", {"arxiv_id": PRIMARY.external_id})
    ]


@pytest.mark.parametrize(
    "failure",
    [
        MCPToolTimeout("tool timed out"),
        MCPTransportError("transport closed"),
    ],
    ids=["timeout", "transport"],
)
def test_real_tool_node_preserves_transient_mcp_failure_identity(failure) -> None:
    def fail_download(_arguments: dict[str, object]) -> str:
        raise failure

    mcp_tools, calls = _mcp_tools(download_result=fail_download)
    model = _ScriptedResearchChatModel(script="prepare_only")
    context, _factory, _unused = _context(
        lambda _tools, _attempt: None,
        model=model,
        mcp_tools=mcp_tools,
    )

    with pytest.raises(type(failure)) as exc_info:
        run_research_agent(STATE, context)

    assert exc_info.value is failure
    assert model.invoke_count == 1
    assert model.tool_trace == ["prepare_paper"]
    assert calls == [
        ("mcp__arxiv__download_paper", {"arxiv_id": PRIMARY.external_id})
    ]
