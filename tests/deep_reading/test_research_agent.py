"""Bounded LangChain research-agent contracts without network or model calls."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Sequence
from xml.etree import ElementTree

import pytest
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
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

from paperpilot.deep_reading.nodes import DeepReadingContext
from paperpilot.deep_reading.context_management.editing import ResearchContextMiddleware
from paperpilot.deep_reading.research_agent import (
    _emit_tool_call,
    clear_tool_fingerprints,
    repeated_tool_calls,
    AgentResearchDecision,
    DeepReadingTaskError,
    ResearchContractError,
    _research_messages,
    run_research_agent,
)
from paperpilot.deep_reading.research_status import (
    ResearchStatusMiddleware,
    ResearchToolBudgetMiddleware,
    ResearchTodoMiddleware,
)
from paperpilot.web.config import ContextManagementConfig
from paperpilot.papers import PaperCandidate
from paperpilot.tools.mcp_client import (
    MCPToolError,
    MCPToolTimeout,
    MCPTransportError,
)
from paperpilot.tools.types import Tool


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
        attempt_number: int,
    ) -> None:
        self.tools = {item.name: item for item in tools}
        self.behavior = behavior
        self.attempt_number = attempt_number
        self.invocations: list[tuple[object, object]] = []

    def invoke(self, model_input: object, *, config: object) -> object:
        self.invocations.append((model_input, config))
        return self.behavior(self.tools, self.attempt_number)


class _AgentFactory:
    def __init__(
        self,
        behavior: Callable[[Mapping[str, Any], int], object],
    ) -> None:
        self.behavior = behavior
        self.calls: list[dict[str, object]] = []
        self.agent: _Agent | None = None
        self.agents: list[_Agent] = []

    def __call__(self, **kwargs: object) -> _Agent:
        self.calls.append(kwargs)
        self.agent = _Agent(
            list(kwargs["tools"]),
            self.behavior,
            len(self.agents) + 1,
        )  # type: ignore[arg-type]
        self.agents.append(self.agent)
        return self.agent


class _ScriptedResearchChatModel(BaseChatModel):
    """Drive LangChain's real Agent/ToolNode loop without provider I/O."""

    script: str = "full"
    _invoke_count: int = PrivateAttr(default=0)
    _tool_trace: list[str] = PrivateAttr(default_factory=list)
    _bound_tool_names: list[str] = PrivateAttr(default_factory=list)
    _received_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

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

    @property
    def received_messages(self) -> list[list[BaseMessage]]:
        return [list(messages) for messages in self._received_messages]

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
        self._received_messages.append(list(messages))
        if self.script == "budget_continuation":
            status = messages[-1]
            assert isinstance(status, HumanMessage)
            if ElementTree.fromstring(str(status.content)).attrib["attempt"] == "2":
                repair = next(
                    message
                    for message in messages
                    if isinstance(message, HumanMessage)
                    and message.additional_kwargs.get("paperpilot_source")
                    == "structured_response_repair"
                )
                evidence = ElementTree.fromstring(str(repair.content)).find(
                    "authoritative_ledger/evidence_items/evidence"
                )
                assert evidence is not None
                name = AgentResearchDecision.__name__
                self._tool_trace.append(name)
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": name,
                                        "args": {
                                            "selected_evidence_ids": [evidence.attrib["id"]],
                                            "paper_uses": [
                                                {
                                                    "external_id": evidence.attrib[
                                                        "paper_external_id"
                                                    ],
                                                    "role": "comparison",
                                                    "evidence_ids": [evidence.attrib["id"]],
                                                }
                                            ],
                                            "limitations": [],
                                        },
                                        "id": "budget-continuation-final",
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        )
                    ]
                )
        if self.script == "attempt_isolation":
            status = messages[-1]
            assert isinstance(status, HumanMessage)
            root = ElementTree.fromstring(str(status.content))
            attempt = int(root.attrib["attempt"])
            sequence = int(root.attrib["sequence"])
            if (attempt, sequence) == (1, 1):
                name = "search_related_papers"
                arguments = {
                    "query": "related method",
                    "limit": 1,
                }
            elif (attempt, sequence) == (1, 2):
                name = AgentResearchDecision.__name__
                arguments = {"selected_evidence_ids": []}
            elif (attempt, sequence) == (2, 1):
                name = AgentResearchDecision.__name__
                arguments = {
                    "selected_evidence_ids": [],
                    "paper_uses": [],
                    "limitations": ["No evidence selected."],
                }
            else:  # pragma: no cover - the scenario has only these logical calls
                raise AssertionError(
                    f"unexpected attempt/sequence pair: {(attempt, sequence)}"
                )
            self._tool_trace.append(name)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": name,
                                    "args": arguments,
                                    "id": f"attempt-isolation-{self._invoke_count}",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            )
        if self.script == "todo_and_final":
            final_arguments = {
                "selected_evidence_ids": [],
                "paper_uses": [],
                "limitations": ["No evidence selected."],
            }
            if self._invoke_count == 1:
                calls = [
                    {
                        "name": "write_todos",
                        "args": {
                            "todos": [
                                {
                                    "id": "todo_1",
                                    "content": "检索证据",
                                    "status": "in_progress",
                                }
                            ]
                        },
                        "id": "todo-and-final-todo",
                        "type": "tool_call",
                    },
                    {
                        "name": AgentResearchDecision.__name__,
                        "args": final_arguments,
                        "id": "todo-and-final-result",
                        "type": "tool_call",
                    },
                ]
                self._tool_trace.extend(call["name"] for call in calls)
            elif self._invoke_count == 2:
                rejected = [
                    message
                    for message in messages
                    if isinstance(message, ToolMessage)
                    and message.status == "error"
                    and message.tool_call_id
                    in {"todo-and-final-todo", "todo-and-final-result"}
                ]
                assert len(rejected) == 2
                calls = [
                    {
                        "name": AgentResearchDecision.__name__,
                        "args": final_arguments,
                        "id": "final-after-rejection",
                        "type": "tool_call",
                    }
                ]
                self._tool_trace.append(AgentResearchDecision.__name__)
            else:  # pragma: no cover - rejection must consume exactly one extra call
                raise AssertionError("same-turn rejection did not terminate on call two")
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content="", tool_calls=calls))
                ]
            )
        scripted_step = self._invoke_count
        if self.script == "eight_calls" and scripted_step >= 7:
            scripted_step -= 2
        if self.script == "prepare_only":
            name = "prepare_paper"
            arguments: dict[str, object] = {"external_id": PRIMARY.external_id}
        elif self.script == "eight_calls" and self._invoke_count in (5, 6):
            name = "search_related_papers"
            arguments = {"query": "additional related method", "limit": 1}
        elif scripted_step == 1:
            name = "write_todos"
            arguments = {
                "todos": [
                    {
                        "id": "todo_1",
                        "content": "检索并核对相关论文证据",
                        "status": "in_progress",
                    }
                ]
            }
        elif scripted_step == 2:
            last_tool = _last_tool_message(messages)
            assert last_tool.name == "write_todos"
            name = "search_related_papers"
            arguments = {"query": "related method", "limit": 1}
        elif scripted_step == 3:
            last_tool = _last_tool_message(messages)
            assert last_tool.name == "search_related_papers"
            found = json.loads(str(last_tool.content))
            assert found[0]["external_id"] == RELATED.external_id
            name = "prepare_paper"
            arguments = {"external_id": RELATED.external_id}
        elif scripted_step == 4:
            last_tool = _last_tool_message(messages)
            assert last_tool.name == "prepare_paper"
            name = "retrieve_paper_evidence"
            arguments = {
                "question": "How do the methods differ?",
                "external_id": RELATED.external_id,
                "top_k_each": 3,
                "summary_k": 2,
            }
        elif scripted_step == 5:
            last_tool = _last_tool_message(messages)
            assert last_tool.name == (
                "search_related_papers"
                if self.script == "eight_calls"
                else "retrieve_paper_evidence"
            )
            name = "write_todos"
            arguments = {
                "todos": [
                    {
                        "id": "todo_1",
                        "content": "检索并核对相关论文证据",
                        "status": "completed",
                    }
                ]
            }
        elif scripted_step == 6:
            retrieved_message = _last_tool_message_named(
                messages,
                "retrieve_paper_evidence",
            )
            retrieved = json.loads(str(retrieved_message.content))
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
        else:  # pragma: no cover - a bounded successful run must stop at call six
            raise AssertionError("real Agent made an unexpected seventh model call")

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


def _last_tool_message_named(messages: list[BaseMessage], name: str) -> ToolMessage:
    selected = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, ToolMessage) and message.name == name
        ),
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


def test_research_messages_use_stable_layered_prefix() -> None:
    summary = {
        "confirmed_facts": ["Fact"],
        "paper_findings": [],
        "comparison_context": [],
        "open_questions": ["Question"],
    }
    first = _research_messages(
        {
            "conversation_summary": summary,
            "messages": [HumanMessage(content="first question", id="h-1")],
        },
        primary_external_id=PRIMARY.external_id,
        active_external_ids=[
            RELATED.external_id,
            ACTIVE.external_id,
            PRIMARY.external_id,
            ACTIVE.external_id,
        ],
    )
    second = _research_messages(
        {
            "conversation_summary": summary,
            "messages": [HumanMessage(content="second question", id="h-2")],
        },
        primary_external_id=PRIMARY.external_id,
        active_external_ids=[
            ACTIVE.external_id,
            PRIMARY.external_id,
            RELATED.external_id,
        ],
    )

    assert [message.type for message in first] == [
        "system",
        "human",
        "human",
        "human",
    ]
    assert first[0].content == second[0].content
    assert PRIMARY.external_id not in first[0].content
    assert ACTIVE.external_id not in first[0].content
    assert first[1].content == second[1].content == (
        "PaperPilot Runtime Context:\n"
        '{"active_paper_external_ids":["2401.10001v1",'
        '"2401.10002v1","2401.10003v1"],'
        '"primary_paper_external_id":"2401.10001v1",'
        '"schema_version":"paperpilot-runtime-context-v1"}'
    )
    assert first[2].content.startswith(
        "PaperPilot Conversation Summary (model-generated):\n"
    )
    assert "Confirmed conversation summary" not in first[2].content
    assert first[-1].content == "first question"
    assert second[-1].content == "second question"


def test_research_system_message_uses_semantic_xml_sections() -> None:
    messages = _research_messages(
        {"messages": [HumanMessage(content="question", id="h-1")]},
        primary_external_id=PRIMARY.external_id,
        active_external_ids=[PRIMARY.external_id],
    )

    content = messages[0].content
    assert isinstance(content, str)
    root = ElementTree.fromstring(content)

    assert root.tag == "research_agent"
    assert [child.tag for child in root] == [
        "role",
        "objective",
        "user_memory",
        "definitions",
        "instruction_priority",
        "decision_policy",
        "evidence_acceptance",
        "coverage_policy",
        "conflict_policy",
        "paper_role_policy",
        "output_contract",
        "planning_and_status_policy",
        "trust_boundaries",
        "completion_criteria",
    ]
    decision_policy = root.find("decision_policy")
    assert decision_policy is not None
    assert [child.tag for child in decision_policy] == [
        "question_classification",
        "paper_scope",
        "search_policy",
        "preparation_policy",
        "retrieval_policy",
    ]
    trust_boundaries = root.find("trust_boundaries")
    assert trust_boundaries is not None
    assert [child.tag for child in trust_boundaries] == [
        "runtime_context",
        "conversation_summary",
        "conversation_history",
        "tool_results",
        "agent_status_bar",
        "todo_plan",
    ]
    assert "research-v6" not in content
    assert "<planning_and_status_policy>" in content
    assert "call `write_todos` before the first research" in content
    assert "do not call `write_todos` again merely to reword" in content
    assert "preserve every existing TODO" in content
    assert "`id` and `content`\n  exactly" in content


def test_research_messages_without_summary_keep_history_after_runtime() -> None:
    history = [HumanMessage(content="question", id="h-1")]

    messages = _research_messages(
        {"messages": history},
        primary_external_id=PRIMARY.external_id,
        active_external_ids=[PRIMARY.external_id],
    )

    assert [message.type for message in messages] == ["system", "human", "human"]
    assert messages[2] is history[0]


def test_research_messages_use_context_view_instead_of_legacy_summary_or_history() -> None:
    messages = _research_messages(
        {
            "conversation_summary": {"confirmed_facts": ["legacy"]},
            "context_view": {
                "active_projection": {"current_goal": "new goal"},
                "messages": [{"role": "user", "content": "new goal"}],
            },
            "messages": [HumanMessage(content="old history", id="old")],
        },
        primary_external_id=PRIMARY.external_id,
        active_external_ids=[PRIMARY.external_id],
    )

    assert len(messages) == 3
    assert "PaperPilot Context View:" in messages[-1].content
    assert "legacy" not in messages[-1].content
    assert "old history" not in messages[-1].content
    assert "new goal" in messages[-1].content


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
        "run_computation",
        "collect_evidence_parallel",
        "analyze_paper_page",
        "search_user_memory",
    ]
    response_format = create_call["response_format"]
    assert isinstance(response_format, ToolStrategy)
    assert response_format.schema is AgentResearchDecision
    assert response_format.handle_errors is False
    assert factory.agent is not None
    assert factory.agent.invocations[0][1] == {
        "recursion_limit": 33,
        "tags": ["paperpilot:model"],
        "metadata": {
            "paperpilot_stage": "research",
            "prompt_version": "research-v9",
        },
    }
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


def test_real_agent_completes_planned_three_tool_chain_in_exactly_six_model_calls() -> None:
    model = _ScriptedResearchChatModel()
    context, _factory, mcp_calls = _context(
        lambda _tools, _attempt: None,
        model=model,
        search=lambda _query, _limit: [RELATED],
    )

    context = replace(context, research_model_call_limit=12)
    result = run_research_agent(STATE, context)

    assert model.invoke_count == 6
    assert model.tool_trace == [
        "write_todos",
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
        "write_todos",
        AgentResearchDecision.__name__,
    ]
    assert "write_todos" in model.bound_tool_names
    assert "search_related_papers" in model.bound_tool_names
    assert "prepare_paper" in model.bound_tool_names
    assert "retrieve_paper_evidence" in model.bound_tool_names
    assert AgentResearchDecision.__name__ in model.bound_tool_names
    assert [name for name, _arguments in mcp_calls] == [
        "mcp__arxiv__download_paper",
        "mcp__colbert__build_index",
        "mcp__colbert__planned_retrieval",
    ]
    assert len(result.evidence_items) == 1
    assert result.evidence_items[0].chunk_text == f"Evidence from {RELATED.external_id}"
    assert result.used_papers[0].paper == RELATED
    assert result.limitations == ["Only one related paper was compared."]
    assert len(model.received_messages) == 6
    for sequence, current_messages in enumerate(model.received_messages, start=1):
        status = current_messages[-1]
        assert isinstance(status, HumanMessage)
        assert status.additional_kwargs == {"paperpilot_source": "agent_status_bar"}
        root = ElementTree.fromstring(str(status.content))
        assert root.attrib["attempt"] == "1"
        assert root.attrib["sequence"] == str(sequence)
        if sequence > 1:
            previous_messages = model.received_messages[sequence - 2]
            assert current_messages[: len(previous_messages)] == previous_messages


def test_real_agent_default_graph_budget_accommodates_eight_model_calls() -> None:
    model = _ScriptedResearchChatModel(script="eight_calls")
    context, _factory, _mcp_calls = _context(
        lambda _tools, _attempt: None,
        model=model,
        search=lambda _query, _limit: [RELATED],
    )

    result = run_research_agent(STATE, context)

    assert model.invoke_count == 8
    assert len(result.evidence_items) == 1
    assert model.tool_trace == [
        "write_todos",
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
        "search_related_papers",
        "search_related_papers",
        "write_todos",
        AgentResearchDecision.__name__,
    ]


def test_real_agent_rejects_write_todos_with_final_result_in_one_model_turn() -> None:
    model = _ScriptedResearchChatModel(script="todo_and_final")
    context, _factory, mcp_calls = _context(
        lambda _tools, _attempt: None,
        model=model,
    )

    result = run_research_agent(STATE, context)

    assert model.invoke_count == 2
    assert model.tool_trace == [
        "write_todos",
        AgentResearchDecision.__name__,
        AgentResearchDecision.__name__,
    ]
    assert mcp_calls == []
    assert result.evidence_items == []
    assert result.limitations == ["No evidence selected."]


def test_real_agent_uses_reserved_attempt_after_primary_model_limit() -> None:
    model = _ScriptedResearchChatModel(script="budget_continuation")
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

    result = run_research_agent(STATE, old_budget_context)

    assert len(result.evidence_items) == 1
    assert model.invoke_count == 5
    assert model.tool_trace == [
        "write_todos",
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
        AgentResearchDecision.__name__,
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

    context = replace(context, research_model_call_limit=12)
    run_research_agent(STATE, context, create_agent_factory=factory)

    middleware = factory.calls[0]["middleware"]
    assert isinstance(middleware, list)
    assert [type(item) for item in middleware] == [
        ModelCallLimitMiddleware,
        ResearchToolBudgetMiddleware,
        ResearchTodoMiddleware,
        ResearchStatusMiddleware,
        ModelRetryMiddleware,
    ]
    assert len(middleware) == 5
    model_limit, tool_budget, todo_middleware, status_middleware, model_retry = middleware
    assert isinstance(model_limit, ModelCallLimitMiddleware)
    assert model_limit.thread_limit is None
    assert model_limit.run_limit == 8
    assert model_limit.exit_behavior == "error"
    assert isinstance(tool_budget, ResearchToolBudgetMiddleware)
    assert tool_budget.run_limit == 8
    assert isinstance(todo_middleware, ResearchTodoMiddleware)
    assert isinstance(status_middleware, ResearchStatusMiddleware)
    assert isinstance(model_retry, ModelRetryMiddleware)
    assert model_retry.max_retries == 1
    assert model_retry.on_failure == "error"
    assert [tool.name for tool in factory.calls[0]["tools"]] == [  # type: ignore[union-attr]
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
        "run_computation",
        "collect_evidence_parallel",
        "analyze_paper_page",
        "search_user_memory",
    ]
    assert [tool.name for tool in todo_middleware.tools] == ["write_todos"]


def test_enabled_context_registers_read_tools_and_fixed_middleware_order() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        return {
            "structured_response": {
                "selected_evidence_ids": [],
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _calls = _context(behavior)
    context = replace(
        context,
        context_management=ContextManagementConfig(enabled=True),
    )
    run_research_agent(STATE, context, create_agent_factory=factory)
    middleware = factory.calls[0]["middleware"]
    assert [type(item) for item in middleware] == [
        ModelCallLimitMiddleware,
        ResearchToolBudgetMiddleware,
        ResearchTodoMiddleware,
        ResearchContextMiddleware,
        ResearchStatusMiddleware,
        ModelRetryMiddleware,
    ]
    assert [tool.name for tool in factory.calls[0]["tools"]] == [
        "search_related_papers",
        "prepare_paper",
        "retrieve_paper_evidence",
        "run_computation",
        "collect_evidence_parallel",
        "analyze_paper_page",
        "search_user_memory",
        "read_artifact_slice",
        "search_artifact",
    ]


def test_artifact_tools_clamp_model_requests_to_runtime_budgets() -> None:
    read_calls: list[dict[str, object]] = []
    search_calls: list[dict[str, object]] = []

    class ArtifactStore:
        def read_slice(self, artifact_id, *, conversation_id, cursor, max_tokens):
            read_calls.append(
                {
                    "artifact_id": artifact_id,
                    "conversation_id": conversation_id,
                    "cursor": cursor,
                    "max_tokens": max_tokens,
                }
            )
            return SimpleNamespace(
                artifact_id=artifact_id,
                text="bounded",
                next_cursor=None,
                actual_tokens=max_tokens,
                sha256="sha-read",
            )

        def search(self, artifact_id, *, conversation_id, query, max_matches):
            search_calls.append(
                {
                    "artifact_id": artifact_id,
                    "conversation_id": conversation_id,
                    "query": query,
                    "max_matches": max_matches,
                }
            )
            return SimpleNamespace(
                artifact_id=artifact_id,
                matches=[],
                actual_tokens=0,
                sha256="sha-search",
            )

    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        read_result = tools["read_artifact_slice"].invoke(
            {"artifact_id": "artifact-1", "cursor": -10, "max_tokens": 99_999}
        )
        search_result = tools["search_artifact"].invoke(
            {"artifact_id": "artifact-1", "query": "method", "max_matches": 999}
        )
        assert read_result["actual_tokens"] == 321
        assert search_result["matches"] == []
        return {
            "structured_response": {
                "selected_evidence_ids": [],
                "paper_uses": [],
                "limitations": ["no_direct_evidence: probe - bounded reads only"],
            }
        }

    context, factory, _calls = _context(behavior)
    context = replace(
        context,
        context_management=ContextManagementConfig(
            enabled=True,
            artifact_read_max_tokens=321,
        ),
        context_management_runtime=SimpleNamespace(artifact_store=ArtifactStore()),
    )

    run_research_agent(STATE, context, create_agent_factory=factory)

    assert read_calls == [
        {
            "artifact_id": "artifact-1",
            "conversation_id": "conversation-1",
            "cursor": 0,
            "max_tokens": 321,
        }
    ]
    assert search_calls == [
        {
            "artifact_id": "artifact-1",
            "conversation_id": "conversation-1",
            "query": "method",
            "max_matches": 10,
        }
    ]


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


def test_prepare_accepts_explicit_arxiv_id_not_in_catalog() -> None:
    """User-named explicit ids are legitimate preparation targets even when
    catalog/search never surfaced them (multipaper deep-research flows)."""
    context, factory, mcp_calls = _context(lambda tools, attempt: None)

    import paperpilot.deep_reading.research_agent as ra

    prepared: dict = {}
    tool = ra._build_prepare_tool(context, {}, prepared)
    result = tool.invoke({"external_id": "2401.99999v1"})
    assert result["external_id"] == "2401.99999v1"
    assert mcp_calls, "explicit id should reach the downloader"


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


def test_retrieval_guard_guides_prepare_first_for_unprepared_paper() -> None:
    def behavior(tools: Mapping[str, Any], _attempt: int) -> object:
        result = tools["retrieve_paper_evidence"].invoke(
            {
                "question": "question",
                "external_id": PRIMARY.external_id,
                "top_k_each": 2,
                "summary_k": 2,
            }
        )
        # The precondition breach is returned as correctable guidance, not a
        # fatal contract error, so the model can prepare and retry.
        assert "prepare_paper" in result["error"]

    context, factory, mcp_calls = _context(behavior)

    with pytest.raises(Exception):  # scripted model never returns a decision
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


def test_ledger_invalid_decision_gets_one_fresh_structured_attempt() -> None:
    retrieved_ids: list[str] = []

    def behavior(tools: Mapping[str, Any], attempt: int) -> object:
        if attempt == 1:
            tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
            retrieved = tools["retrieve_paper_evidence"].invoke(
                {
                    "question": "question",
                    "external_id": PRIMARY.external_id,
                    "top_k_each": 2,
                    "summary_k": 2,
                }
            )
            retrieved_ids.append(retrieved["evidence_items"][0]["id"])
            selected_ids = ["ev-hallucinated"]
        else:
            selected_ids = [retrieved_ids[0]]
        return {
            "structured_response": {
                "selected_evidence_ids": selected_ids,
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _calls = _context(behavior)

    outcome = run_research_agent(STATE, context, create_agent_factory=factory)

    assert [item.id for item in outcome.evidence_items] == retrieved_ids
    assert len(factory.agents) == 2
    second_input = factory.agents[1].invocations[0][0]
    repair = second_input["messages"][-1]
    assert isinstance(repair, HumanMessage)
    assert repair.additional_kwargs == {
        "paperpilot_source": "structured_response_repair"
    }
    assert retrieved_ids[0] in str(repair.content)
    assert PRIMARY.external_id in str(repair.content)
    assert "ResearchContractError" in str(repair.content)


@pytest.mark.parametrize(
    "budget_error",
    [
        GraphRecursionError("recursion limit reached"),
        ModelCallLimitExceededError(
            thread_count=0,
            run_count=9,
            thread_limit=None,
            run_limit=8,
        ),
        ToolCallLimitExceededError(
            thread_count=0,
            run_count=9,
            thread_limit=None,
            run_limit=8,
        ),
    ],
)
def test_budget_exhaustion_uses_reserved_attempt_with_authoritative_ledger(
    budget_error,
) -> None:
    retrieved_ids: list[str] = []

    def behavior(tools: Mapping[str, Any], attempt: int) -> object:
        if attempt == 1:
            tools["prepare_paper"].invoke({"external_id": PRIMARY.external_id})
            retrieved = tools["retrieve_paper_evidence"].invoke(
                {
                    "question": "question",
                    "external_id": PRIMARY.external_id,
                    "top_k_each": 2,
                    "summary_k": 2,
                }
            )
            retrieved_ids.append(retrieved["evidence_items"][0]["id"])
            raise budget_error
        return {
            "structured_response": {
                "selected_evidence_ids": [retrieved_ids[0]],
                "paper_uses": [],
                "limitations": [],
            }
        }

    context, factory, _calls = _context(behavior)

    outcome = run_research_agent(STATE, context, create_agent_factory=factory)

    assert [item.id for item in outcome.evidence_items] == retrieved_ids
    assert len(factory.agents) == 2
    second_input = factory.agents[1].invocations[0][0]
    continuation = second_input["messages"][-1]
    assert isinstance(continuation, HumanMessage)
    assert retrieved_ids[0] in str(continuation.content)
    assert "bounded continuation" in str(continuation.content)


def test_invalid_structured_response_is_attempted_at_most_twice() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        return {"structured_response": {"selected_evidence_ids": []}}

    context, factory, _calls = _context(behavior)

    with pytest.raises(DeepReadingTaskError, match="structured response") as exc_info:
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert len(factory.agents) == 2
    assert [len(agent.invocations) for agent in factory.agents] == [1, 1]
    assert exc_info.value.error_code == "agent_budget_exhausted"
    model_limits = [
        next(
            item.run_limit
            for item in call["middleware"]
            if isinstance(item, ModelCallLimitMiddleware)
        )
        for call in factory.calls
    ]
    tool_limits = [
        next(
            item.run_limit
            for item in call["middleware"]
            if isinstance(item, ResearchToolBudgetMiddleware)
        )
        for call in factory.calls
    ]
    assert model_limits == [8, 4]
    assert sum(model_limits) == 12
    assert tool_limits == [8, 4]
    assert sum(tool_limits) == 12


def test_structured_attempts_isolate_status_state_but_retain_business_ledgers() -> None:
    def behavior(tools: Mapping[str, Any], attempt: int) -> object:
        tools["search_related_papers"].invoke({"query": "related", "limit": 1})
        if attempt == 1:
            return {"structured_response": {"selected_evidence_ids": []}}
        return {
            "structured_response": {
                "selected_evidence_ids": [],
                "paper_uses": [],
                "limitations": ["No evidence selected."],
            }
        }

    context, factory, _calls = _context(
        behavior,
        search=lambda _query, _limit: [RELATED],
    )
    context = replace(context, research_model_call_limit=12)

    result = run_research_agent(STATE, context, create_agent_factory=factory)

    assert result.evidence_items == []
    assert len(factory.agents) == 2
    first_middleware = factory.calls[0]["middleware"]
    second_middleware = factory.calls[1]["middleware"]
    assert isinstance(first_middleware, list)
    assert isinstance(second_middleware, list)
    first_status = first_middleware[3]
    second_status = second_middleware[3]
    first_budget = first_middleware[1]
    second_budget = second_middleware[1]
    assert isinstance(first_status, ResearchStatusMiddleware)
    assert isinstance(second_status, ResearchStatusMiddleware)
    assert first_status is not second_status
    assert isinstance(first_budget, ResearchToolBudgetMiddleware)
    assert isinstance(second_budget, ResearchToolBudgetMiddleware)
    assert first_budget is not second_budget

    first_status.tracker.next_snapshot(
        [{"id": "todo_1", "content": "计划", "status": "in_progress"}],
        0,
        6,
        0,
        6,
    )
    assert first_status.tracker.sequence == 1
    assert second_status.tracker.sequence == 0
    second_snapshot = second_status.tracker.next_snapshot([], 0, 6, 0, 6)
    assert second_snapshot.attempt == 2
    assert second_snapshot.sequence == 1
    assert second_snapshot.mode == "unplanned"
    assert second_snapshot.ledger.candidate_ids
    assert second_snapshot.last_event is None
    assert second_snapshot.alerts == ()
    assert second_budget.used == 0


def test_real_structured_attempt_retry_resets_status_and_keeps_candidate_ledger() -> None:
    model = _ScriptedResearchChatModel(script="attempt_isolation")
    context, _factory, _mcp_calls = _context(
        lambda _tools, _attempt: None,
        model=model,
        search=lambda _query, _limit: [RELATED],
    )

    result = run_research_agent(STATE, context)

    assert result.evidence_items == []
    assert result.limitations == ["No evidence selected."]
    assert model.tool_trace == [
        "search_related_papers",
        AgentResearchDecision.__name__,
        AgentResearchDecision.__name__,
        AgentResearchDecision.__name__,
    ]
    received = model.received_messages
    roots = [ElementTree.fromstring(str(messages[-1].content)) for messages in received]
    assert [
        (root.attrib["attempt"], root.attrib["sequence"]) for root in roots
    ] == [("1", "1"), ("1", "2"), ("1", "2"), ("2", "1")]
    assert received[1][-1] is received[2][-1]

    first_candidate_count = int(
        roots[0].find("execution_state/candidate_papers").attrib["count"]
    )
    second_attempt_candidate_count = int(
        roots[-1].find("execution_state/candidate_papers").attrib["count"]
    )
    assert first_candidate_count == 2
    assert second_attempt_candidate_count == 3
    assert roots[-1].find("task_progress").attrib["mode"] == "unplanned"
    assert roots[-1].find("side_channel/last_event").attrib == {"available": "false"}
    second_attempt_statuses = [
        message
        for message in received[-1]
        if isinstance(message, HumanMessage)
        and message.additional_kwargs.get("paperpilot_source") == "agent_status_bar"
    ]
    assert len(second_attempt_statuses) == 1


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

    assert len(factory.agents) == 2
    assert [len(agent.invocations) for agent in factory.agents] == [1, 1]
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
        if retrieval_count % 2 == 0:
            # Every second replay returns conflicting content, so the
            # contract error fires on both the first attempt and the repair.
            pool = payload["evidence_pool"]
            assert isinstance(pool, dict)
            pool["items"][0]["chunk_text"] = "conflicting evidence text"
        return json.dumps(payload)

    mcp_tools, _calls = _mcp_tools(retrieval_result=retrieval)

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
        # The second call replays ev_1 with conflicting content and raises
        # ResearchContractError, which now routes into the repair attempt.
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
    assert len(factory.agents) == 2
    assert [len(agent.invocations) for agent in factory.agents] == [1, 1]


def test_structured_output_error_stops_after_two_agent_invocations() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        raise StructuredOutputError("invalid structured tool output")

    context, factory, _calls = _context(behavior)

    with pytest.raises(DeepReadingTaskError, match="structured response"):
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert len(factory.agents) == 2
    assert [len(agent.invocations) for agent in factory.agents] == [1, 1]


def test_graph_recursion_exhaustion_becomes_deep_reading_task_error() -> None:
    def behavior(_tools: Mapping[str, Any], _attempt: int) -> object:
        raise GraphRecursionError("recursion limit reached")

    context, factory, _calls = _context(behavior)

    with pytest.raises(DeepReadingTaskError, match="budget exhausted") as exc_info:
        run_research_agent(STATE, context, create_agent_factory=factory)

    assert exc_info.value.error_code == "agent_budget_exhausted"
    assert "recursion limit reached" not in exc_info.value.public_message
    assert factory.agent is not None
    assert len(factory.agents) == 2
    assert [len(agent.invocations) for agent in factory.agents] == [1, 1]


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
    assert len(factory.agents) == 2
    assert [len(agent.invocations) for agent in factory.agents] == [1, 1]


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
    # One repair attempt runs before the contract error becomes terminal;
    # the scripted behavior repeats on both attempts.
    assert model.invoke_count == 2
    assert model.tool_trace == ["prepare_paper", "prepare_paper"]
    assert calls == [
        ("mcp__arxiv__download_paper", {"arxiv_id": PRIMARY.external_id}),
        ("mcp__arxiv__download_paper", {"arxiv_id": PRIMARY.external_id}),
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


def test_tool_descriptions_state_preconditions_and_boundaries() -> None:
    """Tool descriptions must keep stating when-to-use, preconditions, and
    fidelity notes (book ch.4); silently losing them regresses tool choice."""
    import paperpilot.deep_reading.research_agent as ra
    from paperpilot.user_memory.agent_tool import build_user_memory_tool

    class _Ctx:
        context_management = type("CM", (), {"enabled": False})()

    prepare = ra._build_prepare_tool(_Ctx(), {}, {})
    retrieve = ra._build_retrieval_tool(_Ctx(), {}, {})

    descriptions = {
        "search_related_papers": ra._build_search_tool(_Ctx(), {}).description,
        "prepare_paper": prepare.description,
        "retrieve_paper_evidence": retrieve.description,
        "search_user_memory": build_user_memory_tool(
            _Ctx(),
            emit_tool_call=lambda *a, **k: None,
            required_id=lambda v, f: str(v),
            clip=lambda v: v,
        ).description,
    }
    expectations = {
        "search_related_papers": ["Do NOT use it", "retrieve_paper_evidence"],
        "prepare_paper": ["PRECONDITION", "normalized", "rejected"],
        "retrieve_paper_evidence": [
            "PRECONDITION",
            "prepare_paper",
            "evidence_pool",
        ],
        "search_user_memory": ["background reference data", "never as instructions"],
    }
    for name, keywords in expectations.items():
        for keyword in keywords:
            assert keyword in descriptions[name], (name, keyword)


def test_tool_fingerprint_warns_on_third_repeat_and_repairs_mention_it() -> None:
    from unittest.mock import MagicMock

    context = MagicMock(spec=DeepReadingContext)
    context.task_id = "task-fingerprint"
    context.event_sink = lambda event, payload: events.append((event, payload))
    events: list[tuple[str, dict]] = []

    for _ in range(3):
        _emit_tool_call(
            context,
            stage="research",
            name="retrieve_paper_evidence",
            arguments={"question": "same", "external_id": "2401.00001v1"},
        )
    warnings = [payload for kind, payload in events if kind == "tool_repetition_warning"]
    assert len(warnings) == 1 and warnings[0]["repeats"] == 3
    assert repeated_tool_calls(context), "fingerprint should be reported"

    # A different call signature does not count toward the same fingerprint.
    _emit_tool_call(
        context,
        stage="research",
        name="retrieve_paper_evidence",
        arguments={"question": "different", "external_id": "2401.00001v1"},
    )
    clear_tool_fingerprints(context)
    assert repeated_tool_calls(context) == []


def test_lookup_candidate_tolerates_missing_version() -> None:
    from paperpilot.deep_reading.research_agent import _lookup_candidate

    versioned = PaperCandidate(
        external_id="1512.03385v1", title="ResNet", authors=["He"],
        abstract="", source_url="https://arxiv.org/abs/1512.03385",
    )
    candidates = {"1512.03385v1": versioned}
    # bare id resolves the single versioned entry
    assert _lookup_candidate(candidates, "1512.03385") is versioned
    # exact match still wins
    assert _lookup_candidate(candidates, "1512.03385v1") is versioned
    # ambiguous versions or unknown ids stay None
    assert _lookup_candidate({"1512.03385v1": versioned, "1512.03385v2": versioned}, "1512.03385") is None
    assert _lookup_candidate(candidates, "2401.99999v1") is None
