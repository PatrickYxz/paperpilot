"""Fixed orchestration and checkpoint semantics for the deep-reading graph."""
from __future__ import annotations

from typing import Any

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import paperpilot.deep_reading.graph as graph_module
from paperpilot.deep_reading.graph import build_deep_reading_graph
from paperpilot.deep_reading.nodes import DeepReadingContext
from paperpilot.deep_reading.schemas import (
    AnswerCitation,
    AnswerDraft,
    EvidenceItem,
    PaperUse,
    ResearchResult,
)
from paperpilot.papers import PaperCandidate
from paperpilot.web.task_store import TaskStore


def _context(
    *,
    task_id: str,
    message_id: str,
    threshold: int,
    store: object | None = None,
    user_id: str = "user-1",
    conversation_id: str = "conv-two-turn",
    base_checkpoint_id: str | None = None,
) -> DeepReadingContext:
    return DeepReadingContext(
        user_id=user_id,
        conversation_id=conversation_id,
        task_id=task_id,
        current_user_message_id=message_id,
        base_checkpoint_id=base_checkpoint_id,
        task_store=store or object(),  # type: ignore[arg-type]
        model=object(),
        mcp_tools={},
        paper_search=lambda _query, _limit: [],
        event_sink=lambda _event, _payload: None,
        summary_token_threshold=threshold,
    )


def test_graph_follows_fixed_path_with_only_optional_summary(monkeypatch) -> None:
    for threshold, expected in (
        (
            32_000,
            [
                "initialize_turn",
                "prepare_primary_paper",
                "research_evidence",
                "write_answer",
                "publish_result",
            ],
        ),
        (
            1,
            [
                "initialize_turn",
                "summarize_history",
                "prepare_primary_paper",
                "research_evidence",
                "write_answer",
                "publish_result",
            ],
        ),
    ):
        seen: list[str] = []

        def fake_node(name: str):
            def run(state: object, runtime: object) -> dict[str, object]:
                del state, runtime
                seen.append(name)
                return {}

            return run

        monkeypatch.setattr(
            graph_module, "initialize_turn", fake_node("initialize_turn")
        )
        monkeypatch.setattr(
            graph_module, "summarize_history", fake_node("summarize_history")
        )
        monkeypatch.setattr(
            graph_module,
            "prepare_primary_paper",
            fake_node("prepare_primary_paper"),
        )
        monkeypatch.setattr(
            graph_module, "research_evidence", fake_node("research_evidence")
        )
        monkeypatch.setattr(graph_module, "write_answer", fake_node("write_answer"))
        monkeypatch.setattr(
            graph_module, "publish_result", fake_node("publish_result")
        )

        graph = build_deep_reading_graph(checkpointer=None)
        graph.invoke(
            {"messages": [HumanMessage(content="A sufficiently long question")]},
            context=_context(
                task_id=f"task-{threshold}",
                message_id=f"message-{threshold}",
                threshold=threshold,
            ),
        )

        assert seen == expected


def test_in_memory_saver_preserves_history_and_long_lived_state_across_two_turns(
    monkeypatch,
) -> None:
    """InMemorySaver is intentionally only a unit-test checkpointer here."""
    from paperpilot.deep_reading.nodes import initialize_turn as real_initialize_turn

    writer_states: list[dict[str, Any]] = []
    research_entries: list[dict[str, Any]] = []

    def fake_summary(state: dict[str, Any], runtime: object) -> dict[str, object]:
        del state, runtime
        return {
            "conversation_summary": {
                "confirmed_facts": ["first-turn fact"],
                "paper_findings": ["first-turn finding"],
                "comparison_context": [],
                "open_questions": ["follow-up"],
            }
        }

    def fake_prepare(state: dict[str, Any], runtime: object) -> dict[str, object]:
        del runtime
        return {
            "primary_paper_id": state.get("primary_paper_id", "paper-primary"),
            "active_paper_ids": state.get(
                "active_paper_ids", ["paper-primary", "paper-related"]
            ),
        }

    def fake_research(state: dict[str, Any], runtime: Any) -> dict[str, object]:
        research_entries.append(dict(state))
        return {
            "research_result": {
                "evidence_items": [],
                "used_papers": [],
                "limitations": [f"research-{runtime.context.task_id}"],
            }
        }

    def fake_writer(state: dict[str, Any], runtime: Any) -> dict[str, object]:
        writer_states.append(dict(state))
        return {
            "answer_draft": {
                "content": f"answer-{runtime.context.task_id}",
                "citations": [],
                "result_quality": "partial",
            }
        }

    def fake_publish(state: dict[str, Any], runtime: Any) -> dict[str, object]:
        message_id = f"assistant-{runtime.context.task_id}"
        return {
            "published_message_id": message_id,
            "active_paper_ids": list(state["active_paper_ids"]),
            "messages": [
                {
                    "role": "assistant",
                    "content": state["answer_draft"]["content"],
                    "id": message_id,
                }
            ],
        }

    monkeypatch.setattr(graph_module, "initialize_turn", real_initialize_turn)
    monkeypatch.setattr(graph_module, "summarize_history", fake_summary)
    monkeypatch.setattr(graph_module, "prepare_primary_paper", fake_prepare)
    monkeypatch.setattr(graph_module, "research_evidence", fake_research)
    monkeypatch.setattr(graph_module, "write_answer", fake_writer)
    monkeypatch.setattr(graph_module, "publish_result", fake_publish)

    graph = build_deep_reading_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "conv-two-turn"}}
    first = graph.invoke(
        {
            "messages": [
                HumanMessage(content="first question", id="message-first")
            ],
            "primary_paper_id": "paper-primary",
        },
        config=config,
        context=_context(
            task_id="task-first",
            message_id="message-first",
            threshold=1,
        ),
    )
    second = graph.invoke(
        {
            "messages": [
                HumanMessage(content="second question", id="message-second")
            ],
            "primary_paper_id": "paper-primary",
        },
        config=config,
        context=_context(
            task_id="task-second",
            message_id="message-second",
            threshold=32_000,
        ),
    )

    assert first["published_message_id"] == "assistant-task-first"
    assert second["current_task_id"] == "task-second"
    assert second["published_message_id"] == "assistant-task-second"
    assert second["research_result"]["limitations"] == ["research-task-second"]
    assert second["answer_draft"]["content"] == "answer-task-second"
    assert second["conversation_summary"]["confirmed_facts"] == ["first-turn fact"]
    assert second["active_paper_ids"] == ["paper-primary", "paper-related"]
    assert [message.id for message in second["messages"]] == [
        "message-first",
        "assistant-task-first",
        "message-second",
        "assistant-task-second",
    ]

    second_research_entry = research_entries[1]
    assert second_research_entry["research_result"] is None
    assert second_research_entry["answer_draft"] is None
    assert second_research_entry["published_message_id"] is None
    second_writer_state = writer_states[1]
    assert second_writer_state["conversation_summary"]["confirmed_facts"] == [
        "first-turn fact"
    ]
    assert second_writer_state["active_paper_ids"] == [
        "paper-primary",
        "paper-related",
    ]
    assert second_writer_state["research_result"]["limitations"] == [
        "research-task-second"
    ]
    assert second_writer_state["answer_draft"] is None
    assert second_writer_state["published_message_id"] is None
    assert [message.id for message in second_writer_state["messages"]] == [
        "message-first",
        "assistant-task-first",
        "message-second",
    ]


def _replay_result(suffix: str) -> ResearchResult:
    paper = PaperCandidate(
        external_id=f"2401.8000{suffix}v1",
        title=f"Related paper {suffix}",
        authors=[f"Author {suffix}"],
        abstract=f"Abstract {suffix}",
        source_url=f"https://arxiv.org/abs/2401.8000{suffix}v1",
    )
    evidence_id = f"ev-result-{suffix}"
    return ResearchResult(
        evidence_items=[
            EvidenceItem(
                id=evidence_id,
                paper_external_id=paper.external_id,
                paper_title=paper.title,
                chunk_text=f"Evidence {suffix}",
                score=0.8,
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


def test_same_task_replay_keeps_first_sqlite_publication_authoritative(
    tmp_path,
    monkeypatch,
) -> None:
    store = TaskStore(tmp_path / "business.sqlite3")
    user = store.create_user(
        username="graph-replay",
        password_hash="hash",
        password_salt="salt",
    )
    conversation = store.create_conversation(
        user_id=user.id,
        paper=PaperCandidate(
            external_id="2401.80000v1",
            title="Primary paper",
            authors=["Primary Author"],
            abstract="Primary abstract",
            source_url="https://arxiv.org/abs/2401.80000v1",
        ),
    )
    turn = store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content="Compare the papers.",
        depth="deep",
        expected_head_message_id=None,
    )
    assert store.claim_task(turn.task.id) is not None

    results = [_replay_result("1"), _replay_result("2")]
    drafts = [
        AnswerDraft(
            content="answer-1",
            citations=[
                AnswerCitation(evidence_id="ev-result-1", label="result-1")
            ],
            result_quality="complete",
        ),
        AnswerDraft(
            content="answer-2",
            citations=[
                AnswerCitation(evidence_id="ev-result-2", label="result-2")
            ],
            result_quality="partial",
        ),
    ]
    research_calls = 0
    writer_calls = 0

    def fake_prepare(state: dict[str, Any], runtime: object) -> dict[str, object]:
        del runtime
        return {
            "primary_paper_id": conversation.primary_paper_id,
            "active_paper_ids": state.get(
                "active_paper_ids", [conversation.primary_paper_id]
            ),
        }

    def fake_research(state: dict[str, Any], runtime: object) -> dict[str, object]:
        nonlocal research_calls
        del state, runtime
        result = results[research_calls]
        research_calls += 1
        return {"research_result": result.model_dump(mode="json")}

    def fake_writer(state: dict[str, Any], runtime: object) -> dict[str, object]:
        nonlocal writer_calls
        del state, runtime
        draft = drafts[writer_calls]
        writer_calls += 1
        return {"answer_draft": draft.model_dump(mode="json")}

    monkeypatch.setattr(graph_module, "prepare_primary_paper", fake_prepare)
    monkeypatch.setattr(graph_module, "research_evidence", fake_research)
    monkeypatch.setattr(graph_module, "write_answer", fake_writer)

    graph = build_deep_reading_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": conversation.id}}
    context = _context(
        task_id=turn.task.id,
        message_id=turn.user_message.id,
        threshold=32_000,
        store=store,
        user_id=user.id,
        conversation_id=conversation.id,
        base_checkpoint_id=None,
    )
    graph_input = {
        "messages": [
            HumanMessage(content=turn.user_message.content, id=turn.user_message.id)
        ],
        "primary_paper_id": conversation.primary_paper_id,
    }

    first = graph.invoke(graph_input, config=config, context=context)
    second = graph.invoke(graph_input, config=config, context=context)

    assert first["answer_draft"] == drafts[0].model_dump(mode="json")
    assert second["answer_draft"] == drafts[0].model_dump(mode="json")
    assert second["research_result"] == results[0].model_dump(mode="json")
    assert [
        message.content for message in second["messages"] if message.type == "ai"
    ] == ["answer-1"]
    assistant = store.get_task_message(turn.task.id, "assistant")
    assert assistant is not None
    assert assistant.content == "answer-1"
    assert assistant.metadata["result_quality"] == "complete"
    assert assistant.metadata["research_result"] == results[0].model_dump(mode="json")
    artifacts = store.list_artifacts_page(
        turn.task.id,
        user_id=user.id,
        after_id=0,
        limit=10,
    )
    assert artifacts is not None
    assert len(artifacts.items) == 1
    assert artifacts.items[0].content == "answer-1"
    assert artifacts.items[0].payload == assistant.metadata

    with store.engine.connect() as connection:
        external_ids = list(
            connection.exec_driver_sql(
                """
                SELECT papers.external_id
                FROM conversation_papers
                JOIN papers ON papers.id = conversation_papers.paper_id
                WHERE conversation_papers.conversation_id = ?
                ORDER BY papers.external_id
                """,
                (conversation.id,),
            ).scalars()
        )
    assert external_ids == ["2401.80000v1", "2401.80001v1"]
