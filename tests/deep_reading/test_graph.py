"""Fixed orchestration and checkpoint semantics for the deep-reading graph."""
from __future__ import annotations

from typing import Any

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import paperpilot.deep_reading.graph as graph_module
from paperpilot.deep_reading.graph import build_deep_reading_graph
from paperpilot.deep_reading.nodes import DeepReadingContext


def _context(
    *,
    task_id: str,
    message_id: str,
    threshold: int,
) -> DeepReadingContext:
    return DeepReadingContext(
        user_id="user-1",
        conversation_id="conv-two-turn",
        task_id=task_id,
        current_user_message_id=message_id,
        base_checkpoint_id=None,
        task_store=object(),  # type: ignore[arg-type]
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
