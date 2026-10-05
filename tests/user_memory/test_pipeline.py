"""End-to-end extraction pipeline tests against a real TaskStore."""
from __future__ import annotations

from unittest.mock import MagicMock

from paperpilot.papers import PaperCandidate
from paperpilot.user_memory.extractor import MemoryCandidate, MemoryExtractionOutput
from paperpilot.user_memory.pipeline import run_memory_extraction
from paperpilot.web.task_store import TaskStore

PAPER = PaperCandidate(
    external_id="2401.12345v1",
    title="Paper",
    authors=["Author"],
    abstract="Abstract",
    source_url="https://arxiv.org/abs/2401.12345v1",
)

USER_TEXT = "I focus on model distillation. Avoid papers without code."
ASSISTANT_TEXT = "Filtered to distillation methods with released code."


def _setup(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    user = store.create_user(
        username="alice", password_hash="hash", password_salt="salt"
    )
    conversation = store.create_conversation(user_id=user.id, paper=PAPER)
    detail = store.get_conversation_detail(conversation.id, user_id=user.id)
    turn = store.create_conversation_turn(
        user_id=user.id,
        conversation_id=conversation.id,
        content=USER_TEXT,
        depth="standard",
        expected_head_message_id=detail.conversation.head_message_id,
    )
    return store, user, turn


def _model(memories):
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = (
        MemoryExtractionOutput(memories=memories)
    )
    return model


def test_pipeline_writes_verified_memories(tmp_path):
    store, user, turn = _setup(tmp_path)
    model = _model(
        [
            MemoryCandidate(
                kind="fact",
                content="User focuses on model distillation.",
                support_span="I focus on model distillation.",
            ),
            MemoryCandidate(
                kind="fact",
                content="User is based in Tokyo.",
                support_span="I am based in Tokyo.",
            ),
        ]
    )

    written = run_memory_extraction(
        store=store,
        model=model,
        task=turn.task,
        user_message_id=turn.user_message.id,
        user_text=USER_TEXT,
        assistant_text=ASSISTANT_TEXT,
    )

    assert written == 1  # unsupported candidate dropped
    memories = store.list_user_memories(user.id)
    assert len(memories) == 1
    assert memories[0].source_task_id == turn.task.id
    assert memories[0].support_span == "I focus on model distillation."


def test_pipeline_is_idempotent_per_source_task(tmp_path):
    store, user, turn = _setup(tmp_path)
    model = _model(
        [
            MemoryCandidate(
                kind="preference",
                content="User avoids papers without code.",
                support_span="Avoid papers without code.",
            )
        ]
    )
    first = run_memory_extraction(
        store=store, model=model, task=turn.task,
        user_message_id=turn.user_message.id,
        user_text=USER_TEXT, assistant_text=ASSISTANT_TEXT,
    )
    second = run_memory_extraction(
        store=store, model=model, task=turn.task,
        user_message_id=turn.user_message.id,
        user_text=USER_TEXT, assistant_text=ASSISTANT_TEXT,
    )

    assert first == 1
    assert second == 0  # idempotent: LLM not even invoked again
    model.with_structured_output.return_value.invoke.assert_called_once()
    assert len(store.list_user_memories(user.id)) == 1
