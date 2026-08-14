"""Paper and conversation persistence tests."""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier, BrokenBarrierError, Event
from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError

from paperpilot.papers import PaperCandidate
from paperpilot.web import task_store as task_store_module
from paperpilot.web.store import messages as messages_module
from paperpilot.web.task_store import TaskStore

def _create_test_user(store: TaskStore, username: str):
    return store.create_user(
        username=username,
        password_hash="hash",
        password_salt="salt",
    )

def _paper(**overrides: object) -> PaperCandidate:
    values: dict[str, object] = {
        "external_id": "2401.12345v2",
        "title": "A Test Paper",
        "authors": ["Ada Lovelace"],
        "abstract": "abstract",
        "source_url": "https://arxiv.org/abs/2401.12345v2",
    }
    values.update(overrides)
    return PaperCandidate(**values)

def _table_count(db_path, table: str) -> int:
    with sqlite3.connect(db_path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

def _insert_completed_turn(
    store: TaskStore,
    *,
    conversation_id: str,
    suffix: str,
    parent_message_id: str | None,
    make_head: bool = False,
    assistant_status: str = "complete",
) -> tuple[str, str, str]:
    task_id = f"task_{suffix}"
    user_message_id = f"msg_{suffix}_user"
    assistant_message_id = f"msg_{suffix}_assistant"
    with store.engine.begin() as connection:
        user_id = connection.exec_driver_sql(
            "SELECT user_id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).scalar_one()
        connection.exec_driver_sql(
            """
            INSERT INTO research_tasks (
                id, question, depth, status, created_at, updated_at, user_id,
                conversation_id, base_checkpoint_id, final_checkpoint_id,
                result_quality
            ) VALUES (?, ?, 'standard', 'completed', ?, ?, ?, ?, ?, ?, 'complete')
            """,
            (
                task_id,
                f"Question {suffix}",
                f"2026-08-07T08:00:{suffix[-1]}0+00:00",
                f"2026-08-07T08:00:{suffix[-1]}1+00:00",
                user_id,
                conversation_id,
                f"cp_base_{suffix}" if parent_message_id else None,
                f"cp_final_{suffix}",
            ),
        )
        connection.exec_driver_sql(
            """
            INSERT INTO messages (
                id, conversation_id, task_id, parent_message_id, role,
                content, status, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, 'user', ?, 'complete', '{}', ?)
            """,
            (
                user_message_id,
                conversation_id,
                task_id,
                parent_message_id,
                f"Question {suffix}",
                f"2026-08-07T08:00:{suffix[-1]}0+00:00",
            ),
        )
        connection.exec_driver_sql(
            """
            INSERT INTO messages (
                id, conversation_id, task_id, parent_message_id, role,
                content, status, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, 'assistant', ?, ?, '{}', ?)
            """,
            (
                assistant_message_id,
                conversation_id,
                task_id,
                user_message_id,
                f"Answer {suffix}",
                assistant_status,
                f"2026-08-07T08:00:{suffix[-1]}1+00:00",
            ),
        )
        if make_head:
            connection.exec_driver_sql(
                """
                UPDATE conversations
                SET head_message_id = ?, head_checkpoint_id = ?
                WHERE id = ?
                """,
                (assistant_message_id, f"cp_final_{suffix}", conversation_id),
            )
    return task_id, user_message_id, assistant_message_id

def _used_paper(
    *,
    external_id: str,
    role: str = "comparison",
) -> task_store_module.UsedPaperInput:
    return task_store_module.UsedPaperInput(
        paper=_paper(
            external_id=external_id,
            title=f"Related {external_id}",
            source_url=f"https://arxiv.org/abs/{external_id}",
        ),
        role=role,
    )

def _publish_turn(
    store: TaskStore,
    task_id: str,
    *,
    used_papers: list[task_store_module.UsedPaperInput] | None = None,
):
    return store.publish_conversation_result(
        task_id=task_id,
        content="Evidence-backed answer.",
        metadata={
            "citations": [
                {"evidence_id": "ev-1", "label": "Related evidence"}
            ]
        },
        used_papers=used_papers or [],
    )

@contextmanager
def _synchronize_reservation_or_legacy_read(
    stores: tuple[TaskStore, TaskStore],
    *,
    reservation_prefix: str,
    legacy_read_fragment: str,
):
    """Synchronize the real reservation SQL, with a RED-only legacy fallback."""
    reservation_seen = Event()
    reservation_writers_ready = Barrier(2)
    legacy_readers_ready = Barrier(2)

    def synchronize_reservation(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        normalized = " ".join(statement.split())
        if not normalized.startswith(reservation_prefix):
            return
        reservation_seen.set()
        try:
            reservation_writers_ready.wait(timeout=5)
        except BrokenBarrierError as exc:
            raise AssertionError("both writers did not reach reservation SQL") from exc

    def synchronize_legacy_read(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        normalized = " ".join(statement.split())
        if reservation_seen.is_set() or legacy_read_fragment not in normalized:
            return
        try:
            legacy_readers_ready.wait(timeout=5)
        except BrokenBarrierError as exc:
            raise AssertionError("both legacy writers did not complete first read") from exc

    for store in stores:
        event.listen(store.engine, "before_cursor_execute", synchronize_reservation)
        event.listen(store.engine, "after_cursor_execute", synchronize_legacy_read)
    try:
        yield
    finally:
        for store in stores:
            event.remove(
                store.engine,
                "before_cursor_execute",
                synchronize_reservation,
            )
            event.remove(
                store.engine,
                "after_cursor_execute",
                synchronize_legacy_read,
            )

def _call_or_exception(callable_):
    try:
        return callable_()
    except Exception as exc:  # preserve both real transaction outcomes
        return exc

def test_publish_conversation_result_is_task_idempotent_and_keeps_head_stable(
    tmp_path,
):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-publish")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30001v1"),
    )
    turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Compare this paper.",
        depth="deep",
        expected_head_message_id=None,
    )
    assert store.claim_task(turn.task.id) is not None
    used_paper = _used_paper(external_id="2401.30002v1")

    first = _publish_turn(store, turn.task.id, used_papers=[used_paper])
    second = _publish_turn(store, turn.task.id, used_papers=[used_paper])

    assert first == second
    assert first.message.parent_message_id == turn.user_message.id
    assert first.message.metadata == {
        "citations": [{"evidence_id": "ev-1", "label": "Related evidence"}]
    }
    assert first.artifact.kind == "result"
    assert first.artifact.payload == first.message.metadata
    assert _table_count(store.db_path, "messages") == 2
    assert _table_count(store.db_path, "task_artifacts") == 1
    assert _table_count(store.db_path, "papers") == 2
    assert _table_count(store.db_path, "conversation_papers") == 2

    detail = store.get_conversation_detail(conversation.id, user_id=alice.id)
    assert detail is not None
    assert detail.conversation.head_message_id is None
    assert detail.conversation.head_checkpoint_id is None
    assert [paper.id for paper in detail.active_papers] == [
        conversation.primary_paper_id
    ]
    assert first.active_paper_ids[0] == conversation.primary_paper_id
    assert len(first.active_paper_ids) == 2
    # get_conversation_detail exposes active associations only, so inspect the
    # persisted inactive publication association directly.
    with sqlite3.connect(store.db_path) as connection:
        publication_link = connection.execute(
            """
            SELECT source_task_id, source_message_id, role, added_by, is_active
            FROM conversation_papers
            WHERE conversation_id = ? AND paper_id != ?
            """,
            (conversation.id, conversation.primary_paper_id),
        ).fetchone()
    assert publication_link == (
        turn.task.id,
        first.message.id,
        "comparison",
        "agent",
        0,
    )

def test_concurrent_publish_serializes_one_assistant_and_result_artifact(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    first_store = TaskStore(db_path)
    alice = _create_test_user(first_store, "alice-concurrent-publish")
    conversation = first_store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30016v1"),
    )
    turn = first_store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Publish once under redelivery.",
        depth="deep",
        expected_head_message_id=None,
    )
    first_store.claim_task(turn.task.id)
    second_store = TaskStore(db_path)
    write_reservation_seen = Event()
    reservation_writers_ready = Barrier(2)
    assistant_readers_ready = Barrier(2)

    def synchronize_write_reservation(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        normalized = " ".join(statement.split())
        if not normalized.startswith(
            "UPDATE research_tasks SET updated_at=research_tasks.updated_at"
        ):
            return
        write_reservation_seen.set()
        reservation_writers_ready.wait(timeout=5)

    def synchronize_unserialized_assistant_reads(
        _connection,
        _cursor,
        statement,
        parameters,
        _context,
        _executemany,
    ) -> None:
        normalized = " ".join(statement.split())
        if (
            write_reservation_seen.is_set()
            or "FROM messages" not in normalized
            or "assistant" not in parameters
        ):
            return
        assistant_readers_ready.wait(timeout=5)

    for store in (first_store, second_store):
        event.listen(
            store.engine,
            "before_cursor_execute",
            synchronize_write_reservation,
        )
        event.listen(
            store.engine,
            "after_cursor_execute",
            synchronize_unserialized_assistant_reads,
        )

    used = [_used_paper(external_id="2401.30017v1")]

    def publish(store: TaskStore):
        try:
            return _publish_turn(store, turn.task.id, used_papers=used)
        except Exception as exc:  # preserve both real race outcomes for assertions
            return exc

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(publish, first_store),
                executor.submit(publish, second_store),
            ]
            results = [future.result() for future in futures]
    finally:
        for store in (first_store, second_store):
            event.remove(
                store.engine,
                "before_cursor_execute",
                synchronize_write_reservation,
            )
            event.remove(
                store.engine,
                "after_cursor_execute",
                synchronize_unserialized_assistant_reads,
            )

    assert all(
        isinstance(result, task_store_module.PublishedConversationResult)
        for result in results
    ), results
    assert results[0] == results[1]
    assert _table_count(db_path, "messages") == 2
    assert _table_count(db_path, "task_artifacts") == 1
    assert _table_count(db_path, "conversation_papers") == 2

def test_concurrent_identical_finalize_is_idempotent_with_one_completed_event(
    tmp_path,
):
    db_path = tmp_path / "tasks.sqlite3"
    first_store = TaskStore(db_path)
    alice = _create_test_user(first_store, "alice-concurrent-finalize")
    conversation = first_store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30018v1"),
    )
    turn = first_store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Finalize once under redelivery.",
        depth="deep",
        expected_head_message_id=None,
    )
    first_store.claim_task(turn.task.id)
    published = _publish_turn(
        first_store,
        turn.task.id,
        used_papers=[_used_paper(external_id="2401.30019v1")],
    )
    second_store = TaskStore(db_path)
    stores = (first_store, second_store)

    def finalize(store: TaskStore):
        return store.finalize_conversation_task(
            task_id=turn.task.id,
            assistant_message_id=published.message.id,
            final_checkpoint_id="cp-concurrent-finalize",
            result_quality="complete",
            active_paper_ids=published.active_paper_ids,
        )

    with _synchronize_reservation_or_legacy_read(
        stores,
        reservation_prefix=(
            "UPDATE research_tasks SET updated_at=research_tasks.updated_at"
        ),
        legacy_read_fragment=(
            "FROM research_tasks WHERE research_tasks.id = ?"
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(finalize, store) for store in stores]
            results = [future.result() for future in futures]

    assert all(
        isinstance(result, task_store_module.FinalizedConversationTask)
        for result in results
    )
    assert results[0] == results[1]
    assert results[0].task.status == "completed"
    assert results[0].conversation.head_message_id == published.message.id
    with sqlite3.connect(db_path) as connection:
        completed_events = connection.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND type = 'completed'",
            (turn.task.id,),
        ).fetchone()[0]
    assert completed_events == 1

def test_concurrent_identical_fail_is_idempotent_with_one_failed_event(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    first_store = TaskStore(db_path)
    alice = _create_test_user(first_store, "alice-concurrent-fail")
    conversation = first_store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30020v1"),
    )
    turn = first_store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Fail once under redelivery.",
        depth="standard",
        expected_head_message_id=None,
    )
    first_store.claim_task(turn.task.id)
    second_store = TaskStore(db_path)
    stores = (first_store, second_store)

    def fail(store: TaskStore):
        return store.fail_conversation_task(
            task_id=turn.task.id,
            message="One stable failure.",
            stage="research",
            payload={"retryable": True},
        )

    with _synchronize_reservation_or_legacy_read(
        stores,
        reservation_prefix=(
            "UPDATE research_tasks SET updated_at=research_tasks.updated_at"
        ),
        legacy_read_fragment=(
            "FROM research_tasks WHERE research_tasks.id = ?"
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(fail, store) for store in stores]
            results = [future.result() for future in futures]

    assert all(isinstance(result, task_store_module.ResearchTask) for result in results)
    assert results[0] == results[1]
    assert results[0].status == "failed"
    with sqlite3.connect(db_path) as connection:
        failed_events = connection.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND type = 'failed'",
            (turn.task.id,),
        ).fetchone()[0]
    assert failed_events == 1

def test_concurrent_finalize_and_fail_have_one_semantic_terminal_winner(tmp_path):
    db_path = tmp_path / "tasks.sqlite3"
    first_store = TaskStore(db_path)
    alice = _create_test_user(first_store, "alice-terminal-race")
    conversation = first_store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30021v1"),
    )
    turn = first_store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Only one terminal outcome may win.",
        depth="deep",
        expected_head_message_id=None,
    )
    first_store.claim_task(turn.task.id)
    published = _publish_turn(
        first_store,
        turn.task.id,
        used_papers=[_used_paper(external_id="2401.30022v1")],
    )
    second_store = TaskStore(db_path)
    stores = (first_store, second_store)

    finalize = lambda: first_store.finalize_conversation_task(
        task_id=turn.task.id,
        assistant_message_id=published.message.id,
        final_checkpoint_id="cp-terminal-race",
        result_quality="complete",
        active_paper_ids=published.active_paper_ids,
    )
    fail = lambda: second_store.fail_conversation_task(
        task_id=turn.task.id,
        message="Concurrent terminal failure.",
        stage="finalize",
    )

    with _synchronize_reservation_or_legacy_read(
        stores,
        reservation_prefix=(
            "UPDATE research_tasks SET updated_at=research_tasks.updated_at"
        ),
        legacy_read_fragment=(
            "FROM research_tasks WHERE research_tasks.id = ?"
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_call_or_exception, finalize),
                executor.submit(_call_or_exception, fail),
            ]
            results = [future.result() for future in futures]

    successes = [result for result in results if not isinstance(result, Exception)]
    failures = [result for result in results if isinstance(result, Exception)]
    assert len(successes) == 1, results
    assert len(failures) == 1, results
    assert isinstance(failures[0], ValueError), results

    persisted_task = first_store.get_task(turn.task.id)
    detail = first_store.get_conversation_detail(conversation.id, user_id=alice.id)
    assert persisted_task is not None
    assert detail is not None
    with sqlite3.connect(db_path) as connection:
        terminal_events = connection.execute(
            """
            SELECT type FROM task_events
            WHERE task_id = ? AND type IN ('completed', 'failed')
            ORDER BY id
            """,
            (turn.task.id,),
        ).fetchall()
    assert len(terminal_events) == 1
    if persisted_task.status == "completed":
        assert isinstance(
            successes[0],
            task_store_module.FinalizedConversationTask,
        )
        assert terminal_events == [("completed",)]
        assert persisted_task.final_checkpoint_id == "cp-terminal-race"
        assert detail.conversation.head_message_id == published.message.id
        assert {paper.id for paper in detail.active_papers} == set(
            published.active_paper_ids
        )
    else:
        assert isinstance(successes[0], task_store_module.ResearchTask)
        assert persisted_task.status == "failed"
        assert terminal_events == [("failed",)]
        assert persisted_task.final_checkpoint_id is None
        assert persisted_task.result_quality is None
        assert detail.conversation.head_message_id is None
        assert [paper.id for paper in detail.active_papers] == [
            conversation.primary_paper_id
        ]

def test_concurrent_identical_head_switch_has_one_stale_loser_and_coherent_state(
    tmp_path,
):
    db_path = tmp_path / "tasks.sqlite3"
    first_store = TaskStore(db_path)
    alice = _create_test_user(first_store, "alice-concurrent-switch")
    conversation = first_store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30023v1"),
    )
    first_turn = first_store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="First stable branch.",
        depth="standard",
        expected_head_message_id=None,
    )
    first_store.claim_task(first_turn.task.id)
    first_published = _publish_turn(
        first_store,
        first_turn.task.id,
        used_papers=[_used_paper(external_id="2401.30024v1")],
    )
    first_final = first_store.finalize_conversation_task(
        task_id=first_turn.task.id,
        assistant_message_id=first_published.message.id,
        final_checkpoint_id="cp-concurrent-switch-target",
        result_quality="complete",
        active_paper_ids=first_published.active_paper_ids,
    )
    second_turn = first_store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Current stable branch.",
        depth="standard",
        expected_head_message_id=first_final.conversation.head_message_id,
    )
    first_store.claim_task(second_turn.task.id)
    second_published = _publish_turn(
        first_store,
        second_turn.task.id,
        used_papers=[_used_paper(external_id="2401.30025v1")],
    )
    second_final = first_store.finalize_conversation_task(
        task_id=second_turn.task.id,
        assistant_message_id=second_published.message.id,
        final_checkpoint_id="cp-concurrent-switch-current",
        result_quality="complete",
        active_paper_ids=second_published.active_paper_ids,
    )
    second_store = TaskStore(db_path)
    stores = (first_store, second_store)

    def switch(store: TaskStore):
        return store.switch_conversation_head(
            conversation.id,
            user_id=alice.id,
            expected_head_message_id=second_final.conversation.head_message_id,
            target_message_id=first_published.message.id,
            target_checkpoint_id="cp-concurrent-switch-target",
            active_paper_ids=first_published.active_paper_ids,
        )

    with _synchronize_reservation_or_legacy_read(
        stores,
        reservation_prefix=(
            "UPDATE conversations SET updated_at=conversations.updated_at"
        ),
        legacy_read_fragment=(
            "FROM conversations WHERE conversations.id = ?"
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_call_or_exception, lambda store=store: switch(store))
                for store in stores
            ]
            results = [future.result() for future in futures]

    successes = [result for result in results if not isinstance(result, Exception)]
    failures = [result for result in results if isinstance(result, Exception)]
    assert len(successes) == 1, results
    assert len(failures) == 1, results
    assert isinstance(successes[0], task_store_module.ConversationRecord)
    assert isinstance(failures[0], task_store_module.StaleConversationHeadError)
    detail = first_store.get_conversation_detail(conversation.id, user_id=alice.id)
    assert detail is not None
    assert detail.conversation.head_message_id == first_published.message.id
    assert detail.conversation.head_checkpoint_id == "cp-concurrent-switch-target"
    assert {paper.id for paper in detail.active_papers} == set(
        first_published.active_paper_ids
    )

def test_finalize_conversation_task_atomically_completes_and_is_idempotent(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-finalize")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30003v1"),
    )
    turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Finish this turn.",
        depth="standard",
        expected_head_message_id=None,
    )
    store.claim_task(turn.task.id)
    published = _publish_turn(
        store,
        turn.task.id,
        used_papers=[_used_paper(external_id="2401.30004v1")],
    )

    first = store.finalize_conversation_task(
        task_id=turn.task.id,
        assistant_message_id=published.message.id,
        final_checkpoint_id="cp-final-1",
        result_quality="complete",
        active_paper_ids=published.active_paper_ids,
    )
    second = store.finalize_conversation_task(
        task_id=turn.task.id,
        assistant_message_id=published.message.id,
        final_checkpoint_id="cp-final-1",
        result_quality="complete",
        active_paper_ids=published.active_paper_ids,
    )

    assert first == second
    assert first.task.status == "completed"
    assert first.task.final_checkpoint_id == "cp-final-1"
    assert first.task.result_quality == "complete"
    assert first.conversation.head_message_id == published.message.id
    assert first.conversation.head_checkpoint_id == "cp-final-1"
    detail = store.get_conversation_detail(conversation.id, user_id=alice.id)
    assert detail is not None
    assert {paper.id for paper in detail.active_papers} == set(
        published.active_paper_ids
    )
    assert _table_count(store.db_path, "messages") == 2
    assert _table_count(store.db_path, "task_artifacts") == 1
    with sqlite3.connect(store.db_path) as connection:
        completed_events = connection.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND type = 'completed'",
            (turn.task.id,),
        ).fetchone()[0]
    assert completed_events == 1

def test_publish_crash_window_reuses_rows_before_successful_finalization(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-publish-crash")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30005v1"),
    )
    turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Recover after publication.",
        depth="standard",
        expected_head_message_id=None,
    )
    store.claim_task(turn.task.id)
    used = [_used_paper(external_id="2401.30006v1")]

    published_before_crash = _publish_turn(store, turn.task.id, used_papers=used)
    with pytest.raises(RuntimeError, match="simulated crash"):
        raise RuntimeError("simulated crash after publish before finalization")

    recovered = _publish_turn(store, turn.task.id, used_papers=used)
    final = store.finalize_conversation_task(
        task_id=turn.task.id,
        assistant_message_id=recovered.message.id,
        final_checkpoint_id="cp-after-publish-crash",
        result_quality="partial",
        active_paper_ids=recovered.active_paper_ids,
    )

    assert recovered.message.id == published_before_crash.message.id
    assert final.task.status == "completed"
    assert final.conversation.head_message_id == recovered.message.id
    assert _table_count(store.db_path, "messages") == 2
    assert _table_count(store.db_path, "task_artifacts") == 1

def test_finalize_transaction_failure_rolls_back_and_retry_completes(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-finalize-crash")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30007v1"),
    )
    turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Recover finalization.",
        depth="deep",
        expected_head_message_id=None,
    )
    store.claim_task(turn.task.id)
    published = _publish_turn(
        store,
        turn.task.id,
        used_papers=[_used_paper(external_id="2401.30008v1")],
    )
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TRIGGER reject_finalization_head
            BEFORE UPDATE OF head_message_id ON conversations
            BEGIN
                SELECT RAISE(ABORT, 'simulated finalization crash');
            END
            """
        )

    with pytest.raises(IntegrityError, match="simulated finalization crash"):
        store.finalize_conversation_task(
            task_id=turn.task.id,
            assistant_message_id=published.message.id,
            final_checkpoint_id="cp-known-before-crash",
            result_quality="complete",
            active_paper_ids=published.active_paper_ids,
        )

    rolled_back_task = store.get_task(turn.task.id)
    rolled_back_detail = store.get_conversation_detail(
        conversation.id,
        user_id=alice.id,
    )
    assert rolled_back_task is not None
    assert rolled_back_task.status == "running"
    assert rolled_back_task.final_checkpoint_id is None
    assert rolled_back_detail is not None
    assert rolled_back_detail.conversation.head_message_id is None
    assert [paper.id for paper in rolled_back_detail.active_papers] == [
        conversation.primary_paper_id
    ]
    with store.engine.begin() as connection:
        connection.exec_driver_sql("DROP TRIGGER reject_finalization_head")

    final = store.finalize_conversation_task(
        task_id=turn.task.id,
        assistant_message_id=published.message.id,
        final_checkpoint_id="cp-known-before-crash",
        result_quality="complete",
        active_paper_ids=published.active_paper_ids,
    )
    assert final.task.status == "completed"
    assert final.conversation.head_checkpoint_id == "cp-known-before-crash"
    assert _table_count(store.db_path, "messages") == 2
    assert _table_count(store.db_path, "task_artifacts") == 1

def test_finalize_rejects_stale_head_and_broken_assistant_parent_chain(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-delayed-worker")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30009v1"),
    )
    turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="A delayed result.",
        depth="standard",
        expected_head_message_id=None,
    )
    store.claim_task(turn.task.id)
    published = _publish_turn(store, turn.task.id)
    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE conversations SET head_checkpoint_id = 'cp-new-path' WHERE id = ?",
            (conversation.id,),
        )

    with pytest.raises(task_store_module.StaleConversationHeadError):
        store.finalize_conversation_task(
            task_id=turn.task.id,
            assistant_message_id=published.message.id,
            final_checkpoint_id="cp-delayed",
            result_quality="complete",
            active_paper_ids=published.active_paper_ids,
        )
    assert store.get_task(turn.task.id).status == "running"

    with store.engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE conversations SET head_checkpoint_id = NULL WHERE id = ?",
            (conversation.id,),
        )
        connection.exec_driver_sql(
            "UPDATE messages SET parent_message_id = NULL WHERE id = ?",
            (published.message.id,),
        )
    with pytest.raises(ValueError, match="parent"):
        store.finalize_conversation_task(
            task_id=turn.task.id,
            assistant_message_id=published.message.id,
            final_checkpoint_id="cp-broken-parent",
            result_quality="complete",
            active_paper_ids=published.active_paper_ids,
        )
    assert store.get_task(turn.task.id).status == "running"

def test_fail_conversation_task_is_idempotent_and_never_overwrites_completed(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-failure")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30010v1"),
    )
    failed_turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="This worker fails.",
        depth="quick",
        expected_head_message_id=None,
    )
    store.claim_task(failed_turn.task.id)

    first = store.fail_conversation_task(
        task_id=failed_turn.task.id,
        message="Research provider failed.",
        stage="research",
        payload={"retryable": True},
    )
    second = store.fail_conversation_task(
        task_id=failed_turn.task.id,
        message="Research provider failed.",
        stage="research",
        payload={"retryable": True},
    )

    assert first == second
    assert first is not None
    assert first.status == "failed"
    with sqlite3.connect(store.db_path) as connection:
        failure_events = connection.execute(
            """
            SELECT type, stage, message, payload_json
            FROM task_events WHERE task_id = ? AND type = 'failed'
            """,
            (failed_turn.task.id,),
        ).fetchall()
    assert len(failure_events) == 1
    assert failure_events[0][:3] == (
        "failed",
        "research",
        "Research provider failed.",
    )
    assert json.loads(failure_events[0][3]) == {"retryable": True}

    retry = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="This worker succeeds.",
        depth="quick",
        expected_head_message_id=None,
    )
    store.claim_task(retry.task.id)
    published = _publish_turn(store, retry.task.id)
    final = store.finalize_conversation_task(
        task_id=retry.task.id,
        assistant_message_id=published.message.id,
        final_checkpoint_id="cp-completed",
        result_quality="complete",
        active_paper_ids=published.active_paper_ids,
    )
    with pytest.raises(ValueError, match="completed"):
        store.fail_conversation_task(
            task_id=final.task.id,
            message="Late failure must not win.",
        )
    assert store.get_task(final.task.id).status == "completed"

def test_switch_conversation_head_atomically_changes_path_and_active_papers(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-switch")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30011v1"),
    )
    first_turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="First branch.",
        depth="standard",
        expected_head_message_id=None,
    )
    store.claim_task(first_turn.task.id)
    first_published = _publish_turn(
        store,
        first_turn.task.id,
        used_papers=[_used_paper(external_id="2401.30012v1")],
    )
    first_final = store.finalize_conversation_task(
        task_id=first_turn.task.id,
        assistant_message_id=first_published.message.id,
        final_checkpoint_id="cp-first-branch",
        result_quality="complete",
        active_paper_ids=first_published.active_paper_ids,
    )
    second_turn = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Second branch.",
        depth="standard",
        expected_head_message_id=first_final.conversation.head_message_id,
    )
    store.claim_task(second_turn.task.id)
    second_published = _publish_turn(
        store,
        second_turn.task.id,
        used_papers=[_used_paper(external_id="2401.30013v1")],
    )
    second_final = store.finalize_conversation_task(
        task_id=second_turn.task.id,
        assistant_message_id=second_published.message.id,
        final_checkpoint_id="cp-second-branch",
        result_quality="complete",
        active_paper_ids=second_published.active_paper_ids,
    )

    switched = store.switch_conversation_head(
        conversation.id,
        user_id=alice.id,
        expected_head_message_id=second_final.conversation.head_message_id,
        target_message_id=first_published.message.id,
        target_checkpoint_id="cp-first-branch",
        # Checkpoint state may omit the primary paper; business storage must
        # still keep the conversation's primary association active.
        active_paper_ids=first_published.active_paper_ids[1:],
    )

    assert switched is not None
    assert switched.head_message_id == first_published.message.id
    assert switched.head_checkpoint_id == "cp-first-branch"
    detail = store.get_conversation_detail(conversation.id, user_id=alice.id)
    assert detail is not None
    assert {paper.id for paper in detail.active_papers} == set(
        first_published.active_paper_ids
    )
    assert conversation.primary_paper_id in {
        paper.id for paper in detail.active_papers
    }
    assert store.get_message(
        conversation.id,
        second_published.message.id,
        user_id=alice.id,
    ) == second_published.message
    assert _table_count(store.db_path, "messages") == 4

def test_switch_conversation_head_rejects_conflicts_without_partial_changes(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-switch-conflict")
    bob = _create_test_user(store, "bob-switch-conflict")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30014v1"),
    )
    task_id, user_message_id, assistant_message_id = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="rollback",
        parent_message_id=None,
        make_head=True,
    )
    original = store.get_conversation_detail(conversation.id, user_id=alice.id)
    assert original is not None

    assert store.switch_conversation_head(
        conversation.id,
        user_id=bob.id,
        expected_head_message_id=assistant_message_id,
        target_message_id=assistant_message_id,
        target_checkpoint_id="cp-owned-by-alice",
        active_paper_ids=[],
    ) is None
    with pytest.raises(task_store_module.StaleConversationHeadError):
        store.switch_conversation_head(
            conversation.id,
            user_id=alice.id,
            expected_head_message_id="msg-stale",
            target_message_id=assistant_message_id,
            target_checkpoint_id="cp-stale",
            active_paper_ids=[],
        )
    with pytest.raises(ValueError, match="assistant"):
        store.switch_conversation_head(
            conversation.id,
            user_id=alice.id,
            expected_head_message_id=assistant_message_id,
            target_message_id=user_message_id,
            target_checkpoint_id="cp-user-message",
            active_paper_ids=[],
        )

    active = store.create_conversation_turn(
        user_id=alice.id,
        conversation_id=conversation.id,
        content="Active work blocks rollback.",
        depth="quick",
        expected_head_message_id=assistant_message_id,
    )
    with pytest.raises(task_store_module.ConversationBusyError):
        store.switch_conversation_head(
            conversation.id,
            user_id=alice.id,
            expected_head_message_id=assistant_message_id,
            target_message_id=assistant_message_id,
            target_checkpoint_id="cp-busy",
            active_paper_ids=[],
        )
    after = store.get_conversation_detail(conversation.id, user_id=alice.id)
    assert after is not None
    assert after.conversation == original.conversation
    assert after.active_papers == original.active_papers
    assert store.get_message(
        conversation.id,
        user_message_id,
        user_id=alice.id,
    ) is not None
    assert store.get_task(task_id).status == "completed"
    assert store.get_task(active.task.id).status == "pending"

def test_switch_conversation_head_rejects_archived_conversation(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    alice = _create_test_user(store, "alice-switch-archived")
    conversation = store.create_conversation(
        user_id=alice.id,
        paper=_paper(external_id="2401.30015v1"),
    )
    _, _, assistant_message_id = _insert_completed_turn(
        store,
        conversation_id=conversation.id,
        suffix="archived",
        parent_message_id=None,
        make_head=True,
    )
    store.update_conversation(conversation.id, user_id=alice.id, archived=True)

    with pytest.raises(ValueError, match="archived"):
        store.switch_conversation_head(
            conversation.id,
            user_id=alice.id,
            expected_head_message_id=assistant_message_id,
            target_message_id=assistant_message_id,
            target_checkpoint_id="cp-archived",
            active_paper_ids=[],
        )
