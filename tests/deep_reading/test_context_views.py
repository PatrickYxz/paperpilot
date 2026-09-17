"""Active Projection, Archive retrieval, and model-view contracts."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

from paperpilot.deep_reading.context_management.models import ArtifactRef
from paperpilot.deep_reading.context_management.views import (
    ActiveProjectionBuilder,
    ArchiveRetriever,
    ContextViewBuilder,
    render_context_view,
)


class _Counter:
    def count_messages(self, messages, *, tool_schemas=()) -> int:
        return self.count_text(str((list(messages), list(tool_schemas))))

    def count_text(self, value: str) -> int:
        return max(1, len(value) // 4)


def _archive(
    archive_id: str,
    *,
    created_at: str,
    goal: str,
    paper_id: str = "2401.12345v1",
    constraints: list[dict] | None = None,
    verification: list[str] | None = None,
    supersedes: list[dict] | None = None,
    narrative: str | None = None,
    version: str = "turn-archive-v1",
):
    def protected(protected_id: str, exact_text: str) -> dict[str, str]:
        return {
            "protected_id": protected_id,
            "exact_text": exact_text,
            "sha256": hashlib.sha256(exact_text.encode()).hexdigest(),
        }

    seed = {
        "archive_id": archive_id,
        "user_goal": protected(f"goal-{archive_id}", goal),
        "constraints": constraints or [],
        "decisions": [],
        "paper_findings": [f"{paper_id}:comparison:ev-{archive_id}"],
        "evidence_refs": [f"ev-{archive_id}"],
        "artifact_refs": [],
        "failed_paths": [],
        "verification": verification or ["citation:pass"],
        "unresolved_todos": [],
        "rollback_notes": [],
        "supersedes": supersedes or [],
        "archive_version": version,
    }
    return SimpleNamespace(
        archive_id=archive_id,
        archive_version=version,
        terminal_status="success",
        seed_json=seed,
        narrative_summary=narrative,
        created_at=created_at,
        updated_at=created_at,
    )


def test_projection_uses_latest_goal_and_only_non_superseded_protected_text() -> None:
    old = _archive(
        "archive-old",
        created_at="2026-09-01T00:00:00+00:00",
        goal="old goal",
        constraints=[
            {
                "protected_id": "constraint-old",
                "exact_text": "use baseline",
                "sha256": hashlib.sha256(b"use baseline").hexdigest(),
            }
        ],
    )
    new = _archive(
        "archive-new",
        created_at="2026-09-01T01:00:00+00:00",
        goal="new goal",
        constraints=[
            {
                "protected_id": "constraint-new",
                "exact_text": "use corrected method",
                "sha256": hashlib.sha256(b"use corrected method").hexdigest(),
            }
        ],
        supersedes=[
            {
                "target_protected_id": "constraint-old",
                "source_message_id": "message-2",
                "exact_text": "use corrected method",
                "sha256": hashlib.sha256(b"use corrected method").hexdigest(),
            }
        ],
    )

    projection = ActiveProjectionBuilder().build(
        current_goal="current request",
        active_paper_ids=["paper-primary"],
        archives=[old, new],
    )

    assert projection.current_goal == "current request"
    assert projection.active_constraints == ["use corrected method"]
    assert "use baseline" not in projection.active_constraints
    assert projection.active_paper_ids == ["paper-primary"]
    assert projection.protected_items[0].protected_id == "constraint-new"


def test_retriever_prioritizes_exact_id_limits_budget_and_hides_old_superseded() -> None:
    archives = [
        _archive(
            "archive-old",
            created_at="2026-09-01T00:00:00+00:00",
            goal="old goal",
            paper_id="2401.99999v1",
        ),
        _archive(
            "archive-exact",
            created_at="2026-09-01T00:30:00+00:00",
            goal="exact archive",
            paper_id="2401.99999v1",
            narrative="long narrative " * 200,
        ),
        _archive(
            "archive-recent",
            created_at="2026-09-01T02:00:00+00:00",
            goal="recent goal",
            paper_id="2401.12345v1",
        ),
    ]
    retriever = ArchiveRetriever(
        token_counter=_Counter(),
        budget_tokens=500,
        max_records=5,
        recent_records=2,
    )

    result = retriever.retrieve(
        "Please explain archive-exact",
        archives=archives,
        active_paper_ids=["2401.12345v1"],
    )

    assert [item.archive_id for item in result][0] == "archive-exact"
    assert len(result) <= 5
    assert "archive-old" not in [item.archive_id for item in result]
    assert all(item.seed for item in result)
    assert all(item.narrative_summary is None for item in result if item.archive_id == "archive-exact")


def test_history_query_loads_both_sides_of_scoped_supersession() -> None:
    old = _archive(
        "archive-old",
        created_at="2026-09-01T00:00:00+00:00",
        goal="old goal",
    )
    new = _archive(
        "archive-new",
        created_at="2026-09-01T01:00:00+00:00",
        goal="new goal",
        supersedes=[
            {
                "target_protected_id": "goal-archive-old",
                "source_message_id": "message-2",
                "exact_text": "new goal",
                "sha256": hashlib.sha256(b"new goal").hexdigest(),
            }
        ],
    )
    result = ArchiveRetriever(token_counter=_Counter()).retrieve(
        "what changed for goal-archive-old?",
        archives=[old, new],
        active_paper_ids=[],
    )
    assert {item.archive_id for item in result} == {"archive-old", "archive-new"}


def test_retriever_never_injects_outer_versioned_legacy_summary_archive() -> None:
    legacy = _archive(
        "legacy-summary",
        created_at="2026-09-01T00:00:00+00:00",
        goal="legacy goal",
        version="legacy-summary-v1",
    )
    legacy.seed_json = {
        "archive_type": "legacy-summary-v1",
        "untrusted": True,
        "legacy_summary": {"confirmed_facts": ["untrusted candidate"]},
    }

    result = ArchiveRetriever(token_counter=_Counter()).retrieve(
        "why did legacy-summary change?",
        archives=[legacy],
        active_paper_ids=[],
    )

    assert result == []


def test_retriever_exactly_matches_uuid_artifact_reference() -> None:
    archive = _archive(
        "archive-with-artifact",
        created_at="2026-09-01T00:00:00+00:00",
        goal="artifact lookup",
    )
    artifact_id = "550e8400-e29b-41d4-a716-446655440000"
    archive.seed_json["artifact_refs"] = [
        {
            "artifact_id": artifact_id,
            "sha256": "a" * 64,
            "token_estimate": 10,
            "preview": "preview",
        }
    ]

    result = ArchiveRetriever(
        token_counter=_Counter(),
        recent_records=0,
    ).retrieve(
        f"show artifact {artifact_id}",
        archives=[archive],
        active_paper_ids=[],
    )

    assert [item.archive_id for item in result] == ["archive-with-artifact"]


def test_context_view_renderer_is_stable_and_capsule_keeps_recent_turns() -> None:
    view = ContextViewBuilder(token_counter=_Counter(), recent_turns=2).build(
        current_goal="current goal",
        active_paper_ids=["paper-2", "paper-1"],
        messages=[
            {"role": "user", "id": "m1", "content": "first"},
            {"role": "assistant", "id": "a1", "content": "answer"},
            {"role": "user", "id": "m2", "content": "current goal"},
        ],
        archives=[],
        continuation_capsule={"current_goal": "capsule goal"},
    )
    rendered = render_context_view(view)
    assert rendered == render_context_view(view)
    assert '"current_goal":"capsule goal"' in rendered
    assert view.messages == [
        {"role": "user", "id": "m1", "content": "first"},
        {"role": "assistant", "id": "a1", "content": "answer"},
        {"role": "user", "id": "m2", "content": "current goal"},
    ]
    assert view.active_projection is not None
    assert view.active_projection.active_paper_ids == ["paper-1", "paper-2"]


def test_capsule_view_keeps_configured_completed_turns_plus_current_request() -> None:
    messages = [
        {"role": "user", "id": "u1", "content": "q1"},
        {"role": "assistant", "id": "a1", "content": "a1"},
        {"role": "user", "id": "u2", "content": "q2"},
        {"role": "assistant", "id": "a2", "content": "a2"},
        {"role": "user", "id": "u3", "content": "q3"},
        {"role": "assistant", "id": "a3", "content": "a3"},
        {"role": "user", "id": "u4", "content": "current"},
    ]

    view = ContextViewBuilder(token_counter=_Counter(), recent_turns=2).build(
        current_goal="current",
        active_paper_ids=[],
        messages=messages,
        archives=[],
        continuation_capsule={"current_goal": "q3"},
    )

    assert [item["id"] for item in view.messages] == ["u2", "a2", "u3", "a3", "u4"]


def test_context_view_budget_counts_complete_model_request_envelope() -> None:
    class RecordingCounter:
        def __init__(self) -> None:
            self.messages = None
            self.tool_schemas = None

        def count_messages(self, messages, *, tool_schemas=()) -> int:
            self.messages = list(messages)
            self.tool_schemas = list(tool_schemas)
            return 123

        def count_text(self, value: str) -> int:
            return max(1, len(value) // 4)

    counter = RecordingCounter()
    tool_schemas = [
        {
            "name": "retrieve_paper_evidence",
            "parameters": {"type": "object"},
        }
    ]
    view = ContextViewBuilder(
        token_counter=counter,
        fixed_messages=[{"role": "system", "content": "fixed policy"}],
        tool_schemas=tool_schemas,
    ).build(
        current_goal="current goal",
        active_paper_ids=[],
        messages=[{"role": "user", "id": "m1", "content": "current goal"}],
        archives=[],
    )

    assert view.input_tokens == 123
    assert counter.messages is not None
    assert counter.messages[0] == {"role": "system", "content": "fixed policy"}
    context_message = counter.messages[-1]
    assert context_message.type == "human"
    assert context_message.content.startswith("PaperPilot Context View:\n")
    assert '"input_tokens"' not in context_message.content
    assert counter.tool_schemas == tool_schemas


def test_context_view_carries_hash_bound_artifact_and_evidence_authority() -> None:
    archive = _archive(
        "archive-1",
        created_at="2026-09-01T00:00:00+00:00",
        goal="goal",
    )
    artifact = ArtifactRef(
        artifact_id="artifact-1",
        sha256="a" * 64,
        token_estimate=10,
        preview="preview",
    )
    archive.seed_json["artifact_refs"] = [artifact.model_dump(mode="json")]
    archive.seed_json["evidence_refs"] = ["evidence-1"]

    view = ContextViewBuilder(token_counter=_Counter()).build(
        current_goal="goal",
        active_paper_ids=["paper-1"],
        messages=[{"role": "user", "content": "goal"}],
        archives=[archive],
    )

    assert view.authority_artifact_ids == ["artifact-1"]
    assert view.authority_artifact_refs == [artifact]
    assert view.authority_evidence_ids == ["evidence-1"]


def test_minimal_safe_view_keeps_one_completed_turn_current_request_and_refs() -> None:
    artifact = ArtifactRef(
        artifact_id="artifact-1",
        sha256="a" * 64,
        token_estimate=10,
        preview="preview",
    )
    view = ContextViewBuilder(token_counter=_Counter()).build(
        current_goal="current",
        active_paper_ids=["paper-1"],
        messages=[
            {"role": "user", "id": "u1", "content": "q1"},
            {"role": "assistant", "id": "a1", "content": "a1"},
            {"role": "user", "id": "u2", "content": "q2"},
            {"role": "assistant", "id": "a2", "content": "a2"},
            {"role": "user", "id": "u3", "content": "current"},
        ],
        archives=[],
        continuation_capsule={
            "current_goal": "q2",
            "artifact_refs": [artifact.model_dump(mode="json")],
            "evidence_refs": ["evidence-1"],
        },
    )

    minimal = ContextViewBuilder(token_counter=_Counter()).build_minimal_safe_view(view)

    assert [item["id"] for item in minimal.messages] == ["u2", "a2", "u3"]
    assert minimal.authority_artifact_refs == [artifact]
    assert minimal.authority_evidence_ids == ["evidence-1"]
