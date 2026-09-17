"""Strict domain contracts for context management."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from paperpilot.deep_reading.context_management.models import (
    ActiveProjection,
    ArtifactRef,
    ContinuationCapsule,
    FutureRetention,
    InitialAction,
    ProtectedText,
    ToolResultDisposition,
)


def test_context_action_and_retention_enums_are_closed():
    assert [item.value for item in InitialAction] == [
        "DROP_NOW",
        "EXTERNALIZE_NOW",
        "KEEP_INLINE",
    ]
    assert [item.value for item in FutureRetention] == [
        "PROTECTED",
        "CLEARABLE_AFTER_USE",
    ]


def test_context_models_forbid_extra_fields_and_blank_ids():
    with pytest.raises(ValidationError):
        ArtifactRef(
            artifact_id="",
            sha256="a" * 64,
            token_estimate=1,
            preview="preview",
            unexpected="field",
        )

    with pytest.raises(ValidationError):
        ArtifactRef(
            artifact_id="artifact-1",
            sha256="a" * 64,
            token_estimate=1,
            preview=" ",
        )


def test_protected_text_recomputes_utf8_hash():
    text = "约束：必须保留否定词"
    protected = ProtectedText(
        protected_id="constraint-1",
        exact_text=text,
        sha256=__import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
    )
    assert protected.sha256
    with pytest.raises(ValidationError, match="sha256"):
        ProtectedText(
            protected_id="constraint-1",
            exact_text=text,
            sha256="b" * 64,
        )


def test_tool_result_disposition_is_frozen_and_references_are_unique():
    artifact = ArtifactRef(
        artifact_id="artifact-1",
        sha256="a" * 64,
        token_estimate=10,
        preview="preview",
    )
    disposition = ToolResultDisposition(
        tool_call_id="call-1",
        content_sha256="b" * 64,
        initial_action=InitialAction.EXTERNALIZE_NOW,
        future_retention=FutureRetention.CLEARABLE_AFTER_USE,
        preview="preview",
        artifact_ref=artifact,
    )
    with pytest.raises(ValidationError):
        disposition.preview = "changed"

    with pytest.raises(ValidationError, match="unique"):
        ContinuationCapsule(
            current_goal="goal",
            exact_constraints=[],
            decisions_and_rationales=[],
            active_papers=[],
            evidence_refs=[],
            artifact_refs=[artifact, artifact],
            completed_work=[],
            verification=[],
            failed_paths=[],
            unresolved_todos=[],
            rollback_notes=[],
            source_archive_ids=["archive-1"],
            capsule_version="continuation-v1",
        )

    with pytest.raises(ValidationError, match="unique"):
        ContinuationCapsule(
            current_goal="goal",
            evidence_refs=["evidence-1", "evidence-1"],
        )


def test_projection_rejects_duplicate_references():
    with pytest.raises(ValidationError, match="unique"):
        ActiveProjection(
            current_goal="goal",
            active_constraints=[],
            active_decisions=[],
            active_paper_ids=["paper-1", "paper-1"],
            open_todos=[],
            failed_verifications=[],
            unresolved_questions=[],
        )
