"""Deterministic tool-result lifecycle and preview policy tests."""
from __future__ import annotations

import json

import pytest

from paperpilot.deep_reading.context_management.artifacts import (
    DeterministicNoiseClassifier,
    EvidencePreviewRenderer,
    SearchPreviewRenderer,
    ToolResultIngestRequest,
    ToolResultIngestor,
    ToolResultPolicy,
    render_drop,
    render_externalized,
)
from paperpilot.deep_reading.context_management.budget import ModelAwareTokenCounter
from paperpilot.deep_reading.context_management.models import (
    FutureRetention,
    InitialAction,
)
from paperpilot.web.context_artifacts import ArtifactPutRequest


class _ArtifactStore:
    def __init__(self, record=None, failure=None):
        self.record = record
        self.failure = failure
        self.requests = []

    def put(self, request):
        self.requests.append(request)
        if self.failure is not None:
            raise self.failure
        return self.record


class _FixedCounter(ModelAwareTokenCounter):
    def __init__(self, values):
        super().__init__()
        self.values = iter(values)
        self.model_calls = 0

    def count_text(self, text):
        self.model_calls += 1
        return next(self.values)


def test_noise_classifier_drops_only_deterministic_noise_without_a_model():
    classifier = DeterministicNoiseClassifier()
    assert classifier.classify("   ") == "empty_result"
    assert classifier.classify({"status": "progress", "percent": 5}) == "progress_metadata"
    assert classifier.classify({"debug": "trace"}) == "debug_metadata"
    assert classifier.classify({"protocol": "tool_call_echo"}) == "protocol_echo"
    assert classifier.classify({"valid": False}, business_rejected=True) == "business_rejected"
    assert classifier.classify({"state": "known"}, state_duplicate=True) == "state_duplicate"
    assert classifier.classify({"value": "ambiguous"}) is None


def test_policy_externalizes_semantically_uncertain_content_and_freezes_result():
    policy = ToolResultPolicy(inline_max_tokens=2_000)
    first = policy.decide(
        tool_call_id="call-1",
        tool_name="other_tool",
        raw_result={"value": "unknown"},
        token_estimate=1,
        artifact_ref=None,
        current_step_needs_content=False,
    )
    second = policy.decide(
        tool_call_id="call-1",
        tool_name="other_tool",
        raw_result={"value": "unknown"},
        token_estimate=9_999,
        artifact_ref=None,
        current_step_needs_content=True,
    )
    assert first == second
    assert first.initial_action is InitialAction.EXTERNALIZE_NOW


def test_policy_has_exact_inline_boundary_and_large_failure_is_not_truncated():
    policy = ToolResultPolicy(inline_max_tokens=2_000)
    inline = policy.decide(
        tool_call_id="call-inline",
        tool_name="retrieve_paper_evidence",
        raw_result={"chunk": "x"},
        token_estimate=2_000,
        artifact_ref=None,
        current_step_needs_content=True,
    )
    assert inline.initial_action is InitialAction.KEEP_INLINE
    with pytest.raises(ValueError, match="externalize"):
        ToolResultIngestor(
            artifact_store=_ArtifactStore(failure=RuntimeError("disk")),
            token_counter=ModelAwareTokenCounter(),
            inline_max_tokens=2_000,
        ).ingest(
            ToolResultIngestRequest(
                conversation_id="conv-1",
                task_id="task-1",
                tool_call_id="call-large",
                tool_name="other_tool",
                raw_result={"large": True},
                current_step_needs_content=True,
            ),
            token_estimate=2_001,
        )


def test_search_preview_keeps_all_ids_titles_and_only_five_abstracts():
    items = [
        {
            "external_id": f"paper-{index}",
            "title": f"Title {index}",
            "authors": ["Author"],
            "abstract": "a" * 500,
        }
        for index in range(8)
    ]
    rendered = SearchPreviewRenderer().render(items)
    payload = json.loads(rendered)
    assert [item["external_id"] for item in payload["items"]] == [
        f"paper-{index}" for index in range(8)
    ]
    assert [item["title"] for item in payload["items"]] == [
        f"Title {index}" for index in range(8)
    ]
    assert sum("abstract" in item for item in payload["items"]) == 5
    assert all(len(item.get("abstract", "")) <= 300 for item in payload["items"])
    assert rendered == SearchPreviewRenderer().render(items)


def test_evidence_preview_preserves_metadata_and_prioritizes_summary_items():
    payload = {
        "evidence_items": [
            {
                "id": "ev-1",
                "paper_external_id": "paper-1",
                "paper_title": "Paper",
                "score": 0.9,
                "supports": ["method"],
                "chunk_text": "summary chunk",
            },
            {
                "id": "ev-2",
                "paper_external_id": "paper-1",
                "paper_title": "Paper",
                "score": 0.8,
                "supports": ["result"],
                "chunk_text": "other chunk",
            },
        ],
        "summary_item_ids": ["ev-1"],
    }
    rendered = EvidencePreviewRenderer(ModelAwareTokenCounter()).render(payload)
    result = json.loads(rendered)
    assert result["summary_item_ids"] == ["ev-1"]
    assert [item["id"] for item in result["evidence_items"]] == ["ev-1", "ev-2"]
    assert result["evidence_items"][0]["chunk_text"] == "summary chunk"


def test_drop_and_externalized_renderers_use_stable_safe_references():
    policy = ToolResultPolicy()
    dropped = policy.decide(
        tool_call_id="call<&",
        tool_name="search_related_papers",
        raw_result=" ",
        token_estimate=0,
        artifact_ref=None,
        current_step_needs_content=True,
    )
    drop_text = render_drop(dropped, reason="duplicate_result", result_id="result-1")
    assert drop_text.startswith("<tool_result ")
    assert "status=\"dropped\"" in drop_text
    assert "call&lt;&amp;" in drop_text
    assert set(drop_text.split("<tool_result ", 1)[1].split(" />", 1)[0].split('"')[0:0]) == set()

    with pytest.raises(ValueError, match="artifact"):
        render_externalized(dropped)


def test_ingestor_emits_redacted_drop_event_without_tool_body():
    events = []
    ingestor = ToolResultIngestor(
        artifact_store=_ArtifactStore(),
        token_counter=ModelAwareTokenCounter(),
        event_sink=lambda kind, payload: events.append((kind, payload)),
    )

    ingestor.ingest(
        ToolResultIngestRequest(
            conversation_id="conv-1",
            task_id="task-1",
            tool_call_id="call-drop",
            tool_name="search_related_papers",
            raw_result="   ",
        ),
        token_estimate=17,
    )

    assert [kind for kind, _payload in events] == ["tool_result_dropped"]
    payload = events[0][1]
    assert payload["stage"] == "research"
    assert payload["before_tokens"] == 17
    assert payload["after_tokens"] == 0
    assert payload["reclaimed_tokens"] == 17
    assert payload["reason"] == "empty_result"
    assert "raw_result" not in payload
    assert "   " not in json.dumps(payload)
