"""Deterministic tool-result classification, previews, and ingestion."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping
from typing import Any
from xml.sax.saxutils import escape

from paperpilot.deep_reading.context_management.models import (
    ArtifactRef,
    FutureRetention,
    InitialAction,
    ToolResultDisposition,
)
from paperpilot.web.context_artifacts import ArtifactPutRequest


BUSINESS_PREVIEW_TOOLS = {
    "search_related_papers",
    "prepare_paper",
    "retrieve_paper_evidence",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class DeterministicNoiseClassifier:
    """Classify only facts that can be proven without a model call."""

    def classify(
        self,
        raw_result: Any,
        *,
        existing_artifact_id: str | None = None,
        business_rejected: bool = False,
        state_duplicate: bool = False,
        same_tool_fingerprint: bool = False,
    ) -> str | None:
        if raw_result is None or (isinstance(raw_result, str) and not raw_result.strip()):
            return "empty_result"
        if business_rejected:
            return "business_rejected"
        if state_duplicate:
            return "state_duplicate"
        if same_tool_fingerprint and existing_artifact_id is not None:
            return "duplicate_result"
        if isinstance(raw_result, Mapping):
            keys = {str(key).casefold() for key in raw_result}
            values = {str(value).casefold() for value in raw_result.values()}
            if "protocol" in keys and any("echo" in value for value in values):
                return "protocol_echo"
            if "status" in keys and str(raw_result.get("status")).casefold() in {
                "progress",
                "heartbeat",
            }:
                return "progress_metadata"
            if keys and keys <= {"debug", "trace", "request_id", "elapsed_ms"}:
                return "debug_metadata"
        return None


class ToolResultPolicy:
    def __init__(self, *, inline_max_tokens: int = 2_000) -> None:
        if inline_max_tokens < 1:
            raise ValueError("inline_max_tokens must be positive")
        self._inline_max_tokens = inline_max_tokens
        self._frozen: dict[tuple[str, str], ToolResultDisposition] = {}
        self._authority_by_digest: dict[tuple[str, str], ArtifactRef] = {}
        self._noise = DeterministicNoiseClassifier()

    def decide(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        raw_result: Any,
        token_estimate: int,
        artifact_ref: ArtifactRef | None,
        current_step_needs_content: bool,
        existing_artifact_id: str | None = None,
        business_rejected: bool = False,
        state_duplicate: bool = False,
        same_tool_fingerprint: bool = False,
        protected: bool = False,
    ) -> ToolResultDisposition:
        content_sha256 = _digest(raw_result)
        key = (tool_call_id, content_sha256)
        frozen = self._frozen.get(key)
        if frozen is not None:
            if frozen.artifact_ref is None and artifact_ref is not None:
                frozen = ToolResultDisposition(
                    tool_call_id=frozen.tool_call_id,
                    content_sha256=frozen.content_sha256,
                    initial_action=frozen.initial_action,
                    future_retention=frozen.future_retention,
                    preview=artifact_ref.preview,
                    artifact_ref=artifact_ref,
                    result_id=frozen.result_id,
                    reason=frozen.reason,
                )
                self._frozen[key] = frozen
                self._authority_by_digest[(tool_name, content_sha256)] = artifact_ref
            return frozen

        authority = self._authority_by_digest.get((tool_name, content_sha256))
        reason = self._noise.classify(
            raw_result,
            existing_artifact_id=existing_artifact_id or (
                authority.artifact_id if authority is not None else None
            ),
            business_rejected=business_rejected,
            state_duplicate=state_duplicate,
            same_tool_fingerprint=same_tool_fingerprint,
        )
        if reason is None and authority is not None:
            reason = "duplicate_result"
            existing_artifact_id = authority.artifact_id
            artifact_ref = authority

        if reason is not None:
            disposition = ToolResultDisposition(
                tool_call_id=tool_call_id,
                content_sha256=content_sha256,
                initial_action=InitialAction.DROP_NOW,
                future_retention=FutureRetention.PROTECTED,
                preview="dropped",
                artifact_ref=artifact_ref,
                result_id=f"result-{content_sha256[:16]}",
                reason=reason,
            )
        else:
            action = (
                InitialAction.KEEP_INLINE
                if token_estimate <= self._inline_max_tokens
                and current_step_needs_content
                and artifact_ref is None
                else InitialAction.EXTERNALIZE_NOW
            )
            retention = (
                FutureRetention.PROTECTED
                if protected or tool_name not in BUSINESS_PREVIEW_TOOLS
                else FutureRetention.CLEARABLE_AFTER_USE
            )
            disposition = ToolResultDisposition(
                tool_call_id=tool_call_id,
                content_sha256=content_sha256,
                initial_action=action,
                future_retention=retention,
                preview=artifact_ref.preview if artifact_ref is not None else "pending",
                artifact_ref=artifact_ref,
                result_id=f"result-{content_sha256[:16]}",
                reason=action.value.lower(),
            )
        self._frozen[key] = disposition
        if disposition.artifact_ref is not None:
            self._authority_by_digest[(tool_name, content_sha256)] = disposition.artifact_ref
        return disposition


@dataclass(frozen=True)
class ToolResultIngestRequest:
    conversation_id: str
    task_id: str
    tool_call_id: str
    tool_name: str
    raw_result: Any
    current_step_needs_content: bool = True
    business_rejected: bool = False
    state_duplicate: bool = False
    same_tool_fingerprint: bool = False
    protected: bool = False


@dataclass(frozen=True)
class IngestedToolResult:
    disposition: ToolResultDisposition
    model_content: str


class SearchPreviewRenderer:
    def render(self, raw_result: Any) -> str:
        items = raw_result.get("items", []) if isinstance(raw_result, Mapping) else raw_result
        if not isinstance(items, list):
            items = []
        rendered: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            entry: dict[str, Any] = {
                "external_id": item.get("external_id", item.get("id", "")),
                "title": item.get("title", ""),
            }
            if index < 5 and item.get("abstract") is not None:
                entry["abstract"] = str(item["abstract"])[:300]
            rendered.append(entry)
        return _canonical_json({"items": rendered})


class PreparePreviewRenderer:
    def render(self, raw_result: Any) -> str:
        if isinstance(raw_result, Mapping):
            value = {
                "external_id": raw_result.get("external_id", raw_result.get("paper_id", "")),
                "status": raw_result.get("status", "prepared"),
            }
        else:
            value = {"external_id": "", "status": "prepared"}
        return _canonical_json(value)


class EvidencePreviewRenderer:
    def __init__(self, token_counter) -> None:
        self._token_counter = token_counter

    def render(self, raw_result: Any) -> str:
        if not isinstance(raw_result, Mapping):
            return _canonical_json({"evidence_items": [], "summary_item_ids": []})
        pool = raw_result.get("evidence_pool", raw_result)
        if not isinstance(pool, Mapping):
            pool = {}
        source = raw_result.get("evidence_items", pool.get("items", []))
        items = source if isinstance(source, list) else []
        summary_ids = raw_result.get("summary_item_ids", pool.get("summary_items", []))
        summary_ids = list(summary_ids) if isinstance(summary_ids, list) else []
        by_id = {
            str(item.get("id")): item
            for item in items
            if isinstance(item, Mapping) and item.get("id") is not None
        }
        order = [item_id for item_id in summary_ids if str(item_id) in by_id]
        order.extend(
            str(item.get("id"))
            for item in items
            if isinstance(item, Mapping) and str(item.get("id")) not in order
        )
        output: list[dict[str, Any]] = []
        chunk_budget = 2_000
        for item_id in order:
            source_item = by_id[item_id]
            entry = {
                "id": source_item.get("id"),
                "paper_external_id": source_item.get(
                    "paper_external_id", source_item.get("paper_id", "")
                ),
                "paper_title": source_item.get("paper_title", ""),
                "score": source_item.get("score", source_item.get("best_score", 0)),
                "supports": source_item.get("supports", []),
            }
            chunk = source_item.get("chunk_text")
            if chunk is not None and chunk_budget > 0:
                chunk_text = str(chunk)
                chunk_tokens = self._token_counter.count_text(chunk_text)
                if chunk_tokens > chunk_budget:
                    chunk_text = self._token_counter.truncate_text(chunk_text, chunk_budget)
                    chunk_tokens = self._token_counter.count_text(chunk_text)
                if chunk_text:
                    entry["chunk_text"] = chunk_text
                    chunk_budget -= chunk_tokens
            output.append(entry)
        return _canonical_json(
            {"evidence_items": output, "summary_item_ids": [str(item) for item in summary_ids]}
        )


class ToolResultIngestor:
    def __init__(
        self,
        *,
        artifact_store,
        token_counter,
        inline_max_tokens: int = 2_000,
        event_sink=None,
    ) -> None:
        self._artifact_store = artifact_store
        self._token_counter = token_counter
        self._policy = ToolResultPolicy(inline_max_tokens=inline_max_tokens)
        self._event_sink = event_sink or (lambda _kind, _payload: None)

    def ingest(
        self,
        request: ToolResultIngestRequest,
        *,
        token_estimate: int | None = None,
    ) -> IngestedToolResult:
        canonical = _canonical_json(request.raw_result)
        estimated = (
            self._token_counter.count_text(canonical)
            if token_estimate is None
            else token_estimate
        )
        preliminary = self._policy.decide(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            raw_result=request.raw_result,
            token_estimate=estimated,
            artifact_ref=None,
            current_step_needs_content=request.current_step_needs_content,
            business_rejected=request.business_rejected,
            state_duplicate=request.state_duplicate,
            same_tool_fingerprint=request.same_tool_fingerprint,
            protected=request.protected,
        )
        if preliminary.initial_action is InitialAction.DROP_NOW:
            content = render_drop(preliminary)
            self._event_sink(
                "tool_result_dropped",
                {
                    "stage": "research",
                    "tool_name": request.tool_name,
                    "tool_call_id": request.tool_call_id,
                    "result_id": preliminary.result_id,
                    "sha256": preliminary.content_sha256,
                    "reason": preliminary.reason,
                    "before_tokens": estimated,
                    "after_tokens": 0,
                    "reclaimed_tokens": estimated,
                },
            )
            return IngestedToolResult(preliminary, content)
        if request.tool_name == "prepare_paper":
            return IngestedToolResult(
                preliminary,
                PreparePreviewRenderer().render(request.raw_result),
            )
        if preliminary.initial_action is InitialAction.KEEP_INLINE:
            return IngestedToolResult(preliminary, canonical)

        preview = _render_preview(request.tool_name, request.raw_result, self._token_counter)
        try:
            record = self._artifact_store.put(
                ArtifactPutRequest(
                    conversation_id=request.conversation_id,
                    task_id=request.task_id,
                    tool_call_id=request.tool_call_id,
                    tool_name=request.tool_name,
                    kind=request.tool_name,
                    payload=request.raw_result,
                    preview=preview,
                    initial_action=InitialAction.EXTERNALIZE_NOW.value,
                    future_retention=preliminary.future_retention.value,
                )
            )
        except Exception as exc:
            raise ValueError("unable to externalize tool result") from exc
        artifact_ref = ArtifactRef(
            artifact_id=record.artifact_id,
            sha256=record.sha256,
            token_estimate=record.token_estimate,
            preview=record.preview,
        )
        # Freeze the final disposition under the same key before rendering it.
        final = self._policy.decide(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            raw_result=request.raw_result,
            token_estimate=estimated,
            artifact_ref=artifact_ref,
            current_step_needs_content=request.current_step_needs_content,
            protected=request.protected,
        )
        content = render_externalized(final)
        after_tokens = self._token_counter.count_text(content)
        self._event_sink(
            "artifact_externalized",
            {
                "stage": "research",
                "tool_name": request.tool_name,
                "tool_call_id": request.tool_call_id,
                "artifact_id": record.artifact_id,
                "sha256": record.sha256,
                "token_estimate": record.token_estimate,
                "reason": final.reason,
                "before_tokens": estimated,
                "after_tokens": after_tokens,
                "reclaimed_tokens": max(0, estimated - after_tokens),
            },
        )
        return IngestedToolResult(final, content)


def render_drop(
    disposition: ToolResultDisposition,
    *,
    reason: str | None = None,
    result_id: str | None = None,
) -> str:
    attributes = (
        ("status", "dropped"),
        ("reason", reason or disposition.reason),
        ("result_id", result_id or disposition.result_id),
        ("tool_call_id", disposition.tool_call_id),
        ("sha256", disposition.content_sha256),
    )
    rendered = " ".join(
        f'{name}="{escape(str(value))}"' for name, value in attributes
    )
    return f"<tool_result {rendered} />"


def render_externalized(disposition: ToolResultDisposition) -> str:
    if disposition.artifact_ref is None:
        raise ValueError("externalized disposition requires an artifact")
    ref = disposition.artifact_ref
    preview = escape(ref.preview)
    return (
        '<tool_result status="externalized">'
        f'<artifact_ref id="{escape(ref.artifact_id)}" '
        f'sha256="{escape(ref.sha256)}" '
        f'token_estimate="{ref.token_estimate}" />'
        f"<preview>{preview}</preview>"
        "<read_tools>read_artifact_slice,search_artifact</read_tools>"
        "</tool_result>"
    )


def _render_preview(tool_name: str, raw_result: Any, token_counter) -> str:
    if tool_name == "search_related_papers":
        return SearchPreviewRenderer().render(raw_result)
    if tool_name == "retrieve_paper_evidence":
        return EvidencePreviewRenderer(token_counter).render(raw_result)
    return _canonical_json(raw_result)


def _drop_reason(raw_result: Any) -> str:
    reason = DeterministicNoiseClassifier().classify(raw_result)
    return reason or "dropped"
