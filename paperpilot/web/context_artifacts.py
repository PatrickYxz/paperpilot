"""Atomic local storage for immutable internal context artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any

from paperpilot.deep_reading.context_management.ports import ContextArtifactPort
from paperpilot.deep_reading.context_management.ports import TokenCounter
from paperpilot.web.store.records import ContextArtifactRecord, NewContextArtifact
from paperpilot.web.task_store import TaskStore


class ArtifactIntegrityError(ValueError):
    """Raised when an immutable artifact is missing or has a changed hash."""


@dataclass(frozen=True)
class ArtifactPutRequest:
    conversation_id: str
    task_id: str
    tool_call_id: str
    tool_name: str
    kind: str
    payload: Any
    preview: str
    initial_action: str
    future_retention: str


@dataclass(frozen=True)
class ArtifactSlice:
    artifact_id: str
    text: str
    next_cursor: int | None
    actual_tokens: int
    sha256: str


@dataclass(frozen=True)
class ArtifactMatch:
    start: int
    end: int
    snippet: str


@dataclass(frozen=True)
class ArtifactMatches:
    artifact_id: str
    matches: list[ArtifactMatch]
    actual_tokens: int
    sha256: str


class LocalContextArtifactStore(ContextArtifactPort):
    def __init__(
        self,
        *,
        root: Path,
        task_store: TaskStore,
        token_counter: TokenCounter,
        read_max_tokens: int,
    ) -> None:
        if read_max_tokens < 1:
            raise ValueError("read_max_tokens must be at least 1")
        self._root = Path(root)
        self._task_store = task_store
        self._token_counter = token_counter
        self._read_max_tokens = read_max_tokens

    def put(self, request: ArtifactPutRequest) -> ContextArtifactRecord:
        payload_bytes = _canonical_json(request.payload).encode("utf-8")
        digest = _sha256(payload_bytes)
        existing = self._task_store.find_context_artifact_by_digest(
            conversation_id=request.conversation_id,
            tool_name=request.tool_name,
            sha256=digest,
        )
        if existing is not None:
            return existing

        artifact_id = str(uuid.uuid4())
        storage_key = f"{artifact_id}.json"
        root = self._root
        root.mkdir(parents=True, exist_ok=True)
        final_path = _safe_artifact_path(root, storage_key)
        if final_path.exists() or final_path.is_symlink():
            raise ValueError("artifact destination already exists")
        temporary_path = root / f".{artifact_id}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary_path.open("xb") as handle:
                handle.write(payload_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            written_digest = _sha256(temporary_path.read_bytes())
            if written_digest != digest:
                raise ArtifactIntegrityError("artifact hash verification failed")
            os.replace(temporary_path, final_path)
            record = NewContextArtifact(
                artifact_id=artifact_id,
                conversation_id=request.conversation_id,
                task_id=request.task_id,
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                kind=request.kind,
                storage_key=storage_key,
                sha256=digest,
                byte_size=len(payload_bytes),
                token_estimate=self._token_counter.count_text(
                    payload_bytes.decode("utf-8")
                ),
                preview=request.preview,
                initial_action=request.initial_action,
                future_retention=request.future_retention,
                created_at=_utc_now(),
            )
            return self._task_store.create_context_artifact(record=record)
        except Exception:
            _unlink_if_unreferenced(
                self._task_store,
                artifact_id=artifact_id,
                path=final_path,
            )
            raise
        finally:
            if temporary_path.exists() and not temporary_path.is_symlink():
                temporary_path.unlink()

    def get(
        self,
        artifact_id: str,
        *,
        conversation_id: str,
    ) -> ContextArtifactRecord | None:
        return self._task_store.get_context_artifact(
            artifact_id,
            conversation_id=conversation_id,
        )

    def read_slice(
        self,
        artifact_id: str,
        *,
        conversation_id: str,
        cursor: int,
        max_tokens: int,
    ) -> ArtifactSlice:
        if cursor < 0:
            raise ValueError("cursor must be non-negative")
        self._validate_read_max_tokens(max_tokens)
        record, text = self._load_text(artifact_id, conversation_id=conversation_id)
        if cursor > len(text):
            raise ValueError("cursor exceeds artifact length")
        chunk = self._token_counter.truncate_text(text[cursor:], max_tokens)
        next_cursor = None if cursor + len(chunk) >= len(text) else cursor + len(chunk)
        return ArtifactSlice(
            artifact_id=record.artifact_id,
            text=chunk,
            next_cursor=next_cursor,
            actual_tokens=self._token_counter.count_text(chunk),
            sha256=record.sha256,
        )

    def search(
        self,
        artifact_id: str,
        *,
        conversation_id: str,
        query: str,
        max_matches: int,
    ) -> ArtifactMatches:
        if not query.strip():
            raise ValueError("query must not be empty")
        if max_matches < 1 or max_matches > 10:
            raise ValueError("max_matches must be between 1 and 10")
        record, text = self._load_text(artifact_id, conversation_id=conversation_id)
        folded_text = text.casefold()
        folded_query = query.casefold()
        matches: list[ArtifactMatch] = []
        offset = 0
        while len(matches) < max_matches:
            start = folded_text.find(folded_query, offset)
            if start < 0:
                break
            end = start + len(folded_query)
            snippet = text[max(0, start - 80) : min(len(text), end + 80)]
            matches.append(ArtifactMatch(start=start, end=end, snippet=snippet))
            offset = max(end, start + 1)

        bounded: list[ArtifactMatch] = []
        used_tokens = 0
        for match in matches:
            remaining = self._read_max_tokens - used_tokens
            if remaining <= 0:
                break
            snippet = self._token_counter.truncate_text(match.snippet, remaining)
            if not snippet and match.snippet:
                break
            used_tokens += self._token_counter.count_text(snippet)
            bounded.append(
                ArtifactMatch(
                    start=match.start,
                    end=match.end,
                    snippet=snippet,
                )
            )
        return ArtifactMatches(
            artifact_id=record.artifact_id,
            matches=bounded,
            actual_tokens=used_tokens,
            sha256=record.sha256,
        )

    def _load_text(
        self,
        artifact_id: str,
        *,
        conversation_id: str,
    ) -> tuple[ContextArtifactRecord, str]:
        record = self.get(artifact_id, conversation_id=conversation_id)
        if record is None:
            raise ValueError("unknown artifact or conversation mismatch")
        path = _safe_artifact_path(self._root, record.storage_key)
        if not path.exists() or path.is_symlink() or not path.is_file():
            raise ArtifactIntegrityError("artifact file is missing")
        data = path.read_bytes()
        if _sha256(data) != record.sha256:
            raise ArtifactIntegrityError("artifact hash mismatch")
        try:
            return record, data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactIntegrityError("artifact is not valid UTF-8") from exc

    def _validate_read_max_tokens(self, max_tokens: int) -> None:
        if max_tokens < 1 or max_tokens > self._read_max_tokens:
            raise ValueError(
                f"max_tokens must be between 1 and {self._read_max_tokens}"
            )


def _safe_artifact_path(root: Path, storage_key: str) -> Path:
    if not storage_key or Path(storage_key).is_absolute():
        raise ValueError("invalid artifact storage key")
    if "/" in storage_key or "\\" in storage_key or Path(storage_key).name != storage_key:
        raise ValueError("invalid artifact storage key")
    if not storage_key.endswith(".json"):
        raise ValueError("invalid artifact storage key")
    try:
        uuid.UUID(storage_key[:-5])
    except ValueError as exc:
        raise ValueError("invalid artifact storage key") from exc
    root_resolved = root.resolve()
    path = root / storage_key
    if path.is_symlink():
        raise ValueError("artifact storage key points to a symlink")
    if path.resolve().parent != root_resolved:
        raise ValueError("invalid artifact storage key")
    return path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unlink_if_unreferenced(
    task_store: TaskStore,
    *,
    artifact_id: str,
    path: Path,
) -> None:
    try:
        referenced = task_store.is_context_artifact_referenced(artifact_id=artifact_id)
    except Exception:
        referenced = True
    if not referenced and path.exists() and not path.is_symlink():
        path.unlink()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
