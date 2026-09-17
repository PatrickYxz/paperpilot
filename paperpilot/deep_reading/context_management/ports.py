"""Ports separating context policy from Web persistence and providers."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class TokenCounter(Protocol):
    def count_messages(self, messages: Sequence[Any], *, tool_schemas=()) -> int: ...

    def count_text(self, text: str) -> int: ...

    def truncate_text(self, text: str, max_tokens: int) -> str: ...


class ContextArtifactPort(Protocol):
    def put(self, request: Any) -> Any: ...

    def get(self, artifact_id: str, *, conversation_id: str) -> Any | None: ...

    def read_slice(
        self,
        artifact_id: str,
        *,
        conversation_id: str,
        cursor: int,
        max_tokens: int,
    ) -> Any: ...

    def search(
        self,
        artifact_id: str,
        *,
        conversation_id: str,
        query: str,
        max_matches: int,
    ) -> Any: ...


class TurnArchivePort(Protocol):
    def seed(self, seed: Any) -> Any: ...

    def list_for_conversation(self, conversation_id: str) -> list[Any]: ...


class CompressionStatePort(Protocol):
    def read(self, conversation_id: str, compressor_version: str) -> Any: ...

    def record_outcome(self, outcome: Any) -> Any: ...


@dataclass(frozen=True)
class EditedRequest:
    messages: tuple[Any, ...]
    input_tokens: int


class ContextEditingAdapter(Protocol):
    def edit(self, request: Any, dispositions: Sequence[Any]) -> EditedRequest: ...
