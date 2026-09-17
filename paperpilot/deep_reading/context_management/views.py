"""Authoritative ActiveProjection, Archive retrieval, and model-view rendering."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import re
from typing import Any

from langchain.messages import HumanMessage

from .models import (
    ActiveProjection,
    ArtifactRef,
    ContextView,
    ProtectedText,
    RetrievedArchiveView,
)
from .ports import TokenCounter


_HISTORY_TERMS = ("历史", "原因", "变化", "之前", "history", "why", "changed", "before")
_ID_PATTERN = re.compile(
    r"(?:archive[-_][A-Za-z0-9_-]+|artifact[-_][A-Za-z0-9_-]+|"
    r"(?:evg|paper)[-_][A-Za-z0-9_-]+|\d{4}\.\d{4,5}(?:v\d+)?|"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[1-5][0-9A-Fa-f]{3}-"
    r"[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12})"
)


class ContextCapacityExhaustedError(RuntimeError):
    """The deterministic protected minimum cannot fit the model input budget."""

    error_code = "context_capacity_exhausted"
    public_message = "The conversation context exceeds the available model capacity."


class ActiveProjectionBuilder:
    """Rebuild current effective state from current business inputs and archives."""

    def build(
        self,
        *,
        current_goal: str,
        active_paper_ids: Sequence[str],
        archives: Sequence[object],
        open_todos: Sequence[str] = (),
        failed_verifications: Sequence[str] = (),
        unresolved_questions: Sequence[str] = (),
    ) -> ActiveProjection:
        superseded_ids = _superseded_ids(archives)
        protected: dict[str, ProtectedText] = {}
        for archive in sorted(archives, key=_archive_sort_key):
            seed = _seed(archive)
            for field in ("constraints", "decisions"):
                values = seed.get(field, [])
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, Mapping):
                        continue
                    item = ProtectedText.model_validate(value)
                    if item.protected_id not in superseded_ids:
                        protected[item.protected_id] = item

        constraints = _effective_texts(archives, "constraints", superseded_ids)
        decisions = _effective_texts(archives, "decisions", superseded_ids)
        todo_values = list(open_todos)
        failed_values = list(failed_verifications)
        question_values = list(unresolved_questions)
        for archive in archives:
            seed = _seed(archive)
            todo_values.extend(_strings(seed.get("unresolved_todos", [])))
            failed_values.extend(
                item
                for item in _strings(seed.get("verification", []))
                if "fail" in item.casefold()
            )
            question_values.extend(_strings(seed.get("failed_paths", [])))
        return ActiveProjection(
            current_goal=current_goal,
            active_constraints=_unique(constraints),
            active_decisions=_unique(decisions),
            active_paper_ids=sorted(_unique(active_paper_ids)),
            open_todos=_unique(todo_values),
            failed_verifications=_unique(failed_values),
            unresolved_questions=_unique(question_values),
            protected_items=[protected[key] for key in sorted(protected)],
        )


class ArchiveRetriever:
    """Select relevant archives under deterministic count and token budgets."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        budget_tokens: int = 4_000,
        max_records: int = 5,
        recent_records: int = 2,
    ) -> None:
        if budget_tokens < 1 or max_records < 1 or recent_records < 0:
            raise ValueError("invalid archive retrieval budget")
        self._token_counter = token_counter
        self._budget_tokens = budget_tokens
        self._max_records = max_records
        self._recent_records = recent_records

    def retrieve(
        self,
        request: str,
        *,
        archives: Sequence[object],
        active_paper_ids: Sequence[str],
    ) -> list[RetrievedArchiveView]:
        if not request.strip():
            return []
        ordered = sorted(archives, key=_archive_sort_key)
        superseded_ids = _superseded_ids(ordered)
        exact_ids = {item.casefold() for item in _ID_PATTERN.findall(request)}
        request_folded = request.casefold()
        history_query = any(term in request_folded for term in _HISTORY_TERMS)
        eligible_recent = [
            item for item in reversed(ordered) if not _is_superseded(item, superseded_ids)
        ][: self._recent_records]
        recent_ids = {id(item) for item in eligible_recent}
        ranked: list[tuple[tuple[int, int, str], object]] = []
        for archive in ordered:
            if _archive_version(archive) == "legacy-summary-v1":
                continue
            if _is_superseded(archive, superseded_ids) and not history_query:
                continue
            identifiers = _identifiers(archive)
            exact = any(identifier.casefold() in exact_ids for identifier in identifiers)
            active = any(
                paper_id.casefold() in _archive_text(archive).casefold()
                for paper_id in active_paper_ids
            )
            related_history = history_query and _relation_matches(
                archive,
                request_folded,
                superseded_ids,
            )
            if not (exact or active or id(archive) in recent_ids or related_history):
                continue
            priority = 0 if exact else 1 if active else 2 if id(archive) in recent_ids else 3
            recency = -ordered.index(archive)
            ranked.append(((priority, recency, _archive_id(archive)), archive))
        ranked.sort(key=lambda item: item[0])

        selected: list[RetrievedArchiveView] = []
        used_tokens = 0
        for _rank, archive in ranked:
            if len(selected) >= self._max_records:
                break
            full = RetrievedArchiveView(
                archive_id=_archive_id(archive),
                terminal_status=_terminal_status(archive),
                seed=_seed(archive),
                narrative_summary=_narrative(archive),
                superseded=_is_superseded(archive, superseded_ids),
            )
            candidate = full
            candidate_tokens = self._token_counter.count_text(_canonical(candidate.model_dump(mode="json")))
            if used_tokens + candidate_tokens > self._budget_tokens:
                candidate = full.model_copy(
                    update={
                        "seed": _minimal_seed(full.seed),
                        "narrative_summary": None,
                    }
                )
                candidate_tokens = self._token_counter.count_text(
                    _canonical(candidate.model_dump(mode="json"))
                )
            if used_tokens + candidate_tokens > self._budget_tokens:
                continue
            selected.append(candidate)
            used_tokens += candidate_tokens
        return selected


class ContextViewBuilder:
    """Assemble the only dynamic model view used by Research and Answer nodes."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter,
        archive_retriever: ArchiveRetriever | None = None,
        fixed_messages: Sequence[Any] = (),
        trailing_messages: Sequence[Any] = (),
        tool_schemas: Sequence[Any] = (),
        recent_turns: int = 2,
    ) -> None:
        if recent_turns < 1:
            raise ValueError("recent_turns must be positive")
        self._token_counter = token_counter
        self._archive_retriever = archive_retriever or ArchiveRetriever(
            token_counter=token_counter
        )
        self._fixed_messages = tuple(fixed_messages)
        self._trailing_messages = tuple(trailing_messages)
        self._tool_schemas = tuple(tool_schemas)
        self._recent_turns = recent_turns

    def build(
        self,
        *,
        current_goal: str,
        active_paper_ids: Sequence[str],
        messages: Sequence[Mapping[str, Any]],
        archives: Sequence[object],
        open_todos: Sequence[str] = (),
        failed_verifications: Sequence[str] = (),
        unresolved_questions: Sequence[str] = (),
        archive_query: str | None = None,
        continuation_capsule: Mapping[str, Any] | None = None,
    ) -> ContextView:
        projection = ActiveProjectionBuilder().build(
            current_goal=current_goal,
            active_paper_ids=active_paper_ids,
            archives=archives,
            open_todos=open_todos,
            failed_verifications=failed_verifications,
            unresolved_questions=unresolved_questions,
        )
        selected_messages = [dict(message) for message in messages]
        if continuation_capsule is not None:
            selected_messages = select_recent_turns(
                selected_messages,
                current_goal=current_goal,
                completed_turn_count=self._recent_turns,
            )
        retrieved = self._archive_retriever.retrieve(
            archive_query or current_goal,
            archives=archives,
            active_paper_ids=active_paper_ids,
        )
        retrieved_ids = {item.archive_id for item in retrieved}
        authority_archives = [
            archive for archive in archives if _archive_id(archive) in retrieved_ids
        ]
        artifact_refs = _authority_artifact_refs(
            authority_archives,
            continuation_capsule,
        )
        artifact_ids = [item.artifact_id for item in artifact_refs]
        evidence_ids = _authority_strings(
            authority_archives,
            continuation_capsule,
            "evidence_refs",
        )
        verification = _authority_strings(
            authority_archives,
            continuation_capsule,
            "verification",
        )
        failed_paths = _authority_strings(
            authority_archives,
            continuation_capsule,
            "failed_paths",
        )
        rollback_notes = _authority_strings(
            authority_archives,
            continuation_capsule,
            "rollback_notes",
        )
        archive_ids = sorted({_archive_id(item) for item in archives})
        view = ContextView(
            messages=selected_messages,
            active_projection=projection,
            retrieved_archives=retrieved,
            continuation_capsule=(
                dict(continuation_capsule) if continuation_capsule is not None else None
            ),
            authority_artifact_ids=artifact_ids,
            authority_artifact_refs=artifact_refs,
            authority_evidence_ids=evidence_ids,
            authority_archive_ids=archive_ids,
            authority_verification=verification,
            authority_failed_paths=failed_paths,
            authority_rollback_notes=rollback_notes,
        )
        return view.model_copy(update={"input_tokens": self.count_view(view)})

    def count_view(self, view: ContextView) -> int:
        """Count the complete request envelope that will be sent to the model."""
        messages = [
            *self._fixed_messages,
            HumanMessage(
                content="PaperPilot Context View:\n" + render_context_view(view)
            ),
            *self._trailing_messages,
        ]
        return self._token_counter.count_messages(
            messages,
            tool_schemas=self._tool_schemas,
        )

    def build_minimal_safe_view(self, view: ContextView) -> ContextView:
        """Keep protected projection, refs, one business turn, and one archive."""
        current_goal = (
            view.active_projection.current_goal
            if view.active_projection is not None
            else ""
        )
        minimal_messages = select_recent_turns(
            view.messages,
            current_goal=current_goal,
            completed_turn_count=1,
        )
        archives = [
            item.model_copy(update={"narrative_summary": None})
            for item in view.retrieved_archives[-1:]
        ]
        minimal = view.model_copy(
            deep=True,
            update={
                "messages": minimal_messages,
                "retrieved_archives": archives,
                "continuation_capsule": None,
                "input_tokens": 0,
            },
        )
        return minimal.model_copy(update={"input_tokens": self.count_view(minimal)})

    def build_minimal_safe_view_with_budget(
        self,
        view: ContextView,
        *,
        usable_input_budget: int,
    ) -> ContextView:
        minimal = self.build_minimal_safe_view(view)
        if minimal.input_tokens > usable_input_budget:
            raise ContextCapacityExhaustedError()
        return minimal


def render_context_view(view: ContextView) -> str:
    """Render deterministic JSON for a HumanMessage technical context slot."""
    return json.dumps(
        view.model_dump(mode="json", exclude={"input_tokens"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _seed(archive: object) -> dict[str, Any]:
    value = archive.get("seed_json", {}) if isinstance(archive, Mapping) else getattr(archive, "seed_json", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _archive_id(archive: object) -> str:
    value = archive.get("archive_id") if isinstance(archive, Mapping) else getattr(archive, "archive_id", None)
    return str(value or "")


def _archive_version(archive: object) -> str:
    value = (
        archive.get("archive_version")
        if isinstance(archive, Mapping)
        else getattr(archive, "archive_version", None)
    )
    if isinstance(value, str) and value:
        return value
    seed = _seed(archive)
    fallback = seed.get("archive_version", seed.get("archive_type", ""))
    return str(fallback)


def _terminal_status(archive: object) -> str:
    value = archive.get("terminal_status") if isinstance(archive, Mapping) else getattr(archive, "terminal_status", "failed")
    return str(value)


def _narrative(archive: object) -> str | None:
    value = archive.get("narrative_summary") if isinstance(archive, Mapping) else getattr(archive, "narrative_summary", None)
    return value if isinstance(value, str) and value.strip() else None


def _archive_sort_key(archive: object) -> tuple[str, str]:
    created = archive.get("created_at", "") if isinstance(archive, Mapping) else getattr(archive, "created_at", "")
    return str(created), _archive_id(archive)


def _archive_text(archive: object) -> str:
    return _canonical(_seed(archive))


def _identifiers(archive: object) -> set[str]:
    result = {_archive_id(archive)}
    result.update(_ID_PATTERN.findall(_archive_text(archive)))
    for value in _walk_strings(_seed(archive)):
        if value.strip():
            result.add(value)
    return result


def _walk_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _superseded_ids(archives: Sequence[object]) -> set[str]:
    result: set[str] = set()
    for archive in archives:
        for relation in _seed(archive).get("supersedes", []):
            if isinstance(relation, Mapping):
                target = relation.get("target_protected_id")
                if isinstance(target, str):
                    result.add(target)
    return result


def _is_superseded(archive: object, superseded_ids: set[str]) -> bool:
    if not superseded_ids:
        return False
    seed = _seed(archive)
    for field in ("user_goal", "constraints", "decisions"):
        values = seed.get(field, [])
        if isinstance(values, Mapping):
            values = [values]
        if isinstance(values, list):
            for value in values:
                if isinstance(value, Mapping) and value.get("protected_id") in superseded_ids:
                    return True
    return False


def _relation_matches(
    archive: object,
    request: str,
    superseded_ids: set[str],
) -> bool:
    if not superseded_ids:
        return False
    return any(target.casefold() in request for target in superseded_ids) and (
        _is_superseded(archive, superseded_ids)
        or any(
            isinstance(item, Mapping)
            and item.get("target_protected_id") in superseded_ids
            for item in _seed(archive).get("supersedes", [])
        )
    )


def _effective_texts(
    archives: Sequence[object],
    field: str,
    superseded_ids: set[str],
) -> list[str]:
    result: list[str] = []
    for archive in sorted(archives, key=_archive_sort_key):
        values = _seed(archive).get(field, [])
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping) or value.get("protected_id") in superseded_ids:
                continue
            exact_text = value.get("exact_text")
            if isinstance(exact_text, str) and exact_text.strip():
                result.append(exact_text)
    return result


def _minimal_seed(seed: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        key: seed.get(key, [])
        for key in (
            "archive_id",
            "archive_version",
            "terminal_status",
            "user_goal",
            "constraints",
            "decisions",
            "evidence_refs",
            "artifact_refs",
            "verification",
            "supersedes",
        )
    }
    return result


def select_recent_turns(
    messages: Sequence[Mapping[str, Any]],
    *,
    current_goal: str,
    completed_turn_count: int,
) -> list[dict[str, Any]]:
    """Keep N completed user turns plus the current request and its tail."""
    if completed_turn_count < 1:
        raise ValueError("completed_turn_count must be positive")
    copied = [dict(message) for message in messages]
    current_index = next(
        (
            index
            for index in range(len(copied) - 1, -1, -1)
            if copied[index].get("role") in {"user", "human"}
            and copied[index].get("content") == current_goal
        ),
        None,
    )
    if current_index is None:
        current_index = next(
            (
                index
                for index in range(len(copied) - 1, -1, -1)
                if copied[index].get("role") in {"user", "human"}
            ),
            len(copied),
        )
    prior_user_indexes = [
        index
        for index in range(current_index)
        if copied[index].get("role") in {"user", "human"}
    ]
    if len(prior_user_indexes) <= completed_turn_count:
        return copied
    start = prior_user_indexes[-completed_turn_count]
    return copied[start:]


def _strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


def _authority_artifact_refs(
    archives: Sequence[object],
    continuation_capsule: Mapping[str, Any] | None,
) -> list[ArtifactRef]:
    values: list[object] = [
        item
        for archive in archives
        for item in (
            _seed(archive).get("artifact_refs", [])
            if isinstance(_seed(archive).get("artifact_refs", []), list)
            else []
        )
    ]
    if continuation_capsule is not None:
        capsule_refs = continuation_capsule.get("artifact_refs", [])
        if isinstance(capsule_refs, list):
            values.extend(capsule_refs)
    by_id: dict[str, ArtifactRef] = {}
    for value in values:
        try:
            artifact_ref = ArtifactRef.model_validate(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid artifact authority reference") from exc
        existing = by_id.get(artifact_ref.artifact_id)
        if existing is not None and existing != artifact_ref:
            raise ValueError(
                f"conflicting artifact authority: {artifact_ref.artifact_id}"
            )
        by_id[artifact_ref.artifact_id] = artifact_ref
    return [by_id[artifact_id] for artifact_id in sorted(by_id)]


def _authority_strings(
    archives: Sequence[object],
    continuation_capsule: Mapping[str, Any] | None,
    field_name: str,
) -> list[str]:
    values = [
        item
        for archive in archives
        for item in _strings(_seed(archive).get(field_name, []))
    ]
    if continuation_capsule is not None:
        values.extend(_strings(continuation_capsule.get(field_name, [])))
    return sorted(set(values))


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item.strip()))


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
