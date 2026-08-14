"""Conversation CRUD HTTP handlers.

This module owns authenticated Conversation creation, listing, detail reads,
and title/archive updates, including PaperReference resolution.  It does not
own the Message tree, executor admission, rollback validation, or response
presentation policy beyond calling the pure presenters.  Store methods remain
the owners of database transactions and user isolation.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from paperpilot.papers import PaperCandidate, normalize_arxiv_id
from paperpilot.web.routes.auth import RequireUser
from paperpilot.web.routes.conversations.presenters import (
    conversation_dict,
    detail_dict,
)
from paperpilot.web.routes.papers import PaperSearch
from paperpilot.web.schemas import (
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    CreateConversationRequest,
    PaperReference,
    UpdateConversationRequest,
)
from paperpilot.web.task_store import ConversationDetail, TaskStore, WebUser


def build_crud_router(
    *,
    store: TaskStore,
    require_user: RequireUser,
    paper_search: PaperSearch,
) -> APIRouter:
    router = APIRouter()

    @router.get("/{conversation_id}", response_model=ConversationDetailResponse)
    def get_conversation(
        conversation_id: str,
        user: WebUser = Depends(require_user),
    ) -> dict:
        detail = _owned_detail(store, conversation_id, user.id)
        return detail_dict(detail)

    @router.patch("/{conversation_id}", response_model=ConversationResponse)
    def update_conversation(
        conversation_id: str,
        payload: UpdateConversationRequest,
        user: WebUser = Depends(require_user),
    ) -> dict:
        if store.get_conversation_detail(conversation_id, user_id=user.id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        try:
            record = store.update_conversation(
                conversation_id,
                user_id=user.id,
                title=payload.title,
                archived=payload.archived,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return conversation_dict(record)

    return router


def register_collection_routes(
    router: APIRouter,
    *,
    store: TaskStore,
    require_user: RequireUser,
    paper_search: PaperSearch,
) -> None:
    """Register collection routes on the already-prefixed root router.

    FastAPI rejects an included child route whose path and include prefix are
    both empty.  The parent owns `/api/conversations`, so the two collection
    operations are registered there while item routes remain in the child
    CRUD router.
    """

    @router.post("", response_model=ConversationResponse, status_code=201)
    def create_conversation(
        payload: CreateConversationRequest,
        user: WebUser = Depends(require_user),
    ) -> dict:
        paper = _resolve_paper_reference(payload.paper, paper_search)
        try:
            conversation = store.create_conversation(
                user_id=user.id,
                paper=paper,
                title=payload.title,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return conversation_dict(conversation)

    @router.get("", response_model=ConversationListResponse)
    def list_conversations(
        include_archived: bool = Query(default=False),
        limit: int = Query(default=100, ge=1, le=100),
        user: WebUser = Depends(require_user),
    ) -> dict:
        records = store.list_conversations(
            user_id=user.id,
            include_archived=include_archived,
            limit=limit,
        )
        return {"items": [conversation_dict(record) for record in records]}


def _resolve_paper_reference(
    reference: PaperReference,
    paper_search: PaperSearch,
) -> PaperCandidate:
    normalized = normalize_arxiv_id(reference.external_id)
    if normalized is None:
        raise HTTPException(status_code=422, detail="invalid arXiv id")
    candidates = paper_search(normalized, 1)
    for candidate in candidates:
        if (
            candidate.source == reference.source
            and normalize_arxiv_id(candidate.external_id) == normalized
        ):
            return candidate
    raise HTTPException(status_code=404, detail="paper not found")


def _owned_detail(
    store: TaskStore,
    conversation_id: str,
    user_id: str,
) -> ConversationDetail:
    detail = store.get_conversation_detail(conversation_id, user_id=user_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return detail
