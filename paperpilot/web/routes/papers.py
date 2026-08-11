"""Authenticated paper-search HTTP routes."""
from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Query

from paperpilot.papers import (
    PaperCandidate,
    normalize_arxiv_id,
    resolve_arxiv_candidate,
    search_arxiv_candidates,
)
from paperpilot.web.routes.auth import RequireUser
from paperpilot.web.schemas import PaperSearchResponse
from paperpilot.web.task_store import WebUser


PaperSearch = Callable[[str, int], list[PaperCandidate]]


def default_web_paper_search(query: str, limit: int) -> list[PaperCandidate]:
    """Use exact arXiv resolution for IDs/URLs and catalog search otherwise."""
    normalized = normalize_arxiv_id(query)
    if normalized is not None:
        try:
            return [resolve_arxiv_candidate(normalized)]
        except LookupError:
            return []
    return search_arxiv_candidates(query, limit)


def build_paper_router(
    *,
    require_user: RequireUser,
    paper_search: PaperSearch,
) -> APIRouter:
    router = APIRouter(prefix="/api/papers")

    @router.get("/search", response_model=PaperSearchResponse)
    def search_papers(
        q: str = Query(..., min_length=1),
        limit: int = Query(default=10, ge=1, le=20),
        user: WebUser = Depends(require_user),
    ) -> dict:
        del user
        if not q.strip():
            raise HTTPException(status_code=422, detail="query must not be empty")
        candidates = paper_search(q, limit)
        return {"items": [_candidate_dict(candidate) for candidate in candidates]}

    return router


def _candidate_dict(candidate: PaperCandidate) -> dict:
    return {
        "source": candidate.source,
        "external_id": candidate.external_id,
        "title": candidate.title,
        "authors": list(candidate.authors),
        "abstract": candidate.abstract,
        "source_url": candidate.source_url,
    }
