"""Authenticated paper-search HTTP routes."""
from __future__ import annotations

import re
from collections.abc import Callable

import arxiv
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

# arXiv 查询语法片段:字段前缀(ti: 等)与布尔词。出现任一即视为用户显式
# 写了查询语法,不再包短语引号。
_ARXIV_FIELD_PREFIX_RE = re.compile(r"\b(?:ti|au|abs|co|jr|cat|rn|sr|all):")
_ARXIV_BOOLEAN_RE = re.compile(r"\b(?:AND|OR|ANDNOT)\b")


def _phrase_query(query: str) -> str | None:
    """Return an `all:"..."` phrase query for plain multi-word text, else None."""
    stripped = query.strip()
    if len(stripped.split()) < 2:
        return None
    if (
        '"' in stripped
        or _ARXIV_FIELD_PREFIX_RE.search(stripped)
        or _ARXIV_BOOLEAN_RE.search(stripped)
    ):
        return None
    return f'all:"{stripped}"'


def default_web_paper_search(
    query: str, limit: int, client: arxiv.Client | None = None
) -> list[PaperCandidate]:
    """Use exact arXiv resolution for IDs/URLs and catalog search otherwise.

    arXiv 把不带引号的多词查询当词袋 AND 排序,标题类查询召回大量仿名
    论文,因此自由文本先按短语查询,短语无结果时回退原样查询。
    """
    normalized = normalize_arxiv_id(query)
    if normalized is not None:
        try:
            return [resolve_arxiv_candidate(normalized, client=client)]
        except LookupError:
            return []
    phrase = _phrase_query(query)
    if phrase is not None:
        phrase_results = search_arxiv_candidates(phrase, limit, client=client)
        if phrase_results:
            return phrase_results
    return search_arxiv_candidates(query, limit, client=client)


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
