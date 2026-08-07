"""Reusable structured paper catalog backed by arXiv."""
from __future__ import annotations

import re
from typing import Literal

import arxiv
from pydantic import BaseModel


class PaperCandidate(BaseModel):
    """A paper result suitable for application and agent consumers."""

    source: Literal["arxiv"] = "arxiv"
    external_id: str
    title: str
    authors: list[str]
    abstract: str | None = None
    source_url: str
    pdf_url: str | None = None
    published: str | None = None
    primary_category: str | None = None


_NEW_ARXIV_ID_RE = r"\d{4}\.\d{4,5}(?:v\d+)?"
_OLD_ARXIV_ID_RE = r"[A-Za-z-]+(?:\.[A-Za-z-]+)?/\d{7}(?:v\d+)?"
_ARXIV_ID_RE = re.compile(rf"^(?:{_NEW_ARXIV_ID_RE}|{_OLD_ARXIV_ID_RE})$")
_ARXIV_URL_RE = re.compile(
    rf"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/"
    rf"(?P<arxiv_id>{_NEW_ARXIV_ID_RE}|{_OLD_ARXIV_ID_RE})(?:\.pdf)?$"
)


def normalize_arxiv_id(value: str) -> str | None:
    """Return a canonical arXiv id from a supported id or arXiv URL."""
    candidate = value.strip()
    url_match = _ARXIV_URL_RE.fullmatch(candidate)
    if url_match:
        return url_match.group("arxiv_id")
    if _ARXIV_ID_RE.fullmatch(candidate):
        return candidate
    return None


def search_arxiv_candidates(
    query: str,
    limit: int,
    client: arxiv.Client | None = None,
    *,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
) -> list[PaperCandidate]:
    """Search arXiv and map its results to structured paper candidates."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")

    search = arxiv.Search(query=query, max_results=limit, sort_by=sort_by)
    arxiv_client = client or arxiv.Client()
    return [_to_candidate(result) for result in arxiv_client.results(search)]


def resolve_arxiv_candidate(
    external_id: str, client: arxiv.Client | None = None
) -> PaperCandidate:
    """Resolve an arXiv id or arXiv URL to its structured paper candidate."""
    normalized = normalize_arxiv_id(external_id)
    if normalized is None:
        raise ValueError(f"invalid arXiv id: {external_id!r}")

    search = arxiv.Search(id_list=[normalized], max_results=1)
    arxiv_client = client or arxiv.Client()
    result = next(arxiv_client.results(search), None)
    if result is None:
        raise LookupError(f"arXiv paper not found: {normalized}")
    return _to_candidate(result)


def _to_candidate(result: arxiv.Result) -> PaperCandidate:
    published = getattr(result, "published", None)
    return PaperCandidate(
        external_id=result.get_short_id(),
        title=result.title.strip(),
        authors=[author.name for author in result.authors],
        abstract=result.summary.strip() or None,
        source_url=result.entry_id,
        pdf_url=result.pdf_url,
        published=published.date().isoformat() if published else None,
        primary_category=result.primary_category,
    )
