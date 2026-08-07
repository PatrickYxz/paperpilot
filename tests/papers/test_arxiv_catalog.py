from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from paperpilot.papers import (
    normalize_arxiv_id,
    resolve_arxiv_candidate,
    search_arxiv_candidates,
)


class FakeResult:
    entry_id = "https://arxiv.org/abs/2401.12345v2"
    title = "  Structured arXiv Catalog  "
    authors = [SimpleNamespace(name="Ada Lovelace"), SimpleNamespace(name="Grace Hopper")]
    summary = "  A searchable paper summary.  "
    pdf_url = "https://arxiv.org/pdf/2401.12345v2"
    published = datetime(2024, 1, 31, 12, 0, tzinfo=timezone.utc)
    primary_category = "cs.AI"

    def get_short_id(self) -> str:
        return "2401.12345v2"


class FakeArxivClient:
    def __init__(self, results: list[FakeResult] | None = None) -> None:
        self.results_to_return = results or [FakeResult()]
        self.searches = []

    def results(self, search):
        self.searches.append(search)
        return iter(self.results_to_return)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2401.12345", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345v2", "2401.12345v2"),
        ("https://arxiv.org/pdf/2401.12345v2.pdf", "2401.12345v2"),
        ("cs.AI/0501001v3", "cs.AI/0501001v3"),
        ("not an arxiv id", None),
    ],
)
def test_normalize_arxiv_id(raw: str, expected: str | None) -> None:
    assert normalize_arxiv_id(raw) == expected


def test_search_maps_arxiv_result_to_structured_candidate() -> None:
    client = FakeArxivClient()

    candidates = search_arxiv_candidates("structured catalog", limit=2, client=client)

    assert len(candidates) == 1
    assert candidates[0].model_dump() == {
        "source": "arxiv",
        "external_id": "2401.12345v2",
        "title": "Structured arXiv Catalog",
        "authors": ["Ada Lovelace", "Grace Hopper"],
        "abstract": "A searchable paper summary.",
        "source_url": "https://arxiv.org/abs/2401.12345v2",
        "pdf_url": "https://arxiv.org/pdf/2401.12345v2",
        "published": "2024-01-31",
        "primary_category": "cs.AI",
    }
    search = client.searches[0]
    assert search.query == "structured catalog"
    assert search.max_results == 2
    assert search.id_list == []


def test_resolve_uses_normalized_id_list_and_preserves_version() -> None:
    client = FakeArxivClient()

    candidate = resolve_arxiv_candidate(
        "https://arxiv.org/pdf/2401.12345v2.pdf", client=client
    )

    assert candidate.external_id == "2401.12345v2"
    search = client.searches[0]
    assert search.query == ""
    assert search.id_list == ["2401.12345v2"]


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(query: str) -> None:
    with pytest.raises(ValueError, match="query"):
        search_arxiv_candidates(query, limit=1, client=FakeArxivClient())


def test_search_allows_shared_catalog_maximum_limit() -> None:
    client = FakeArxivClient()

    candidates = search_arxiv_candidates("catalog", limit=50, client=client)

    assert [candidate.external_id for candidate in candidates] == ["2401.12345v2"]
    assert client.searches[0].max_results == 50


@pytest.mark.parametrize("limit", [0, 51])
def test_search_rejects_limit_outside_catalog_range(limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        search_arxiv_candidates("catalog", limit=limit, client=FakeArxivClient())


def test_search_propagates_client_error_without_fallback() -> None:
    class FailingArxivClient(FakeArxivClient):
        def results(self, search):
            raise RuntimeError("arxiv unavailable")

    with pytest.raises(RuntimeError, match="arxiv unavailable"):
        search_arxiv_candidates("catalog", limit=1, client=FailingArxivClient())
