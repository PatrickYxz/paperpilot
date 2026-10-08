"""Unit tests for the default web paper search (phrase quoting + fallback)."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from paperpilot.web.routes.papers import default_web_paper_search


class FakeResult:
    def __init__(self, short_id: str, title: str) -> None:
        self._short_id = short_id
        self.entry_id = f"https://arxiv.org/abs/{short_id}"
        self.title = title
        self.authors = [SimpleNamespace(name="Ada Lovelace")]
        self.summary = "  A searchable paper summary.  "
        self.pdf_url = f"https://arxiv.org/pdf/{short_id}"
        self.published = datetime(2024, 1, 31, 12, 0, tzinfo=timezone.utc)
        self.primary_category = "cs.AI"

    def get_short_id(self) -> str:
        return self._short_id


class FakeArxivClient:
    """Returns canned results per keyword query or per resolved arXiv id."""

    def __init__(
        self,
        keyword_results: dict[str, list[FakeResult]] | None = None,
        id_results: dict[str, FakeResult] | None = None,
    ) -> None:
        self.keyword_results = keyword_results or {}
        self.id_results = id_results or {}
        self.keyword_queries: list[str] = []
        self.id_lookups: list[str] = []

    def results(self, search):
        if search.id_list:
            self.id_lookups.append(search.id_list[0])
            result = self.id_results.get(search.id_list[0])
            return iter([result] if result else [])
        self.keyword_queries.append(search.query)
        return iter(self.keyword_results.get(search.query, []))


def test_multi_word_free_text_searches_phrase_first() -> None:
    phrase = 'all:"attention is all you need"'
    expected = FakeResult("1706.03762v7", "Attention Is All You Need")
    client = FakeArxivClient(keyword_results={phrase: [expected]})

    candidates = default_web_paper_search(
        "attention is all you need", 10, client=client
    )

    assert client.keyword_queries == [phrase]
    assert [candidate.external_id for candidate in candidates] == ["1706.03762v7"]


def test_empty_phrase_results_fall_back_to_raw_query() -> None:
    raw = "attention is all u need"
    fallback = FakeResult("2401.12345v2", "Loose Match")
    client = FakeArxivClient(keyword_results={raw: [fallback]})

    candidates = default_web_paper_search(raw, 10, client=client)

    assert client.keyword_queries == ['all:"attention is all u need"', raw]
    assert [candidate.external_id for candidate in candidates] == ["2401.12345v2"]


def test_empty_phrase_and_raw_queries_return_no_candidates() -> None:
    client = FakeArxivClient()

    candidates = default_web_paper_search(
        "nonexistent phrase query", 10, client=client
    )

    assert candidates == []
    assert client.keyword_queries == [
        'all:"nonexistent phrase query"',
        "nonexistent phrase query",
    ]


@pytest.mark.parametrize(
    "query",
    [
        "transformers",
        "au:vaswani",
        "ti:attention AND cat:cs.CL",
        'all:"attention is all you need"',
        '"attention is all you need"',
    ],
)
def test_query_syntax_bypasses_phrase_quoting(query: str) -> None:
    expected = FakeResult("2401.12345v2", "Direct Query")
    client = FakeArxivClient(keyword_results={query: [expected]})

    candidates = default_web_paper_search(query, 10, client=client)

    assert client.keyword_queries == [query]
    assert [candidate.external_id for candidate in candidates] == ["2401.12345v2"]


def test_arxiv_id_resolves_without_keyword_search() -> None:
    expected = FakeResult("1706.03762v7", "Attention Is All You Need")
    client = FakeArxivClient(id_results={"1706.03762": expected})

    candidates = default_web_paper_search("1706.03762", 10, client=client)

    assert client.keyword_queries == []
    assert client.id_lookups == ["1706.03762"]
    assert [candidate.external_id for candidate in candidates] == ["1706.03762v7"]
