"""arxiv.download_paper 单测。mock urllib + 真跑 pymupdf 解析 fixture PDF。"""
from __future__ import annotations

import urllib.error
from hashlib import sha256
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import arxiv as arxiv_sdk

from paperpilot.mcp_servers import arxiv as arxiv_mod

FIXTURE_PDF = Path(__file__).parent.parent / "fixtures" / "sample_paper.pdf"


def _fake_urlopen_ctx(payload: bytes) -> MagicMock:
    """构造 urllib.request.urlopen 的上下文管理器返回值,模拟 HTTP body。"""
    resp = MagicMock()
    resp.read.return_value = payload
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    return resp


def test_parse_real_pdf(tmp_path, monkeypatch):
    """提 text 含 BERT 关键词;落盘到 cache 路径。"""
    monkeypatch.setattr(arxiv_mod, "_PAPERS_DIR", tmp_path / "papers")
    pdf_bytes = FIXTURE_PDF.read_bytes()

    with patch("urllib.request.urlopen", return_value=_fake_urlopen_ctx(pdf_bytes)):
        result = arxiv_mod._download_paper_impl("1810.04805v2")

    assert result["paper_id"] == "1810.04805v2"
    assert "BERT" in result["text"]
    assert "transformer" in result["text"].lower()
    assert (tmp_path / "papers" / "1810.04805v2.txt").exists()


def test_cache_hit(tmp_path, monkeypatch):
    """二次调用同 id 直接读盘,不再调 urlopen。"""
    cache_dir = tmp_path / "papers"
    cache_dir.mkdir()
    (cache_dir / "2401.12345.txt").write_text("cached text content", encoding="utf-8")
    monkeypatch.setattr(arxiv_mod, "_PAPERS_DIR", cache_dir)

    spy = MagicMock()
    with patch("urllib.request.urlopen", spy):
        result = arxiv_mod._download_paper_impl("2401.12345")

    assert result == {"paper_id": "2401.12345", "text": "cached text content"}
    spy.assert_not_called()


def test_legacy_id_uses_unique_top_level_cache_and_second_call_hits_cache(
    tmp_path,
    monkeypatch,
):
    cache_dir = tmp_path / "papers"
    monkeypatch.setattr(arxiv_mod, "_PAPERS_DIR", cache_dir)
    pdf_bytes = FIXTURE_PDF.read_bytes()
    urlopen = MagicMock(return_value=_fake_urlopen_ctx(pdf_bytes))

    with patch("urllib.request.urlopen", urlopen):
        first = arxiv_mod._download_paper_impl("cs.AI/0501001v3")
        second = arxiv_mod._download_paper_impl("cs.AI/0501001v3")
        assert urlopen.call_count == 1
        other = arxiv_mod._download_paper_impl("cs.AI/0501002v3")

    assert first == second
    assert first["paper_id"] == "cs.AI/0501001v3"
    assert other["paper_id"] == "cs.AI/0501002v3"
    assert urlopen.call_count == 2
    cache_files = sorted(cache_dir.iterdir())
    assert len(cache_files) == 2
    assert all(path.is_file() and path.parent == cache_dir for path in cache_files)
    assert len({path.name for path in cache_files}) == 2
    assert all("/" not in path.name for path in cache_files)
    assert all("\\" not in path.name for path in cache_files)
    assert all(".." not in path.name for path in cache_files)
    assert not (cache_dir / "cs.AI").exists()


def test_legacy_ids_differing_only_by_case_use_casefold_distinct_caches(
    tmp_path,
    monkeypatch,
):
    cache_dir = tmp_path / "papers"
    monkeypatch.setattr(arxiv_mod, "_PAPERS_DIR", cache_dir)
    fetch = MagicMock(side_effect=lambda canonical_id: canonical_id.encode("utf-8"))
    monkeypatch.setattr(arxiv_mod, "_fetch_pdf", fetch)
    monkeypatch.setattr(
        arxiv_mod,
        "_extract_text",
        lambda canonical_id, pdf_bytes: pdf_bytes.decode("utf-8"),
    )

    upper_first = arxiv_mod._download_paper_impl("cs.AI/0501001v3")
    lower_first = arxiv_mod._download_paper_impl("cs.ai/0501001v3")
    upper_cached = arxiv_mod._download_paper_impl("cs.AI/0501001v3")
    lower_cached = arxiv_mod._download_paper_impl("cs.ai/0501001v3")

    assert upper_first == upper_cached == {
        "paper_id": "cs.AI/0501001v3",
        "text": "cs.AI/0501001v3",
    }
    assert lower_first == lower_cached == {
        "paper_id": "cs.ai/0501001v3",
        "text": "cs.ai/0501001v3",
    }
    assert fetch.call_count == 2
    cache_names = [path.name for path in cache_dir.iterdir()]
    assert set(cache_names) == {
        f"legacy-{sha256(canonical_id.encode('utf-8')).hexdigest()}.txt"
        for canonical_id in ("cs.AI/0501001v3", "cs.ai/0501001v3")
    }
    assert len({name.casefold() for name in cache_names}) == 2


@pytest.mark.parametrize(
    "unsafe_id",
    [
        "",
        "   ",
        "not-an-arxiv-id",
        "../../outside",
        "cs.AI/../../outside",
        r"..\outside",
        "https://example.com/abs/2401.12345",
    ],
)
def test_invalid_or_traversal_id_is_rejected_before_fetch_or_write(
    unsafe_id,
    tmp_path,
    monkeypatch,
):
    cache_dir = tmp_path / "papers"
    fetch = MagicMock(side_effect=AssertionError("fetch boundary was reached"))
    monkeypatch.setattr(arxiv_mod, "_PAPERS_DIR", cache_dir)
    monkeypatch.setattr(arxiv_mod, "_fetch_pdf", fetch)

    with pytest.raises(ValueError, match="invalid arXiv id"):
        arxiv_mod._download_paper_impl(unsafe_id)

    fetch.assert_not_called()
    assert not cache_dir.exists()


def test_arxiv_404(tmp_path, monkeypatch):
    """arxiv 返回 404 → 抛 ArxivNotFoundError。"""
    monkeypatch.setattr(arxiv_mod, "_PAPERS_DIR", tmp_path / "papers")
    err = urllib.error.HTTPError(
        url="https://arxiv.org/pdf/9999.99999",
        code=404, msg="Not Found", hdrs=None, fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(arxiv_mod.ArxivNotFoundError):
            arxiv_mod._download_paper_impl("9999.99999")


def test_pdf_corrupt(tmp_path, monkeypatch):
    """非 PDF 字节流 → 抛 PDFParseError。"""
    monkeypatch.setattr(arxiv_mod, "_PAPERS_DIR", tmp_path / "papers")
    bad = b"this is not a PDF file at all, just plain text"
    with patch("urllib.request.urlopen", return_value=_fake_urlopen_ctx(bad)):
        with pytest.raises(arxiv_mod.PDFParseError):
            arxiv_mod._download_paper_impl("2401.12345")


def test_search_papers_keeps_legacy_text_output_with_catalog(monkeypatch):
    """搜索仍使用旧 MCP 字段和分段格式，但数据来自 catalog。"""
    from datetime import datetime
    from types import SimpleNamespace

    class FakeResult:
        title = "Catalog paper"
        authors = [SimpleNamespace(name="Ada")]
        summary = "An abstract"
        pdf_url = "https://arxiv.org/pdf/2401.12345v2"
        published = datetime(2024, 1, 31)
        primary_category = "cs.AI"
        entry_id = "https://arxiv.org/abs/2401.12345v2"

        def get_short_id(self):
            return "2401.12345v2"

    class FakeClient:
        searches = []

        def results(self, search):
            self.searches.append(search)
            return iter([FakeResult()])

    client = FakeClient()
    monkeypatch.setattr(arxiv_mod, "_client", client)

    result = arxiv_mod.search_papers("catalog", max_results=50)

    assert result == (
        "arxiv_id: 2401.12345v2\n"
        "title: Catalog paper\n"
        "authors: Ada\n"
        "published: 2024-01-31\n"
        "primary_category: cs.AI\n"
        "pdf_url: https://arxiv.org/pdf/2401.12345v2\n"
        "abstract: An abstract"
    )
    assert client.searches[0].max_results == 50


def test_search_papers_keeps_legacy_sort_semantics(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.searches = []

        def results(self, search):
            self.searches.append(search)
            return iter([])

    client = FakeClient()
    monkeypatch.setattr(arxiv_mod, "_client", client)

    arxiv_mod.search_papers("catalog", max_results=1, sort_by="submittedDate")

    assert client.searches[0].sort_by is arxiv_sdk.SortCriterion.SubmittedDate
