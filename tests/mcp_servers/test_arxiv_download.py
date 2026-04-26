"""arxiv.download_paper 单测。mock urllib + 真跑 pymupdf 解析 fixture PDF。"""
from __future__ import annotations

import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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
