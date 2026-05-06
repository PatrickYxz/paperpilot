"""PageRenderer unit tests. urllib + fitz are mocked; file IO uses tmp_path."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paperpilot.mcp_servers.vlm import page_renderer as pr_module


@pytest.fixture
def patched_root(tmp_path, monkeypatch):
    monkeypatch.setattr(pr_module.PageRenderer, "CACHE_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def patched_io(monkeypatch):
    fake_pdf_bytes = b"%PDF-1.4 fake"
    download_calls: list[str] = []

    def _fake_fetch(arxiv_id: str) -> bytes:
        download_calls.append(arxiv_id)
        return fake_pdf_bytes

    monkeypatch.setattr(pr_module, "_fetch_pdf", _fake_fetch)

    fake_pix = MagicMock()
    fake_pix.tobytes.return_value = b"\x89PNG\r\n\x1a\nFAKEPNG"

    fake_page = MagicMock()
    fake_page.get_pixmap.return_value = fake_pix

    fake_doc = MagicMock()
    fake_doc.page_count = 5
    fake_doc.load_page.return_value = fake_page

    fitz_open = MagicMock(return_value=fake_doc)
    monkeypatch.setattr(pr_module.fitz, "open", fitz_open)

    return {
        "download_calls": download_calls,
        "fake_doc": fake_doc,
        "fake_pix": fake_pix,
        "fitz_open": fitz_open,
    }


def test_cold_render_writes_png_and_returns_bytes(patched_root, patched_io):
    renderer = pr_module.PageRenderer()
    out = renderer.get_page_png("p1", 2)

    assert out == b"\x89PNG\r\n\x1a\nFAKEPNG"
    paper_dirs = [p for p in patched_root.iterdir() if p.is_dir()]
    assert len(paper_dirs) == 1
    png_path = paper_dirs[0] / "page_2.png"
    assert png_path.exists()
    assert png_path.read_bytes() == b"\x89PNG\r\n\x1a\nFAKEPNG"
    assert patched_io["download_calls"] == ["p1"]
    patched_io["fake_doc"].load_page.assert_called_once_with(1)


def test_disk_hit_skips_render(patched_root, patched_io):
    renderer = pr_module.PageRenderer()
    paper_dir = patched_root / pr_module._paper_key("p1")
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "page_3.png").write_bytes(b"PRECACHED")

    out = renderer.get_page_png("p1", 3)

    assert out == b"PRECACHED"
    assert patched_io["download_calls"] == []
    patched_io["fitz_open"].assert_not_called()


def test_pdf_bytes_memo_skips_second_download(patched_root, patched_io):
    renderer = pr_module.PageRenderer()
    renderer.get_page_png("p1", 1)
    renderer.get_page_png("p1", 2)

    assert patched_io["download_calls"] == ["p1"]


def test_page_num_out_of_range_raises(patched_root, patched_io):
    renderer = pr_module.PageRenderer()
    with pytest.raises(pr_module.PageOutOfRange, match="out of 1..5"):
        renderer.get_page_png("p1", 10)


def test_paper_key_escapes_path_separators(patched_root, patched_io):
    renderer = pr_module.PageRenderer()
    renderer.get_page_png("cs/0501001", 1)

    paper_dirs = [p for p in patched_root.iterdir() if p.is_dir()]
    assert len(paper_dirs) == 1
    assert "/" not in paper_dirs[0].name
    assert "\\" not in paper_dirs[0].name
