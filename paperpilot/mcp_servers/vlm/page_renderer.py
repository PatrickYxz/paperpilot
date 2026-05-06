"""vlm-mcp page renderer.

Each page gets persisted as data/vlm_cache/<paper_key>/page_<N>.png.
PDF bytes are cached in process memory for the session only.
"""
from __future__ import annotations

import hashlib
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import certifi
import fitz

PDF_DPI = 150
_SAFE_PAPER_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ArxivNotFoundError(RuntimeError):
    """arxiv returned 404 / invalid id."""


class PDFParseError(RuntimeError):
    """fitz failed to parse PDF bytes."""


class PageOutOfRange(RuntimeError):
    """page_num is outside the PDF page range."""


class PageRenderer:
    PDF_DPI = PDF_DPI
    CACHE_ROOT = Path("data/vlm_cache")

    def __init__(self) -> None:
        self.CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self._pdf_bytes_mem: dict[str, bytes] = {}

    def get_page_png(self, arxiv_id: str, page_num: int) -> bytes:
        """Return PNG bytes for the given 1-based page. Cache-first."""
        png_path = self._png_path(arxiv_id, page_num)
        if png_path.exists():
            return png_path.read_bytes()

        pdf = self._get_pdf_bytes(arxiv_id)
        png_bytes = self._render_page(pdf, page_num)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(png_bytes)
        return png_bytes

    def _get_pdf_bytes(self, arxiv_id: str) -> bytes:
        if arxiv_id not in self._pdf_bytes_mem:
            self._pdf_bytes_mem[arxiv_id] = _fetch_pdf(arxiv_id)
        return self._pdf_bytes_mem[arxiv_id]

    def _render_page(self, pdf_bytes: bytes, page_num: int) -> bytes:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise PDFParseError(f"failed to open PDF: {e}") from e
        try:
            if page_num < 1 or page_num > doc.page_count:
                raise PageOutOfRange(
                    f"page {page_num} out of 1..{doc.page_count}"
                )
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=self.PDF_DPI)
            return pix.tobytes("png")
        finally:
            doc.close()

    def _png_path(self, arxiv_id: str, page_num: int) -> Path:
        return self.CACHE_ROOT / _paper_key(arxiv_id) / f"page_{page_num}.png"


def _fetch_pdf(arxiv_id: str) -> bytes:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ArxivNotFoundError(f"arxiv paper not found: {arxiv_id}") from e
        raise


def _paper_key(arxiv_id: str) -> str:
    safe = _SAFE_PAPER_KEY_RE.sub("_", arxiv_id).strip("._")
    digest = hashlib.sha1(arxiv_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe or 'paper'}-{digest}"
