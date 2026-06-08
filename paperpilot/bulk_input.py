"""Detect and rewrite oversized user-pasted paper comparison requests."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")


@dataclass(frozen=True)
class BulkPaperInput:
    document_text: str
    target_paper_id: str


class BulkPaperInputDetector:
    def __init__(self, *, min_chars: int | None = None) -> None:
        self.min_chars = min_chars or _env_int("BULK_INPUT_MIN_CHARS", 20_000)

    def detect(self, user_text: str) -> BulkPaperInput | None:
        if len(user_text) < self.min_chars:
            return None
        if not _looks_like_paper(user_text):
            return None
        if not _has_similarity_intent(user_text):
            return None
        target = _extract_target_paper_id(user_text)
        if target is None:
            return None
        return BulkPaperInput(document_text=user_text, target_paper_id=target)


def _looks_like_paper(text: str) -> bool:
    lower = text.lower()
    markers = [
        "abstract",
        "introduction",
        "references",
        "experiment",
        "evaluation",
        "conclusion",
    ]
    return sum(1 for marker in markers if marker in lower) >= 2


def _has_similarity_intent(text: str) -> bool:
    lower = text.lower()
    return any(
        marker in lower
        for marker in [
            "similarity",
            "similar",
            "compare",
            "comparison",
            "对比",
            "比较",
            "相似",
            "相似度",
        ]
    )


def _extract_target_paper_id(text: str) -> str | None:
    match = ARXIV_ID_RE.search(text)
    if match:
        return match.group(0)
    paper_id_match = re.search(
        r"\b(?:paper_id|target_paper_id)\s*[:=]\s*([A-Za-z0-9_.:-]+)",
        text,
        flags=re.IGNORECASE,
    )
    if paper_id_match:
        return paper_id_match.group(1)
    return None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
