"""Lightweight local storage and lexical search for user-pasted documents."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_USER_DOCUMENT_ROOT = Path("data/user_documents")


@dataclass(frozen=True)
class StoredDocument:
    doc_id: str
    path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    section_hint: str | None
    score: float
    text: str


class DocumentStore:
    def __init__(self, root: Path | str = DEFAULT_USER_DOCUMENT_ROOT) -> None:
        self.root = Path(root)

    def save_user_paste(self, text: str, *, target_paper_id: str) -> StoredDocument:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = f"userdoc-{digest[:12]}"
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{doc_id}.txt"
        metadata_path = self.root / f"{doc_id}.json"
        metadata = {
            "doc_id": doc_id,
            "sha256": digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "user_paste",
            "char_count": len(text),
            "approx_tokens": math.ceil(len(text) / 4),
            "detected_target": target_paper_id,
        }
        path.write_text(text, encoding="utf-8")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return StoredDocument(
            doc_id=doc_id,
            path=path,
            metadata_path=metadata_path,
            metadata=metadata,
        )

    def load_text(self, doc_id: str) -> str:
        path = self._text_path(doc_id)
        if not path.exists():
            raise FileNotFoundError(f"user document not found: {doc_id}")
        return path.read_text(encoding="utf-8")

    def search(
        self,
        doc_id: str,
        query: str,
        *,
        top_k: int = 5,
        max_chars_per_hit: int = 900,
    ) -> list[SearchHit]:
        text = self.load_text(doc_id)
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        chunks = _chunk_document(text)
        scored: list[SearchHit] = []
        for index, chunk in enumerate(chunks):
            chunk_terms = _tokenize(chunk.text)
            score = _score(query_terms, chunk_terms)
            if score <= 0:
                continue
            scored.append(SearchHit(
                chunk_id=f"{doc_id}#chunk-{index}",
                section_hint=chunk.section_hint,
                score=round(score, 4),
                text=_trim(chunk.text, max_chars_per_hit),
            ))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:max(1, top_k)]

    def _text_path(self, doc_id: str) -> Path:
        _validate_doc_id(doc_id)
        return self.root / f"{doc_id}.txt"


@dataclass(frozen=True)
class _Chunk:
    text: str
    section_hint: str | None


def _chunk_document(text: str, *, target_chars: int = 1600) -> list[_Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[_Chunk] = []
    current: list[str] = []
    current_len = 0
    section_hint: str | None = None
    current_section: str | None = None

    for paragraph in paragraphs:
        heading = _heading(paragraph)
        if heading:
            current_section = heading
        if current and current_len + len(paragraph) > target_chars:
            chunks.append(_Chunk("\n\n".join(current), section_hint))
            current = []
            current_len = 0
            section_hint = current_section
        if not current:
            section_hint = current_section
        current.append(paragraph)
        current_len += len(paragraph)

    if current:
        chunks.append(_Chunk("\n\n".join(current), section_hint))
    return chunks


def _heading(paragraph: str) -> str | None:
    line = paragraph.strip().splitlines()[0][:120].strip()
    normalized = line.lower().strip(".:")
    known = {
        "abstract",
        "introduction",
        "related work",
        "method",
        "methods",
        "approach",
        "experiments",
        "experiment",
        "evaluation",
        "results",
        "conclusion",
        "conclusions",
        "references",
    }
    if normalized in known:
        return line
    if re.match(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z -]{2,80}$", line):
        return line
    return None


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|\d+(?:\.\d+)?", text)
    ]


def _score(query_terms: list[str], chunk_terms: list[str]) -> float:
    if not chunk_terms:
        return 0.0
    query_unique = set(query_terms)
    chunk_counts: dict[str, int] = {}
    for term in chunk_terms:
        chunk_counts[term] = chunk_counts.get(term, 0) + 1
    score = 0.0
    for term in query_unique:
        count = chunk_counts.get(term, 0)
        if count:
            score += 1.0 + math.log(count)
    coverage = sum(1 for term in query_unique if term in chunk_counts) / len(query_unique)
    return score * (1.0 + coverage)


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _validate_doc_id(doc_id: str) -> None:
    if not re.fullmatch(r"userdoc-[a-f0-9]{12}", doc_id):
        raise ValueError(f"invalid user document id: {doc_id!r}")
