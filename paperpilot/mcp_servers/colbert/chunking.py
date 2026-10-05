"""Structure-aware chunking and contextual prefixes for paper text.

Splitting prefers document boundaries: paragraphs first, sentence windows
for oversized paragraphs, and a proportional character fallback for
pathological single sentences. Every chunk carries metadata (section hint,
character span) used to build contextual prefixes at indexing time — the
template version of Anthropic-style contextual retrieval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

CHUNK_SIZE = 256
OVERLAP = 32
_HEADING_MAX_LEN = 80
_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+\S+|abstract\b|introduction\b|conclusions?\b"
    r"|references\b|related work\b|acknowledg(e)?ments?\b)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")

TokenCounter = Callable[[str], int]


@dataclass
class ChunkRecord:
    text: str
    section_hint: str
    char_start: int
    char_end: int
    index: int = 0

    @property
    def char_span(self) -> list[int]:
        return [self.char_start, self.char_end]


@dataclass
class _Piece:
    text: str
    start: int
    end: int
    token_count: int


def chunk_document(
    text: str,
    counter: TokenCounter,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[ChunkRecord]:
    """Split text into structure-aware chunks. Empty text yields no chunks."""
    if not text.strip():
        return []

    section = ""
    groups: list[tuple[list[tuple[_Piece, str]], str]] = []
    current: list[tuple[_Piece, str]] = []
    current_tokens = 0

    for para_text, start, end in _paragraphs(text):
        hint = _heading_hint(para_text)
        if hint:
            section = hint
        para_tokens = counter(para_text.strip())
        pieces = (
            [_Piece(para_text.strip(), start, end, para_tokens)]
            if para_tokens <= chunk_size
            else _sentence_pieces(para_text, start, counter, chunk_size)
        )
        for piece in pieces:
            if current and current_tokens + piece.token_count > chunk_size:
                groups.append((current, _section_of(current)))
                # Carry the trailing piece into the next group as overlap when
                # it is small enough; paragraph-sized pieces stay exclusive.
                tail = (
                    current[-1]
                    if current[-1][0].token_count <= overlap
                    else None
                )
                current = [tail] if tail else []
                current_tokens = tail[0].token_count if tail else 0
            current.append((piece, section))
            current_tokens += piece.token_count

    if current:
        groups.append((current, _section_of(current)))

    records: list[ChunkRecord] = []
    for group, group_section in groups:
        records.append(
            ChunkRecord(
                text="\n".join(piece.text for piece, _ in group),
                section_hint=group_section,
                char_start=group[0][0].start,
                char_end=group[-1][0].end,
            )
        )
    for i, record in enumerate(records):
        record.index = i
    return records


def _section_of(group: list[tuple[_Piece, str]]) -> str:
    for _, section in group:
        if section:
            return section
    return ""


def default_context_prefix(paper_id: str, record: ChunkRecord, total: int) -> str:
    """Template contextual prefix anchoring a chunk to its source context."""
    parts = [f"source: paper {paper_id}"]
    if record.section_hint:
        parts.append(f"section: {record.section_hint}")
    parts.append(f"part {record.index + 1}/{total}")
    return "[" + "; ".join(parts) + "]"


def _paragraphs(text: str) -> list[tuple[str, int, int]]:
    """Split on blank lines, keeping absolute character offsets."""
    paragraphs: list[tuple[str, int, int]] = []
    lines = text.splitlines(keepends=True)
    current: list[str] = []
    current_start = 0
    pos = 0
    for line in lines:
        if line.strip():
            if not current:
                current_start = pos
            current.append(line)
        elif current:
            paragraphs.append(("".join(current), current_start, pos))
            current = []
        pos += len(line)
    if current:
        paragraphs.append(("".join(current), current_start, pos))
    return paragraphs


def _heading_hint(para_text: str) -> str:
    stripped = para_text.strip()
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0].strip()
    if not first_line or len(first_line) > _HEADING_MAX_LEN:
        return ""
    if _HEADING_RE.match(first_line):
        return first_line
    letters = [char for char in first_line if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        return first_line
    return ""


def _sentence_pieces(
    para_text: str,
    para_start: int,
    counter: TokenCounter,
    chunk_size: int,
) -> list[_Piece]:
    sentences = _sentences_with_offsets(para_text.strip(), para_start)
    if not sentences:
        return []

    counts = [counter(sent) for sent, _, _ in sentences]
    if max(counts, default=0) <= chunk_size:
        return [
            _Piece(sent, start, end, count)
            for (sent, start, end), count in zip(sentences, counts)
        ]

    # Pathological single sentence longer than the budget: proportional
    # character split so downstream windows stay bounded.
    oversized = sentences[counts.index(max(counts))]
    sent, start, end = oversized
    ratio = chunk_size / max(max(counts), 1)
    cut = max(int(len(sent) * ratio), 1)
    head, tail = sent[:cut].strip(), sent[cut:].strip()
    pieces = [_Piece(head, start, start + cut, counter(head))]
    if tail:
        pieces.append(_Piece(tail, start + cut, end, counter(tail)))
    return pieces


def _sentences_with_offsets(
    text: str, offset_base: int
) -> list[tuple[str, int, int]]:
    sentences: list[tuple[str, int, int]] = []
    pos = 0
    for sent in _SENTENCE_SPLIT_RE.split(text):
        if not sent:
            continue
        found = text.find(sent, pos)
        start = found if found >= 0 else pos
        sentences.append((sent, offset_base + start, offset_base + start + len(sent)))
        pos = start + len(sent)
    return sentences
