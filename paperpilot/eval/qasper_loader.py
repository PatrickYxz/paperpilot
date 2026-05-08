"""Load QASPER paper dumps into Day 16 EvalCase rows."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    arxiv_id: str
    paper_title: str
    abstract: str
    full_text: str
    question: str
    oracle_spans: tuple[str, ...]


def extract_arxiv_id(url: str | None) -> str | None:
    if not url:
        return None
    match = ARXIV_RE.search(url)
    return match.group(1) if match else None


def _join_full_text(full_text: list[dict] | None) -> str:
    if not full_text:
        return ""

    parts: list[str] = []
    for section in full_text:
        if not isinstance(section, dict):
            continue
        name = section.get("section_name") or ""
        paragraphs = section.get("paragraphs") or []
        if name:
            parts.append(f"## {name}")
        parts.extend(p for p in paragraphs if isinstance(p, str) and p)
    return "\n\n".join(parts)


def _answer_spans(answer_record: dict[str, Any]) -> list[str]:
    inner = answer_record.get("answer")
    if not isinstance(inner, dict):
        inner = answer_record
    spans = inner.get("extractive_spans") if isinstance(inner, dict) else None
    if not spans:
        return []
    return [s for s in spans if isinstance(s, str) and s.strip()]


def _collect_extractive_qas(qas: list) -> list[tuple[str, tuple[str, ...]]]:
    out: list[tuple[str, tuple[str, ...]]] = []
    for qa in qas or []:
        if not isinstance(qa, dict):
            continue
        question = qa.get("question") or ""
        spans: list[str] = []
        for answer in qa.get("answers") or []:
            if isinstance(answer, dict):
                spans.extend(_answer_spans(answer))
        if not question or not spans:
            continue

        seen: set[str] = set()
        uniq: list[str] = []
        for span in spans:
            if span in seen:
                continue
            seen.add(span)
            uniq.append(span)
        out.append((question, tuple(uniq)))
    return out


def load_qasper_cases(qasper_path: Path) -> list[EvalCase]:
    """Parse QASPER JSON and return three extractive QA cases per arXiv paper."""
    raw = json.loads(qasper_path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []

    for paper in raw.values():
        if not isinstance(paper, dict):
            continue
        arxiv_id = extract_arxiv_id(paper.get("paper_url"))
        if arxiv_id is None:
            continue

        qa_list = _collect_extractive_qas(paper.get("qas") or [])
        if len(qa_list) < 3:
            continue

        full_text = _join_full_text(paper.get("full_text"))
        for idx, (question, spans) in enumerate(qa_list[:3]):
            cases.append(EvalCase(
                case_id=f"qasper-{arxiv_id}-q{idx}",
                arxiv_id=arxiv_id,
                paper_title=paper.get("title") or "",
                abstract=paper.get("abstract") or "",
                full_text=full_text,
                question=question,
                oracle_spans=spans,
            ))
    return cases
