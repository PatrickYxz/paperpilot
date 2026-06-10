"""Load QASPER paper dumps into Day 16 EvalCase rows."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)
ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})$")


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    arxiv_id: str
    paper_title: str
    abstract: str
    full_text: str
    question: str
    oracle_spans: tuple[str, ...]


@dataclass(frozen=True)
class QasperAnswer:
    annotation_id: str
    extractive_spans: tuple[str, ...]
    free_form_answer: str
    yes_no: bool | None
    unanswerable: bool
    evidence: tuple[str, ...]
    highlighted_evidence: tuple[str, ...]


@dataclass(frozen=True)
class EnrichedEvalCase:
    case_id: str
    arxiv_id: str
    paper_title: str
    abstract: str
    full_text: str
    question: str
    oracle_spans: tuple[str, ...]
    answers: tuple[QasperAnswer, ...]


def extract_arxiv_id(s: str | None) -> str | None:
    """Accept either a raw arxiv id (QASPER paper key, e.g. '1909.00694')
    or an arxiv URL ('https://arxiv.org/abs/...')."""
    if not s:
        return None
    m = ARXIV_ID_RE.match(s)
    if m:
        return m.group(1)
    m = ARXIV_URL_RE.search(s)
    return m.group(1) if m else None


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


def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item.strip())


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _parse_qasper_answer(answer_record: dict[str, Any]) -> QasperAnswer:
    inner = answer_record.get("answer")
    if not isinstance(inner, dict):
        inner = answer_record
    return QasperAnswer(
        annotation_id=str(answer_record.get("annotation_id") or ""),
        extractive_spans=_string_list(inner.get("extractive_spans")),
        free_form_answer=str(inner.get("free_form_answer") or ""),
        yes_no=_optional_bool(inner.get("yes_no")),
        unanswerable=bool(inner.get("unanswerable", False)),
        evidence=_string_list(inner.get("evidence")),
        highlighted_evidence=_string_list(inner.get("highlighted_evidence")),
    )


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

    for paper_id, paper in raw.items():
        if not isinstance(paper, dict):
            continue
        # Real QASPER uses arxiv id as the dict key; fixtures use paper_url.
        arxiv_id = (
            extract_arxiv_id(paper_id)
            or extract_arxiv_id(paper.get("paper_url"))
        )
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


def load_qasper_enriched_cases(qasper_path: Path) -> list[EnrichedEvalCase]:
    """Parse QASPER JSON and preserve answer/evidence metadata.

    This uses the same extractive-only policy as load_qasper_cases(): a paper is
    included only if it has at least three QAs with extractive spans, and only
    the first three such QAs are returned.
    """
    raw = json.loads(qasper_path.read_text(encoding="utf-8"))
    cases: list[EnrichedEvalCase] = []

    for paper_id, paper in raw.items():
        if not isinstance(paper, dict):
            continue
        arxiv_id = (
            extract_arxiv_id(paper_id)
            or extract_arxiv_id(paper.get("paper_url"))
        )
        if arxiv_id is None:
            continue

        full_text = _join_full_text(paper.get("full_text"))
        qa_cases: list[tuple[str, tuple[str, ...], tuple[QasperAnswer, ...]]] = []
        for qa in paper.get("qas") or []:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question") or ""
            answers = tuple(
                _parse_qasper_answer(answer)
                for answer in qa.get("answers") or []
                if isinstance(answer, dict)
            )
            spans: list[str] = []
            for answer in answers:
                spans.extend(answer.extractive_spans)
            if not question or not spans:
                continue

            seen: set[str] = set()
            uniq: list[str] = []
            for span in spans:
                if span in seen:
                    continue
                seen.add(span)
                uniq.append(span)
            qa_cases.append((question, tuple(uniq), answers))

        if len(qa_cases) < 3:
            continue

        for idx, (question, spans, answers) in enumerate(qa_cases[:3]):
            cases.append(EnrichedEvalCase(
                case_id=f"qasper-{arxiv_id}-q{idx}",
                arxiv_id=arxiv_id,
                paper_title=paper.get("title") or "",
                abstract=paper.get("abstract") or "",
                full_text=full_text,
                question=question,
                oracle_spans=spans,
                answers=answers,
            ))
    return cases
