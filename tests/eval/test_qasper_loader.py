"""Tests for paperpilot.eval.qasper_loader."""
from pathlib import Path

from paperpilot.eval.qasper_loader import (
    EvalCase,
    extract_arxiv_id,
    load_qasper_cases,
)

FIXTURE = Path(__file__).parent / "fixtures" / "qasper_mini.json"


def test_extract_arxiv_id_abs() -> None:
    assert extract_arxiv_id("https://arxiv.org/abs/2001.12345") == "2001.12345"


def test_extract_arxiv_id_pdf() -> None:
    assert extract_arxiv_id("https://arxiv.org/pdf/2001.12345.pdf") == "2001.12345"


def test_extract_arxiv_id_non_arxiv_returns_none() -> None:
    assert extract_arxiv_id("https://aclweb.org/foo") is None
    assert extract_arxiv_id(None) is None
    assert extract_arxiv_id("") is None


def test_extract_arxiv_id_raw_id() -> None:
    """Real QASPER uses the bare arxiv id as the dict key."""
    assert extract_arxiv_id("1909.00694") == "1909.00694"
    assert extract_arxiv_id("2001.12345") == "2001.12345"
    assert extract_arxiv_id("paper_aaa") is None


def test_load_qasper_filters_papers_with_lt_3_extractive_or_no_arxiv() -> None:
    cases = load_qasper_cases(FIXTURE)
    arxiv_ids = {c.arxiv_id for c in cases}
    assert arxiv_ids == {"2001.12345"}


def test_load_qasper_takes_first_three_qa_in_order() -> None:
    cases = load_qasper_cases(FIXTURE)
    a_cases = [c for c in cases if c.arxiv_id == "2001.12345"]
    assert len(a_cases) == 3
    assert [c.question for c in a_cases] == ["Q1?", "Q2?", "Q3?"]
    assert a_cases[0].case_id == "qasper-2001.12345-q0"
    assert a_cases[2].case_id == "qasper-2001.12345-q2"


def test_load_qasper_full_text_concatenated() -> None:
    cases = load_qasper_cases(FIXTURE)
    a = next(c for c in cases if c.arxiv_id == "2001.12345")
    assert "## Introduction" in a.full_text
    assert "Intro p1." in a.full_text
    assert "## Method" in a.full_text
    assert "Method p1." in a.full_text
    assert isinstance(a, EvalCase)


def test_load_qasper_multi_span_oracle_preserved() -> None:
    cases = load_qasper_cases(FIXTURE)
    a_cases = [c for c in cases if c.arxiv_id == "2001.12345"]
    q2 = next(c for c in a_cases if c.question == "Q2?")
    assert q2.oracle_spans == ("span-2a", "span-2b")
