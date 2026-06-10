"""Tests for the enriched QASPER subset writer."""
import json
from dataclasses import asdict
from pathlib import Path

from paperpilot.eval.qasper_loader import load_qasper_enriched_cases
from scripts.day19_prepare_enriched_eval import (
    _index_enriched_cases,
    _load_subset_case_ids,
    _serialize_enriched_case,
)

FIXTURE = Path(__file__).parent / "fixtures" / "qasper_mini.json"


def test_load_subset_case_ids_preserves_file_order(tmp_path: Path) -> None:
    subset = tmp_path / "subset.jsonl"
    subset.write_text(
        "\n".join([
            json.dumps({"case_id": "qasper-2001.12345-q2"}),
            json.dumps({"case_id": "qasper-2001.12345-q0"}),
        ]) + "\n",
        encoding="utf-8",
    )

    assert _load_subset_case_ids(subset) == [
        "qasper-2001.12345-q2",
        "qasper-2001.12345-q0",
    ]


def test_index_enriched_cases_by_case_id() -> None:
    cases = load_qasper_enriched_cases(FIXTURE)
    indexed = _index_enriched_cases(cases)

    assert indexed["qasper-2001.12345-q0"].question == "Q1?"
    assert indexed["qasper-2001.12345-q2"].answers[1].free_form_answer == "Free answer 3b."


def test_serialize_enriched_case_converts_tuples_to_lists() -> None:
    case = load_qasper_enriched_cases(FIXTURE)[0]
    row = _serialize_enriched_case(case)

    assert row["case_id"] == "qasper-2001.12345-q0"
    assert row["oracle_spans"] == ["span-1a"]
    assert row["answers"][0]["extractive_spans"] == ["span-1a"]
    assert row["answers"][0]["evidence"] == ["Evidence paragraph 1."]
    assert asdict(case)["oracle_spans"] == ("span-1a",)
