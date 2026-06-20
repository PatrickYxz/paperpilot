"""Tests for eval evidence selection helpers."""
import json
from pathlib import Path
from types import SimpleNamespace

from paperpilot.eval.evidence_selection import (
    apply_selector_action,
    build_selector_prompt,
    extract_retrieved_chunks,
    parse_selector_output,
    run_evidence_selector,
)


def test_extract_retrieved_chunks_pairs_search_calls_and_results(tmp_path: Path) -> None:
    trace = tmp_path / "case.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "kind": "tool_call",
                "payload": {
                    "name": "mcp__colbert__search",
                    "arguments": {
                        "query": "dataset used in experiment",
                        "paper_id": "1910.04601",
                        "top_k": 2,
                    },
                },
            },
            {
                "kind": "tool_result",
                "payload": {
                    "name": "mcp__colbert__search",
                    "content": json.dumps([
                        {
                            "paper_id": "1910.04601",
                            "chunk_text": "Our study uses WikiHop.",
                            "score": 0.9,
                        },
                        {
                            "paper_id": "1910.04601",
                            "chunk_text": "HotpotQA is a related dataset.",
                            "score": 0.8,
                        },
                    ]),
                },
            },
        ],
    )

    chunks = extract_retrieved_chunks(trace)

    assert len(chunks) == 2
    assert chunks[0]["query"] == "dataset used in experiment"
    assert chunks[0]["rank"] == 1
    assert chunks[0]["chunk_text"] == "Our study uses WikiHop."
    assert chunks[1]["rank"] == 2


def test_extract_retrieved_chunks_accepts_concatenated_json_objects(tmp_path: Path) -> None:
    trace = tmp_path / "case.jsonl"
    content = (
        '{"paper_id": "p1", "chunk_text": "First chunk.", "score": 1.0}\n'
        '{"paper_id": "p1", "chunk_text": "Second chunk.", "score": 0.9}'
    )
    _write_jsonl(
        trace,
        [
            {
                "kind": "tool_call",
                "payload": {
                    "name": "mcp__colbert__search",
                    "arguments": {"query": "q", "paper_id": "p1", "top_k": 2},
                },
            },
            {
                "kind": "tool_result",
                "payload": {
                    "name": "mcp__colbert__search",
                    "content": content,
                },
            },
        ],
    )

    chunks = extract_retrieved_chunks(trace)

    assert [chunk["chunk_text"] for chunk in chunks] == [
        "First chunk.",
        "Second chunk.",
    ]


def test_build_selector_prompt_contains_core_inputs() -> None:
    prompt = build_selector_prompt(
        question="What dataset was used?",
        candidate_answer="Short answer: HotpotQA.",
        retrieved_chunks=[
            {
                "query": "dataset",
                "rank": 1,
                "paper_id": "p1",
                "chunk_text": "Our study uses WikiHop.",
            }
        ],
    )

    assert "What dataset was used?" in prompt
    assert "Short answer: HotpotQA." in prompt
    assert "Our study uses WikiHop." in prompt
    assert "Return JSON only" in prompt
    assert "smallest list governed" in prompt


def test_parse_selector_output_accepts_fenced_json() -> None:
    selection = parse_selector_output(
        """```json
        {
          "question_type": "entity",
          "answer_supported": false,
          "selected_answer": "WikiHop",
          "supporting_evidence": "Our study uses WikiHop.",
          "rejected_candidates": [],
          "action": "replace_answer",
          "confidence": "high",
          "reason": "direct support"
        }
        ```"""
    )

    assert selection["action"] == "replace_answer"
    assert selection["confidence"] == "high"
    assert selection["selected_answer"] == "WikiHop"


def test_parse_selector_output_invalid_json_becomes_uncertain() -> None:
    selection = parse_selector_output("not json")

    assert selection["action"] == "selector_uncertain"
    assert selection["confidence"] == "low"


def test_apply_selector_action_keeps_answer() -> None:
    answer, rewritten = apply_selector_action(
        "Short answer: WikiHop.",
        {"action": "keep_answer"},
    )

    assert answer == "Short answer: WikiHop."
    assert rewritten is False


def test_apply_selector_action_replaces_answer_from_selected_evidence() -> None:
    answer, rewritten = apply_selector_action(
        "Short answer: HotpotQA.",
        {
            "action": "replace_answer",
            "confidence": "high",
            "selected_answer": "WikiHop",
            "supporting_evidence": "Our study uses WikiHop.",
        },
    )

    assert rewritten is True
    assert answer == "Short answer: WikiHop\n\nEvidence: Our study uses WikiHop."


def test_apply_selector_action_does_not_rewrite_low_confidence_selection() -> None:
    answer, rewritten = apply_selector_action(
        "Short answer: HotpotQA.",
        {
            "action": "replace_answer",
            "confidence": "low",
            "selected_answer": "WikiHop",
            "supporting_evidence": "Our study uses WikiHop.",
        },
    )

    assert answer == "Short answer: HotpotQA."
    assert rewritten is False


def test_apply_selector_action_insufficient_evidence_is_conservative() -> None:
    answer, rewritten = apply_selector_action(
        "Short answer: HotpotQA.",
        {
            "action": "insufficient_evidence",
            "confidence": "medium",
            "supporting_evidence": "",
        },
    )

    assert rewritten is True
    assert "Short answer: Evidence is insufficient." in answer
    assert "retrieved passages" in answer


def test_run_evidence_selector_invalid_response_does_not_raise() -> None:
    client = FakeClient("not json")

    selection, error = run_evidence_selector(
        question="What dataset was used?",
        candidate_answer="Short answer: HotpotQA.",
        retrieved_chunks=[{"chunk_text": "Our study uses WikiHop."}],
        client=client,
    )

    assert error is None
    assert selection["action"] == "selector_uncertain"
    assert len(client.calls) == 1


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def call(self, messages: list[dict], tools: list, *, system: str):
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        return SimpleNamespace(text=self.text)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )
