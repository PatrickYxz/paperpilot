"""Tests for eval baseline result shaping."""
from types import SimpleNamespace

from paperpilot.eval import baselines
from paperpilot.eval.qasper_loader import EvalCase


CASE = EvalCase(
    case_id="case-1",
    arxiv_id="1234.5678",
    paper_title="Test Paper",
    abstract="Abstract",
    full_text="Full text",
    question="What dataset was used?",
    oracle_spans=("WikiHop",),
)


class FakeRepairClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def call(self, messages: list[dict], tools: list, *, system: str):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "system": system,
        })
        return SimpleNamespace(text=self.text)


def test_run_paperpilot_repairs_severe_quality_issue(monkeypatch, tmp_path) -> None:
    raw = "Step 4\nShort answer: HotpotQA.\n\nEvidence: The draft mentions HotpotQA."
    repaired = (
        "Short answer: HotpotQA.\n\n"
        "Evidence: The draft mentions HotpotQA as the dataset used."
    )
    repair_client = FakeRepairClient(repaired)

    monkeypatch.setattr(baselines, "_TRACE_DIR", tmp_path)
    _patch_agent_run(monkeypatch, raw)

    result = baselines.run_paperpilot(CASE, repair_client=repair_client)

    assert result["predicted_raw"] == raw
    assert result["predicted"] == repaired
    assert result["answer_repaired"] is True
    assert result["answer_quality"]["severity"] == "severe"
    assert result["repair_answer_quality"]["severity"] == "none"
    assert result["repair_error"] is None
    assert len(repair_client.calls) == 1
    assert repair_client.calls[0]["tools"] == []


def test_run_paperpilot_does_not_repair_risk_only_issue(monkeypatch, tmp_path) -> None:
    raw = (
        "Short answer: A, B, C, D, E, F, G, H are used.\n\n"
        "Evidence: The paper directly lists A, B, C, D, E, F, G, H as the used methods."
    )
    repair_client = FakeRepairClient("unused")

    monkeypatch.setattr(baselines, "_TRACE_DIR", tmp_path)
    _patch_agent_run(monkeypatch, raw)

    result = baselines.run_paperpilot(CASE, repair_client=repair_client)

    assert result["predicted_raw"] == raw
    assert result["predicted"] == raw
    assert result["answer_repaired"] is False
    assert result["answer_quality"]["severity"] == "risk"
    assert result["repair_answer_quality"] is None
    assert repair_client.calls == []


def test_run_paperpilot_keeps_raw_answer_when_repair_fails(monkeypatch, tmp_path) -> None:
    raw = "Step 4\nThe answer is WikiHop."

    class FailingRepairClient:
        def call(self, messages: list[dict], tools: list, *, system: str):
            raise RuntimeError("repair unavailable")

    monkeypatch.setattr(baselines, "_TRACE_DIR", tmp_path)
    _patch_agent_run(monkeypatch, raw)

    result = baselines.run_paperpilot(CASE, repair_client=FailingRepairClient())

    assert result["predicted_raw"] == raw
    assert result["predicted"] == raw
    assert result["answer_repaired"] is False
    assert result["repair_answer_quality"] is None
    assert result["repair_error"] == "RuntimeError: repair unavailable"


def test_run_paperpilot_rewrites_with_supported_selector_answer(monkeypatch, tmp_path) -> None:
    raw = (
        "Short answer: HotpotQA.\n\n"
        "Evidence: The paper mentions HotpotQA as a dataset."
    )
    selector_client = FakeRepairClient(
        """
        {
          "question_type": "entity",
          "answer_supported": false,
          "selected_answer": "WikiHop",
          "supporting_evidence": "Our study uses WikiHop.",
          "rejected_candidates": [
            {
              "candidate": "HotpotQA",
              "reason": "Mentioned nearby, but not as the direct dataset used."
            }
          ],
          "action": "replace_answer",
          "confidence": "high",
          "reason": "WikiHop has direct relation support."
        }
        """
    )

    monkeypatch.setattr(baselines, "_TRACE_DIR", tmp_path)
    _patch_agent_run_with_search(monkeypatch, raw)

    result = baselines.run_paperpilot(CASE, evidence_client=selector_client)

    assert result["predicted_raw"] == raw
    assert result["predicted"] == (
        "Short answer: WikiHop\n\nEvidence: Our study uses WikiHop."
    )
    assert result["evidence_rewritten"] is True
    assert result["evidence_selection"]["action"] == "replace_answer"
    assert result["evidence_selection_error"] is None
    assert len(selector_client.calls) == 1
    assert selector_client.calls[0]["tools"] == []


def test_run_paperpilot_keeps_answer_when_selector_fails(monkeypatch, tmp_path) -> None:
    raw = (
        "Short answer: HotpotQA.\n\n"
        "Evidence: The paper mentions HotpotQA as a dataset."
    )

    class FailingSelectorClient:
        def call(self, messages: list[dict], tools: list, *, system: str):
            raise RuntimeError("selector unavailable")

    monkeypatch.setattr(baselines, "_TRACE_DIR", tmp_path)
    _patch_agent_run_with_search(monkeypatch, raw)

    result = baselines.run_paperpilot(CASE, evidence_client=FailingSelectorClient())

    assert result["predicted"] == raw
    assert result["evidence_rewritten"] is False
    assert result["evidence_selection"]["action"] == "selector_uncertain"
    assert result["evidence_selection_error"] == "RuntimeError: selector unavailable"


def _patch_agent_run(monkeypatch, final_text: str) -> None:
    import paperpilot.main

    def fake_run(query: str, *, max_iter: int, on_event):
        on_event("tool_call", {"name": "load_skill", "arguments": {}})
        return [{"role": "assistant", "content": final_text}]

    monkeypatch.setattr(paperpilot.main, "run", fake_run)


def _patch_agent_run_with_search(monkeypatch, final_text: str) -> None:
    import json

    import paperpilot.main

    def fake_run(query: str, *, max_iter: int, on_event):
        on_event("tool_call", {
            "name": "mcp__colbert__search",
            "arguments": {
                "query": "dataset used in experiment",
                "paper_id": "1234.5678",
                "top_k": 2,
            },
        })
        on_event("tool_result", {
            "name": "mcp__colbert__search",
            "content": json.dumps([
                {
                    "paper_id": "1234.5678",
                    "chunk_text": "Our study uses WikiHop.",
                    "score": 0.9,
                },
                {
                    "paper_id": "1234.5678",
                    "chunk_text": "HotpotQA is mentioned as a related dataset.",
                    "score": 0.8,
                },
            ]),
        })
        return [{"role": "assistant", "content": final_text}]

    monkeypatch.setattr(paperpilot.main, "run", fake_run)
