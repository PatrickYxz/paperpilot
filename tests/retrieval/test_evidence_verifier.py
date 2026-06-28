from paperpilot.retrieval.evidence_verifier import (
    EvidenceVerificationDecision,
    parse_verifier_output,
)


def test_parse_verifier_output_normalizes_valid_json() -> None:
    text = """
    {
      "decisions": [
        {
          "requirement_id": "req_dataset",
          "evidence_id": "ev_2",
          "support": "direct",
          "confidence": "high",
          "answer_atoms": ["WikiHop"],
          "risks": ["dataset_role_clear"],
          "reason": "The chunk states the dataset used."
        }
      ]
    }
    """

    decisions, error = parse_verifier_output(text)

    assert error is None
    assert decisions == [
        EvidenceVerificationDecision(
            requirement_id="req_dataset",
            evidence_id="ev_2",
            support="direct",
            confidence="high",
            answer_atoms=["WikiHop"],
            risks=["dataset_role_clear"],
            reason="The chunk states the dataset used.",
            score=100.0,
        )
    ]


def test_parse_verifier_output_extracts_fenced_json() -> None:
    text = """```json
    {
      "decisions": [
        {
          "requirement_id": "req_num",
          "evidence_id": "ev_1",
          "support": "partial",
          "confidence": "medium",
          "answer_atoms": ["58%"],
          "risks": [],
          "reason": "The number appears, but the metric is unclear."
        }
      ]
    }
    ```"""

    decisions, error = parse_verifier_output(text)

    assert error is None
    assert len(decisions) == 1
    assert decisions[0].support == "partial"
    assert decisions[0].confidence == "medium"
    assert decisions[0].score == 45.0


def test_parse_verifier_output_reports_invalid_json() -> None:
    decisions, error = parse_verifier_output("not json")

    assert decisions == []
    assert error == "verifier did not return valid JSON"


def test_parse_verifier_output_drops_invalid_decisions() -> None:
    text = """
    {
      "decisions": [
        {
          "requirement_id": "req_1",
          "evidence_id": "ev_1",
          "support": "maybe",
          "confidence": "high",
          "answer_atoms": [],
          "risks": [],
          "reason": "invalid"
        },
        {
          "requirement_id": "req_1",
          "evidence_id": "ev_2",
          "support": "no",
          "confidence": "low",
          "answer_atoms": [],
          "risks": [],
          "reason": "related only"
        }
      ]
    }
    """

    decisions, error = parse_verifier_output(text)

    assert error is None
    assert [item.evidence_id for item in decisions] == ["ev_2"]
    assert decisions[0].score == 0.0
