from paperpilot.eval.qasper_loader import EvalCase
from scripts.day24_rerun_paperpilot_cases import _record


def test_record_includes_planned_retrieval_metadata() -> None:
    case = EvalCase(
        case_id="case-1",
        arxiv_id="1234.5678",
        paper_title="Paper",
        abstract="Abstract",
        full_text="Full text",
        question="What is used?",
        oracle_spans=("answer",),
    )
    ans = {
        "predicted": "answer",
        "elapsed_s": 1.2,
        "trace_path": "trace.jsonl",
        "tool_calls": ["mcp__colbert__planned_retrieval"],
        "error": None,
        "planned_retrieval_used": True,
        "planned_retrieval_stats": {
            "raw_result_count": 20,
            "deduped_count": 10,
        },
        "planned_retrieval_missing_requirements": [],
        "planned_retrieval_query_plan_meta": {"fallback_used": False},
        "planned_retrieval_query_errors": [],
    }

    record = _record(case, ans, use_query_plan=False)

    assert record["planned_retrieval_used"] is True
    assert record["planned_retrieval_stats"] == {
        "raw_result_count": 20,
        "deduped_count": 10,
    }
    assert record["planned_retrieval_missing_requirements"] == []
    assert record["planned_retrieval_query_plan_meta"] == {"fallback_used": False}
    assert record["planned_retrieval_query_errors"] == []


def test_record_includes_verification_metadata() -> None:
    case = EvalCase(
        case_id="case-1",
        arxiv_id="1234.5678",
        paper_title="Paper",
        abstract="Abstract",
        full_text="Full text",
        question="What dataset was used?",
        oracle_spans=("WikiHop",),
    )
    ans = {
        "predicted": "WikiHop",
        "elapsed_s": 1.0,
        "trace_path": "trace.jsonl",
        "tool_calls": ["mcp__colbert__planned_retrieval"],
        "error": None,
        "evidence_verification_requested": True,
        "evidence_verification_used": True,
        "verified_summary_count": 1,
        "direct_support_count": 1,
        "partial_support_count": 0,
        "unsupported_count": 0,
        "missing_verified_requirements": [],
        "verification_conflicts": [],
    }

    record = _record(case, ans, use_query_plan=True, verify_evidence=True)

    assert record["baseline"] == "paperpilot_query_plan_v1_verified_rerun_cases"
    assert record["evidence_verification_requested"] is True
    assert record["evidence_verification_used"] is True
    assert record["verified_summary_count"] == 1
