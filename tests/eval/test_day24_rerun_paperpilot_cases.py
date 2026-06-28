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
