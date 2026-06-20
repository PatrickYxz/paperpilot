"""Three baselines for Day 16 deep-read eval."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from paperpilot.core import LLMClient
from paperpilot.eval.answer_quality import (
    build_repair_prompt,
    evaluate_answer_quality,
    should_repair_answer,
)
from paperpilot.eval.evidence_selection import (
    apply_selector_action,
    extract_retrieved_chunks,
    run_evidence_selector,
)
from paperpilot.eval.jsonl_tracer import make_jsonl_tracer
from paperpilot.eval.query_planner import QueryPlan, plan_queries
from paperpilot.eval.qasper_loader import EvalCase

_TRACE_DIR = Path("data/traces")
# About 60K tokens at a rough 4 chars/token average, without adding tiktoken.
_FULL_TEXT_CHAR_BUDGET = 240_000


def run_abstract_only(case: EvalCase, client: LLMClient | None = None) -> dict[str, Any]:
    client = client or LLMClient()
    prompt = (
        f"你是学术论文助手。下面是论文 \"{case.paper_title}\" 的 abstract:\n\n"
        f"{case.abstract}\n\n"
        f"请基于 abstract 简洁回答以下问题。"
        f"如果 abstract 不含答案,直接说\"abstract 中未提及\":\n\n"
        f"Q: {case.question}\nA:"
    )
    return _one_shot(prompt, client)


def run_full_text_dump(case: EvalCase, client: LLMClient | None = None) -> dict[str, Any]:
    client = client or LLMClient()
    text = case.full_text or case.abstract
    truncated = len(text) > _FULL_TEXT_CHAR_BUDGET
    if truncated:
        body = text[:_FULL_TEXT_CHAR_BUDGET]
        notice = (
            f"全文过长,已截至约 {_FULL_TEXT_CHAR_BUDGET // 4} tokens, "
            "后段省略。\n\n"
        )
    else:
        body = text
        notice = ""

    prompt = (
        f"{notice}你是学术论文助手。下面是论文 \"{case.paper_title}\" 的全文:\n\n"
        f"{body}\n\n"
        f"请基于全文回答以下问题:\n\nQ: {case.question}\nA:"
    )
    return _one_shot(prompt, client)


def run_paperpilot(
    case: EvalCase,
    max_iter: int = 12,
    repair_client: LLMClient | None = None,
    evidence_client: LLMClient | None = None,
    use_query_plan: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Run PaperPilot end to end while collecting tool calls and JSONL trace."""
    from paperpilot.main import run as agent_run

    tool_calls: list[str] = []
    run_id = trace_id or case.case_id
    file_tracer = make_jsonl_tracer(run_id, _TRACE_DIR)
    trace_path = _TRACE_DIR / f"{run_id}.jsonl"

    def composite_tracer(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            tool_calls.append(payload.get("name", ""))
        file_tracer(kind, payload)

    query_plan: QueryPlan | None = None
    if use_query_plan:
        query_plan = plan_queries(
            case.question,
            title=case.paper_title,
            abstract=case.abstract,
        )

    prompt = _build_paperpilot_prompt(case, query_plan=query_plan)

    t0 = time.time()
    predicted = ""
    error: str | None = None
    try:
        messages = agent_run(prompt, max_iter=max_iter, on_event=composite_tracer)
        predicted = _extract_final_text(messages[-1].get("content"))
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    elapsed = round(time.time() - t0, 1)

    predicted_raw = predicted
    answer_quality = evaluate_answer_quality(predicted_raw, question=case.question)
    answer_repaired = False
    repair_answer_quality: dict[str, Any] | None = None
    repair_error: str | None = None
    evidence_selection: dict[str, Any] | None = None
    evidence_rewritten = False
    evidence_selection_error: str | None = None

    if error is None and should_repair_answer(answer_quality):
        repaired, repair_answer_quality, repair_error = _repair_final_answer(
            case=case,
            raw_answer=predicted_raw,
            quality=answer_quality,
            repair_client=repair_client,
        )
        if repaired:
            predicted = repaired
            answer_repaired = True

    if error is None:
        predicted, evidence_selection, evidence_rewritten, evidence_selection_error = (
            _select_evidence_supported_answer(
                case=case,
                candidate_answer=predicted,
                trace_path=trace_path,
                evidence_client=evidence_client,
            )
        )

    planned_retrieval_meta = _extract_planned_retrieval_metadata(trace_path)

    return {
        "predicted_raw": predicted_raw,
        "predicted": predicted,
        "elapsed_s": elapsed,
        "trace_path": str(trace_path),
        "tool_calls": tool_calls,
        "error": error,
        "answer_quality": answer_quality,
        "answer_repaired": answer_repaired,
        "repair_answer_quality": repair_answer_quality,
        "repair_error": repair_error,
        "evidence_selection": evidence_selection,
        "evidence_rewritten": evidence_rewritten,
        "evidence_selection_error": evidence_selection_error,
        **planned_retrieval_meta,
        "query_plan_used": use_query_plan,
        "query_plan_version": "deterministic_v1" if query_plan is not None else None,
        "query_plan": query_plan.to_dict() if query_plan is not None else None,
    }


def _one_shot(prompt: str, client: LLMClient) -> dict[str, Any]:
    t0 = time.time()
    predicted = ""
    error: str | None = None
    try:
        resp = client.call(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system="",
        )
        predicted = resp.text or ""
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    elapsed = round(time.time() - t0, 1)
    return {
        "predicted": predicted,
        "elapsed_s": elapsed,
        "trace_path": None,
        "tool_calls": None,
        "error": error,
    }


def _extract_final_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: list[str] = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(p for p in parts if p)


def _build_paperpilot_prompt(case: EvalCase, *, query_plan: QueryPlan | None = None) -> str:
    guidance = _format_query_plan_guidance(query_plan) if query_plan is not None else ""
    return (
        f"请精读 arxiv:{case.arxiv_id}(标题《{case.paper_title}》),"
        "回答下面的问题。使用 deep-read-paper skill 的工作流"
        "(load_skill -> download_paper -> build_index -> "
        "planned_retrieval -> 必要时补充 colbert.search -> 综合)。\n\n"
        f"{guidance}"
        f"Q: {case.question}\nA:"
    )


def _format_query_plan_guidance(query_plan: QueryPlan) -> str:
    data = query_plan.to_dict()
    must_find = _bullet_lines(data.get("must_find") or [])
    avoid = _bullet_lines(data.get("avoid") or [])
    searches = _numbered_query_lines(data.get("queries") or [])

    avoid_block = f"\nAvoid:\n{avoid}" if avoid else ""
    return (
        "Retrieval guidance generated before the run:\n\n"
        f"Question type: {data['question_type']}\n"
        f"Expected answer shape: {data['answer_shape']}\n\n"
        f"Must find:\n{must_find}"
        f"{avoid_block}\n\n"
        f"Planned searches:\n{searches}\n\n"
        "During deep-read:\n"
        "- First run the literal question search.\n"
        "- Then run at least two planned searches that target the expected answer shape.\n"
        "- Prefer evidence that directly expresses the relation asked by the question.\n"
        "- Do not answer from a related mention unless it directly supports the question.\n\n"
    )


def _extract_planned_retrieval_metadata(trace_path: Path) -> dict[str, Any]:
    defaults = {
        "planned_retrieval_used": False,
        "planned_retrieval_stats": None,
        "planned_retrieval_missing_requirements": None,
        "planned_retrieval_query_plan_meta": None,
        "planned_retrieval_query_errors": None,
    }
    if not trace_path.exists():
        return defaults

    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "tool_result":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("name") != "mcp__colbert__planned_retrieval":
            continue
        content = _parse_json_content(payload.get("content"))
        if not isinstance(content, dict):
            continue
        pool = content.get("evidence_pool")
        if not isinstance(pool, dict):
            pool = {}
        return {
            "planned_retrieval_used": True,
            "planned_retrieval_stats": pool.get("stats"),
            "planned_retrieval_missing_requirements": pool.get(
                "missing_requirements"
            ),
            "planned_retrieval_query_plan_meta": content.get("query_plan_meta"),
            "planned_retrieval_query_errors": content.get("query_errors"),
        }
    return defaults


def _parse_json_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    text = content.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text)
        return parsed
    except json.JSONDecodeError:
        return None


def _bullet_lines(values: list[Any]) -> str:
    return "\n".join(f"- {value}" for value in values if str(value).strip()) or "- direct evidence"


def _numbered_query_lines(queries: list[Any]) -> str:
    lines: list[str] = []
    for index, item in enumerate(queries, start=1):
        query = item.get("query", "") if isinstance(item, dict) else ""
        if str(query).strip():
            lines.append(f"{index}. {query}")
    return "\n".join(lines) or "1. literal question"


def _repair_final_answer(
    *,
    case: EvalCase,
    raw_answer: str,
    quality: dict[str, Any],
    repair_client: LLMClient | None = None,
) -> tuple[str, dict[str, Any] | None, str | None]:
    prompt = build_repair_prompt(
        question=case.question,
        raw_answer=raw_answer,
        quality=quality,
    )
    client = repair_client or LLMClient()
    try:
        response = client.call(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system="",
        )
        repaired = response.text or ""
    except Exception as e:  # noqa: BLE001
        return "", None, f"{type(e).__name__}: {e}"

    repaired_quality = evaluate_answer_quality(repaired, question=case.question)
    if not repaired.strip():
        return "", repaired_quality, "repair returned empty answer"
    return repaired, repaired_quality, None


def _select_evidence_supported_answer(
    *,
    case: EvalCase,
    candidate_answer: str,
    trace_path: Path,
    evidence_client: LLMClient | None = None,
) -> tuple[str, dict[str, Any] | None, bool, str | None]:
    try:
        retrieved_chunks = extract_retrieved_chunks(trace_path)
        selection, error = run_evidence_selector(
            question=case.question,
            candidate_answer=candidate_answer,
            retrieved_chunks=retrieved_chunks,
            client=evidence_client,
        )
        selected_answer, rewritten = apply_selector_action(candidate_answer, selection)
        return selected_answer, selection, rewritten, error
    except Exception as e:  # noqa: BLE001
        return candidate_answer, None, False, f"{type(e).__name__}: {e}"
