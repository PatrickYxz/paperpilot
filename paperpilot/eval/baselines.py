"""Three baselines for Day 16 deep-read eval."""
from __future__ import annotations

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
from paperpilot.eval.jsonl_tracer import make_jsonl_tracer
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
) -> dict[str, Any]:
    """Run PaperPilot end to end while collecting tool calls and JSONL trace."""
    from paperpilot.main import run as agent_run

    tool_calls: list[str] = []
    file_tracer = make_jsonl_tracer(case.case_id, _TRACE_DIR)
    trace_path = _TRACE_DIR / f"{case.case_id}.jsonl"

    def composite_tracer(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            tool_calls.append(payload.get("name", ""))
        file_tracer(kind, payload)

    prompt = (
        f"请精读 arxiv:{case.arxiv_id}(标题《{case.paper_title}》),"
        "回答下面的问题。使用 deep-read-paper skill 的工作流"
        "(load_skill -> download_paper -> build_index -> "
        "多次 colbert.search -> 综合)。\n\n"
        f"Q: {case.question}\nA:"
    )

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
