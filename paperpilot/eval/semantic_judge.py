"""LLM-based semantic judge helpers for QASPER eval audits."""
from __future__ import annotations

import json
import re
from typing import Any

from paperpilot.core.adapter import LLMClient
from paperpilot.eval.semantic_audit import (
    ALLOWED_SEMANTIC_LABELS,
    SemanticAuditRecord,
)

ALLOWED_CONFIDENCES: set[str] = {"high", "medium", "low"}
AUDIT_VERSION = "v1"
MAX_ITEMS_PER_FIELD = 10
MAX_TEXT_CHARS = 1200

JUDGE_SYSTEM = """You are a semantic evaluator for scientific-paper QA.
Judge whether a predicted answer correctly answers the question using only the
provided QASPER gold answer metadata and gold evidence. Do not apply substring
matching or exact-span matching. Do not reward fluency, length, or unrelated
explanation. Return only JSON."""


def build_judge_prompt(enriched_case: dict[str, Any], result_row: dict[str, Any]) -> str:
    """Build the semantic judge prompt for one joined QASPER/result row."""
    strict_pass = _strict_pass_from_result(result_row)
    answers = enriched_case.get("answers") or []
    lines: list[str] = [
        "Evaluate the predicted answer against the gold QASPER answer and evidence.",
        "",
        "Labels:",
        "- correct: directly and completely answers the question, consistent with gold evidence.",
        "- partial: partly correct but missing a required answer part or too vague.",
        "- incorrect: wrong fact or does not answer the question.",
        "- contradictory: contradicts the gold answer/evidence.",
        "- unverifiable: gold evidence is insufficient to decide.",
        "- judge_uncertain: cannot confidently assign another label.",
        "",
        "Judging rules:",
        "- This is a semantic audit, not the original strict scorer.",
        "- The strict scorer result is diagnostic context only; do not use it as the semantic label.",
        "- Do not apply substring, exact-span, or exact-granularity matching.",
        "- Do not require exact wording or the same granularity as the gold spans.",
        "- Judge semantic equivalence of the predicted answer's core claim to the gold answer.",
        "- If the prediction contains the gold answer's core definition or a direct paraphrase, label it correct unless it contradicts that core definition.",
        "- A specific count or list of examples from the same category does not contradict a more general gold definition.",
        "- Extra detail is acceptable when the direct answer is complete and the extra detail does not contradict the gold answer/evidence.",
        "- Unsupported extra details are not evaluated in this semantic layer; mention them in the reason if needed, but label the core answer.",
        "- Do not reject an otherwise correct core answer only because extra details are not present in the gold evidence.",
        "- Do not mark an answer incorrect only because it is more specific than the gold answer.",
        "- If the predicted answer is marked (empty prediction), label it incorrect unless the gold answer is unanswerable.",
        "- Never infer or reconstruct a predicted answer from the gold metadata.",
        "- Use partial when the prediction answers only part of a multi-part gold answer.",
        "",
        "Return only JSON with keys: semantic_label, confidence, reason.",
        'confidence must be one of: "high", "medium", "low".',
        "",
        f"Case ID: {enriched_case.get('case_id', result_row.get('case_id', ''))}",
        f"Paper title: {_truncate(str(enriched_case.get('paper_title') or ''))}",
        f"ArXiv ID: {_truncate(str(enriched_case.get('arxiv_id') or ''))}",
        f"Question: {_truncate(str(enriched_case.get('question') or result_row.get('question') or ''))}",
        f"Strict scorer result (diagnostic only): {'pass' if strict_pass else 'fail'}",
        "",
        "Gold extractive spans:",
        _format_list(enriched_case.get("oracle_spans") or result_row.get("oracle_spans") or []),
        "",
        "Gold answers and evidence:",
        _format_answers(answers),
        "",
        "Additional full-paper context for extra details:",
        _format_full_text_context(enriched_case, result_row),
        "",
        "Predicted answer being judged, verbatim:",
        "```text",
        _format_prediction(result_row.get("predicted")),
        "```",
    ]
    return "\n".join(lines)


def parse_judge_response(text: str) -> dict[str, str]:
    """Parse and validate a strict JSON judge response."""
    raw = _extract_json_object(text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid judge JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")

    label = str(payload.get("semantic_label") or "")
    confidence = str(payload.get("confidence") or "")
    reason = str(payload.get("reason") or "").strip()

    if label not in ALLOWED_SEMANTIC_LABELS:
        raise ValueError(f"invalid semantic_label: {label!r}")
    if confidence not in ALLOWED_CONFIDENCES:
        raise ValueError(f"invalid confidence: {confidence!r}")
    if not reason:
        raise ValueError("judge reason is required")
    return {
        "semantic_label": label,
        "confidence": confidence,
        "reason": reason,
    }


def create_audit_record(
    *,
    result_row: dict[str, Any],
    judge_payload: dict[str, str],
    audit_model: str | None,
    audit_version: str = AUDIT_VERSION,
) -> SemanticAuditRecord:
    """Convert a parsed judge payload into the stored audit schema."""
    return SemanticAuditRecord(
        case_id=str(result_row["case_id"]),
        baseline=str(result_row["baseline"]),
        strict_pass=_strict_pass_from_result(result_row),
        semantic_label=judge_payload["semantic_label"],  # type: ignore[arg-type]
        confidence=judge_payload["confidence"],
        reason=judge_payload["reason"],
        used_gold_evidence=True,
        audit_model=audit_model,
        audit_version=audit_version,
    )


class SemanticJudge:
    """Run the semantic audit judge through the existing LLM client."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        audit_model: str | None = None,
        audit_version: str = AUDIT_VERSION,
    ) -> None:
        self.client = client or LLMClient()
        self.audit_model = audit_model or getattr(self.client, "model", None)
        self.audit_version = audit_version

    def audit(
        self,
        enriched_case: dict[str, Any],
        result_row: dict[str, Any],
    ) -> SemanticAuditRecord:
        prompt = build_judge_prompt(enriched_case, result_row)
        response = self.client.call(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system=JUDGE_SYSTEM,
        )
        payload = parse_judge_response(response.text or "")
        return create_audit_record(
            result_row=result_row,
            judge_payload=payload,
            audit_model=self.audit_model,
            audit_version=self.audit_version,
        )


def _strict_pass_from_result(result_row: dict[str, Any]) -> bool:
    strict_pass = result_row.get("strict_pass", result_row.get("passed"))
    if not isinstance(strict_pass, bool):
        raise ValueError("strict_pass or passed must be a boolean")
    return strict_pass


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("judge response did not contain a JSON object")
    return stripped[start : end + 1]


def _format_answers(answers: list[dict[str, Any]]) -> str:
    if not answers:
        return "- (none)"
    blocks: list[str] = []
    for idx, answer in enumerate(answers[:MAX_ITEMS_PER_FIELD], start=1):
        blocks.append(f"Answer {idx}:")
        blocks.append(
            f"- extractive_spans: {_format_inline_list(answer.get('extractive_spans') or [])}"
        )
        free_form = str(answer.get("free_form_answer") or "")
        if free_form:
            blocks.append(f"- free_form_answer: {_truncate(free_form)}")
        if answer.get("yes_no") is not None:
            blocks.append(f"- yes_no: {answer.get('yes_no')}")
        if answer.get("unanswerable"):
            blocks.append("- unanswerable: true")
        blocks.append("- evidence:")
        blocks.append(_format_list(answer.get("evidence") or []))
        blocks.append("- highlighted_evidence:")
        blocks.append(_format_list(answer.get("highlighted_evidence") or []))
    return "\n".join(blocks)


def _format_full_text_context(
    enriched_case: dict[str, Any],
    result_row: dict[str, Any],
) -> str:
    full_text = str(enriched_case.get("full_text") or "")
    if not full_text.strip():
        return "- (none)"

    needles = _context_needles(enriched_case, result_row)
    snippets: list[str] = []
    seen: set[str] = set()
    for needle in needles:
        match = re.search(re.escape(needle), full_text, flags=re.IGNORECASE)
        if not match:
            continue
        start = max(0, match.start() - 300)
        end = min(len(full_text), match.end() + 300)
        snippet = _truncate(full_text[start:end], max_chars=700)
        if snippet in seen:
            continue
        seen.add(snippet)
        snippets.append(f"- {snippet}")
        if len(snippets) >= 5:
            break
    if not snippets:
        return "- (none)"
    return "\n".join(snippets)


def _context_needles(
    enriched_case: dict[str, Any],
    result_row: dict[str, Any],
) -> list[str]:
    raw_needles: list[str] = []
    raw_needles.extend(str(span) for span in enriched_case.get("oracle_spans") or [])
    predicted = str(result_row.get("predicted") or "")
    raw_needles.extend(
        match.group(1)
        for match in re.finditer(r'["“]([^"”]{8,140})["”]', predicted)
    )

    needles: list[str] = []
    seen: set[str] = set()
    for raw in raw_needles:
        clean = " ".join(raw.split())
        if len(clean) < 8 or clean in seen:
            continue
        seen.add(clean)
        needles.append(clean)
    return needles


def _format_list(items: list[Any] | tuple[Any, ...]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(
        f"- {_truncate(str(item))}" for item in list(items)[:MAX_ITEMS_PER_FIELD]
    )


def _format_inline_list(items: list[Any] | tuple[Any, ...]) -> str:
    if not items:
        return "(none)"
    return "; ".join(_truncate(str(item), max_chars=240) for item in items)


def _format_prediction(predicted: Any) -> str:
    text = str(predicted or "").strip()
    if not text:
        return "(empty prediction)"
    return _truncate(text)


def _truncate(text: str, *, max_chars: int = MAX_TEXT_CHARS) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 15].rstrip() + " ...[truncated]"
