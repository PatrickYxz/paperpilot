"""Evidence support selection helpers for eval-time PaperPilot answers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paperpilot.core import LLMClient

ALLOWED_ACTIONS = {
    "keep_answer",
    "replace_answer",
    "insufficient_evidence",
    "selector_uncertain",
}

ALLOWED_CONFIDENCE = {"high", "medium", "low"}

MAX_CHUNKS_FOR_PROMPT = 12
MAX_CHUNK_CHARS = 1800


def extract_retrieved_chunks(trace_path: str | Path) -> list[dict[str, Any]]:
    """Extract ColBERT search chunks from an agent JSONL trace.

    The trace records `tool_call` and `tool_result` separately. Search query
    metadata is attached to the preceding tool call, while retrieved chunk text
    is stored in the result content.
    """
    path = Path(trace_path)
    if not path.exists():
        return []

    chunks: list[dict[str, Any]] = []
    pending_search_call: dict[str, Any] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        kind = record.get("kind")
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        name = str(payload.get("name", ""))

        if kind == "tool_call" and _is_colbert_search(name):
            args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
            pending_search_call = {
                "query": str(args.get("query", "")),
                "paper_id": str(args.get("paper_id", "")),
                "top_k": args.get("top_k"),
            }
            continue

        if kind == "tool_result" and _is_colbert_search(name):
            result_chunks = _parse_search_result_content(payload.get("content"))
            query_meta = pending_search_call or {}
            for index, chunk in enumerate(result_chunks, start=1):
                normalized = _normalize_chunk(chunk)
                if not normalized.get("chunk_text"):
                    continue
                chunks.append({
                    "query": query_meta.get("query", ""),
                    "rank": index,
                    "paper_id": normalized.get("paper_id") or query_meta.get("paper_id", ""),
                    "chunk_text": normalized["chunk_text"],
                    "score": normalized.get("score"),
                })
            pending_search_call = None
            continue

        if kind == "tool_result" and _is_colbert_planned_retrieval(name):
            chunks.extend(_extract_planned_retrieval_chunks(payload.get("content")))

    return chunks


def build_selector_prompt(
    *,
    question: str,
    candidate_answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Build the strict JSON-only selector prompt."""
    chunk_lines = []
    for index, chunk in enumerate(retrieved_chunks[:MAX_CHUNKS_FOR_PROMPT], start=1):
        text = str(chunk.get("chunk_text", ""))[:MAX_CHUNK_CHARS]
        chunk_lines.append(
            f"[{index}] query={chunk.get('query', '')!r} "
            f"rank={chunk.get('rank', '')!r} paper_id={chunk.get('paper_id', '')!r}\n"
            f"{text}"
        )
    chunks_text = "\n\n".join(chunk_lines) or "(no retrieved chunks)"

    return (
        "You are an evidence support selector for a single-paper QA system.\n"
        "Your job is not to judge against a gold answer. Your job is only to decide "
        "whether the candidate answer is directly supported by the retrieved chunks.\n\n"
        "Rules:\n"
        "- Use only the retrieved chunks below.\n"
        "- Do not use prior knowledge.\n"
        "- Do not use or infer any benchmark gold answer.\n"
        "- Check whether the answer follows from the local evidence relation, not just "
        "whether similar words appear nearby.\n"
        "- Also check whether the candidate answer contains extra items, extra metrics, "
        "or extra task results that the question did not ask for. If it does, use "
        "replace_answer with the narrower directly supported answer.\n"
        "- For entity questions, the evidence must state the relation asked by the question.\n"
        "- For numeric questions, the same number and the same comparison or measurement "
        "relation must appear in the supporting evidence; do not keep additional numbers "
        "from other datasets, tables, or downstream tasks unless the question asks for them.\n"
        "- For list questions, every direct-answer item must belong to the requested category.\n"
        "- For list questions with a narrow category phrase, return the smallest list governed "
        "by that phrase. Do not include separate groups introduced by wording such as "
        "'also', 'further compare', 'other', 'baseline', or 'besides' unless the question "
        "explicitly asks for all compared methods or all baselines.\n"
        "- Do not keep a candidate answer merely because all of its facts appear somewhere "
        "in the retrieved chunks; the facts must be directly requested by the question.\n"
        "- Return JSON only, with no markdown fences.\n\n"
        "Allowed actions:\n"
        "- keep_answer: candidate answer is directly supported by retrieved evidence.\n"
        "- replace_answer: retrieved evidence supports a better direct answer.\n"
        "- insufficient_evidence: retrieved evidence does not directly support a reliable answer.\n"
        "- selector_uncertain: you are not confident enough to change the candidate answer.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "question_type": "entity|numeric|list|general",\n'
        '  "answer_supported": true,\n'
        '  "selected_answer": "",\n'
        '  "supporting_evidence": "",\n'
        '  "rejected_candidates": [{"candidate": "", "reason": ""}],\n'
        '  "action": "keep_answer|replace_answer|insufficient_evidence|selector_uncertain",\n'
        '  "confidence": "high|medium|low",\n'
        '  "reason": ""\n'
        "}\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{candidate_answer}\n\n"
        f"Retrieved chunks:\n{chunks_text}\n"
    )


def run_evidence_selector(
    *,
    question: str,
    candidate_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    client: LLMClient | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Run the LLM selector and return `(selection, error)`."""
    if not retrieved_chunks:
        return _uncertain_selection("No retrieved chunks were available."), None

    selector_client = client or LLMClient()
    prompt = build_selector_prompt(
        question=question,
        candidate_answer=candidate_answer,
        retrieved_chunks=retrieved_chunks,
    )
    try:
        response = selector_client.call(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            system="",
        )
    except Exception as e:  # noqa: BLE001
        return _uncertain_selection("Selector call failed."), f"{type(e).__name__}: {e}"

    text = response.text or ""
    selection = parse_selector_output(text)
    return selection, None


def parse_selector_output(text: str) -> dict[str, Any]:
    """Parse selector JSON defensively and normalize unknown fields."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json_object(text)
        if extracted is None:
            return _uncertain_selection("Selector did not return valid JSON.")
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError:
            return _uncertain_selection("Selector did not return valid JSON.")

    if not isinstance(data, dict):
        return _uncertain_selection("Selector JSON was not an object.")

    action = str(data.get("action", "selector_uncertain"))
    if action not in ALLOWED_ACTIONS:
        action = "selector_uncertain"

    confidence = str(data.get("confidence", "low"))
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "low"

    rejected = data.get("rejected_candidates")
    if not isinstance(rejected, list):
        rejected = []

    return {
        "question_type": str(data.get("question_type", "general")),
        "answer_supported": bool(data.get("answer_supported", False)),
        "selected_answer": str(data.get("selected_answer", "")).strip(),
        "supporting_evidence": str(data.get("supporting_evidence", "")).strip(),
        "rejected_candidates": rejected,
        "action": action,
        "confidence": confidence,
        "reason": str(data.get("reason", "")).strip(),
    }


def apply_selector_action(
    candidate_answer: str,
    selection: dict[str, Any],
) -> tuple[str, bool]:
    """Apply a selector decision to a candidate answer.

    Returns `(prediction, rewritten)`.
    """
    action = selection.get("action")
    if selection.get("confidence") == "low" and action in {
        "replace_answer",
        "insufficient_evidence",
    }:
        return candidate_answer, False

    if action == "replace_answer":
        selected_answer = str(selection.get("selected_answer", "")).strip()
        evidence = str(selection.get("supporting_evidence", "")).strip()
        if selected_answer and evidence:
            return (
                f"Short answer: {selected_answer}\n\n"
                f"Evidence: {evidence}",
                True,
            )
        return candidate_answer, False

    if action == "insufficient_evidence":
        evidence = str(selection.get("supporting_evidence", "")).strip()
        if not evidence:
            evidence = "The retrieved passages do not directly answer the question."
        return (
            "Short answer: Evidence is insufficient.\n\n"
            f"Evidence: {evidence}",
            True,
        )

    return candidate_answer, False


def _is_colbert_search(tool_name: str) -> bool:
    return "colbert" in tool_name and tool_name.endswith("__search")


def _is_colbert_planned_retrieval(tool_name: str) -> bool:
    return "colbert" in tool_name and tool_name.endswith("__planned_retrieval")


def _extract_planned_retrieval_chunks(content: Any) -> list[dict[str, Any]]:
    payload = _parse_json_dict_content(content)
    pool = payload.get("evidence_pool") if isinstance(payload, dict) else {}
    if not isinstance(pool, dict):
        return []

    items = pool.get("items")
    if not isinstance(items, list):
        return []

    item_by_id = {
        str(item.get("id")): item
        for item in items
        if isinstance(item, dict) and item.get("id") is not None
    }
    summary_ids = pool.get("summary_items")
    if isinstance(summary_ids, list) and summary_ids:
        ordered_items = [
            item_by_id[str(item_id)]
            for item_id in summary_ids
            if str(item_id) in item_by_id
        ]
    else:
        ordered_items = [item for item in items if isinstance(item, dict)]

    chunks: list[dict[str, Any]] = []
    for rank, item in enumerate(ordered_items, start=1):
        normalized = _normalize_evidence_item(item)
        if not normalized.get("chunk_text"):
            continue
        chunks.append({
            "query": normalized["query"],
            "rank": rank,
            "paper_id": normalized["paper_id"],
            "chunk_text": normalized["chunk_text"],
            "score": normalized["score"],
        })
    return chunks


def _parse_search_result_content(content: Any) -> list[Any]:
    if isinstance(content, (list, dict)):
        parsed = content
    elif isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return _parse_json_object_stream(content)
    else:
        return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _parse_json_dict_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return {}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_object_stream(text: str) -> list[Any]:
    """Parse adjacent JSON objects such as ``{"a": 1}\n{"a": 2}``."""
    decoder = json.JSONDecoder()
    index = 0
    values: list[Any] = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        values.append(value)
        index = end
    return values


def _normalize_chunk(chunk: Any) -> dict[str, Any]:
    if not isinstance(chunk, dict):
        return {}
    return {
        "paper_id": str(chunk.get("paper_id", "")),
        "chunk_text": str(chunk.get("chunk_text", "")),
        "score": chunk.get("score"),
    }


def _normalize_evidence_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    return {
        "paper_id": str(item.get("paper_id", "")),
        "chunk_text": str(item.get("chunk_text", "")),
        "score": item.get("best_score"),
        "query": _first_matched_query(item.get("matched_queries")),
    }


def _first_matched_query(matched_queries: Any) -> str:
    if not isinstance(matched_queries, list):
        return ""
    for match in matched_queries:
        if isinstance(match, dict) and str(match.get("query", "")).strip():
            return str(match["query"])
    return ""


def _extract_json_object(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _uncertain_selection(reason: str) -> dict[str, Any]:
    return {
        "question_type": "general",
        "answer_supported": False,
        "selected_answer": "",
        "supporting_evidence": "",
        "rejected_candidates": [],
        "action": "selector_uncertain",
        "confidence": "low",
        "reason": reason,
    }
