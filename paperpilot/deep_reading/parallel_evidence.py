"""Parallel per-paper evidence workers (book ch.10, manager pattern).

The main research agent stays the Manager: it decides when to fan out.
Each spawned worker is a bounded sub-agent for ONE prepared paper —
retrieve once, have a small model call read the pool and write a
paper-level note, and return selected evidence plus the note. Chunk
texts land in the shared evidence ledger (for decision/write-answer)
but never in the Manager's message context; per-worker failures are
isolated and reported per paper.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Mapping

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict

from paperpilot.papers import PaperCandidate

MAX_PARALLEL_WORKERS = 4

_NOTE_SYSTEM_PROMPT = (
    "You are a per-paper evidence reviewer. Given one paper's retrieved "
    "evidence pool for a research question, write ONE sentence note on "
    "how relevant this paper is to the question and what its key "
    "evidence says, and list the ids of the evidence items worth citing "
    "(only ids present in the pool; empty list if none are relevant)."
)


class _PaperNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    note: str
    selected_evidence_ids: list[str]


def _run_paper_worker(
    *,
    question: str,
    candidate: PaperCandidate,
    top_k_each: int,
    summary_k: int,
    call_mcp_json: Callable[..., dict],  # context pre-bound by the agent
    model: object,
) -> dict:
    """Bounded sub-agent for one paper: retrieve, read, note."""
    payload = call_mcp_json(
        name="mcp__colbert__planned_retrieval",
        arguments={
            "question": question,
            "paper_id": candidate.external_id,
            "paper_title": candidate.title,
            "abstract": candidate.abstract or "",
            "top_k_each": top_k_each,
            "summary_k": summary_k,
        },
        stage="research",
    )
    pool = payload.get("evidence_pool") or {}
    items = pool.get("items", []) if isinstance(pool, dict) else []
    catalog = "\n".join(
        f"- {item.get('id', '?')}: {str(item.get('chunk_text', ''))[:160]}"
        for item in items
    )
    runnable = model.with_structured_output(_PaperNote)
    note = _PaperNote.model_validate(
        runnable.invoke(
            [
                SystemMessage(content=_NOTE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Question: {question}\nPaper: {candidate.title}\n"
                        f"Evidence pool:\n{catalog}"
                    )
                ),
            ]
        )
    )
    valid_ids = {item.get("id") for item in items if item.get("id")}
    return {
        "paper_id": candidate.external_id,
        "note": note.note,
        "selected_evidence_ids": [
            eid for eid in note.selected_evidence_ids if eid in valid_ids
        ],
        "payload": payload,
    }


def build_parallel_evidence_tool(
    context,
    *,
    prepared: Mapping[str, PaperCandidate],
    evidence: dict,
    call_mcp_json: Callable[..., dict],
    decode_evidence_pool: Callable[..., tuple],
    contract_error: Callable[[str], Exception],
    emit_tool_call: Callable[..., None],
    clip: Callable[[str], str],
    required_id: Callable[[object, str], str],
    require_agent_limit: Callable[[int, str], None],
) -> BaseTool:
    """Build collect_evidence_parallel bound to one research run."""

    @tool("collect_evidence_parallel")
    def collect_evidence_parallel(
        question: str,
        top_k_each: int = 3,
        summary_k: int = 6,
    ) -> object:
        """Fan out one bounded evidence worker per PREPARED paper, in parallel.

        Use when the question needs evidence from SEVERAL prepared papers
        (comparisons, "which paper says...") — each worker retrieves and
        reviews one paper in its own context and returns a one-line note
        plus citable evidence ids; full chunk texts go to the shared
        ledger, not this conversation. For a single paper, plain
        retrieve_paper_evidence is cheaper. Papers must be prepared
        first; papers with no active workers are listed in the result.
        """
        cleaned_question = required_id(question, "research question")
        require_agent_limit(top_k_each, "top_k_each")
        require_agent_limit(summary_k, "summary_k")
        emit_tool_call(
            context,
            stage="research",
            name="collect_evidence_parallel",
            arguments={
                "query": clip(cleaned_question),
                "papers": sorted(prepared),
            },
        )
        if not prepared:
            return {
                "error": "no papers prepared; call prepare_paper first",
                "prepared_external_ids": [],
            }

        candidates = sorted(prepared.values(), key=lambda c: c.external_id)
        results: list[dict] = []
        with ThreadPoolExecutor(
            max_workers=min(MAX_PARALLEL_WORKERS, len(candidates))
        ) as pool_exec:
            futures = {
                pool_exec.submit(
                    _run_paper_worker,
                    question=cleaned_question,
                    candidate=candidate,
                    top_k_each=top_k_each,
                    summary_k=summary_k,
                    call_mcp_json=call_mcp_json,
                    model=context.model,
                ): candidate
                for candidate in candidates
            }
            for future, candidate in futures.items():
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 — isolate failures
                    results.append(
                        {
                            "paper_id": candidate.external_id,
                            "error": f"worker failed: {type(exc).__name__}",
                            "selected_evidence_ids": [],
                        }
                    )

        report: list[dict] = []
        for result in results:
            if "payload" not in result:
                report.append(result)
                continue
            candidate = prepared[result["paper_id"]]
            parsed_items, _summary_ids = decode_evidence_pool(
                result["payload"],
                candidate,
                question=cleaned_question,
                top_k_each=top_k_each,
                summary_k=summary_k,
            )
            for item in parsed_items:
                existing = evidence.get(item.id)
                if existing is not None and existing != item:
                    raise contract_error(
                        f"conflicting evidence for global ID: {item.id}"
                    )
                evidence.update(((item.id, item),))
            selected = [
                {
                    "id": item.id,
                    "preview": item.chunk_text[:200],
                }
                for item in parsed_items
                if item.id in set(result["selected_evidence_ids"])
            ]
            report.append(
                {
                    "paper_id": result["paper_id"],
                    "note": result["note"],
                    "selected_evidence": selected,
                }
            )
        return json.dumps(
            {"papers": report}, ensure_ascii=False
        )

    return collect_evidence_parallel
