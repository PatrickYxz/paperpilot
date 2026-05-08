"""paper_deep_read built-in tool.

Each paper is read by an independent agent loop with isolated context. Workers
run in parallel via ThreadPoolExecutor; ColBERT MCP gives each paper its own
index so concurrent build/search calls do not race.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from paperpilot.core.adapter import LLMClient, Tool
from paperpilot.core.guardrail import Guardrail
from paperpilot.core.loop import EventCallback, agent_loop

MAX_PAPERS = 8
SUBAGENT_MAX_ITER = 8
SUBAGENT_BUDGET_TOKENS = 120_000
THREAD_POOL_SIZE = 3

SUBAGENT_TOOL_NAMES = {
    "mcp__arxiv__download_paper",
    "mcp__colbert__build_index",
    "mcp__colbert__search",
}

PAPER_DEEP_READ_NUDGE = """
## Multi-paper deep reading
When the user needs comparison or synthesis across 3-8 papers that are worth
reading in detail, call paper_deep_read(paper_ids=[...], user_query="...").
Each paper is handled by an isolated subagent and returns a markdown summary.
Do not use it for a single paper, fewer than 3 papers, or cases where abstracts
are enough.
""".strip()

SUBAGENT_SYSTEM = """
You are a paper deep-reading subagent. Focus on exactly one paper and answer
the user's query through detailed reading.

Recommended workflow:
1. Call mcp__arxiv__download_paper(arxiv_id="<paper_id>").
2. Call mcp__colbert__build_index(documents=[download_result]).
3. Call mcp__colbert__search(query="...", paper_id="<paper_id>", top_k=5)
   at least 3 times with different targeted queries before final synthesis,
   unless a tool error or the iteration limit prevents it.
   - Search 1: the user's direct question.
   - Search 2: key terms, synonyms, abbreviations, datasets, metrics,
     baselines, or method names from the question.
   - Search 3: likely evidence locations such as method, experiment setup,
     table, appendix, evaluation, or ablation sections.
   If results are off-topic, rewrite the query and search again.
4. Write the final answer as markdown text, starting with a short answer.

Final markdown format, around 500 tokens:
## Answer Span Candidates
<1-3 exact short phrases copied from retrieved passages: numbers, ranges,
dataset names, method names, metrics, or key finding sentences. Preserve
English wording when the query is in English or asks for a paper fact.>

## Short Answer
<one sentence with the most direct answer. It must include at least one phrase
from Answer Span Candidates verbatim; if evidence is missing, say so explicitly>

## Evidence
<compact bullets grounded in retrieved passages>

## Core Method
<method summary>

## Key Findings
<important experiments, evidence, and conclusions>

## Relevance to Query
<how this paper helps answer the user's query>

Constraints:
- Work only on the assigned paper. Do not download or search other paper IDs.
- Only the three listed MCP tools are available.
- Do not move to the final answer after only 1-2 searches if more iterations
  are available.
- For dataset, number, method-name, baseline, or metric questions, put the
  exact atomic fact in Short Answer before explanation.
- For quantity/range questions, list all relevant candidate numbers or ranges
  found in evidence before choosing the direct answer.
- For patterns/findings/observations questions, copy the paper's compact
  finding sentence when available before paraphrasing it.
- Near the iteration limit, stop tool use and write the best available summary.
- Do not return JSON. Put the summary in the final assistant text.
""".strip()


def _filter_subagent_tools(mcp_tools: list[Tool]) -> list[Tool]:
    """Return the exact MCP tools allowed inside a deep-read subagent."""
    return [tool for tool in mcp_tools if tool.name in SUBAGENT_TOOL_NAMES]


def _extract_last_text(messages: list[dict]) -> str | None:
    """Extract text from the last assistant turn."""
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content or None
        if isinstance(content, list):
            texts = [
                getattr(block, "text", "")
                for block in content
                if getattr(block, "type", None) == "text"
            ]
            joined = "\n".join(text for text in texts if text)
            return joined or None
    return None


def _run_one(
    paper_id: str,
    user_query: str,
    *,
    client_factory: Callable[[], LLMClient],
    tools: list[Tool],
    on_event: EventCallback,
    max_iter: int = SUBAGENT_MAX_ITER,
    budget_tokens: int = SUBAGENT_BUDGET_TOKENS,
) -> dict[str, str]:
    """Run one isolated subagent and return a renderable result dict."""
    sub_messages: list[dict] = [{
        "role": "user",
        "content": (
            f"Deep-read paper {paper_id}. User query: {user_query}. "
            "Extract the core method, key findings, and relevance to the query."
        ),
    }]
    guard = Guardrail(max_iterations=max_iter, budget_tokens=budget_tokens)
    on_event("subagent_start", {})
    try:
        agent_loop(
            sub_messages,
            system=SUBAGENT_SYSTEM,
            tools=tools,
            client=client_factory(),
            guardrail=guard,
            on_event=on_event,
        )
    except Exception as exc:
        status = f"error: {type(exc).__name__}: {exc}"
        on_event("subagent_done", {"status": status})
        return {
            "paper_id": paper_id,
            "summary": _extract_last_text(sub_messages) or "",
            "status": status,
        }

    summary = _extract_last_text(sub_messages) or ""
    status = "max_iter_reached" if guard.stop_reason() else "ok"
    on_event("subagent_done", {"status": status})
    return {"paper_id": paper_id, "summary": summary, "status": status}


def _render_results(results: list[dict[str, str]]) -> str:
    lines = [f"## Paper Deep Read Results ({len(results)} papers)", ""]
    for result in results:
        lines.append(f"### {result['paper_id']} (status: {result['status']})")
        lines.append(result["summary"] or "_(no summary produced)_")
        lines.append("")
    return "\n".join(lines).rstrip()


def paper_deep_read_tool(
    *,
    client_factory: Callable[[], LLMClient],
    mcp_tools: list[Tool],
    on_event: EventCallback,
) -> Tool:
    """Build the paper_deep_read tool."""
    subagent_tools = _filter_subagent_tools(mcp_tools)

    def _handler(args: dict) -> str:
        paper_ids = args["paper_ids"]
        user_query = args["user_query"]
        if not (1 <= len(paper_ids) <= MAX_PAPERS):
            raise ValueError(
                f"paper_ids count must be 1..{MAX_PAPERS}, got {len(paper_ids)}"
            )

        emit_lock = threading.Lock()

        def make_sub_emit(paper_id: str) -> EventCallback:
            def sub_emit(kind: str, payload: dict) -> None:
                with emit_lock:
                    on_event(kind, {**payload, "subagent_paper_id": paper_id})

            return sub_emit

        results: list[dict[str, str] | None] = [None] * len(paper_ids)
        max_workers = min(THREAD_POOL_SIZE, len(paper_ids))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_one,
                    paper_id,
                    user_query,
                    client_factory=client_factory,
                    tools=subagent_tools,
                    on_event=make_sub_emit(paper_id),
                ): index
                for index, paper_id in enumerate(paper_ids)
            }
            for future in as_completed(futures):
                index = futures[future]
                paper_id = paper_ids[index]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    results[index] = {
                        "paper_id": paper_id,
                        "summary": "",
                        "status": f"error: {type(exc).__name__}: {exc}",
                    }

        return _render_results([result for result in results if result is not None])

    return Tool(
        name="paper_deep_read",
        description=(
            "Deep-read 1-8 papers with isolated subagents and return markdown "
            "summaries for comparison or synthesis. Use for multi-paper "
            "questions where abstracts are not enough."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PAPERS,
                    "items": {"type": "string", "minLength": 1},
                },
                "user_query": {"type": "string", "minLength": 1},
            },
            "required": ["paper_ids", "user_query"],
            "additionalProperties": False,
        },
        handler=_handler,
    )
