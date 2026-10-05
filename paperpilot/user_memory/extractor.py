"""LLM extraction of long-term memory candidates with span verification.

The extractor may only propose; every candidate must carry a support span
that literally appears in this turn's messages, or the pipeline drops it.
"""
from __future__ import annotations

import re
from typing import Literal, Sequence

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

MEMORY_KINDS = ("preference", "fact", "project", "paper_note")

_EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable long-term memories about the user from one "
    "research-assistant conversation turn. Follow three rules: "
    "SELECTIVITY — keep only facts useful in future conversations, never "
    "one-off details of this turn (e.g. which PDF page was read, how many "
    "options a search returned); ABSTRACTION — generalize this turn's "
    "behavior into stable traits ('prefers methods with code released') "
    "instead of replaying events; STRUCTURE — output concise, "
    "self-contained statements resolvable without this conversation. "
    "For every memory, copy a support_span: a verbatim substring from the "
    "conversation that proves it. If nothing durable appears, return an "
    "empty list."
)


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["preference", "fact", "project", "paper_note"]
    content: str = Field(min_length=4, max_length=400)
    support_span: str = Field(min_length=8, max_length=400)


class MemoryExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[MemoryCandidate]


def build_extraction_prompt(*, user_text: str, assistant_text: str) -> str:
    return (
        "Conversation turn:\n\n"
        f"USER:\n{user_text}\n\n"
        f"ASSISTANT:\n{assistant_text}\n\n"
        "Extract durable user memories as JSON per the system rules. "
        "Kinds: preference (long-term likes/dislikes), fact (stable traits: "
        "research fields, affiliations, expertise), project (ongoing work "
        "and its goals), paper_note (the user's settled takeaway about one "
        "specific paper)."
    )


def extract_memory_candidates(
    *,
    user_text: str,
    assistant_text: str,
    model: object,
) -> list[MemoryCandidate]:
    """Run one structured LLM call; any failure yields no candidates."""
    try:
        runnable = model.with_structured_output(MemoryExtractionOutput)
        response = runnable.invoke(
            [
                SystemMessage(content=_EXTRACTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_extraction_prompt(
                        user_text=user_text, assistant_text=assistant_text
                    )
                ),
            ]
        )
        extraction = MemoryExtractionOutput.model_validate(response)
    except Exception:
        return []
    return list(extraction.memories)


def verify_candidates(
    candidates: Sequence[MemoryCandidate],
    texts: Sequence[str],
) -> list[MemoryCandidate]:
    """Keep only candidates whose support span occurs verbatim in the texts."""
    haystack = _normalize(" \n ".join(texts))
    verified: list[MemoryCandidate] = []
    for candidate in candidates:
        if _normalize(candidate.support_span) in haystack:
            verified.append(candidate)
    return verified


def _normalize(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"\s+", " ", lowered).strip()
