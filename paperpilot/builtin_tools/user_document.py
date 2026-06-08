"""Built-in tools for user-pasted documents."""
from __future__ import annotations

import json

from paperpilot.core.adapter import Tool
from paperpilot.document_store import DocumentStore


USER_DOCUMENT_NUDGE = """
## User-pasted paper comparison
When the user uploads a long paper as user_doc_id and asks to compare it with
an arXiv id or paper_id, do not use the phrase "compare similarity" as the
retrieval query. First build a lightweight target-paper profile from title and
abstract: problem, method keywords, architecture terms, datasets, metrics, and
claims. Then call search_user_document with 4-8 focused queries derived from
that profile, such as problem, method, architecture, evaluation, and claims.
Compare only from retrieved evidence chunks and cite chunk_id values.
""".strip()


def search_user_document_tool(store: DocumentStore | None = None) -> Tool:
    document_store = store or DocumentStore()

    def _handler(args: dict) -> str:
        doc_id = str(args["doc_id"])
        query = str(args["query"])
        top_k = int(args.get("top_k", 5))
        hits = document_store.search(doc_id, query, top_k=top_k)
        return json.dumps({
            "doc_id": doc_id,
            "query": query,
            "hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "section_hint": hit.section_hint,
                    "score": hit.score,
                    "text": hit.text,
                }
                for hit in hits
            ],
        }, ensure_ascii=False)

    return Tool(
        name="search_user_document",
        description=(
            "Search a user-pasted paper saved as user_doc_id. Use profile-derived "
            "queries for paper similarity comparison; do not pass generic task "
            "phrases like 'compare similarity'. Returns short evidence chunks."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "User document id, e.g. userdoc-abc123def456.",
                },
                "query": {
                    "type": "string",
                    "description": "Focused retrieval query derived from target paper content.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of evidence chunks to return.",
                    "default": 5,
                },
            },
            "required": ["doc_id", "query"],
            "additionalProperties": False,
        },
        handler=_handler,
    )
