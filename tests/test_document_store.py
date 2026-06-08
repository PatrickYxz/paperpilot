"""User document store tests."""
from __future__ import annotations

import json

from paperpilot.builtin_tools.user_document import search_user_document_tool
from paperpilot.document_store import DocumentStore


def test_save_user_paste_uses_stable_doc_id_and_metadata(tmp_path):
    store = DocumentStore(tmp_path)
    text = "Abstract\nRetrieval augmented generation.\n\nIntroduction\nRAG."

    first = store.save_user_paste(text, target_paper_id="2402.13718")
    second = store.save_user_paste(text, target_paper_id="2402.13718")

    assert first.doc_id == second.doc_id
    assert first.path.read_text(encoding="utf-8") == text
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["doc_id"] == first.doc_id
    assert metadata["detected_target"] == "2402.13718"


def test_search_returns_relevant_limited_chunks(tmp_path):
    store = DocumentStore(tmp_path)
    stored = store.save_user_paste(
        "Abstract\nGraph neural retrieval overview.\n\n"
        "Methods\nWe use ColBERT late interaction for passage ranking.\n\n"
        "Experiments\nWe evaluate on BEIR datasets.",
        target_paper_id="2402.13718",
    )

    hits = store.search(stored.doc_id, "ColBERT late interaction ranking", top_k=1)

    assert len(hits) == 1
    assert "ColBERT" in hits[0].text
    assert hits[0].chunk_id.startswith(stored.doc_id)


def test_search_user_document_tool_returns_json_hits(tmp_path):
    store = DocumentStore(tmp_path)
    stored = store.save_user_paste(
        "Abstract\nA transformer model.\n\nMethods\nSparse attention transformer.",
        target_paper_id="2402.13718",
    )
    tool = search_user_document_tool(store)

    payload = json.loads(tool.handler({
        "doc_id": stored.doc_id,
        "query": "sparse attention transformer",
        "top_k": 2,
    }))

    assert payload["doc_id"] == stored.doc_id
    assert payload["query"] == "sparse attention transformer"
    assert payload["hits"]
    assert "chunk_id" in payload["hits"][0]
