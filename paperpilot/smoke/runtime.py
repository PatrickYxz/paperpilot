"""Isolated real-instance construction for smoke runs.

Points every persisted path at a throwaway directory via already-declared
environment variables only (the ``.env.example`` set is exact-matched by an
architecture gate; no new variables may be introduced). Paper metadata from
the scenario file is optionally injected through a stub ``PaperSearch`` —
mirroring the 9-17 smoke pattern where only conversation-creation metadata
was injected while Research/MCP/LLM stayed fully real.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ISOLATED_PATH_ENV: dict[str, str] = {
    "PAPERPILOT_TASK_DB_PATH": "tasks.sqlite3",
    "PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH": "checkpoints.sqlite3",
    "PAPERPILOT_CONTEXT_ARTIFACT_ROOT": "context-artifacts",
    "PAPERPILOT_GRAPH_PATH": "citation_graph.pkl",
}


def build_isolated_app(
    base_dir: Path,
    *,
    context_management: bool,
    model: str | None,
    metadata_by_external_id: dict[str, dict[str, Any]],
) -> Any:
    """Build a real web app on an isolated directory; return a TestClient."""
    for var, rel in ISOLATED_PATH_ENV.items():
        os.environ[var] = str(base_dir / rel)
    os.environ["PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED"] = (
        "true" if context_management else "false"
    )
    os.environ["PAPERPILOT_FULL_COMPACTION_ENABLED"] = "false"
    if model:
        os.environ["PAPERPILOT_RESEARCH_MODEL_NAME"] = model

    from fastapi.testclient import TestClient

    from paperpilot.papers import PaperCandidate, normalize_arxiv_id
    from paperpilot.web.app import create_app

    paper_search = None
    if metadata_by_external_id:

        def paper_search(query: str, k: int) -> list[PaperCandidate]:  # noqa: ARG001
            normalized = normalize_arxiv_id(query)
            for known_id, metadata in metadata_by_external_id.items():
                if normalize_arxiv_id(known_id) == normalized:
                    return [PaperCandidate(**metadata)]
            return []

    app = create_app(paper_search=paper_search)
    return TestClient(app)
