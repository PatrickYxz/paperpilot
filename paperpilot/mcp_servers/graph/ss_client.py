"""Semantic Scholar Graph API client for graph-mcp."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx


class SSAPIError(RuntimeError):
    """Semantic Scholar returned a non-retryable error or malformed response."""


class SSRateLimitError(SSAPIError):
    """Semantic Scholar rate limit persisted after retries."""


class SSClient:
    API_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
    FIELDS = (
        "title,year,authors,externalIds,"
        "references.externalIds,references.title,references.year,references.authors,"
        "citations.externalIds,citations.title,citations.year,citations.authors"
    )

    def __init__(self, timeout: float = 30.0, max_attempts: int = 3):
        self._http = httpx.Client(timeout=timeout)
        self._max_attempts = max_attempts

    def fetch_papers_batch(self, arxiv_ids: list[str]) -> list[dict | None]:
        """Fetch paper metadata by arXiv id, preserving input order."""
        if not arxiv_ids or not all(isinstance(i, str) for i in arxiv_ids):
            raise ValueError("arxiv_ids must be a non-empty list[str]")

        fixture = self._load_fixture(arxiv_ids)
        if fixture is not None:
            return fixture

        payload_ids = [self._format_id(i) for i in arxiv_ids]
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._http.post(
                    self.API_URL,
                    params={"fields": self.FIELDS},
                    json={"ids": payload_ids},
                    headers=headers,
                )
                if response.status_code == 429:
                    raise SSRateLimitError("Semantic Scholar rate limited request")
                if 500 <= response.status_code < 600:
                    raise SSAPIError(f"Semantic Scholar server error: {response.status_code}")
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, list):
                    raise SSAPIError("Semantic Scholar response must be a list")
                return data
            except SSRateLimitError as e:
                last_error = e
                if attempt == self._max_attempts - 1:
                    raise
                self._sleep(attempt)
            except SSAPIError as e:
                last_error = e
                if attempt == self._max_attempts - 1:
                    raise
                self._sleep(attempt)
            except httpx.HTTPStatusError as e:
                raise SSAPIError(str(e)) from e
            except httpx.HTTPError as e:
                last_error = e
                if attempt == self._max_attempts - 1:
                    raise SSAPIError(str(e)) from e
                self._sleep(attempt)

        raise SSAPIError(str(last_error))

    def _load_fixture(self, arxiv_ids: list[str]) -> list[dict | None] | None:
        fixture_dir = os.environ.get("PAPERPILOT_SS_FIXTURE_DIR")
        if not fixture_dir:
            return None

        fixture_path = Path(fixture_dir) / "ss_attention_bert.json"
        with fixture_path.open(encoding="utf-8") as f:
            data = json.load(f)
        by_arxiv_id = {
            ((paper.get("externalIds") or {}).get("ArXiv")): paper
            for paper in data
            if paper is not None
        }
        return [by_arxiv_id.get(arxiv_id) for arxiv_id in arxiv_ids]

    @staticmethod
    def _format_id(arxiv_id: str) -> str:
        return arxiv_id if arxiv_id.upper().startswith("ARXIV:") else f"ARXIV:{arxiv_id}"

    @staticmethod
    def _sleep(attempt: int) -> None:
        time.sleep(float(2 ** (attempt + 1)))
