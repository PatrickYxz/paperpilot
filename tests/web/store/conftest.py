"""Shared Store-test fixture factories and test-only builders."""
from __future__ import annotations

import pytest

from paperpilot.papers import PaperCandidate


@pytest.fixture
def paper_factory():
    """Build isolated PaperCandidate values for Store tests that need one."""
    def build(**overrides: object) -> PaperCandidate:
        values: dict[str, object] = {
            "external_id": "2401.12345v2",
            "title": "A Test Paper",
            "authors": ["Ada Lovelace"],
            "abstract": "abstract",
            "source_url": "https://arxiv.org/abs/2401.12345v2",
        }
        values.update(overrides)
        return PaperCandidate(**values)

    return build
