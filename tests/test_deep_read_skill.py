from pathlib import Path


def test_deep_read_skill_requires_planned_retrieval_first() -> None:
    text = Path("paperpilot/skills/deep-read-paper.md").read_text(encoding="utf-8")

    assert "mcp__colbert__planned_retrieval" in text
    assert "planned_retrieval" in text
    assert "可选补充检索" in text
    assert "mcp__colbert__search" in text
