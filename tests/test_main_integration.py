"""main startup integration tests for skill loading."""
from __future__ import annotations

import pytest

from paperpilot.builtin_tools.skill_loader import SkillRegistry
from paperpilot.main import SKILLS_DIR, _build_system_prompt, _build_tools


def test_build_system_prompt_includes_skills():
    prompt = _build_system_prompt()
    assert "## 可用 skill" in prompt
    assert "deep-read-paper" in prompt
    assert "explore-citations" in prompt
    assert "find-classics" in prompt
    assert "compare-papers" in prompt
    assert "analyze-figures" in prompt
    assert "write-research-report" in prompt
    assert "PaperPilot" in prompt
    assert "build_index" in prompt


@pytest.mark.slow
def test_build_tools_contains_load_skill_research_todo_and_mcp_tools():
    messages: list[dict] = []
    tools, mcp = _build_tools(messages_ref=messages)
    try:
        names = [tool.name for tool in tools]
        assert "load_skill" in names
        assert "research_todo" in names
        assert "paper_deep_read" in names
        assert "compact_context" in names
        assert any(name.startswith("mcp__") for name in names)
    finally:
        mcp.close()


def test_build_system_prompt_includes_research_todo_nudge():
    prompt = _build_system_prompt()
    assert "## 多步任务规划" in prompt
    assert "research_todo" in prompt


def test_build_system_prompt_includes_paper_deep_read_nudge():
    prompt = _build_system_prompt()
    assert "## Multi-paper deep reading" in prompt
    assert "paper_deep_read" in prompt


def test_build_system_prompt_includes_compact_context_nudge():
    prompt = _build_system_prompt()
    assert "## Long conversation" in prompt
    assert "compact_context" in prompt


def test_build_system_prompt_mentions_search_paper_id_required():
    prompt = _build_system_prompt()
    assert "mcp__colbert__search" in prompt
    assert "paper_id" in prompt


def test_deep_read_skill_mentions_targeted_search_and_short_answer():
    skill = SkillRegistry(SKILLS_DIR).load("deep-read-paper")
    assert "默认至少做 3 次差异化 search" in skill
    assert "Short answer:" in skill
    assert "Evidence:" in skill
    assert "原子事实" in skill
