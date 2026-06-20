"""main startup integration tests for skill loading."""
from __future__ import annotations

import pytest

from paperpilot.builtin_tools.skill_loader import SkillRegistry
from paperpilot.main import SKILLS_DIR, _build_system_prompt, _build_tools


def test_build_system_prompt_includes_skills():
    prompt = _build_system_prompt()
    assert "skill" in prompt
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
        assert "search_user_document" in names
        assert "ask_user" in names
        assert any(name.startswith("mcp__") for name in names)
    finally:
        mcp.close()


def test_build_system_prompt_includes_research_todo_nudge():
    prompt = _build_system_prompt()
    assert "research_todo" in prompt


def test_build_system_prompt_includes_paper_deep_read_nudge():
    prompt = _build_system_prompt()
    assert "## Multi-paper deep reading" in prompt
    assert "paper_deep_read" in prompt


def test_build_system_prompt_includes_compact_context_nudge():
    prompt = _build_system_prompt()
    assert "## Long conversation" in prompt
    assert "compact_context" in prompt


def test_build_system_prompt_includes_ask_user_nudge():
    prompt = _build_system_prompt()
    assert "## Ask user" in prompt
    assert "ask_user" in prompt
    assert "required information is missing" in prompt


def test_build_system_prompt_includes_user_document_nudge():
    prompt = _build_system_prompt()
    assert "## User-pasted paper comparison" in prompt
    assert "search_user_document" in prompt
    assert "target-paper profile" in prompt


def test_build_system_prompt_mentions_search_paper_id_required():
    prompt = _build_system_prompt()
    assert "mcp__colbert__search" in prompt
    assert "paper_id" in prompt


def test_deep_read_skill_mentions_planned_retrieval_and_short_answer():
    skill = SkillRegistry(SKILLS_DIR).load("deep-read-paper")
    assert "mcp__colbert__planned_retrieval" in skill
    assert "可选补充检索" in skill
    assert "Short answer:" in skill
    assert "Evidence:" in skill
    assert "Final Answer Contract" in skill
    assert "未被问题询问的方法、数据集、指标、baseline" in skill
    assert "关键数字或实体必须也出现在 `Evidence:`" in skill
    assert "不要在最终回答里输出 `Step 4`" in skill
    assert "Answer span candidates" in skill
