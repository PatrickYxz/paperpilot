"""SkillRegistry, frontmatter parsing, and load_skill_tool tests."""
from __future__ import annotations

import pytest

from paperpilot.builtin_tools.skill_loader import (
    SkillNotFoundError,
    SkillRegistry,
    load_skill_tool,
    render_skill_section,
)


SKILL_OK = """\
---
name: deep-read-paper
description: 深读单篇 arxiv 论文
when_to_use: 用户给定 arxiv id 要求详细讲解
---

# Deep Read Paper

正文内容。
"""

SKILL_OK_2 = """\
---
name: explore-citations
description: 引用拓扑探索
when_to_use: 用户问引用关系
---

正文。
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_registry_scans_skills_dir(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    _write(tmp_path, "explore-citations.md", SKILL_OK_2)
    reg = SkillRegistry(tmp_path)
    metadata = reg.list_metadata()
    assert [m["name"] for m in metadata] == [
        "deep-read-paper",
        "explore-citations",
    ]
    assert metadata[0]["description"] == "深读单篇 arxiv 论文"
    assert metadata[0]["when_to_use"] == "用户给定 arxiv id 要求详细讲解"


def test_frontmatter_missing_delimiter_raises(tmp_path):
    _write(tmp_path, "bad.md", "no frontmatter here\n")
    with pytest.raises(ValueError, match="frontmatter"):
        SkillRegistry(tmp_path)


def test_frontmatter_missing_required_field_raises(tmp_path):
    _write(
        tmp_path,
        "bad.md",
        "---\nname: x\ndescription: y\n---\nbody\n",
    )
    with pytest.raises(ValueError, match="when_to_use"):
        SkillRegistry(tmp_path)


def test_frontmatter_extra_field_ignored(tmp_path):
    _write(
        tmp_path,
        "ok.md",
        (
            "---\nname: x\ndescription: y\nwhen_to_use: z\nextra: ignored\n"
            "---\nbody\n"
        ),
    )
    reg = SkillRegistry(tmp_path)
    metadata = reg.list_metadata()
    assert metadata == [{"name": "x", "description": "y", "when_to_use": "z"}]


def test_load_returns_full_content_with_frontmatter(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    reg = SkillRegistry(tmp_path)
    body = reg.load("deep-read-paper")
    assert body.startswith("---\n")
    assert "Deep Read Paper" in body
    assert "正文内容" in body


def test_load_unknown_skill_raises(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    reg = SkillRegistry(tmp_path)
    with pytest.raises(SkillNotFoundError) as exc:
        reg.load("nonexistent")
    msg = str(exc.value)
    assert "nonexistent" in msg
    assert "deep-read-paper" in msg


def test_duplicate_skill_name_raises(tmp_path):
    _write(tmp_path, "a.md", SKILL_OK)
    _write(tmp_path, "b.md", SKILL_OK)
    with pytest.raises(ValueError, match="duplicate"):
        SkillRegistry(tmp_path)


def test_empty_dir_returns_empty_metadata(tmp_path):
    reg = SkillRegistry(tmp_path)
    assert reg.list_metadata() == []


def test_skills_dir_not_found_raises(tmp_path):
    missing = tmp_path / "no-such-dir"
    with pytest.raises(FileNotFoundError):
        SkillRegistry(missing)


def test_render_skill_section_format():
    metadata = [
        {"name": "a", "description": "desc-a", "when_to_use": "use-a"},
        {"name": "b", "description": "desc-b", "when_to_use": "use-b"},
    ]
    section = render_skill_section(metadata)
    assert "## 可用 skill" in section
    assert "**a**" in section
    assert "desc-a" in section
    assert "use-a" in section
    assert "**b**" in section
    assert "load_skill" in section


def test_render_skill_section_empty():
    assert render_skill_section([]) == ""


def test_load_skill_tool_routes_to_registry(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    reg = SkillRegistry(tmp_path)
    tool = load_skill_tool(reg)
    assert tool.name == "load_skill"
    assert "name" in tool.input_schema["properties"]
    body = tool.handler({"name": "deep-read-paper"})
    assert "Deep Read Paper" in body


def test_load_skill_tool_unknown_name_raises(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    reg = SkillRegistry(tmp_path)
    tool = load_skill_tool(reg)
    with pytest.raises(SkillNotFoundError):
        tool.handler({"name": "ghost"})

