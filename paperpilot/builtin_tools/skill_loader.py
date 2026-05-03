"""SkillRegistry and load_skill built-in tool.

Skills live in ``paperpilot/skills/*.md`` with a small YAML-like frontmatter
block containing name, description, and when_to_use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from paperpilot.core.adapter import Tool

REQUIRED_FIELDS = ("name", "description", "when_to_use")


class SkillNotFoundError(LookupError):
    """Raised when load_skill receives a name that is not registered."""


class SkillRegistry:
    def __init__(self, skills_dir: Path):
        if not skills_dir.exists():
            raise FileNotFoundError(f"skills dir not found: {skills_dir}")
        self._dir = skills_dir
        self._skills: dict[str, dict[str, Any]] = {}
        self._scan()

    def list_metadata(self) -> list[dict[str, str]]:
        return [
            {
                "name": skill["name"],
                "description": skill["description"],
                "when_to_use": skill["when_to_use"],
            }
            for skill in sorted(
                self._skills.values(),
                key=lambda item: item["name"],
            )
        ]

    def load(self, name: str) -> str:
        skill = self._skills.get(name)
        if skill is None:
            available = sorted(self._skills.keys())
            raise SkillNotFoundError(
                f"skill '{name}' not found; available: {available}"
            )
        return skill["path"].read_text(encoding="utf-8")

    def _scan(self) -> None:
        for path in sorted(self._dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            meta = _parse_frontmatter(text, path)
            name = meta["name"]
            if name in self._skills:
                first = self._skills[name]["path"]
                raise ValueError(
                    f"duplicate skill name: {name} in {first} and {path}"
                )
            self._skills[name] = {**meta, "path": path}


def _parse_frontmatter(text: str, path: Path) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(
            f"skill {path} frontmatter invalid: missing leading '---' delimiter"
        )

    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        raise ValueError(
            f"skill {path} frontmatter invalid: missing closing '---' delimiter"
        )

    meta: dict[str, str] = {}
    for raw in lines[1:end]:
        if not raw.strip():
            continue
        if ":" not in raw:
            raise ValueError(
                f"skill {path} frontmatter invalid: line not 'key: value': {raw!r}"
            )
        key, _, value = raw.partition(":")
        meta[key.strip()] = value.strip()

    for field in REQUIRED_FIELDS:
        if field not in meta or not meta[field]:
            raise ValueError(
                f"skill {path} frontmatter missing required field: {field}"
            )

    return {field: meta[field] for field in REQUIRED_FIELDS}


def render_skill_section(metadata: list[dict[str, str]]) -> str:
    if not metadata:
        return ""

    lines = ["", "## 可用 skill (按需调 load_skill 加载完整步骤)", ""]
    for item in metadata:
        lines.append(f"- **{item['name']}**: {item['description']}")
        lines.append(f"  适用: {item['when_to_use']}")
    lines.append("")
    lines.append(
        '看到适合场景时调 load_skill(name="...") 拿完整步骤,再按步骤调 mcp tool。'
    )
    return "\n".join(lines)


def load_skill_tool(registry: SkillRegistry) -> Tool:
    def _handler(args: dict[str, Any]) -> str:
        return registry.load(args["name"])

    return Tool(
        name="load_skill",
        description=(
            "加载一个 skill 的完整说明书。skill 列表见 system prompt 的"
            " '可用 skill' 段。tool_result 是该 skill 的 markdown 全文;"
            "你照着步骤调底层 mcp tool 即可。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "skill name, 例如 deep-read-paper",
                },
            },
            "required": ["name"],
        },
        handler=_handler,
    )

