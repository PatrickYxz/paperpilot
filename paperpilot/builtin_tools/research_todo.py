"""Research todo built-in tool.

L2 built-in tool, parallel to load_skill. It keeps single-process memory state
with whole-list replacement semantics and validates at most one in_progress item.
"""
from __future__ import annotations

from typing import Any

from paperpilot.core.adapter import Tool


RESEARCH_TODO_NUDGE = """
## 多步任务规划
涉及多步研究 (找论文 -> 检索 -> 综合 / 比较多篇 paper / 跨 server 协同) 时, 先调
research_todo 列计划, 每完成一步把对应项 status 标 completed, 推进 in_progress
到下一项。任务简单 (1-2 步) 时不必用。
""".rstrip()


class TodoStore:
    def __init__(self) -> None:
        self._items: list[dict] = []

    def replace(self, todos: list[dict]) -> None:
        self._items = list(todos)

    def items(self) -> list[dict]:
        return list(self._items)


def render(items: list[dict]) -> str:
    if not items:
        return "## Research Todos (empty)"

    lines = [f"## Research Todos ({len(items)} items)", ""]
    for item in items:
        if item["status"] == "completed":
            mark = "[x]"
            suffix = ""
        elif item["status"] == "in_progress":
            mark = "[->]"
            suffix = " (in progress)"
        else:
            mark = "[ ]"
            suffix = ""
        lines.append(f"- {mark} {item['content']}{suffix}")
    return "\n".join(lines)


def research_todo_tool(store: TodoStore) -> Tool:
    def _handler(args: dict[str, Any]) -> str:
        todos = args["todos"]
        in_progress_count = sum(
            1 for item in todos if item["status"] == "in_progress"
        )
        if in_progress_count > 1:
            raise ValueError(
                f"only one in_progress allowed, got {in_progress_count}; "
                "complete or revert the others first"
            )
        store.replace(todos)
        return render(store.items())

    return Tool(
        name="research_todo",
        description=(
            "维护多步研究任务清单。整表覆写语义: 每次调用必须传完整新 list, "
            "后端会替换之前的 list。同一时刻至多 1 个 in_progress。"
            "适用: 用户问题需要 3+ 步骤 (找论文 -> 构图 -> 求共引 -> 综合) 时, "
            "先用此 tool 列出计划再执行; 每完成一步把对应项 status 改 completed。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "minLength": 1},
                            "status": {
                                "type": "string",
                                "enum": [
                                    "pending",
                                    "in_progress",
                                    "completed",
                                ],
                            },
                        },
                        "required": ["content", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["todos"],
        },
        handler=_handler,
    )
