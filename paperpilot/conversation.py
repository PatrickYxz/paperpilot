"""Conversation-level orchestration for PaperPilot."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol

from dotenv import load_dotenv

from paperpilot.bulk_input import BulkPaperInputDetector
from paperpilot.builtin_tools.ask_user import ASK_USER_NUDGE, ask_user_tool
from paperpilot.builtin_tools.compact import (
    COMPACT_CONTEXT_NUDGE,
    compact_context_tool,
)
from paperpilot.builtin_tools.research_todo import (
    RESEARCH_TODO_NUDGE,
    TodoStore,
    research_todo_tool,
)
from paperpilot.builtin_tools.subagent import (
    PAPER_DEEP_READ_NUDGE,
    paper_deep_read_tool,
)
from paperpilot.builtin_tools.user_document import (
    USER_DOCUMENT_NUDGE,
    search_user_document_tool,
)
from paperpilot.builtin_tools.skill_loader import (
    SkillRegistry,
    load_skill_tool,
    render_skill_section,
)
from paperpilot.core import Guardrail, LLMClient, agent_loop
from paperpilot.core.adapter import Tool
from paperpilot.core.loop import EventCallback
from paperpilot.document_store import DocumentStore
from paperpilot.session_store import SessionStore
from paperpilot.tools.mcp_client import MCPClient

MANIFEST_PATH = Path(__file__).parent / "mcp_servers.json"
SKILLS_DIR = Path(__file__).parent / "skills"

SYSTEM_PROMPT_BASE = """你是 PaperPilot,一个学术论文研究助手。
工作原则:
- 有 tool 可用时优先调 tool;不要自己编造论文标题、作者或 arxiv id
- 一次只解决用户问的事,不主动扩展任务范围
- tool 报错时,根据错误信息决定:重试(换参数 / 换工具) / 告诉用户失败原因
- 调 tool 时必须按 schema 传完整必填参数;如果错误提示缺字段,下一轮必须补齐字段,不要重复同一个空参数
- 调 mcp__colbert__build_index 时,documents 必须是非空列表,每项包含 paper_id 和 text;通常直接使用 mcp__arxiv__download_paper 返回的对象组成 documents=[download_result]
- 调 mcp__colbert__search 时,paper_id 必填,值必须是已经 build_index 过的同一个 paper_id
""".strip()


class Closable(Protocol):
    def close(self) -> None: ...


ToolBuilder = Callable[
    [
        SkillRegistry | None,
        TodoStore | None,
        list[dict] | None,
        Callable[[str], str] | None,
        EventCallback | None,
    ],
    tuple[list[Tool], Closable],
]


def _default_logger(kind: str, payload: dict) -> None:
    if kind == "tool_call":
        print(f"  -> {payload['name']}({payload['arguments']})")
    elif kind == "tool_result":
        print(f"  <- {payload['name']}: {payload['content'][:120]}...")
    elif kind == "guardrail_stop":
        print(f"  !! guardrail: {payload['reason']}")


def _default_input_provider(question: str) -> str:
    return input(f"\n? {question}\n> ")


def _build_system_prompt(registry: SkillRegistry | None = None) -> str:
    registry = registry or SkillRegistry(SKILLS_DIR)
    return (
        SYSTEM_PROMPT_BASE
        + render_skill_section(registry.list_metadata())
        + "\n\n"
        + RESEARCH_TODO_NUDGE
        + "\n\n"
        + PAPER_DEEP_READ_NUDGE
        + "\n\n"
        + COMPACT_CONTEXT_NUDGE
        + "\n\n"
        + USER_DOCUMENT_NUDGE
        + "\n\n"
        + ASK_USER_NUDGE
    )


def _build_tools(
    registry: SkillRegistry | None = None,
    todo_store: TodoStore | None = None,
    messages_ref: list[dict] | None = None,
    input_provider: Callable[[str], str] | None = None,
    on_event: EventCallback | None = None,
) -> tuple[list[Tool], MCPClient]:
    """Return (tools, mcp_client); caller is responsible for close()."""
    registry = registry or SkillRegistry(SKILLS_DIR)
    todo_store = todo_store or TodoStore()
    messages_ref = messages_ref if messages_ref is not None else []
    ask_input = input_provider or _default_input_provider
    emit = on_event or _default_logger

    mcp = MCPClient(MANIFEST_PATH)
    try:
        mcp.start()
        mcp_tools = mcp.list_tools()
        tools: list[Tool] = [
            load_skill_tool(registry),
            research_todo_tool(todo_store),
            paper_deep_read_tool(
                client_factory=lambda: LLMClient(),
                mcp_tools=mcp_tools,
                on_event=emit,
            ),
            compact_context_tool(
                messages_ref=messages_ref,
                client_factory=lambda: LLMClient(),
                on_event=emit,
            ),
            search_user_document_tool(),
            ask_user_tool(
                input_provider=ask_input,
                on_event=emit,
            ),
            *mcp_tools,
        ]
        return tools, mcp
    except Exception:
        mcp.close()
        raise


class ConversationSession:
    """A long-lived in-process PaperPilot conversation."""

    def __init__(
        self,
        *,
        max_iter_per_turn: int = 8,
        input_provider: Callable[[str], str] | None = None,
        on_event: EventCallback | None = None,
        client_factory: Callable[[], LLMClient] | None = None,
        tool_builder: ToolBuilder = _build_tools,
        session_name: str | None = None,
        session_store: SessionStore | None = None,
        document_store: DocumentStore | None = None,
        bulk_input_detector: BulkPaperInputDetector | None = None,
        reset_session: bool = False,
    ) -> None:
        self.max_iter_per_turn = max_iter_per_turn
        self.input_provider = input_provider or _default_input_provider
        self.on_event = on_event or _default_logger
        self.client_factory = client_factory or LLMClient
        self.tool_builder = tool_builder
        self.session_name = session_name
        self.session_store = session_store or SessionStore()
        self.document_store = document_store or DocumentStore()
        self.bulk_input_detector = bulk_input_detector or BulkPaperInputDetector()
        self.reset_session = reset_session

        self.registry: SkillRegistry | None = None
        self.todo_store: TodoStore | None = None
        self.messages: list[dict] = []
        self.tools: list[Tool] = []
        self.mcp: Closable | None = None
        self.system_prompt: str | None = None
        self._started = False

    def __enter__(self) -> "ConversationSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        """Start tools and MCP resources for this session."""
        if self._started:
            return

        load_dotenv()
        self.registry = SkillRegistry(SKILLS_DIR)
        self.todo_store = TodoStore()
        self.messages = self._load_initial_messages()
        self.system_prompt = _build_system_prompt(self.registry)
        self.tools, self.mcp = self.tool_builder(
            self.registry,
            self.todo_store,
            self.messages,
            self.input_provider,
            self.on_event,
        )
        self._bind_user_document_tool()
        self._started = True

    def ask(self, user_text: str) -> list[dict]:
        """Append one user turn and run the agent until it stops."""
        if not self._started:
            self.start()
        if self.system_prompt is None:
            raise RuntimeError("ConversationSession is not started")

        user_message = self._prepare_user_message(user_text)
        self.messages.append({"role": "user", "content": user_message})
        messages = agent_loop(
            self.messages,
            system=self.system_prompt,
            tools=self.tools,
            client=self.client_factory(),
            guardrail=Guardrail(
                max_iterations=self.max_iter_per_turn,
                budget_tokens=int(os.environ.get("BUDGET_TOKENS", 50_000)),
            ),
            on_event=self.on_event,
        )
        self._save_if_named()
        return messages

    def compact(self) -> str:
        """Manually compact the current message history."""
        if not self._started:
            self.start()
        for tool in self.tools:
            if tool.name == "compact_context":
                result = str(tool.handler({}))
                self._save_if_named()
                return result
        raise RuntimeError("compact_context tool is not available")

    def reset(self) -> None:
        """Start a new clean conversation in the same process."""
        self.close()
        if self.session_name is not None:
            self.session_store.reset(self.session_name)
            self.reset_session = False
        self.start()

    def close(self) -> None:
        """Close MCP resources for this session."""
        if self.mcp is not None:
            self.mcp.close()
        self.mcp = None
        self.tools = []
        self.system_prompt = None
        self._started = False

    def _load_initial_messages(self) -> list[dict]:
        if self.session_name is None:
            return []
        if self.reset_session:
            self.session_store.reset(self.session_name)
            self.reset_session = False
            return []
        return self.session_store.load(self.session_name)

    def _save_if_named(self) -> None:
        if self.session_name is not None:
            self.session_store.save(self.session_name, self.messages)

    def _bind_user_document_tool(self) -> None:
        replacement = search_user_document_tool(self.document_store)
        for index, tool in enumerate(self.tools):
            if tool.name == replacement.name:
                self.tools[index] = replacement
                return
        self.tools.append(replacement)

    def _prepare_user_message(self, user_text: str) -> str:
        bulk = self.bulk_input_detector.detect(user_text)
        if bulk is None:
            return user_text
        stored = self.document_store.save_user_paste(
            bulk.document_text,
            target_paper_id=bulk.target_paper_id,
        )
        self.on_event("bulk_input_saved", {
            "doc_id": stored.doc_id,
            "target_paper_id": bulk.target_paper_id,
            "char_count": stored.metadata["char_count"],
            "approx_tokens": stored.metadata["approx_tokens"],
        })
        return (
            "用户上传了一篇超长论文全文，已保存为本地文档。\n\n"
            f"user_doc_id: {stored.doc_id}\n"
            "source: user_paste\n"
            f"approx_tokens: {stored.metadata['approx_tokens']}\n"
            f"target_paper_id: {bulk.target_paper_id}\n\n"
            "任务：比较 user_doc_id 与 target_paper_id 的主题、方法、实验和结论"
            "相似度。不要要求用户重新粘贴全文。先根据目标论文 title/abstract "
            "生成内容画像和多组检索 query，再调用 search_user_document 获取"
            "用户论文证据 chunk，并在最终回答中引用 chunk_id。"
        )


def run(query: str, *, max_iter: int = 8, on_event=None) -> list[dict]:
    """Run one complete agent conversation and return final messages."""
    with ConversationSession(max_iter_per_turn=max_iter, on_event=on_event) as session:
        return session.ask(query)
