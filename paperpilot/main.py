"""PaperPilot top-level entrypoint.

CLI:
    python -m paperpilot.main --query "..."
    python -m paperpilot.main --chat

Library:
    from paperpilot.main import run
    run(query, max_iter=8)
"""
from __future__ import annotations

import argparse
from typing import Any

from paperpilot.conversation import (
    MANIFEST_PATH,
    SKILLS_DIR,
    SYSTEM_PROMPT_BASE,
    ConversationSession,
    _build_system_prompt,
    _build_tools,
    _default_logger,
    run,
)
from paperpilot.session_store import SessionStore


def _print_final(messages: list[dict[str, Any]]) -> None:
    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for block in last:
            if hasattr(block, "text"):
                print(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                print(block.get("text", ""))
    else:
        print(last)


def _run_chat(
    max_iter: int,
    *,
    session_name: str | None = None,
    reset_session: bool = False,
) -> None:
    label = f" session={session_name!r}" if session_name else ""
    print(
        "PaperPilot chat"
        f"{label}. 输入 /reset 开始新对话, /compact 压缩上下文, /exit 退出。"
    )
    with ConversationSession(
        max_iter_per_turn=max_iter,
        session_name=session_name,
        reset_session=reset_session,
    ) as session:
        while True:
            try:
                user_text = input("\nPaperPilot> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break

            if not user_text:
                continue
            if user_text in {"/exit", "/quit"}:
                break
            if user_text in {"/reset", "/new"}:
                session.reset()
                print("已重置当前会话。")
                continue
            if user_text == "/compact":
                print(session.compact())
                continue

            messages = session.ask(user_text)
            _print_final(messages)


def _print_sessions() -> None:
    sessions = SessionStore().list_sessions()
    if not sessions:
        print("No saved sessions.")
        return
    for session in sessions:
        print(
            f"{session.name}\tmessages={session.message_count}"
            f"\tupdated={session.updated_at}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="paperpilot")
    parser.add_argument("--query")
    parser.add_argument("--chat", action="store_true")
    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument("--session")
    session_group.add_argument("--new-session")
    parser.add_argument("--list-sessions", action="store_true")
    parser.add_argument("--max-iter", type=int, default=8)
    args = parser.parse_args()

    if args.list_sessions:
        _print_sessions()
        return
    if (args.session or args.new_session) and not args.chat:
        parser.error("--session and --new-session require --chat")
    if args.chat:
        _run_chat(
            args.max_iter,
            session_name=args.session or args.new_session,
            reset_session=bool(args.new_session),
        )
        return
    if not args.query:
        parser.error("--query is required unless --chat is set")

    messages = run(args.query, max_iter=args.max_iter)
    _print_final(messages)


if __name__ == "__main__":
    main()
