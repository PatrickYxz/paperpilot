"""Main agent loop。

参照 learn-claude-code s01 的 while 循环骨架:
  - LLMClient 调 DeepSeek 的 Anthropic 兼容端点
  - guardrail 做 max_iter / token budget / 重复调用保护
  - tools 显式传入,通过 name 注册表分派到 handler
  - on_event 回调暴露 loop 内部状态,后续 demo / 评估可用
"""
from __future__ import annotations

import json
from typing import Callable

from paperpilot.core.adapter import LLMClient, Tool, ToolResult
from paperpilot.core.guardrail import Guardrail, GuardrailStop

EventCallback = Callable[[str, dict], None]


def agent_loop(
    messages: list[dict],
    *,
    system: str,
    tools: list[Tool],
    client: LLMClient,
    guardrail: Guardrail | None = None,
    on_event: EventCallback | None = None,
) -> list[dict]:
    """跑 agent loop,就地修改并返回 messages。"""
    guard = guardrail or Guardrail()
    tool_by_name = {t.name: t for t in tools}
    emit = on_event or (lambda _k, _v: None)
    downloaded_documents: list[dict] = []

    while True:
        if guard.should_stop():
            emit("guardrail_stop", {"reason": guard.stop_reason()})
            break

        response = client.call(messages, tools, system=system)
        guard.record_turn(response.usage)
        client.append_assistant_turn(messages, response)
        emit("turn", {
            "text": response.text,
            "tool_calls": [tc.name for tc in response.tool_calls],
            "usage": response.usage,
        })

        if not response.tool_calls:
            break  # 模型自然停止

        results: list[ToolResult] = []
        stopped = False
        for tc in response.tool_calls:
            spec = tool_by_name.get(tc.name)
            if spec is None:
                content, is_error = f"Error: tool '{tc.name}' not found", True
            else:
                if _should_repair_build_index_args(tc, downloaded_documents):
                    original = dict(tc.arguments)
                    tc.arguments = {
                        **tc.arguments,
                        "documents": list(downloaded_documents),
                    }
                    emit("tool_arg_repair", {
                        "name": tc.name,
                        "original": original,
                        "repaired": {
                            "documents_count": len(downloaded_documents),
                            "paper_ids": [
                                d.get("paper_id") for d in downloaded_documents
                            ],
                        },
                    })
                try:
                    guard.check_repeated_call(tc)
                except GuardrailStop as e:
                    emit("guardrail_stop", {"reason": str(e)})
                    stopped = True
                    break

                emit("tool_call", {"name": tc.name, "arguments": tc.arguments})
                try:
                    content = str(spec.handler(tc.arguments))
                    is_error = False
                    doc = _extract_downloaded_document(tc.name, content)
                    if doc is not None:
                        downloaded_documents = [doc]
                except Exception as e:
                    content = f"Error: {type(e).__name__}: {e}"
                    is_error = True
                emit("tool_result", {"name": tc.name, "content": content[:200]})

            results.append(ToolResult(id=tc.id, content=content, is_error=is_error))

        if results:
            client.append_tool_results(messages, results)
        if stopped:
            break

    return messages


def _should_repair_build_index_args(
    tc: ToolCall, downloaded_documents: list[dict]
) -> bool:
    if not tc.name.endswith("__build_index") or not downloaded_documents:
        return False
    documents = tc.arguments.get("documents")
    return not documents


def _extract_downloaded_document(tool_name: str, content: str) -> dict | None:
    if not tool_name.endswith("__download_paper"):
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if (
        isinstance(data, dict)
        and isinstance(data.get("paper_id"), str)
        and isinstance(data.get("text"), str)
    ):
        return {"paper_id": data["paper_id"], "text": data["text"]}
    return None
