"""Day 6 冒烟: Main Loop ←→ MCPClient ←→ stdio ←→ {arxiv-mcp, colbert-mcp} ←→ LLM
全链路验证。

跑一次 ~$0.02-0.05(LLM 多轮 tool_use); 需联网 + 有效 LLM API key。
首次跑会触发 ColBERT 模型加载(预热过的话 ~10s)。
用法: python scripts/day6_smoke.py
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

EXPECT_TOOLS = {
    "mcp__arxiv__search_papers",
    "mcp__arxiv__download_paper",
    "mcp__colbert__build_index",
    "mcp__colbert__search",
}


def main() -> None:
    saw: set[str] = set()

    def tracer(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            saw.add(payload["name"])
            print(f"  → {payload['name']}({_preview_args(payload.get('arguments', {}))})")
        elif kind == "tool_arg_repair":
            print(f"  ↻ repaired {payload['name']}: {payload['repaired']}")
        elif kind == "tool_result":
            content = payload.get("content", "")
            preview = content[:160] if isinstance(content, str) else str(content)[:160]
            print(f"  ← {payload['name']}: {preview}...")
        elif kind == "guardrail_stop":
            print(f"  ⚠ guardrail: {payload['reason']}")

    messages = run(
        "请按固定流程验证全文检索链路。"
        "第一步必须调用 mcp__arxiv__search_papers,参数用 query='1706.03762', max_results=3。"
        "如果 search 返回 429 或结果不理想,不要继续换 query 搜索,直接进入下一步。"
        "第二步必须调用 mcp__arxiv__download_paper,参数 arxiv_id='1706.03762'。"
        "第三步必须调用 mcp__colbert__build_index,documents 参数必须是一个非空列表,"
        "其中唯一元素就是 download_paper 返回的对象,形如 documents=[{'paper_id': ..., 'text': ...}],"
        "绝不能传空对象 {}。"
        "第四步必须调用 mcp__colbert__search,查询 'multi-head attention definition', top_k=3。"
        "最后用一段话回答 multi-head attention 是怎么定义的,回答必须基于 colbert.search 返回的具体段落。",
        max_iter=8,
        on_event=tracer,
    )

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for b in last:
            if hasattr(b, "text"):
                print(b.text)
    else:
        print(last)

    missing = EXPECT_TOOLS - saw
    assert not missing, f"FAIL: 期望调用的 tool 缺失 {missing};实际只见 {saw}"
    print("\n✅ Day 6 smoke PASSED")


def _preview_args(args: dict) -> dict:
    if "documents" not in args:
        return args
    docs = args.get("documents") or []
    preview_docs = []
    for doc in docs:
        if isinstance(doc, dict):
            preview_docs.append({
                "paper_id": doc.get("paper_id"),
                "text_len": len(doc.get("text", "")),
            })
    return {**args, "documents": preview_docs}


if __name__ == "__main__":
    main()
