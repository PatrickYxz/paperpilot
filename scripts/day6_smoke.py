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
            print(f"  → {payload['name']}({payload.get('arguments', {})})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            preview = content[:160] if isinstance(content, str) else str(content)[:160]
            print(f"  ← {payload['name']}: {preview}...")
        elif kind == "guardrail_stop":
            print(f"  ⚠ guardrail: {payload['reason']}")

    messages = run(
        "搜一篇 attention 相关的 arxiv 论文(最近一两年的就行),下载它的全文,"
        "然后在全文里查 multi-head attention 是怎么定义的,用一段话回答我。"
        "回答必须基于 colbert.search 返回的具体段落,不要只看 abstract。",
        max_iter=10,
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


if __name__ == "__main__":
    main()
