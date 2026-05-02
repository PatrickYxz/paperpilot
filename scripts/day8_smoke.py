"""Day 8 smoke: Main Loop -> MCPClient -> graph-mcp -> Semantic Scholar -> LLM.

Requires a valid LLM API key in .env. Uses real Semantic Scholar API and writes
data/graph/citation_graph.pkl through graph-mcp.
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

EXPECT_TOOLS = {
    "mcp__graph__build_graph",
    "mcp__graph__get_shortest_path",
}


def main() -> None:
    saw: set[str] = set()

    def tracer(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            saw.add(payload["name"])
            print(f"  -> {payload['name']}({payload.get('arguments', {})})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            preview = content[:200] if isinstance(content, str) else str(content)[:200]
            print(f"  <- {payload['name']}: {preview}...")
        elif kind == "guardrail_stop":
            print(f"  !! guardrail: {payload['reason']}")

    messages = run(
        "请使用 graph 工具分析 arXiv:2401.00002 的引用关系。"
        "必须先调用 mcp__graph__build_graph 构建 ['2401.00002'] 的引用图，"
        "然后调用 mcp__graph__get_shortest_path 查询 2401.00002 到 2010.11313 "
        "之间是否存在直接或间接引用路径。最后用一句话说明结果。",
        max_iter=6,
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
    assert not missing, f"FAIL: expected graph tools missing: {missing}; saw {saw}"
    assert Path("data/graph/citation_graph.pkl").exists(), "FAIL: graph pickle missing"
    print("\n✅ Day 8 smoke PASSED")


if __name__ == "__main__":
    main()
