"""Day 7 复现: FastMCP `list[dict]` 序列化对 LLM 的实际影响。

设计 spec: docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md

跑法: python scripts/day7_fastmcp_repro.py 2> repro.stderr
脚本本身往 stdout 打人类摘要 + verdict; mcp_client 的 [REPRO] 块走 stderr。
分析时 stderr 里会有 ≥1 个 [REPRO] 块, Task 2 用 grep 取出。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

# arxiv id 正则: 2401.12345 / 2401.12345v2 等
ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
COMPLAINT_RE = re.compile(
    r"(?i)(invalid json|malformed json|json (parse|format) error|"
    r"unable to parse|extra data|cannot parse)"
)


def main() -> None:
    tool_result_paper_ids: set[str] = set()
    answer_text_chunks: list[str] = []

    def tracer(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            print(f"  → {payload['name']}({list(payload.get('arguments', {}).keys())})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            if payload["name"] == "mcp__colbert__search" and isinstance(content, str):
                # 把 search 工具结果里所有 paper_id 抓出来
                for m in ARXIV_ID_RE.finditer(content):
                    tool_result_paper_ids.add(m.group(0))
            preview = content[:120] if isinstance(content, str) else str(content)[:120]
            print(f"  ← {payload['name']}: {preview}...")
        elif kind == "turn":
            txt = payload.get("text") or ""
            if txt:
                answer_text_chunks.append(txt)

    messages = run(
        "搜 3 篇 multi-head attention transformer 相关的 arxiv 论文(最近一两年),"
        "全部下载全文,然后用 colbert.search 在这些论文里查 "
        "'How does multi-head attention compute the output?',"
        "用一段话回答我。回答必须基于 colbert.search 返回的具体段落,"
        "并在答案里**显式标注每段引用的 paper arxiv id**(如 2401.12345)。",
        max_iter=12,
        on_event=tracer,
    )

    full_answer = "\n".join(answer_text_chunks)
    print("\n=== FINAL ANSWER ===")
    print(full_answer)

    # 信号分析
    cited_ids = set(ARXIV_ID_RE.findall(full_answer))
    c1_unknown = "(C1 走 stderr 分析, 见下文)"
    c2 = bool(tool_result_paper_ids) and cited_ids >= tool_result_paper_ids
    c3 = COMPLAINT_RE.search(full_answer) is None

    print("\n=== 三信号 ===")
    print(f"  tool_result paper_ids: {sorted(tool_result_paper_ids)}")
    print(f"  answer cited paper_ids: {sorted(cited_ids)}")
    print(f"  C1 (结构, 多 JSON 对象拼接): {c1_unknown}")
    print(f"  C2 (LLM 引用全 vs 缺失):   {'PASS' if c2 else 'FAIL'}")
    print(f"  C3 (LLM 不抱怨 JSON):       {'PASS' if c3 else 'FAIL'}")

    if c2 and c3:
        print("\n🟢 NO FIX NEEDED — LLM 实际未受影响 (但需在 spec 记 baseline)")
    else:
        print("\n🔴 FIX REQUIRED — LLM 表现失常, 触发 §3 fix")

    # exit 0 即可,不靠 exit code 判定 (留给人 + spec 决策记录)


if __name__ == "__main__":
    main()
