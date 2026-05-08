"""Day 16 prepare: QASPER to 50 paper x 3 EvalCase JSONL rows.

Usage:
    python scripts/day16_prepare_eval.py

Reads QASPER from data/eval/qasper-source/qasper-train-v0.3.json.
Writes data/eval/qasper_subset.jsonl with up to 50 papers x 3 cases each.
Each candidate paper is dry-run downloaded through the existing arxiv MCP tool.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.eval.qasper_loader import load_qasper_cases
from paperpilot.tools.mcp_client import MCPClient

QASPER_PATH = Path("data/eval/qasper-source/qasper-train-v0.3.json")
OUT_PATH = Path("data/eval/qasper_subset.jsonl")
TARGET_PAPERS = 50
MIN_PAPERS = 40
MANIFEST = Path(__file__).resolve().parent.parent / "paperpilot" / "mcp_servers.json"


def _parse_tool_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"unexpected download_paper result type: {type(result).__name__}")


def _download_tool(mcp: MCPClient):
    for tool in mcp.list_tools():
        if tool.name == "mcp__arxiv__download_paper":
            return tool
    raise RuntimeError("mcp__arxiv__download_paper not found")


def main() -> None:
    if not QASPER_PATH.exists():
        sys.exit(
            f"FAIL: {QASPER_PATH} not found. "
            "Download QASPER per Day 16 plan Step 5.1 first."
        )

    print(f"Loading QASPER from {QASPER_PATH}...")
    all_cases = load_qasper_cases(QASPER_PATH)
    print(f"  {len(all_cases)} candidate cases ({len(all_cases) // 3} papers)")

    papers_in_order: list[str] = []
    cases_by_paper: dict[str, list] = {}
    for case in all_cases:
        if case.arxiv_id not in cases_by_paper:
            papers_in_order.append(case.arxiv_id)
            cases_by_paper[case.arxiv_id] = []
        cases_by_paper[case.arxiv_id].append(case)

    print("\nDry-running mcp__arxiv__download_paper on each paper...")
    mcp = MCPClient(MANIFEST)
    mcp.start()
    accepted_papers: list[str] = []
    rejected: list[tuple[str, str]] = []
    try:
        tool = _download_tool(mcp)
        for arxiv_id in papers_in_order:
            if len(accepted_papers) >= TARGET_PAPERS:
                break
            try:
                result = _parse_tool_result(tool.handler({"arxiv_id": arxiv_id}))
                text = result.get("text")
                if not isinstance(text, str) or len(text) < 500:
                    rejected.append((arxiv_id, "text too short or missing"))
                    continue
                accepted_papers.append(arxiv_id)
                print(
                    f"  [{len(accepted_papers):>2}/{TARGET_PAPERS}] "
                    f"OK arxiv:{arxiv_id} ({len(text)} chars)"
                )
            except Exception as e:  # noqa: BLE001
                rejected.append((arxiv_id, f"{type(e).__name__}: {e}"))
                print(f"  -- skip arxiv:{arxiv_id}: {type(e).__name__}")
    finally:
        mcp.close()

    print(f"\nAccepted: {len(accepted_papers)}, Rejected: {len(rejected)}")
    if len(accepted_papers) < MIN_PAPERS:
        sys.exit(
            f"FAIL: only {len(accepted_papers)} papers accepted "
            f"(need >= {MIN_PAPERS}). Check arxiv quota or try later."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for arxiv_id in accepted_papers:
            for case in cases_by_paper[arxiv_id]:
                row = asdict(case)
                row["oracle_spans"] = list(row["oracle_spans"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(accepted_papers) * 3
    print(f"\nWrote {OUT_PATH} with {len(accepted_papers)} papers x 3 = {total} cases")


if __name__ == "__main__":
    main()
