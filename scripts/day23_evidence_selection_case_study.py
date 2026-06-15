"""Build a small evidence-selection case-study report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CASE_IDS = [
    "qasper-1910.04601-q1",
    "qasper-1701.00185-q1",
    "qasper-1910.07181-q0",
]

DEFAULT_CALIBRATION_PATH = Path("data/eval/semantic_calibration_candidates_20260614.jsonl")
DEFAULT_OUT_PATH = Path("docs/evidence_selection_case_study_20260615.md")

TRACE_CANDIDATES = {
    "qasper-1910.04601-q1": [
        Path("data/traces/qasper-1910.04601-q1.jsonl"),
        Path("data/traces_day16_before_day18/qasper-1910.04601-q1.jsonl"),
    ],
    "qasper-1701.00185-q1": [
        Path("data/traces/qasper-1701.00185-q1.jsonl"),
        Path("data/traces_day16_before_day18/qasper-1701.00185-q1.jsonl"),
    ],
    "qasper-1910.07181-q0": [
        Path("data/traces/qasper-1910.07181-q0.jsonl"),
        Path("data/traces_day16_before_day18/qasper-1910.07181-q0.jsonl"),
    ],
}

DIAGNOSES = {
    "qasper-1910.04601-q1": {
        "title": "WikiHop vs HotpotQA",
        "layer": "retrieval/evidence-selection miss",
        "finding": (
            "The available latest trace searched generic dataset queries and an explicit "
            "`HotpotQA` query. The truncated retrieved snippets and final answer center on "
            "HotpotQA, while the calibration gold evidence says WikiHop. This means the "
            "answer repair loop can clean the output, but it cannot recover the gold answer "
            "because the selected evidence already points to the wrong dataset."
        ),
        "next": (
            "For dataset/entity questions, evidence selection should require the evidence "
            "sentence to express the relation asked by the question and should not inject a "
            "candidate entity into the search query before verifying alternatives."
        ),
    },
    "qasper-1701.00185-q1": {
        "title": "Overbroad clustering-method list",
        "layer": "scope-control failure after relevant evidence",
        "finding": (
            "The gold evidence itself contains the narrow answer clause followed by broader "
            "context about baseline dimensionality-reduction methods and non-biased neural "
            "networks. PaperPilot promoted the broader neighboring context into the direct "
            "answer, so this is not primarily a retrieval miss."
        ),
        "next": (
            "For list questions, final synthesis needs a scope boundary step: identify the "
            "noun phrase being asked for, then include only items governed by that phrase."
        ),
    },
    "qasper-1910.07181-q0": {
        "title": "Wrong improvement percentages",
        "layer": "numeric/entity verification failure",
        "finding": (
            "The answer selected MRR table values and a percentage statement from retrieved "
            "snippets, but calibration says the central percentages should be 50% and 31%, "
            "not 58% and 37%. This needs exact numeric relation verification against the "
            "gold-like evidence sentence, not more formatting cleanup."
        ),
        "next": (
            "For numeric questions, evidence selection should prefer sentences that directly "
            "answer the comparative relation in the question and should verify that the exact "
            "numbers in the short answer occur in that selected sentence."
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    calibration_rows = _load_calibration(args.calibration_path)
    lines = _render_report(calibration_rows, args.calibration_path)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {args.out_path}")


def _load_calibration(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("case_id") in CASE_IDS:
            rows[str(row["case_id"])] = row
    missing = [case_id for case_id in CASE_IDS if case_id not in rows]
    if missing:
        raise SystemExit(f"Missing calibration rows: {missing}")
    return rows


def _render_report(
    calibration_rows: dict[str, dict[str, Any]],
    calibration_path: Path,
) -> list[str]:
    lines = [
        "# Evidence Selection Case Study",
        "",
        "Date: 2026-06-15",
        "",
        f"Calibration source: `{calibration_path}`",
        "",
        "This report inspects three representative QASPER failures before changing PaperPilot behavior again.",
        "",
        "Important trace limitation: current JSONL traces store truncated tool-result content. The report can inspect search queries and snippet heads, but not the full retrieved top-k ranking.",
        "",
        "## Summary",
        "",
        "| case_id | issue | likely failed layer | recommended next check |",
        "|---|---|---|---|",
    ]
    for case_id in CASE_IDS:
        diag = DIAGNOSES[case_id]
        lines.append(
            f"| `{case_id}` | {diag['title']} | {diag['layer']} | {diag['next']} |"
        )

    for case_id in CASE_IDS:
        row = calibration_rows[case_id]
        diag = DIAGNOSES[case_id]
        trace_path = _select_trace(case_id)
        trace = _parse_trace(trace_path) if trace_path else {"searches": [], "final": ""}
        lines.extend([
            "",
            f"## {case_id}: {diag['title']}",
            "",
            f"- Question: {_one_line(row.get('question'))}",
            f"- Review decision: `{row.get('review_decision')}`",
            f"- Manual note: {_one_line(row.get('review_notes'))}",
            f"- Likely failed layer: `{diag['layer']}`",
            f"- Trace used: `{trace_path}`" if trace_path else "- Trace used: `(none found)`",
            "",
            "### Gold Evidence",
            "",
            _render_gold(row),
            "",
            "### Search Queries And Snippet Heads",
            "",
        ])
        if trace["searches"]:
            for index, search in enumerate(trace["searches"], 1):
                lines.extend([
                    f"**Search {index}**",
                    "",
                    f"- Query: `{_one_line(search['query'], max_chars=220)}`",
                    f"- Snippet head: {_quote_block(search['snippet'])}",
                    "",
                ])
        else:
            lines.append("- No usable search trace found.")
            lines.append("")

        lines.extend([
            "### Final Answer Excerpt",
            "",
            _quote_block(trace.get("final") or row.get("predicted") or "", max_chars=900),
            "",
            "### Diagnosis",
            "",
            diag["finding"],
            "",
            "### Recommended Next Action",
            "",
            diag["next"],
            "",
        ])

    lines.extend([
        "## Proposed Evidence Selection V1 Direction",
        "",
        "The next implementation should not expand the repair loop. It should add a narrow evidence-selection or verification step before final synthesis.",
        "",
        "Recommended first slice:",
        "",
        "1. Classify question type into `list`, `numeric`, `entity/dataset/method`, or `general`.",
        "2. For `entity/dataset/method` questions, require a selected evidence sentence to contain both the candidate entity and the relation asked by the question.",
        "3. For `numeric` questions, require the exact number in `Short answer` to appear in the selected evidence sentence.",
        "4. For `list` questions, require the final list to stay inside the requested category phrase.",
        "",
        "Before implementing this, improve trace capture so full retrieved top-k chunks are available for analysis.",
        "",
    ])
    return lines


def _select_trace(case_id: str) -> Path | None:
    for path in TRACE_CANDIDATES[case_id]:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def _parse_trace(path: Path) -> dict[str, Any]:
    searches: list[dict[str, str]] = []
    final = ""
    pending_query = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        payload = row.get("payload") or {}
        if row.get("kind") == "tool_call" and "search" in str(payload.get("name")):
            args = payload.get("arguments") or {}
            pending_query = str(args.get("query") or "")
        elif row.get("kind") == "tool_result" and "search" in str(payload.get("name")):
            searches.append({
                "query": pending_query,
                "snippet": _extract_snippet(str(payload.get("content") or "")),
            })
            pending_query = ""
        elif row.get("kind") == "turn" and not payload.get("tool_calls"):
            text = str(payload.get("text") or "")
            if text:
                final = text
    return {"searches": searches, "final": final}


def _extract_snippet(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _one_line(content, max_chars=500)
    return _one_line(parsed.get("chunk_text") or content, max_chars=500)


def _render_gold(row: dict[str, Any]) -> str:
    evidence: list[str] = []
    for answer in row.get("gold_answers") or []:
        evidence.extend(answer.get("evidence") or [])
    if evidence:
        return _quote_block(" ".join(evidence), max_chars=900)
    spans = row.get("oracle_spans") or []
    return _quote_block(" | ".join(str(span) for span in spans), max_chars=900)


def _quote_block(text: str, *, max_chars: int = 500) -> str:
    rendered = _one_line(text, max_chars=max_chars)
    if not rendered:
        return "> (empty)"
    return "> " + rendered.replace("\n", "\n> ")


def _one_line(value: Any, *, max_chars: int = 300) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + " ...[truncated]"


if __name__ == "__main__":
    main()
