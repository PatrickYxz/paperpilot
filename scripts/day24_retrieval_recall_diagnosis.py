"""Diagnose retrieval recall for representative QASPER failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.eval.evidence_selection import extract_retrieved_chunks

CASE_IDS = [
    "qasper-1910.04601-q1",
    "qasper-1701.00185-q1",
    "qasper-1910.07181-q0",
]

DEFAULT_ENRICHED_PATH = Path("data/eval/qasper_subset_enriched.jsonl")
DEFAULT_REPORT_PATH = Path("docs/retrieval_recall_diagnosis_20260616.md")
DEFAULT_TRACE_DIR = Path("data/traces")


@dataclass(frozen=True)
class ProbeHit:
    query: str
    hit: bool
    oracle_hit: bool
    evidence_hit: bool
    best_rank: int | None
    best_score: float | None
    best_chunk_head: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-path", type=Path, default=DEFAULT_ENRICHED_PATH)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--trace-suffix", default="")
    parser.add_argument("--out-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--report-date", default=date.today().isoformat())
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    case_ids = args.case_ids or CASE_IDS
    rows = _load_enriched_rows(args.enriched_path, case_ids)
    probe_runner = None if args.skip_probe else _ProbeRunner(top_k=args.top_k)

    diagnostics = [
        _diagnose_case(row, args.trace_dir, args.trace_suffix, probe_runner)
        for row in rows
    ]

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(
        "\n".join(_render_report(diagnostics, args.enriched_path, args.report_date)).rstrip() + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_path}")


def _load_enriched_rows(path: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(case_ids)
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("case_id") in wanted:
            rows[str(row["case_id"])] = row
    missing = [case_id for case_id in case_ids if case_id not in rows]
    if missing:
        raise SystemExit(f"Missing enriched rows: {missing}")
    return [rows[case_id] for case_id in case_ids]


def _diagnose_case(
    row: dict[str, Any],
    trace_dir: Path,
    trace_suffix: str,
    probe_runner: "_ProbeRunner | None",
) -> dict[str, Any]:
    targets = _target_texts(row)
    full_text = str(row.get("full_text") or "")
    trace_path = trace_dir / f"{row['case_id']}{trace_suffix}.jsonl"
    trace_chunks = extract_retrieved_chunks(trace_path)

    full_text_hits = _match_targets(full_text, targets)
    trace_hits = _match_targets(
        "\n\n".join(str(chunk.get("chunk_text") or "") for chunk in trace_chunks),
        targets,
    )

    probes: list[ProbeHit] = []
    if probe_runner is not None:
        probes = probe_runner.run(
            paper_id=str(row["arxiv_id"]),
            full_text=full_text,
            queries=_probe_queries(row),
            targets=targets,
        )

    return {
        "case_id": row["case_id"],
        "arxiv_id": row["arxiv_id"],
        "paper_title": row["paper_title"],
        "question": row["question"],
        "oracle_spans": row.get("oracle_spans") or [],
        "targets": targets,
        "full_text_hits": full_text_hits,
        "trace_path": str(trace_path),
        "trace_chunk_count": len(trace_chunks),
        "trace_queries": _unique_queries(trace_chunks),
        "trace_hits": trace_hits,
        "probes": probes,
        "interpretation": _interpret(full_text_hits, trace_hits, probes),
    }


def _target_texts(row: dict[str, Any]) -> list[dict[str, str]]:
    targets: list[dict[str, str]] = []
    for span in row.get("oracle_spans") or []:
        if str(span).strip():
            targets.append({"kind": "oracle_span", "text": str(span).strip()})

    for answer in row.get("answers") or []:
        for evidence in answer.get("highlighted_evidence") or []:
            if str(evidence).strip():
                targets.append({
                    "kind": "highlighted_evidence",
                    "text": _prefix(str(evidence).strip(), 260),
                })
        for evidence in answer.get("evidence") or []:
            if str(evidence).strip():
                targets.append({
                    "kind": "evidence_head",
                    "text": _prefix(str(evidence).strip(), 260),
                })
    return _dedupe_targets(targets)


def _dedupe_targets(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for target in targets:
        key = (target["kind"], _loose_norm(target["text"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(target)
    return out


def _match_targets(text: str, targets: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized_text = _space_norm(text)
    loose_text = _loose_norm(text)
    hits: list[dict[str, Any]] = []
    for target in targets:
        target_text = target["text"]
        exact = _space_norm(target_text) in normalized_text
        loose = _loose_norm(target_text) in loose_text
        hits.append({
            "kind": target["kind"],
            "text": target_text,
            "hit": exact or loose,
            "match_type": "exact" if exact else "loose" if loose else "none",
        })
    return hits


def _probe_queries(row: dict[str, Any]) -> list[str]:
    queries: list[str] = [str(row.get("question") or "")]
    oracle = [str(span) for span in row.get("oracle_spans") or [] if str(span).strip()]
    queries.extend(oracle)
    if oracle:
        queries.append(f"{row.get('question')} {oracle[0]}")

    for answer in row.get("answers") or []:
        for evidence in answer.get("highlighted_evidence") or []:
            if str(evidence).strip():
                queries.append(_prefix(str(evidence), 220))
                break
    return _dedupe_strings(queries)


def _unique_queries(chunks: list[dict[str, Any]]) -> list[str]:
    return _dedupe_strings(str(chunk.get("query") or "") for chunk in chunks)


def _dedupe_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = _space_norm(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _interpret(
    full_text_hits: list[dict[str, Any]],
    trace_hits: list[dict[str, Any]],
    probes: list[ProbeHit],
) -> str:
    full_has_oracle = _has_oracle_hit(full_text_hits)
    full_has_evidence = _has_evidence_hit(full_text_hits)
    trace_has_oracle = _has_oracle_hit(trace_hits)
    trace_has_evidence = _has_evidence_hit(trace_hits)
    probe_evidence_hit = any(probe.evidence_hit for probe in probes)
    probe_oracle_hit = any(probe.oracle_hit for probe in probes)

    if not full_has_oracle and not full_has_evidence:
        return "gold_not_present_in_eval_full_text"
    if trace_has_evidence:
        return "gold_evidence_recalled_in_current_trace"
    if trace_has_oracle:
        return "oracle_span_mentioned_but_gold_evidence_missing_in_current_trace"
    if probe_evidence_hit:
        return "current_query_missed_retrievable_gold_evidence"
    if probe_oracle_hit:
        return "probe_mentions_oracle_but_gold_evidence_missing"
    if probes:
        return "gold_present_but_probe_did_not_retrieve"
    return "gold_present_but_current_trace_missed"


def _any_hit(hits: list[dict[str, Any]], kind: str) -> bool:
    return any(hit["kind"] == kind and hit["hit"] for hit in hits)


def _has_oracle_hit(hits: list[dict[str, Any]]) -> bool:
    return _any_hit(hits, "oracle_span")


def _has_evidence_hit(hits: list[dict[str, Any]]) -> bool:
    return _any_hit(hits, "highlighted_evidence") or _any_hit(hits, "evidence_head")


class _ProbeRunner:
    def __init__(self, *, top_k: int) -> None:
        from paperpilot.mcp_servers.colbert.index_manager import IndexManager

        self._manager = IndexManager()
        self._top_k = top_k

    def run(
        self,
        *,
        paper_id: str,
        full_text: str,
        queries: list[str],
        targets: list[dict[str, str]],
    ) -> list[ProbeHit]:
        probe_paper_id = _probe_paper_id(paper_id, full_text)
        try:
            self._manager.build([{"paper_id": probe_paper_id, "text": full_text}])
        except Exception:
            # Search below will record the concrete error per query.
            pass

        out: list[ProbeHit] = []
        for query in queries:
            try:
                results = self._manager.search(
                    query,
                    paper_id=probe_paper_id,
                    top_k=self._top_k,
                )
            except Exception as e:  # noqa: BLE001
                out.append(ProbeHit(
                    query=query,
                    hit=False,
                    oracle_hit=False,
                    evidence_hit=False,
                    best_rank=None,
                    best_score=None,
                    best_chunk_head=f"probe error: {type(e).__name__}: {e}",
                ))
                continue

            best_rank: int | None = None
            best_score: float | None = None
            best_chunk = ""
            best_evidence_rank: int | None = None
            best_evidence_score: float | None = None
            best_evidence_chunk = ""
            oracle_hit = False
            evidence_hit = False
            for rank, result in enumerate(results, start=1):
                chunk_text = str(result.get("chunk_text") or "")
                chunk_matches = _match_targets(chunk_text, targets)
                if any(match["hit"] for match in chunk_matches):
                    oracle_hit = oracle_hit or _has_oracle_hit(chunk_matches)
                    evidence_hit = evidence_hit or _has_evidence_hit(chunk_matches)
                    best_rank = rank
                    best_score = float(result.get("score", 0.0))
                    best_chunk = chunk_text
                    if _has_evidence_hit(chunk_matches):
                        best_evidence_rank = rank
                        best_evidence_score = float(result.get("score", 0.0))
                        best_evidence_chunk = chunk_text
                        break
            out.append(ProbeHit(
                query=query,
                hit=evidence_hit,
                oracle_hit=oracle_hit,
                evidence_hit=evidence_hit,
                best_rank=best_evidence_rank if evidence_hit else best_rank,
                best_score=best_evidence_score if evidence_hit else best_score,
                best_chunk_head=_head(
                    best_evidence_chunk if evidence_hit else best_chunk,
                    300,
                ),
            ))
        return out


def _render_report(
    diagnostics: list[dict[str, Any]],
    enriched_path: Path,
    report_date: str,
) -> list[str]:
    lines = [
        "# Retrieval Recall Diagnosis",
        "",
        f"Date: {report_date}",
        "",
        f"Enriched source: `{enriched_path}`",
        "",
        "This report checks whether representative QASPER gold/oracle evidence is present in the eval full text, recalled by current PaperPilot traces, and retrievable with gold-oriented diagnostic queries.",
        "",
        "Gold-oriented probe queries are diagnostic only. They must not be used by runtime PaperPilot.",
        "",
        "The probe builds a separate diagnostic ColBERT index from QASPER enriched `full_text`, keyed by a content hash, so it does not reuse the runtime arXiv/PaperPilot index.",
        "",
        "## Summary",
        "",
        "| case_id | full text has gold | current trace recalls gold | probe retrieves gold | interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for diag in diagnostics:
        lines.append(
            f"| `{diag['case_id']}` "
            f"| {_yes_no(_has_gold_hit(diag['full_text_hits']))} "
            f"| {_yes_no(_has_gold_hit(diag['trace_hits']))} "
            f"| {_yes_no(any(probe.evidence_hit for probe in diag['probes'])) if diag['probes'] else 'not run'} "
            f"| `{diag['interpretation']}` |"
        )

    for diag in diagnostics:
        lines.extend(_render_case(diag))
    return lines


def _render_case(diag: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"## {diag['case_id']}",
        "",
        f"- Paper: `{diag['arxiv_id']}` - {diag['paper_title']}",
        f"- Question: {diag['question']}",
        f"- Oracle spans: {', '.join(f'`{span}`' for span in diag['oracle_spans'])}",
        f"- Current trace: `{diag['trace_path']}`",
        f"- Current trace chunk count: {diag['trace_chunk_count']}",
        f"- Interpretation: `{diag['interpretation']}`",
        "",
        "### Targets",
        "",
    ]
    for target in diag["targets"]:
        lines.append(f"- `{target['kind']}`: {_head(target['text'], 220)}")

    lines.extend([
        "",
        "### Full-Text Presence",
        "",
        "| target | hit | match |",
        "|---|---:|---|",
    ])
    lines.extend(_render_hit_rows(diag["full_text_hits"]))

    lines.extend([
        "",
        "### Current Trace Recall",
        "",
        "Queries:",
        "",
    ])
    for query in diag["trace_queries"]:
        lines.append(f"- `{query}`")
    if not diag["trace_queries"]:
        lines.append("- (none)")

    lines.extend([
        "",
        "| target | hit | match |",
        "|---|---:|---|",
    ])
    lines.extend(_render_hit_rows(diag["trace_hits"]))

    lines.extend([
        "",
        "### Gold-Oriented Probe",
        "",
        "Probe source: QASPER enriched `full_text` diagnostic index.",
        "",
    ])
    if not diag["probes"]:
        lines.append("- Probe not run.")
    else:
        lines.extend([
            "| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for probe in diag["probes"]:
            lines.append(
                f"| `{_escape_table(_head(probe.query, 130))}` "
                f"| {_yes_no(probe.evidence_hit)} "
                f"| {_yes_no(probe.oracle_hit and not probe.evidence_hit)} "
                f"| {probe.best_rank if probe.best_rank is not None else ''} "
                f"| {round(probe.best_score, 3) if probe.best_score is not None else ''} "
                f"| {_escape_table(_head(probe.best_chunk_head, 220))} |"
            )
    return lines


def _render_hit_rows(hits: list[dict[str, Any]]) -> list[str]:
    return [
        f"| `{hit['kind']}` | {_yes_no(hit['hit'])} | `{hit['match_type']}` |"
        for hit in hits
    ]


def _has_gold_hit(hits: list[dict[str, Any]]) -> bool:
    return _has_evidence_hit(hits) or (
        not any(hit["kind"] in {"highlighted_evidence", "evidence_head"} for hit in hits)
        and _has_oracle_hit(hits)
    )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _space_norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _loose_norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _head(text: str, max_chars: int) -> str:
    one_line = " ".join(str(text or "").split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 15].rstrip() + " ...[truncated]"


def _prefix(text: str, max_chars: int) -> str:
    return " ".join(str(text or "").split())[:max_chars].rstrip()


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|")


def _probe_paper_id(paper_id: str, full_text: str) -> str:
    digest = hashlib.sha1(full_text.encode("utf-8")).hexdigest()[:10]
    return f"{paper_id}__qasper_probe_{digest}"


if __name__ == "__main__":
    main()
