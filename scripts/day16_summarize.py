"""Day 16 summarize: results_<baseline>.jsonl files to summary.md."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.eval.scorer import cluster_failures

EVAL_DIR = Path("data/eval")
OUT_PATH = EVAL_DIR / "summary.md"
BASELINES = ["abstract_only", "full_text", "paperpilot"]


def _load(baseline: str) -> list[dict]:
    path = EVAL_DIR / f"results_{baseline}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stats(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {"total": 0, "pass": 0, "fail": 0, "err": 0, "rate": 0.0, "avg_s": 0.0}
    err = sum(1 for r in records if r.get("error"))
    passed = sum(1 for r in records if r.get("passed"))
    fail = total - passed - err
    avg_s = mean(r.get("elapsed_s", 0) for r in records)
    return {
        "total": total,
        "pass": passed,
        "fail": fail,
        "err": err,
        "rate": passed / total,
        "avg_s": avg_s,
    }


def _fmt_pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def main() -> None:
    all_records = {baseline: _load(baseline) for baseline in BASELINES}
    stats = {baseline: _stats(records) for baseline, records in all_records.items()}

    if any(s["total"] == 0 for s in stats.values()):
        missing = [baseline for baseline, s in stats.items() if s["total"] == 0]
        sys.exit(f"FAIL: results_{missing[0]}.jsonl is empty. Run day16_run_eval.py first.")

    pp_stats = stats["paperpilot"]
    fail_buckets = cluster_failures(all_records["paperpilot"])
    fail_total = sum(fail_buckets.values())
    fail_rows = sorted(fail_buckets.items(), key=lambda kv: -kv[1])

    lines: list[str] = []
    lines.append("# Day 16 deep-read 召回精度 eval")
    lines.append("")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("**Dataset**: AI2 QASPER NLP subset (extractive QA)")
    lines.append(f"**Cases per baseline**: {pp_stats['total']}")
    lines.append("**LLM**: DeepSeek (via anthropic SDK), shared across all baselines")
    lines.append("")
    lines.append("## 三段对比")
    lines.append("")
    lines.append("| Baseline | Pass | Fail | Error | Pass Rate | Avg latency |")
    lines.append("|---|---|---|---|---|---|")
    for baseline in BASELINES:
        s = stats[baseline]
        lines.append(
            f"| {baseline} | {s['pass']} | {s['fail']} | {s['err']} | "
            f"**{_fmt_pct(s['rate'])}** | {s['avg_s']:.1f}s |"
        )
    lines.append("")
    lines.append("## PaperPilot 失败归因")
    lines.append("")
    lines.append("| 桶 | 计数 | 占比 |")
    lines.append("|---|---|---|")
    for bucket, count in fail_rows:
        pct = (count / fail_total * 100) if fail_total else 0
        lines.append(f"| {bucket} | {count} | {pct:.0f}% |")
    if not fail_rows:
        lines.append("| (无失败) | 0 | - |")
    lines.append("")
    lines.append("## 解读")
    lines.append("")

    abstract_rate = stats["abstract_only"]["rate"]
    full_text_rate = stats["full_text"]["rate"]
    paperpilot_rate = stats["paperpilot"]["rate"]
    if full_text_rate > abstract_rate:
        lines.append(
            f"- abstract -> full_text 提升 {(full_text_rate - abstract_rate) * 100:+.1f}pts:"
            "细节召回需要正文,abstract 远不够"
        )
    else:
        lines.append(
            f"- abstract -> full_text 未提升 {(full_text_rate - abstract_rate) * 100:+.1f}pts:"
            "可能题目对 abstract 已经友好,或 full_text 截断过严"
        )
    if paperpilot_rate > full_text_rate:
        lines.append(
            f"- full_text -> paperpilot 提升 {(paperpilot_rate - full_text_rate) * 100:+.1f}pts:"
            "colbert 选段 + 多次召回比全文一次性塞 LLM 更优,验证 RAG 路线价值"
        )
    else:
        lines.append(
            f"- full_text -> paperpilot 未提升 {(paperpilot_rate - full_text_rate) * 100:+.1f}pts:"
            "Day 17 需要加 BM25 baseline 或换更难题型重跑"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")
    print("\n--- preview ---")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
