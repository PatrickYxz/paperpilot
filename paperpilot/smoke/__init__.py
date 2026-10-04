"""Real-model E2E smoke harness for the PaperPilot conversation pipeline.

Starts a fully real web instance (real DeepSeek/MCP) on an isolated temporary
directory, drives it through real business scenarios, evaluates programmatic
checks, and writes git-tagged JSONL reports for cross-run comparison.

Design principle (entry points keep evolving):
- Scenario data (cases.jsonl) is pure business language with zero API detail.
- All HTTP entry-point knowledge lives in ``adapter.ConversationApi`` only.
- Real-model runs stay behind the README gate: no DEEPSEEK_API_KEY, no run
  (``--dry-run`` excepted).
"""
from paperpilot.smoke.adapter import ConversationApi
from paperpilot.smoke.checks import (
    CheckResult,
    ScenarioOutcome,
    TurnObservation,
    evaluate_checks,
    extract_citations,
    extract_evidence_ids,
)
from paperpilot.smoke.judge import JudgeError, judge_answer, run_judge_check
from paperpilot.smoke.runner import build_report_row, run_scenario
from paperpilot.smoke.runtime import build_isolated_app
from paperpilot.smoke.scenarios import (
    DEFAULT_CASES_PATH,
    Scenario,
    ScenarioError,
    load_scenarios,
)

__all__ = [
    "ConversationApi",
    "CheckResult",
    "ScenarioOutcome",
    "TurnObservation",
    "evaluate_checks",
    "extract_citations",
    "extract_evidence_ids",
    "JudgeError",
    "judge_answer",
    "run_judge_check",
    "build_report_row",
    "run_scenario",
    "build_isolated_app",
    "DEFAULT_CASES_PATH",
    "Scenario",
    "ScenarioError",
    "load_scenarios",
]
