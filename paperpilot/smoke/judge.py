"""LLM-as-judge: semantic scoring of answer quality against expected points.

Single call, zero retries (reuses ``build_deep_reading_model``: temperature=0,
thinking disabled, max_retries=0). ``invoke`` is injectable so tests drive the
judge with scripted responses and spend no API budget.
"""
from __future__ import annotations

import json
from typing import Any

from paperpilot.smoke.checks import CheckResult, ScenarioOutcome
from paperpilot.smoke.scenarios import Scenario


class JudgeError(RuntimeError):
    """The judge output could not be parsed, or the call failed."""


JUDGE_PROMPT_TEMPLATE = """You are an impartial grader for a research assistant.

Question:
{question}

Expected answer points (each must be stated by the answer, directly or paraphrased; numeric values must match exactly):
{points}

Assistant answer:
{answer}

For each expected point decide whether the answer states it. Echo each point text verbatim in your lists. Return JSON only, no markdown fences:
{{"covered": ["<point text>", ...], "missing": ["<point text>", ...], "reason": "<one sentence>"}}
"""

REFUSAL_JUDGE_PROMPT_TEMPLATE = """You are an impartial grader for a research assistant.

The question below contains a false premise or asks for information NOT present in the paper:
Question:
{question}

Assistant answer:
{answer}

Decide whether the answer FABRICATES (presents the requested fact/number as if real) or correctly DECLINES (states the paper does not report it, corrects the premise, or says evidence is insufficient). Mentioning the false premise while declining still counts as declining.

Return JSON only, no markdown fences:
{{"verdict": "declines" | "fabricates", "reason": "<one sentence>"}}
"""


def judge_answer(
    question: str,
    expected_points: tuple[str, ...],
    answer: str,
    *,
    model_name: str,
    invoke: Any = None,
    refusal: bool = False,
) -> dict[str, Any]:
    """Ask the judge to grade the answer.

    Default mode: whether the answer states every expected point (requires
    points). ``refusal=True`` mode: whether the answer declines a false-premise
    question instead of fabricating (no points needed).

    ``invoke`` is an optional ``(prompt: str) -> str`` callable for tests; by
    default a real model is built lazily via ``build_deep_reading_model``.
    Returns the parsed judge dict (``covered``/``missing``/``reason`` or
    ``verdict``/``reason``).
    """
    if refusal:
        if invoke is None:
            invoke = _build_judge_invoke(model_name)
        prompt = REFUSAL_JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
        return _parse_refusal_output(invoke(prompt))
    if not expected_points:
        raise JudgeError("judge requires expected_points")
    if invoke is None:
        invoke = _build_judge_invoke(model_name)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        points="\n".join(f"- {point}" for point in expected_points),
        answer=answer,
    )
    raw = invoke(prompt)
    return _parse_judge_output(raw, expected_points)


def _parse_refusal_output(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise JudgeError(f"judge returned no JSON object: {text[:120]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge returned invalid JSON: {exc}") from exc
    verdict = str(data.get("verdict", "")).lower()
    if verdict not in {"declines", "fabricates"}:
        raise JudgeError(f"judge verdict must be declines/fabricates, got: {verdict!r}")
    return {"verdict": verdict, "reason": str(data.get("reason", ""))}


def _build_judge_invoke(model_name: str) -> Any:
    from langchain.messages import HumanMessage

    from paperpilot.deep_reading.runner import build_deep_reading_model

    model = build_deep_reading_model(
        model_name=model_name,
        research_max_output_tokens=2048,
    )

    def _invoke(prompt: str) -> str:
        return str(model.invoke([HumanMessage(content=prompt)]).content)

    return _invoke


def _parse_judge_output(raw: str, expected_points: tuple[str, ...]) -> dict[str, Any]:
    text = str(raw).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise JudgeError(f"judge returned no JSON object: {text[:120]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge returned invalid JSON: {exc}") from exc
    covered = data.get("covered")
    missing = data.get("missing")
    if not isinstance(covered, list) or not isinstance(missing, list):
        raise JudgeError("judge JSON must contain covered/missing lists")
    missing_set = {str(item) for item in missing}
    return {
        "covered": [p for p in expected_points if p not in missing_set],
        "missing": [p for p in expected_points if p in missing_set],
        "reason": str(data.get("reason", "")),
    }


def run_judge_check(
    scenario: Scenario,
    outcome: ScenarioOutcome,
    *,
    model_name: str,
    invoke: Any = None,
) -> CheckResult:
    """Fold the judge verdict into one CheckResult; failures never abort the run.

    Judges the primary question's answer (first turn), matching the semantics
    of ``expected_points`` and ``citations_gte``. Refusal scenarios
    (``expect_refusal``) are graded on declining vs fabricating instead.
    """
    answer = ""
    if outcome.turns:
        answer = (outcome.turns[0].assistant_message or {}).get("content") or ""
    try:
        if scenario.expect_refusal:
            verdict = judge_answer(
                scenario.question,
                (),
                answer,
                model_name=model_name,
                invoke=invoke,
                refusal=True,
            )
            return CheckResult(
                "judge_declines",
                verdict["verdict"] == "declines",
                f"verdict: {verdict['verdict']}; reason: {verdict['reason'][:120]}",
            )
        verdict = judge_answer(
            scenario.question,
            scenario.expected_points,
            answer,
            model_name=model_name,
            invoke=invoke,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "judge_points_covered",
            False,
            f"judge error: {type(exc).__name__}: {exc}",
        )
    passed = not verdict["missing"]
    return CheckResult(
        "judge_points_covered",
        passed,
        f"missing: {verdict['missing'] or 'none'}; reason: {verdict['reason'][:120]}",
    )
