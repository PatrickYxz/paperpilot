"""Quality checks and repair prompt helpers for evaluated final answers."""
from __future__ import annotations

import re
from typing import Any

Issue = dict[str, str]

SEVERE = "severe"
RISK = "risk"
NONE = "none"

INTERNAL_TRACE_MARKERS = [
    "Step 4",
    "Answer span candidates",
    "Now I have enough evidence",
    "Now I have all the information",
    "Let me synthesize",
    "检索充分",
    "综合回答",
]

SEVERE_ISSUE_CODES = {
    "internal_trace_marker",
    "missing_short_answer_section",
    "missing_evidence_section",
    "short_answer_too_long",
    "numeric_not_supported_in_evidence",
}

SECTION_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?"
    r"(short answer|evidence|notes)"
    r"(?:\*\*)?\s*:",
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")

MAX_SHORT_ANSWER_CHARS = 700
MIN_EVIDENCE_CHARS = 40


def evaluate_answer_quality(answer: str, *, question: str | None = None) -> dict[str, Any]:
    """Return structured quality issues for a final answer.

    The checker is intentionally shallow. It flags output quality risks and
    repair triggers, but it does not decide whether the answer is semantically
    correct.
    """
    text = answer or ""
    sections = _extract_sections(text)
    issues: list[Issue] = []

    for marker in INTERNAL_TRACE_MARKERS:
        if marker.lower() in text.lower():
            issues.append(_issue(
                "internal_trace_marker",
                SEVERE,
                f"Final answer contains internal process marker: {marker}",
            ))
            break

    short_answer = sections.get("short answer")
    evidence = sections.get("evidence")

    if short_answer is None:
        issues.append(_issue(
            "missing_short_answer_section",
            SEVERE,
            "Final answer should include a Short answer section.",
        ))
    elif len(short_answer.strip()) > MAX_SHORT_ANSWER_CHARS:
        issues.append(_issue(
            "short_answer_too_long",
            SEVERE,
            f"Short answer is longer than {MAX_SHORT_ANSWER_CHARS} characters.",
        ))

    if evidence is None:
        issues.append(_issue(
            "missing_evidence_section",
            SEVERE,
            "Final answer should include an Evidence section.",
        ))
    elif len(evidence.strip()) < MIN_EVIDENCE_CHARS:
        issues.append(_issue(
            "evidence_too_thin",
            RISK,
            "Evidence section is very short or underspecified.",
        ))

    if short_answer is not None and evidence is not None:
        unsupported_numbers = _unsupported_numbers(short_answer, evidence)
        if unsupported_numbers:
            issues.append(_issue(
                "numeric_not_supported_in_evidence",
                SEVERE,
                "Numbers in Short answer are not repeated in Evidence: "
                + ", ".join(unsupported_numbers),
            ))
        if _looks_like_list_dump(short_answer):
            issues.append(_issue(
                "list_dumping_risk",
                RISK,
                "Short answer looks list-heavy; verify the question asks for all listed items.",
            ))

    severity = _overall_severity(issues)
    return {
        "passed": severity != SEVERE,
        "severity": severity,
        "repair_required": severity == SEVERE,
        "issue_count": len(issues),
        "issues": issues,
    }


def should_repair_answer(quality: dict[str, Any]) -> bool:
    """Return true when the quality report has any severe repair trigger."""
    return any(
        issue.get("code") in SEVERE_ISSUE_CODES
        and issue.get("severity") == SEVERE
        for issue in quality.get("issues", [])
    )


def build_repair_prompt(
    *,
    question: str,
    raw_answer: str,
    quality: dict[str, Any],
) -> str:
    """Build a narrow one-shot final-answer repair prompt."""
    issue_lines = "\n".join(
        f"- {issue.get('code')}: {issue.get('message')}"
        for issue in quality.get("issues", [])
    ) or "- none"
    return (
        "Rewrite the final answer below into a cleaner evaluated answer.\n\n"
        "Rules:\n"
        "- Do not call tools.\n"
        "- Do not add new facts.\n"
        "- Use only claims and evidence already present in the draft answer.\n"
        "- If the draft does not contain enough evidence, say that evidence is insufficient.\n"
        "- Output exactly these sections: Short answer:, Evidence:, and optional Notes:.\n"
        "- Short answer must directly answer the question first.\n"
        "- Evidence must only include text that directly supports the short answer.\n"
        "- Do not include Step 4, Answer span candidates, tool progress, or process narration.\n"
        "- For numbers in Short answer, repeat the same numbers in Evidence.\n\n"
        f"Question:\n{question}\n\n"
        f"Quality issues to fix:\n{issue_lines}\n\n"
        f"Draft final answer:\n{raw_answer}\n"
    )


def _extract_sections(text: str) -> dict[str, str]:
    matches = list(SECTION_PATTERN.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()
    return sections


def _unsupported_numbers(short_answer: str, evidence: str) -> list[str]:
    evidence_numbers = set(NUMBER_PATTERN.findall(evidence))
    unsupported: list[str] = []
    for number in NUMBER_PATTERN.findall(short_answer):
        if number not in evidence_numbers and number not in unsupported:
            unsupported.append(number)
    return unsupported


def _looks_like_list_dump(short_answer: str) -> bool:
    bullet_lines = sum(
        1
        for line in short_answer.splitlines()
        if line.lstrip().startswith(("-", "*", "1.", "2.", "3.", "4."))
    )
    punctuation_count = short_answer.count(",") + short_answer.count(";")
    return bullet_lines >= 4 or punctuation_count >= 7


def _overall_severity(issues: list[Issue]) -> str:
    if any(issue["severity"] == SEVERE for issue in issues):
        return SEVERE
    if issues:
        return RISK
    return NONE


def _issue(code: str, severity: str, message: str) -> Issue:
    return {
        "code": code,
        "severity": severity,
        "message": message,
    }
