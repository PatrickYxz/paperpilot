"""Deterministic query planning helpers for QASPER-style retrieval evals."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


QUESTION_TYPES = {
    "dataset_used",
    "method_or_baseline_list",
    "metric_or_result",
    "comparison_or_improvement",
    "negative_or_exception",
    "advantage_or_contribution",
    "evaluation_protocol",
    "definition_or_description",
}

ANSWER_SHAPES = {
    "single_entity",
    "list",
    "number",
    "comparison",
    "yes_no",
    "freeform_explanation",
}


@dataclass(frozen=True)
class QueryItem:
    role: str
    query: str


@dataclass(frozen=True)
class QueryConstraints:
    polarity: str = "neutral"
    needs_numbers: bool = False
    needs_comparison: bool = False
    needs_table: bool = False


@dataclass(frozen=True)
class ExpansionHints:
    neighbor_window: int = 1
    prefer_tables: bool = False
    prefer_captions: bool = False


@dataclass(frozen=True)
class QueryPlan:
    question_type: str
    answer_shape: str
    focus_terms: list[str]
    constraints: QueryConstraints
    must_find: list[str]
    avoid: list[str]
    queries: list[QueryItem]
    expansion_hints: ExpansionHints = field(default_factory=ExpansionHints)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_queries(question: str, title: str = "", abstract: str = "") -> QueryPlan:
    """Return a structured retrieval plan using only runtime-available inputs."""
    clean_question = _space_norm(question)
    question_type = classify_question_type(clean_question)
    answer_shape = infer_answer_shape(clean_question)
    focus_terms = extract_focus_terms(clean_question, title=title, abstract=abstract)
    constraints = _constraints_for(clean_question, question_type, answer_shape)
    must_find = _must_find(clean_question, question_type, answer_shape, focus_terms)
    avoid = _avoid(clean_question, question_type)
    queries = _query_items(clean_question, question_type, answer_shape, focus_terms)
    hints = _expansion_hints(question_type, answer_shape)
    return validate_query_plan(QueryPlan(
        question_type=question_type,
        answer_shape=answer_shape,
        focus_terms=focus_terms,
        constraints=constraints,
        must_find=must_find,
        avoid=avoid,
        queries=queries,
        expansion_hints=hints,
    ), clean_question)


def classify_question_type(question: str) -> str:
    q = question.lower()
    if _has_any(q, [
        " not ",
        "does not",
        "do not",
        "did not",
        "n't",
        "except",
        "fail",
        "fails",
        "worse",
        "lag",
        "lags",
        "behind",
        "no improvement",
    ]):
        return "negative_or_exception"
    if _has_any(q, ["advantage", "advantages", "contribution", "benefit", "proposed model"]):
        return "advantage_or_contribution"
    if _has_any(q, ["dataset", "corpus", "corpora", "benchmark", "data set", "data used"]):
        return "dataset_used"
    if _has_any(q, ["manual evaluation", "evaluate", "evaluation", "criteria", "measured", "measure"]):
        return "evaluation_protocol"
    if _has_any(q, ["how much", "accuracy", "f1", "precision", "recall", "score", "result", "%", "percentage"]):
        return "metric_or_result"
    if _has_any(q, ["better", "improve", "improvement", "outperform", "compared", "comparison", "than previous"]):
        return "comparison_or_improvement"
    if _has_any(q, ["method", "methods", "baseline", "baselines", "approach", "approaches", "model", "models"]):
        return "method_or_baseline_list"
    return "definition_or_description"


def infer_answer_shape(question: str) -> str:
    q = question.lower()
    if q.startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ")):
        return "yes_no"
    if _has_any(q, ["how much", "how many", "accuracy", "f1", "score", "percentage", "%"]):
        return "number"
    if _has_any(q, ["which ", "what are", "list", "methods", "baselines", "approaches", "tasks", "corpora"]):
        return "list"
    if _has_any(q, ["better", "compared", "than", "improvement", "outperform"]):
        return "comparison"
    if _has_any(q, ["dataset", "corpus", "benchmark"]):
        return "single_entity"
    return "freeform_explanation"


def extract_focus_terms(question: str, *, title: str = "", abstract: str = "") -> list[str]:
    """Extract conservative surface terms that generated queries should preserve."""
    del abstract
    terms: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*(?:BERT|LDA|QA|IR|NLI|CRF|SOTA)?\b", question):
        term = match.group(0)
        if term.lower() not in _STOPWORDS:
            terms.append(term)

    quoted = re.findall(r'"([^"]+)"|`([^`]+)`', question)
    for a, b in quoted:
        terms.append(a or b)

    title_terms = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*\b", title)
    for term in title_terms[:4]:
        if term.lower() not in _STOPWORDS and len(term) > 2:
            terms.append(term)

    return _dedupe(terms)[:8]


def validate_query_plan(plan: QueryPlan, question: str) -> QueryPlan:
    queries = [item for item in plan.queries if item.query.strip()]
    queries = _dedupe_query_items(queries)
    literal_query = QueryItem("literal", question)
    if not any(_space_norm(item.query).lower() == _space_norm(question).lower() for item in queries):
        queries.insert(0, literal_query)

    if plan.question_type == "negative_or_exception" and not any(_has_negative_signal(item.query) for item in queries):
        queries.append(QueryItem("negative_exception", f"{_focus_prefix(plan.focus_terms)} not improve except lags behind".strip()))

    if plan.constraints.needs_numbers and not any(_has_any(item.query.lower(), ["table", "score", "result", "accuracy", "f1"]) for item in queries):
        queries.append(QueryItem("result_table", f"{_focus_prefix(plan.focus_terms)} results table score accuracy F1".strip()))

    if len(queries) > 6:
        queries = queries[:6]

    if len(queries) < 4:
        queries.extend(_generic_backfill(question, plan.focus_terms, 4 - len(queries)))

    return QueryPlan(
        question_type=plan.question_type,
        answer_shape=plan.answer_shape,
        focus_terms=plan.focus_terms,
        constraints=plan.constraints,
        must_find=_dedupe(plan.must_find),
        avoid=_dedupe(plan.avoid),
        queries=_dedupe_query_items(queries),
        expansion_hints=plan.expansion_hints,
    )


def _constraints_for(question: str, question_type: str, answer_shape: str) -> QueryConstraints:
    q = question.lower()
    return QueryConstraints(
        polarity="negative" if question_type == "negative_or_exception" else "neutral",
        needs_numbers=answer_shape == "number" or _has_any(q, ["how much", "accuracy", "f1", "score", "%"]),
        needs_comparison=question_type in {"comparison_or_improvement", "negative_or_exception"} or _has_any(q, ["than", "compared"]),
        needs_table=answer_shape == "number" or question_type in {"metric_or_result", "comparison_or_improvement"},
    )


def _must_find(question: str, question_type: str, answer_shape: str, focus_terms: list[str]) -> list[str]:
    focus = _focus_prefix(focus_terms)
    if question_type == "dataset_used":
        return [f"{focus} dataset or corpus used in the experiment".strip(), "explicit experiment data source"]
    if question_type == "method_or_baseline_list":
        return [f"{focus} methods baselines approaches compared".strip(), "complete list requested by the question"]
    if question_type == "metric_or_result":
        return [f"{focus} reported numeric result score metric".strip(), "table or result sentence with numbers"]
    if question_type == "comparison_or_improvement":
        return [f"{focus} comparison against previous or baseline results".strip(), "direction and size of improvement"]
    if question_type == "negative_or_exception":
        return [f"{focus} negative exception evidence".strip(), "not improve except lags behind fails to outperform"]
    if question_type == "advantage_or_contribution":
        return [f"{focus} advantages of the proposed model".strip(), "performance or efficiency benefit"]
    if question_type == "evaluation_protocol":
        return [f"{focus} evaluation protocol criteria".strip(), "who evaluated and what scale or metric was used"]
    return [question]


def _avoid(question: str, question_type: str) -> list[str]:
    del question
    if question_type == "negative_or_exception":
        return ["general improvement statements without exception details"]
    if question_type == "metric_or_result":
        return ["qualitative claims without the requested numbers"]
    if question_type == "dataset_used":
        return ["related work datasets not used in the experiment"]
    if question_type == "method_or_baseline_list":
        return ["background methods not directly compared or used"]
    return []


def _query_items(question: str, question_type: str, answer_shape: str, focus_terms: list[str]) -> list[QueryItem]:
    focus = _focus_prefix(focus_terms)
    items = [QueryItem("literal", question)]

    if question_type == "dataset_used":
        items.extend([
            QueryItem("dataset", f"{focus} dataset used in experiment".strip()),
            QueryItem("corpus", f"{focus} corpus benchmark data used".strip()),
            QueryItem("evidence_pattern", f"{focus} we use dataset experiments use corpus".strip()),
        ])
    elif question_type == "method_or_baseline_list":
        items.extend([
            QueryItem("methods", f"{focus} methods baselines approaches compared".strip()),
            QueryItem("comparison", f"{focus} compared with baseline methods".strip()),
            QueryItem("evidence_pattern", f"{focus} baseline methods directly based on popular approaches".strip()),
        ])
    elif question_type == "metric_or_result":
        items.extend([
            QueryItem("result", f"{focus} reported results score accuracy F1".strip()),
            QueryItem("table", f"{focus} results table metric score".strip()),
            QueryItem("evidence_pattern", f"{focus} achieved accuracy F1 precision recall".strip()),
        ])
    elif question_type == "comparison_or_improvement":
        items.extend([
            QueryItem("comparison", f"{focus} compared with previous state-of-the-art results".strip()),
            QueryItem("improvement", f"{focus} improvement over baseline previous results".strip()),
            QueryItem("table", f"{focus} table comparison better than baseline".strip()),
        ])
    elif question_type == "negative_or_exception":
        items.extend([
            QueryItem("negative_exception", f"{focus} does not improve tasks".strip()),
            QueryItem("contrast", f"{focus} lags behind previous state-of-the-art except".strip()),
            QueryItem("evidence_pattern", f"{focus} except for not improve fails to outperform".strip()),
        ])
    elif question_type == "advantage_or_contribution":
        items.extend([
            QueryItem("advantage", f"{focus} advantages of proposed model".strip()),
            QueryItem("performance", f"{focus} outperforms baselines faster efficient".strip()),
            QueryItem("evidence_pattern", f"{focus} proposed model outperforms converges faster".strip()),
        ])
    elif question_type == "evaluation_protocol":
        items.extend([
            QueryItem("evaluation", f"{focus} evaluation criteria measured by".strip()),
            QueryItem("protocol", f"{focus} manual evaluation annotators scale criteria".strip()),
            QueryItem("metric", f"{focus} precision recall F1 accuracy score".strip()),
        ])
    else:
        items.extend([
            QueryItem("description", f"{focus} definition description architecture".strip()),
            QueryItem("mechanism", f"{focus} how it works method".strip()),
            QueryItem("evidence_pattern", question),
        ])

    if answer_shape == "list":
        items.append(QueryItem("answer_shape", f"{focus} list all which items".strip()))
    elif answer_shape == "number":
        items.append(QueryItem("answer_shape", f"{focus} exact number score percentage table".strip()))
    elif answer_shape == "comparison":
        items.append(QueryItem("answer_shape", f"{focus} compared with than difference improvement".strip()))

    return items


def _expansion_hints(question_type: str, answer_shape: str) -> ExpansionHints:
    return ExpansionHints(
        neighbor_window=2 if question_type in {"dataset_used", "negative_or_exception"} else 1,
        prefer_tables=question_type in {"metric_or_result", "comparison_or_improvement"} or answer_shape == "number",
        prefer_captions=question_type in {"metric_or_result", "comparison_or_improvement"},
    )


def _generic_backfill(question: str, focus_terms: list[str], count: int) -> list[QueryItem]:
    focus = _focus_prefix(focus_terms)
    candidates = [
        QueryItem("semantic", f"{focus} answer evidence for question".strip()),
        QueryItem("section", f"{focus} experiment results evaluation".strip()),
        QueryItem("question_terms", question),
    ]
    return candidates[:count]


def _has_negative_signal(text: str) -> bool:
    return _has_any(text.lower(), [" not ", "does not", "except", "fail", "lags", "behind", "worse"])


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def _space_norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _focus_prefix(focus_terms: list[str]) -> str:
    return " ".join(focus_terms[:4])


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _space_norm(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _dedupe_query_items(items: list[QueryItem]) -> list[QueryItem]:
    seen: set[str] = set()
    out: list[QueryItem] = []
    for item in items:
        query = _space_norm(item.query)
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(QueryItem(item.role, query))
    return out


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "did",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "they",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
}
