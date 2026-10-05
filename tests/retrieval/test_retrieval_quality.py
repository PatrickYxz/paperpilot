"""Retrieval quality eval on a fixed synthetic corpus.

Two synthetic papers with deliberately overlapping phrasing ("proposed
approach/scheme reduces ...") make the contextual-prefix ablation meaningful:
ambiguous queries carry paper-level clues that exist only in the prefix.
Everything here runs without the ColBERT model — dense rankings come from a
controlled synonym-based stand-in.
"""
from __future__ import annotations

import json
from pathlib import Path

from paperpilot.retrieval.bm25 import BM25Index, tokenize
from paperpilot.retrieval.fusion import rrf_fuse
from paperpilot.retrieval.metrics import mrr, recall_at_k

FIXTURES = Path(__file__).parent.parent / "fixtures" / "retrieval_eval"
# Widest relevant set has 4 chunks, so recall is measured at k=4 for a fair
# ceiling; single-relevant queries are unaffected by the wider window.
TOP_K = 4


def _load(name: str) -> list[dict]:
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


CORPUS = _load("corpus.jsonl")
QUERIES = _load("queries.jsonl")


def _prefixed(row: dict, part: int, total: int) -> str:
    section = f"; section: {row['section']}" if row["section"] else ""
    return f"[source: paper {row['paper_id']}{section}; part {part}/{total}]"


def _corpora() -> tuple[dict[str, str], dict[str, str]]:
    per_paper_totals: dict[str, int] = {}
    for row in CORPUS:
        per_paper_totals[row["paper_id"]] = (
            per_paper_totals.get(row["paper_id"], 0) + 1
        )
    seen: dict[str, int] = {}
    plain: dict[str, str] = {}
    prefixed: dict[str, str] = {}
    for row in CORPUS:
        paper = row["paper_id"]
        seen[paper] = seen.get(paper, 0) + 1
        prefix = _prefixed(row, seen[paper], per_paper_totals[paper])
        plain[row["chunk_id"]] = row["text"]
        prefixed[row["chunk_id"]] = f"{prefix}\n{row['text']}"
    return plain, prefixed


PLAIN, PREFIXED = _corpora()


def _bm25(corpus: dict[str, str]) -> BM25Index:
    return BM25Index().build(corpus)


def _ranked(index: BM25Index, query: str) -> list[str]:
    return [doc_id for doc_id, _ in index.search(query, top_k=20)]


def _queries_of(kind: str) -> list[dict]:
    return [q for q in QUERIES if q["kind"] == kind]


def test_exact_queries_hit_top_rank_without_prefixes():
    index = _bm25(PLAIN)
    for q in _queries_of("exact"):
        ranked = _ranked(index, q["query"])
        assert recall_at_k(ranked, set(q["relevant"]), TOP_K) == 1.0, q["query"]
    scores = [
        mrr(_ranked(index, q["query"]), set(q["relevant"]))
        for q in _queries_of("exact")
    ]
    assert sum(scores) / len(scores) >= 0.9


def test_contextual_prefixes_do_not_hurt_exact_queries():
    plain_index = _bm25(PLAIN)
    prefixed_index = _bm25(PREFIXED)
    for q in _queries_of("exact"):
        plain_recall = recall_at_k(
            _ranked(plain_index, q["query"]), set(q["relevant"]), TOP_K
        )
        prefixed_recall = recall_at_k(
            _ranked(prefixed_index, q["query"]), set(q["relevant"]), TOP_K
        )
        assert prefixed_recall >= plain_recall, q["query"]


def test_contextual_prefixes_improve_ambiguous_queries():
    plain_index = _bm25(PLAIN)
    prefixed_index = _bm25(PREFIXED)

    plain_recalls, prefixed_recalls = [], []
    plain_mrrs, prefixed_mrrs = [], []
    for q in _queries_of("ambiguous"):
        relevant = set(q["relevant"])
        plain_ranked = _ranked(plain_index, q["query"])
        prefixed_ranked = _ranked(prefixed_index, q["query"])
        plain_recalls.append(recall_at_k(plain_ranked, relevant, TOP_K))
        prefixed_recalls.append(recall_at_k(prefixed_ranked, relevant, TOP_K))
        plain_mrrs.append(mrr(plain_ranked, relevant))
        prefixed_mrrs.append(mrr(prefixed_ranked, relevant))

    avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
    assert avg(prefixed_recalls) > avg(plain_recalls)
    assert avg(prefixed_mrrs) >= avg(plain_mrrs)
    assert avg(prefixed_recalls) >= 0.75


# Controlled dense stand-in: synonym groups emulate semantic matching that
# lexical BM25 cannot see (the query says "parameter-efficient", the corpus
# says "one tenth of the parameters").
_SEMANTIC_GROUPS = {
    "parameter-efficient": ["one percent of all weights", "one tenth of the parameters"],
    "privacy": ["differential privacy", "privacy budget", "privacy accounting"],
    "cheap training": ["GPU hours", "340 MB", "consumer GPU"],
}


def _fake_dense_ranking(query: str, corpus: dict[str, str]) -> list[str]:
    query_tokens = set(tokenize(query))
    scores: list[tuple[float, str]] = []
    for doc_id, text in corpus.items():
        score = 0.0
        for group, phrases in _SEMANTIC_GROUPS.items():
            group_tokens = set(tokenize(group))
            if query_tokens & group_tokens:
                score += sum(
                    1.0
                    for phrase in phrases
                    if all(tok in set(tokenize(text)) for tok in tokenize(phrase))
                )
        scores.append((score, doc_id))
    ranked = sorted(scores, key=lambda kv: (-kv[0], kv[1]))
    return [doc_id for score, doc_id in ranked if score > 0]


def test_rrf_hybrid_recovers_speech_misses():
    """Sparse misses semantic matches; RRF hybrid must recover them."""
    semantic_queries = [
        {
            "query": "parameter-efficient adaptation",
            "relevant": {
                "lora-transfer::chunk_0",
                "lora-transfer::chunk_4",
            },
        },
        {
            "query": "privacy costs and guarantees",
            "relevant": {
                "feddp-aggregation::chunk_0",
                "feddp-aggregation::chunk_2",
                "feddp-aggregation::chunk_8",
            },
        },
    ]
    index = _bm25(PLAIN)
    for q in semantic_queries:
        sparse_ranked = _ranked(index, q["query"])
        dense_ranked = _fake_dense_ranking(q["query"], PLAIN)

        sparse_recall = recall_at_k(sparse_ranked, q["relevant"], TOP_K)
        fused = [
            doc_id
            for doc_id, _ in rrf_fuse(
                {"dense": dense_ranked, "sparse": sparse_ranked}, top_k=TOP_K
            )
        ]
        fused_recall = recall_at_k(fused, q["relevant"], TOP_K)
        assert fused_recall > sparse_recall, q["query"]


def test_hybrid_prefix_and_semantics_compose():
    """Prefixed sparse + semantic dense should answer a mixed-clue query."""
    index = _bm25(PREFIXED)
    query = "parameter-efficient results in the lora-transfer paper"
    relevant = {"lora-transfer::chunk_4"}

    sparse_ranked = _ranked(index, query)
    dense_ranked = _fake_dense_ranking(query, PREFIXED)
    fused = [
        doc_id
        for doc_id, _ in rrf_fuse(
            {"dense": dense_ranked, "sparse": sparse_ranked}, top_k=TOP_K
        )
    ]
    assert recall_at_k(fused, relevant, TOP_K) == 1.0
