"""Self-contained Okapi BM25 sparse retrieval.

No third-party dependency: tokenization covers ASCII words plus CJK character
bigrams, so Chinese queries match without a segmentation library. Scores use
the Lucene-style non-negative IDF variant ``ln(1 + (N - df + 0.5) / (df + 0.5))``.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Mapping

_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def _stem(word: str) -> str:
    """Light English plural normalization (no dependency, conservative)."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercased ASCII words (plural-normalized) plus CJK bigrams."""
    tokens = [_stem(word) for word in _WORD_RE.findall(text.lower())]
    for run in _CJK_RUN_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    """Inverted-index BM25 over a ``{doc_id: text}`` corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_lengths: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}

    @property
    def doc_count(self) -> int:
        return len(self._doc_lengths)

    def build(self, corpus: Mapping[str, str]) -> "BM25Index":
        """Index the corpus, replacing any previous contents."""
        self._postings = defaultdict(dict)
        self._doc_lengths = {}
        df: Counter[str] = Counter()

        for doc_id, text in corpus.items():
            tf = Counter(tokenize(text))
            self._doc_lengths[doc_id] = sum(tf.values())
            for term, count in tf.items():
                self._postings[term][doc_id] = count
                df[term] += 1

        total = sum(self._doc_lengths.values())
        self._avgdl = total / len(self._doc_lengths) if self._doc_lengths else 0.0
        n_docs = len(self._doc_lengths)
        self._idf = {
            term: math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            for term, doc_freq in df.items()
        }
        return self

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` pairs sorted by descending BM25 score."""
        if top_k <= 0 or not self._doc_lengths:
            return []

        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self._postings[term].items():
                norm = self.k1 * (
                    1.0
                    - self.b
                    + self.b * self._doc_lengths[doc_id] / (self._avgdl or 1.0)
                )
                scores[doc_id] += idf * tf * (self.k1 + 1.0) / (tf + norm)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ordered[:top_k]
