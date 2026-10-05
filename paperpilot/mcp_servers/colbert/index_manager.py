"""colbert-mcp index layer.

Each paper gets an isolated persisted PLAID index:
  data/colbert_index/<paper_key>/
    paperpilot_current/
    chunks.json

chunks.json v2 stores raw chunk text plus metadata (section hint, character
span, contextual prefix); v1 files (plain ``{id: text}``) remain readable.
Search defaults to hybrid mode: ColBERT dense retrieval fused with BM25 via
reciprocal rank fusion, with ``mode="dense"`` preserving dense-only behavior.

Windows DLL order still matters: pyarrow/datasets must load before torch gets
pulled in through pylate.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pyarrow  # noqa: F401 (order matters on Windows)
import datasets  # noqa: F401 (order matters on Windows)

from pylate import indexes, models, retrieve

from paperpilot.mcp_servers.colbert.chunking import (
    ChunkRecord,
    chunk_document,
    default_context_prefix,
)
from paperpilot.retrieval.bm25 import BM25Index
from paperpilot.retrieval.fusion import rrf_fuse

INDEX_NAME = "paperpilot_current"
INDEX_ROOT = Path("data/colbert_index")
MODEL_NAME = "lightonai/colbertv2.0"
CHUNKS_SCHEMA_VERSION = 2
_HYBRID_CANDIDATE_MULTIPLIER = 4
_HYBRID_CANDIDATE_FLOOR = 20
_SAFE_PAPER_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")

PrefixProvider = Callable[[str, ChunkRecord, int], str]


class IndexNotFoundError(RuntimeError):
    """Raised when search is called before a paper index exists."""


@dataclass
class _IndexState:
    index: object
    chunk_texts: dict[str, str] = field(default_factory=dict)
    chunk_meta: dict[str, dict] = field(default_factory=dict)
    bm25: BM25Index | None = None

    def bm25_corpus(self) -> dict[str, str]:
        """Contextual-prefixed corpus used for the sparse index."""
        corpus: dict[str, str] = {}
        for full_id, text in self.chunk_texts.items():
            prefix = self.chunk_meta.get(full_id, {}).get("prefix", "")
            corpus[full_id] = f"{prefix}\n{text}" if prefix else text
        return corpus


class IndexManager:
    SEARCH_MODES = ("hybrid", "dense")

    def __init__(
        self, prefix_provider: PrefixProvider = default_context_prefix
    ) -> None:
        INDEX_ROOT.mkdir(parents=True, exist_ok=True)
        self._model = models.ColBERT(model_name_or_path=MODEL_NAME)
        self._prefix_provider = prefix_provider
        self._states: dict[str, _IndexState] = {}

    def build(self, documents: list[dict]) -> dict:
        """Build or load one isolated index per paper_id."""
        if not documents:
            raise ValueError("documents must not be empty")
        cached: list[str] = []
        fresh: list[str] = []
        for document in documents:
            paper_id = document["paper_id"]
            if paper_id in self._states:
                cached.append(paper_id)
                continue
            if self._try_lazy_load(paper_id):
                cached.append(paper_id)
                continue
            self._cold_build(paper_id, document["text"])
            fresh.append(paper_id)

        return {
            "indexed_count": len(documents),
            "index_name": INDEX_NAME,
            "cached_papers": cached,
            "fresh_papers": fresh,
        }

    def search(
        self, query: str, paper_id: str, top_k: int, mode: str = "hybrid"
    ) -> list[dict]:
        """Search one paper's isolated index and return matching chunks."""
        if mode not in self.SEARCH_MODES:
            raise ValueError(
                f"mode must be one of {self.SEARCH_MODES}, got {mode!r}"
            )
        state = self._load_state(paper_id)
        if mode == "dense":
            return [
                self._to_hit(state, full_id, score)
                for full_id, score in self._dense_ranked(state, query, top_k)
            ]
        return self._hybrid_search(state, query, top_k)

    def _load_state(self, paper_id: str) -> _IndexState:
        state = self._states.get(paper_id)
        if state is None:
            if not self._try_lazy_load(paper_id):
                raise IndexNotFoundError(
                    f"no index for {paper_id!r}; call build_index first"
                )
            state = self._states[paper_id]
        return state

    def _dense_ranked(
        self, state: _IndexState, query: str, k: int
    ) -> list[tuple[str, float]]:
        q_emb = self._model.encode([query], is_query=True, show_progress_bar=False)
        retr = retrieve.ColBERT(index=state.index)
        scores = retr.retrieve(queries_embeddings=q_emb, k=k)
        return [
            (result["id"], float(result["score"])) for result in scores[0]
        ]

    def _hybrid_search(
        self, state: _IndexState, query: str, top_k: int
    ) -> list[dict]:
        candidate_k = min(
            max(top_k * _HYBRID_CANDIDATE_MULTIPLIER, _HYBRID_CANDIDATE_FLOOR),
            max(len(state.chunk_texts), 1),
        )
        dense = self._dense_ranked(state, query, candidate_k)
        sparse = (
            state.bm25.search(query, candidate_k)
            if state.bm25 is not None
            else []
        )
        dense_ranks = {
            full_id: rank for rank, (full_id, _) in enumerate(dense, start=1)
        }
        sparse_ranks = {
            full_id: rank for rank, (full_id, _) in enumerate(sparse, start=1)
        }
        fused = rrf_fuse(
            {
                "dense": [full_id for full_id, _ in dense],
                "sparse": [full_id for full_id, _ in sparse],
            },
            top_k=top_k,
        )
        hits: list[dict] = []
        for full_id, fused_score in fused:
            hit = self._to_hit(state, full_id, fused_score)
            hit["fused_score"] = fused_score
            hit["dense_rank"] = dense_ranks.get(full_id)
            hit["sparse_rank"] = sparse_ranks.get(full_id)
            hits.append(hit)
        return hits

    def _to_hit(self, state: _IndexState, full_id: str, score: float) -> dict:
        result_paper_id, chunk_id = full_id.split("::", 1)
        meta = state.chunk_meta.get(full_id, {})
        return {
            "paper_id": result_paper_id,
            "chunk_id": chunk_id,
            "chunk_text": state.chunk_texts[full_id],
            "score": score,
            "context_prefix": meta.get("prefix", ""),
            "section": meta.get("section", ""),
        }

    def _cold_build(self, paper_id: str, text: str) -> None:
        paper_root = self._paper_root(paper_id)
        paper_root.mkdir(parents=True, exist_ok=True)

        records = chunk_document(text, self._token_count)
        chunk_texts: dict[str, str] = {}
        chunk_meta: dict[str, dict] = {}
        encode_texts: list[str] = []
        for record in records:
            full_id = f"{paper_id}::chunk_{record.index}"
            prefix = self._prefix_provider(paper_id, record, len(records))
            chunk_texts[full_id] = record.text
            chunk_meta[full_id] = {
                "section": record.section_hint,
                "chunk_index": record.index,
                "char_span": record.char_span,
                "prefix": prefix,
            }
            encode_texts.append(f"{prefix}\n{record.text}")

        embs = self._model.encode(
            encode_texts, is_query=False, show_progress_bar=False
        )

        index = indexes.PLAID(
            index_folder=str(paper_root),
            index_name=INDEX_NAME,
            override=True,
        )
        index.add_documents(
            documents_ids=list(chunk_texts), documents_embeddings=embs
        )
        self._index_path(paper_id).mkdir(parents=True, exist_ok=True)

        payload = {
            "version": CHUNKS_SCHEMA_VERSION,
            "chunks": {
                full_id: {"text": text_, "meta": chunk_meta[full_id]}
                for full_id, text_ in chunk_texts.items()
            },
        }
        self._chunks_path(paper_id).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        bm25 = BM25Index().build(dict(zip(chunk_texts, encode_texts)))
        self._states[paper_id] = _IndexState(
            index=index,
            chunk_texts=chunk_texts,
            chunk_meta=chunk_meta,
            bm25=bm25,
        )

    def _try_lazy_load(self, paper_id: str) -> bool:
        if not self._index_path(paper_id).exists():
            return False
        if not self._chunks_path(paper_id).exists():
            return False
        try:
            data = json.loads(
                self._chunks_path(paper_id).read_text(encoding="utf-8")
            )
            chunk_texts, chunk_meta = self._parse_chunks_payload(data)
            index = indexes.PLAID(
                index_folder=str(self._paper_root(paper_id)),
                index_name=INDEX_NAME,
                override=False,
            )
        except Exception:
            return False
        state = _IndexState(
            index=index,
            chunk_texts=chunk_texts,
            chunk_meta=chunk_meta,
        )
        state.bm25 = BM25Index().build(state.bm25_corpus())
        self._states[paper_id] = state
        return True

    @staticmethod
    def _parse_chunks_payload(
        data: object,
    ) -> tuple[dict[str, str], dict[str, dict]]:
        if (
            isinstance(data, dict)
            and data.get("version") == CHUNKS_SCHEMA_VERSION
        ):
            chunks = data.get("chunks", {})
            chunk_texts = {
                full_id: record.get("text", "")
                for full_id, record in chunks.items()
            }
            chunk_meta = {
                full_id: record.get("meta", {})
                for full_id, record in chunks.items()
            }
        elif isinstance(data, dict):
            chunk_texts = {
                full_id: text_
                for full_id, text_ in data.items()
                if isinstance(text_, str)
            }
            chunk_meta = {}
        else:
            raise ValueError("unsupported chunks payload")
        if not chunk_texts:
            raise ValueError("empty chunks payload")
        return chunk_texts, chunk_meta

    def _token_count(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def _paper_root(self, paper_id: str) -> Path:
        return INDEX_ROOT / self._paper_key(paper_id)

    def _index_path(self, paper_id: str) -> Path:
        return self._paper_root(paper_id) / INDEX_NAME

    def _chunks_path(self, paper_id: str) -> Path:
        return self._paper_root(paper_id) / "chunks.json"

    def _paper_key(self, paper_id: str) -> str:
        safe = _SAFE_PAPER_KEY_RE.sub("_", paper_id).strip("._")
        digest = hashlib.sha1(paper_id.encode("utf-8")).hexdigest()[:10]
        return f"{safe or 'paper'}-{digest}"
