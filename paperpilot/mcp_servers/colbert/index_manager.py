"""colbert-mcp index layer.

Each paper gets an isolated persisted PLAID index:
  data/colbert_index/<paper_key>/
    paperpilot_current/
    chunks.json

Windows DLL order still matters: pyarrow/datasets must load before torch gets
pulled in through pylate.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow  # noqa: F401 (order matters on Windows)
import datasets  # noqa: F401 (order matters on Windows)

from pylate import indexes, models, retrieve

INDEX_NAME = "paperpilot_current"
INDEX_ROOT = Path("data/colbert_index")
MODEL_NAME = "lightonai/colbertv2.0"
_SAFE_PAPER_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")


class IndexNotFoundError(RuntimeError):
    """Raised when search is called before a paper index exists."""


@dataclass
class _IndexState:
    index: object
    chunk_texts: dict[str, str] = field(default_factory=dict)


class IndexManager:
    CHUNK_SIZE = 256
    OVERLAP = 32

    def __init__(self) -> None:
        INDEX_ROOT.mkdir(parents=True, exist_ok=True)
        self._model = models.ColBERT(model_name_or_path=MODEL_NAME)
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

    def search(self, query: str, paper_id: str, top_k: int) -> list[dict]:
        """Search one paper's isolated index and return matching chunks."""
        state = self._states.get(paper_id)
        if state is None:
            if not self._try_lazy_load(paper_id):
                raise IndexNotFoundError(
                    f"no index for {paper_id!r}; call build_index first"
                )
            state = self._states[paper_id]

        q_emb = self._model.encode(
            [query], is_query=True, show_progress_bar=False
        )
        retr = retrieve.ColBERT(index=state.index)
        scores = retr.retrieve(queries_embeddings=q_emb, k=top_k)

        out: list[dict] = []
        for result in scores[0]:
            full_id = result["id"]
            result_paper_id, chunk_id = full_id.split("::", 1)
            out.append({
                "paper_id": result_paper_id,
                "chunk_id": chunk_id,
                "chunk_text": state.chunk_texts[full_id],
                "score": float(result["score"]),
            })
        return out

    def _cold_build(self, paper_id: str, text: str) -> None:
        paper_root = self._paper_root(paper_id)
        paper_root.mkdir(parents=True, exist_ok=True)

        chunks = self._chunk(text)
        chunk_ids = [f"{paper_id}::chunk_{i}" for i in range(len(chunks))]
        embs = self._model.encode(
            chunks, is_query=False, show_progress_bar=False
        )

        index = indexes.PLAID(
            index_folder=str(paper_root),
            index_name=INDEX_NAME,
            override=True,
        )
        index.add_documents(documents_ids=chunk_ids, documents_embeddings=embs)
        self._index_path(paper_id).mkdir(parents=True, exist_ok=True)

        chunk_texts = dict(zip(chunk_ids, chunks))
        self._chunks_path(paper_id).write_text(
            json.dumps(chunk_texts, ensure_ascii=False),
            encoding="utf-8",
        )
        self._states[paper_id] = _IndexState(index=index, chunk_texts=chunk_texts)

    def _try_lazy_load(self, paper_id: str) -> bool:
        if not self._index_path(paper_id).exists():
            return False
        if not self._chunks_path(paper_id).exists():
            return False
        try:
            chunk_texts = json.loads(
                self._chunks_path(paper_id).read_text(encoding="utf-8")
            )
            index = indexes.PLAID(
                index_folder=str(self._paper_root(paper_id)),
                index_name=INDEX_NAME,
                override=False,
            )
        except Exception:
            return False
        self._states[paper_id] = _IndexState(index=index, chunk_texts=chunk_texts)
        return True

    def _chunk(self, text: str) -> list[str]:
        """Fixed token window. Empty text returns no chunks."""
        tokens = self._model.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return []
        out: list[str] = []
        i = 0
        while i < len(tokens):
            sub = tokens[i : i + self.CHUNK_SIZE]
            out.append(self._model.tokenizer.decode(sub))
            if i + self.CHUNK_SIZE >= len(tokens):
                break
            i += self.CHUNK_SIZE - self.OVERLAP
        return out

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
