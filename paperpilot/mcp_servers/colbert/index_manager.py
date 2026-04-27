"""colbert-mcp 的索引层。唯一接触 PyLate 的地方。

启动期(__init__):
  1. rm -rf data/colbert_index/paperpilot_current/  (Q6)
  2. models.ColBERT(model_name_or_path="lightonai/colbertv2.0")  (Q5)
任何启动期失败 → 直接抛,触发 mcp_client 启动 hard-fail。

Windows DLL 顺序说明:
  pyarrow 必须在 torch 之前加载,否则 Windows 上出现 access violation segfault
  (torch 加载某 DLL 后与 pyarrow 的 DLL 冲突)。
  pylate → sentence_transformers → datasets → pyarrow 的链条在 torch 已加载后触发崩溃。
  解决:在模块顶层先 import pyarrow/datasets,再 import pylate。
  单测中 IndexManager 从不被实例化(mock),此处 import 不影响单测速度。
"""
from __future__ import annotations

import shutil
from pathlib import Path

# Windows DLL 冲突修复: pyarrow 必须在 torch 前加载。
# sentence_transformers.__init__ → datasets → pyarrow; 若 torch 已加载会 segfault。
import pyarrow  # noqa: F401 (order matters on Windows)
import datasets  # noqa: F401 (order matters on Windows)

from pylate import indexes, models, retrieve

INDEX_NAME = "paperpilot_current"
INDEX_ROOT = Path("data/colbert_index")
MODEL_NAME = "lightonai/colbertv2.0"


class IndexNotFoundError(RuntimeError):
    """search 时索引目录不存在(LLM 没先 build_index)。"""


class IndexManager:
    CHUNK_SIZE = 256
    OVERLAP = 32

    def __init__(self) -> None:
        self._clear_stale_index()
        INDEX_ROOT.mkdir(parents=True, exist_ok=True)
        self._model = models.ColBERT(model_name_or_path=MODEL_NAME)
        self._index: indexes.PLAID | None = None
        self._chunk_texts: dict[str, str] = {}

    def build(self, documents: list[dict]) -> dict:
        """对 documents 建立 ColBERT 索引;固定 index_name=paperpilot_current,强制覆盖。

        chunk 在内部完成(256 token 滑窗,overlap 32);chunk_text 留在
        self._chunk_texts 供 search 时回填。LLM 视角是 paper 级,不感知 chunk。
        """
        if not documents:
            raise ValueError("documents must not be empty")

        all_ids: list[str] = []
        all_texts: list[str] = []
        for d in documents:
            for i, ck in enumerate(self._chunk(d["text"])):
                all_ids.append(f"{d['paper_id']}::chunk_{i}")
                all_texts.append(ck)

        embs = self._model.encode(
            all_texts, is_query=False, show_progress_bar=False
        )

        index = indexes.PLAID(
            index_folder=str(INDEX_ROOT),
            index_name=INDEX_NAME,
            override=True,
        )
        index.add_documents(documents_ids=all_ids, documents_embeddings=embs)

        self._index = index
        self._chunk_texts = dict(zip(all_ids, all_texts))

        return {"indexed_count": len(documents), "index_name": INDEX_NAME}

    def search(self, query: str, top_k: int) -> list[dict]:
        """在 paperpilot_current 索引上查 top_k 段落;chunk_text 从内存映射回填。"""
        if self._index is None or not self._index_path().exists():
            raise IndexNotFoundError(
                "no index at paperpilot_current; call build_index first"
            )

        q_emb = self._model.encode(
            [query], is_query=True, show_progress_bar=False
        )
        retr = retrieve.ColBERT(index=self._index)
        scores = retr.retrieve(queries_embeddings=q_emb, k=top_k)
        # shape: list[list[{id, score}]] — 外层 query (len=1),内层 top-k

        return [
            {
                "paper_id": r["id"].split("::", 1)[0],
                "chunk_text": self._chunk_texts[r["id"]],
                "score": float(r["score"]),
            }
            for r in scores[0]
        ]

    def _chunk(self, text: str) -> list[str]:
        """固定 token 滑窗。空 text 返空 list(该 paper 不产生 chunk)。"""
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

    def _clear_stale_index(self) -> None:
        """Q6: 启动时把 paperpilot_current/ 干净清掉。
        清理失败(权限错等)直接抛,启动 hard-fail。
        """
        stale = self._index_path()
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=False)

    def _index_path(self) -> Path:
        return INDEX_ROOT / INDEX_NAME
