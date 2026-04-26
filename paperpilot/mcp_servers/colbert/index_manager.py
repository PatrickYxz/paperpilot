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

from pylate import models

INDEX_NAME = "paperpilot_current"
INDEX_ROOT = Path("data/colbert_index")
MODEL_NAME = "lightonai/colbertv2.0"


class IndexNotFoundError(RuntimeError):
    """search 时索引目录不存在(LLM 没先 build_index)。"""


class IndexManager:
    def __init__(self) -> None:
        self._clear_stale_index()
        INDEX_ROOT.mkdir(parents=True, exist_ok=True)
        self._model = models.ColBERT(model_name_or_path=MODEL_NAME)

    def build(self, documents: list[dict]) -> dict:
        raise NotImplementedError("Task 4 实现 - 使用 self._model.encode + indexes.PLAID")

    def search(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError("Task 4 实现 - 使用 self._model.encode + retrieve.ColBERT")

    def _clear_stale_index(self) -> None:
        """Q6: 启动时把 paperpilot_current/ 干净清掉。
        清理失败(权限错等)直接抛,启动 hard-fail。
        """
        stale = self._index_path()
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=False)

    def _index_path(self) -> Path:
        return INDEX_ROOT / INDEX_NAME
