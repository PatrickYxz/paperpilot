# PaperPilot colbert-mcp Task 4 内部设计:chunking + chunk_text 映射

| 项 | 值 |
|---|---|
| 日期 | 2026-04-27(方案 C Day 6 下半场,Task 4 实施前) |
| 关系 | `2026-04-25-colbert-mcp-design.md` 的子 spec —— 解决 §5 因 ragatouille→PyLate 切换衍生的内部实现空缺 |
| 范围 | (1) `IndexManager` 内部 chunking 策略;(2) chunk_text → search output 映射;(3) 主 spec / plan 同步清单 |
| 不在范围 | colbert-mcp 整体架构 / Tool 签名(主 spec 锁);Task 5/6(plan 已锁) |
| 状态 | Draft,待用户 review |

---

## 1. 触发原因

主 spec(2026-04-25)写时假设 ragatouille `RAGPretrainedModel.index()` 自带 chunking 且 search 返回 chunk 原文。Day 6 上半场把后端切到 PyLate 后,这两条假设失效:

1. **PyLate 不自动 chunk** —— 一篇 paper 全文 ~10K token 直接 `model.encode()` 会撞 ColBERT v2 的 ~512 token 上限被截断
2. **PyLate `index.add_documents` 只存 embedding+id,不存原文** —— search 返回的是 `{id, score}`,没有 chunk 文本

主 spec §5 锁了 search output schema 含 `chunk_text` 字段,且 §6 数据流第 [10] 步要求 LLM 用 chunk 原文组装"带引用的回答" —— 这是核心价值,不能砍。所以需要在 IndexManager 内部补上 chunking + chunk_text 持久化层。

---

## 2. 三个核心决策

| ID | 决策 | 选项 | 主要理由 |
|---|---|---|---|
| Q1 | chunk_text 字段 | **A: 保留真值,IndexManager 维护 `dict[chunk_id, str]`** | 学术研究 agent 核心价值是带引用回答;砍 chunk_text 让 LLM 自己回 download_paper 重读再定位段落,引用质量大幅掉;内存代价 ~MB 级可忽略 |
| Q2 | chunking 策略 | **A: 固定 token 滑窗,size=256, overlap=32, 用 model.tokenizer 切** | ColBERT 多向量本就鲁棒于切割位置(每 token 自带向量);PyMuPDF 出来的 `\n\n` 在 PDF 双栏/页头页脚下噪音大,段落切不可靠;固定窗大小可控不会撞 max_length;依赖零增加 |
| Q3 | chunk_id 格式 + 多 build 行为 | **`f"{paper_id}::chunk_{i}"`;多 build = 全量重建(_index / _chunk_texts 原子替换)** | `::` 在 arxiv id 中永不出现,可做无歧义分隔符;反查 paper_id 一行 split;增量加/删让 LLM 重 build 一次,与主 spec Q1 session-local 语义对齐 |

### YAGNI 列表(防回头加)

| 项 | 推迟触发条件 |
|---|---|
| `_chunk_texts` 持久化到 disk(json/sqlite) | 真出现"build 完进程崩,重启后想直接 search"的需求 —— 主 spec Q2 锁了启动清索引,自相矛盾,不可能触发 |
| chunk size 自适应(按 paper 长度调) | 实测固定 256/32 检索质量明显不行 |
| chunk_id → paper_id 反向 dict | `split("::", 1)[0]` 比 dict 查更便宜,无收益 |
| chunk overlap 用 char 而非 token | tokenizer 切错位置概率几乎为零;ColBERT 多向量本来鲁棒 |
| 同 paper 多 chunk 命中 top_k 时 dedup | 决策让 LLM 做(用户可能就是要一篇多个支持段落) |
| chunk 内带 page/section 反查 | LLM 真要原文定位时再说;arxiv PDF section 提取本身一坨工程 |
| 跨 build 增量加 paper / 按 id 删 | 主 spec Q1 锁了 session-local 全量重建语义 |
| `_chunk` 独立纯函数单测 | 行为由集成测试覆盖,纯函数 over-test |

---

## 3. `IndexManager` 设计

### 3.1 内部状态(进程生命周期内)

```python
class IndexManager:
    _model: pylate.models.ColBERT          # 启动期一次性加载
    _index: pylate.indexes.PLAID | None    # build_index 后填充;None=未 build
    _chunk_texts: dict[str, str]           # chunk_id → 该 chunk 原文
```

`_index` / `_chunk_texts` **同生命周期**:
- 启动期 `rm -rf paperpilot_current/` 时一并 reset (`_index = None; _chunk_texts = {}`)
- build 时三件套(PyLate 索引 + `_index` 引用 + `_chunk_texts` dict)原子替换

### 3.2 `_chunk(text)` —— 内部方法,无对外暴露

```python
CHUNK_SIZE = 256   # 类常量
OVERLAP = 32

def _chunk(self, text: str) -> list[str]:
    tokens = self._model.tokenizer.encode(text, add_special_tokens=False)
    out, i = [], 0
    while i < len(tokens):
        out.append(self._model.tokenizer.decode(tokens[i : i + CHUNK_SIZE]))
        if i + CHUNK_SIZE >= len(tokens):
            break
        i += CHUNK_SIZE - OVERLAP   # = 224
    return out
```

边界:
- text 为空 / 纯空白 → 该 paper 不产生 chunk(跳过,不抛 —— `download_paper` 已 normalize 空白,真出现是上游 PDF 解析挂了)
- token ≤ 256 → 一段 chunk = 整篇
- 输入 `documents` 整个空 → `build` 抛 `ValueError`(主 spec §7.B 已锁)

### 3.3 `build(documents)` 完整流程

```python
def build(self, documents):
    if not documents:
        raise ValueError("documents 不能空")

    all_ids, all_texts = [], []
    for d in documents:
        for i, ck in enumerate(self._chunk(d["text"])):
            all_ids.append(f"{d['paper_id']}::chunk_{i}")
            all_texts.append(ck)

    embs = self._model.encode(
        all_texts, is_query=False, show_progress_bar=False
    )

    index = pylate.indexes.PLAID(
        index_folder=str(INDEX_ROOT),
        index_name=INDEX_NAME,           # "paperpilot_current"
        override=True,
    )
    index.add_documents(documents_ids=all_ids, documents_embeddings=embs)

    # 原子替换三件套
    self._index = index
    self._chunk_texts = dict(zip(all_ids, all_texts))

    return {"indexed_count": len(documents), "index_name": INDEX_NAME}
```

`indexed_count` 按 **paper 数**(`len(documents)`),不是 chunk 数 —— 与主 spec §5 锁的语义一致;LLM 视角是"我建了 5 篇的索引"。

### 3.4 `__init__` 末尾追加(Task 4 实现细节)

现有 `__init__`(Task 3 commit `05ce7da`)只 init `_model`。Task 4 在末尾追加两行:

```python
def __init__(self) -> None:
    self._clear_stale_index()
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    self._model = models.ColBERT(model_name_or_path=MODEL_NAME)
    # ↓ Task 4 新增 ↓
    self._index: indexes.PLAID | None = None
    self._chunk_texts: dict[str, str] = {}
```

`indexes` 在模块顶层 `from pylate import models, indexes` 即可。

### 3.5 `search(query, top_k)` 完整流程

```python
def search(self, query, top_k=5):
    if self._index is None or not self._index_path().exists():
        raise IndexNotFoundError("must call build_index first")

    q_emb = self._model.encode(
        [query], is_query=True, show_progress_bar=False
    )
    retr = pylate.retrieve.ColBERT(index=self._index)
    scores = retr.retrieve(queries_embeddings=q_emb, k=top_k)
    # shape: list[list[{id, score}]] — 外层 query (len=1),内层 top-k

    return [
        {
            "paper_id": r["id"].split("::", 1)[0],
            "chunk_text": self._chunk_texts[r["id"]],   # 必命中,缺即 bug
            "score": float(r["score"]),
        }
        for r in scores[0]
    ]
```

`self._chunk_texts[r["id"]]` 用 `[]` 不用 `.get(..., "")` —— 缺失即逻辑 bug,直接 KeyError 暴露,不掩盖。

---

## 4. 主 spec / plan 同步清单(Task 4 实施前必做,~15 min)

memory `paperpilot_day6_pending.md` 已点出残留 ragatouille 引用位置。具体编辑:

| 文件 | 段落 | 改动 |
|---|---|---|
| `specs/2026-04-25-colbert-mcp-design.md` | §2 隐含决策表 | 新增行:`chunk_text 持久化 / IndexManager._chunk_texts: dict[chunk_id, str], 与 PyLate index 同生命周期, 进程死即丢` |
| 同上 | §2 隐含决策表 | 新增行:`Chunking 参数 / 256 token / overlap 32, 用 _model.tokenizer 切` |
| 同上 | §5 `colbert.build_index` 行为段 | 替换 ragatouille 代码块为本设计 §3.3 |
| 同上 | §5 `colbert.search` 行为段 | 替换 ragatouille 代码块为本设计 §3.5 |
| 同上 | §7.A 启动 hard-fail 表 | `import ragatouille` → `import pylate`;`RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")` → `models.ColBERT("lightonai/colbertv2.0")` |
| `plans/2026-04-25-colbert-mcp.md` | Task 4 step 4.2 | 重写 `IndexManager.build` 实现代码块为本设计 §3.3 |
| 同上 | Task 4 step 4.3 | 重写 `IndexManager.search` 实现代码块为本设计 §3.5 |
| 同上 | Task 4 step 4.4 集成测试 | 索引存在性断言路径 `INDEX_ROOT/colbert/indexes/paperpilot_current` → `INDEX_ROOT/paperpilot_current`(PyLate 不嵌一层 `colbert/indexes/`) |

mechanical edit,不需再 brainstorm。本 design doc 与主 spec / plan 同步 commit。

---

## 5. 测试调整(只调,不增)

plan Task 4 已规划的 `test_build_and_search` 集成测试,断言改两条:

1. (已规划)断言 `search(query)` 返 top-1 `chunk_text` 含 query 关键词 —— 现在依赖 `_chunk_texts` dict 真有内容,**隐式回归保证**
2. (新加)断言 `chunk_text != ""` 且 `len(chunk_text) > 50` —— 单独点出"映射链路通"

不新增 `_chunk` 独立单测 —— 私有纯函数,集成测试覆盖即可,单测就 over-test。

`indexed_count == len(documents)` (按 paper 数) —— 与主 spec §5 一致。

---

## 附:与主 spec 衔接点

- **不动主 spec §1-4 / §6-10**,只 surgical 改 §2 决策表 + §5 行为段 + §7.A 启动期 import 名。架构、tool 签名、数据流、错误分类、工作量预估全保持。
- **不引入新 tool / 新文件 / 新依赖** —— PyLate 已在 requirements.txt(Day 6 Task 1 commit `60a369e`),tokenizer 是 PyLate 模型自带,无新增。
- **本 design doc 的 §3.3/3.4/3.5 全代码块直接搬到 plan Task 4 step 4.1/4.2/4.3 即可** —— Task 4 的 spec→implementation gap 至此为零。
