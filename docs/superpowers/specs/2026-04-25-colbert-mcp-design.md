# PaperPilot colbert-mcp 设计(Day 6)

| 项 | 值 |
|---|---|
| 日期 | 2026-04-25(方案 C Day 5 完工夜,设计 Day 6 colbert-mcp) |
| 范围 | (1) 新建第二个 MCP server `colbert-mcp`;(2) 扩 `arxiv-mcp` 一个 `download_paper` tool;(3) `mcp_servers.json` + 依赖 + smoke |
| 不在范围 | 个人长期 papers 库 ETL;ColBERT 模型本地微调;graph / pdf / vlm server;并发下载;build_index 异步化 |
| 状态 | Draft,待用户 review |

---

## 1. 目标

让 LLM 在 main loop 里能做"段落级精读":

1. 用 `arxiv.search_papers` 撒网 → 拿 metadata
2. 用 `arxiv.download_paper(arxiv_id)` 下载选中的几篇全文(PDF→text)
3. 用 `colbert.build_index(documents)` 在这批 text 上现场建临时索引
4. 用 `colbert.search(query, top_k)` 在索引里捞最相关段落,喂回 LLM 上下文做引用回答

同时严守 Day 5 已锁定的红线:
- **L1 基座层 + Day 5 mcp_client 零改动**(只改一个全局超时常量,见 §7)
- **server 之间不互通信** —— 协作只在 LLM 那一层串联
- **决策由 LLM 做** —— 内部不引入重试 / 退避 / 降级 / 路由分支
- **不做推测性抽象** —— `get_index_stats()` / `chunk_id` / `index_name` 参数 / 多 session 命名空间全砍

---

## 2. 关键设计决策(Q1-Q6)

| ID | 决策 | 选项 | 主要理由 |
|---|---|---|---|
| Q1 | corpus 形态 | **B: session-local 临时索引** | 学术研究 agent 真实工作流是"撒网→精读";A 路线要求先建个人库 ETL,另一个项目体量;C 引入命名空间复杂度无收益 |
| Q2 | 索引生命周期 | **方案 1: 固定 `paperpilot_current` + 强制覆盖 + 进程启动清空** | 任何时候盘上只有一个 ColBERT 索引(~150MB 上限);零跨 session 状态;符合 B 的 session-local 语义;不预留多 session 并存口 |
| Q3 | 全文获取链路 | **C: 扩 arxiv-mcp 加 `download_paper`,colbert-mcp 接 text 输入** | server 单一职责(arxiv 取文档 / colbert 索引检索);A 只索引 abstract 把 ColBERT 用废;B 让 colbert 越权干 arxiv 活;D 推迟全文 ETL 让 smoke 没意义 |
| Q4 | text 缓存策略 | **A: text 永久累加,索引覆盖** | text 是 raw data(下载+解析慢,体积小 50-200KB/篇);索引是 derived data(便宜,30-90s 重建);两者生命周期天然不同 |
| Q5 | 模型加载时机 | **进程启动时立即加载,留内存复用** | 首次模型下载失败立即 hard-fail(对齐 Day 5 启动红线);后续 build_index/search 复用模型实例零加载成本;代价启动多 5-10s |
| Q6 | 启动期索引清理 | **colbert-mcp 进程启动时 `rm -rf data/colbert_index/paperpilot_current/`** | 自审新增。守"session-local"语义:启动 = 干净;新 session 不先 build_index 直接 search → `IndexNotFoundError` → LLM 知道要先 build。否则上次 session 的旧索引会"漏"到新 session,破坏语义 |

### 隐含决策(已锁)

> **2026-04-25 update**: ColBERT 后端从 ragatouille 切到 **PyLate** —— ragatouille 0.0.9 在 Windows + Py3.12 上 dep hell 严重(pyarrow init segfault / langchain 1.x 重构 / transformers 5.x 不兼容 / 默认 splitter 强依赖 llama-index 等),且 ragatouille 0.0.10 自身将切到 PyLate 后端。直接用 PyLate 干净。详见今日 daily log。

| 项 | 值 | 备注 |
|---|---|---|
| Chunk 策略 | **自己实现 paragraph splitter**(Task 4) | PyLate 不自动 chunk(每 doc 一个多向量,~512 token 上限)。学术 paper 全文需切段后逐段喂入 |
| ColBERT 模型 | `lightonai/colbertv2.0` | PyLate 用 sentence-transformers 格式;`colbert-ir/colbertv2.0` 是 Stanford 原生格式不能直接用 |
| ColBERT 后端 | `pylate.indexes.PLAID` + `pylate.retrieve.ColBERT` | 自带 fast-plaid + fastkmeans;不需要 MSVC;不依赖 langchain / llama-index |
| PDF→text 实现 | `pymupdf` (`fitz`) | 不变 |
| build_index 暴露 | 是,LLM 在 loop 里调 | 不变 |
| download_paper 批量 | 否,一次一篇 | 不变 |
| chunk_text 持久化 | `IndexManager._chunk_texts: dict[chunk_id, str]`(进程内,与 PyLate index 同生命周期) | PyLate 索引层只存 embedding+id,不存原文;LLM 引用回答必须看到 chunk 原文(子 spec Q1) |
| Chunking 参数 | 固定 token 滑窗 size=256, overlap=32, 用 `_model.tokenizer` 切 | PyLate 不自动 chunk;ColBERT 多向量鲁棒于切割位置;PyMuPDF 输出的 `\n\n` 不可靠所以不用段落切(子 spec Q2) |

---

## 3. 架构

```
┌──────────────────────────────────────────────────────────────┐
│  Main Loop  (agent_loop.py — 同步,Day 4 落地,Day 6 不动)     │
│      │                                                       │
│      │ tool_use blocks                                       │
│      ▼                                                       │
│  MCPClient  (Day 5 落地;Day 6 仅改一个超时常量)              │
│      │                                                       │
│      ├──► arxiv-mcp 进程  (Day 5 已建,Day 6 扩)              │
│      │       └── search_papers   (existing)                  │
│      │       └── download_paper  ← Day 6 新增                │
│      │                                                       │
│      └──► colbert-mcp 进程  ← Day 6 新建                     │
│              └── build_index                                 │
│              └── search                                      │
│              (内部: PyLate → ColBERT v2.0)                   │
│              (索引根: data/colbert_index/)                   │
└──────────────────────────────────────────────────────────────┘

  data/papers/<arxiv_id>.txt              ← download_paper 长期缓存(永不删)
  data/colbert_index/
      paperpilot_current/                 ← ColBERT 索引(每次 build 覆盖)
```

**两个 server 进程互不通信**。串联只在 LLM 那一层(LLM 拿 download_paper 输出转手喂 build_index)。这是 MCP 架构的核心边界。

---

## 4. 文件布局

### 新增/改动

| 路径 | 状态 | 作用 | 预估行数 |
|---|---|---|---|
| `paperpilot/mcp_servers/arxiv/server.py` | 改 | + `download_paper` tool handler(urllib + pymupdf + 缓存读写) | +~50 |
| `paperpilot/mcp_servers/arxiv/manifest.json` | 改 | + download_paper 描述 | +5 |
| `paperpilot/mcp_servers/colbert/__init__.py` | 新 | 空文件 | 0 |
| `paperpilot/mcp_servers/colbert/server.py` | 新 | MCP 协议入口;2 个 tool handler;**不接触 PyLate** | ~80 |
| `paperpilot/mcp_servers/colbert/index_manager.py` | 新 | 唯一接触 PyLate 的地方;`build()` / `search()` / `_index_path()` | ~80 |
| `paperpilot/mcp_servers/colbert/manifest.json` | 新 | tool 描述 | ~30 |
| `paperpilot/mcp_servers.json` | 改 | + colbert entry | +5 |
| `paperpilot/tools/mcp_client.py` | 改 | 全局 tool 超时 60s → 180s(改一个常量) | +1 -1 |
| `requirements.txt` | 改 | + `PyMuPDF>=1.24.0`、`pylate>=1.1.0` | +2 |
| `scripts/day6_smoke.py` | 新 | 端到端冒烟,加 `sys.stdout.reconfigure(encoding="utf-8")` | ~50 |
| `scripts/day5_smoke.py` | 改 | 顺手加 `sys.stdout.reconfigure(encoding="utf-8")` 修 Windows GBK | +1 |
| `tests/fixtures/sample_paper.pdf` | 新 | 真实 arxiv PDF(BERT 或类似公开老 paper) | (binary) |
| `tests/mcp_servers/test_arxiv_download.py` | 新 | download_paper 单测(mock urllib + 真 pymupdf) | ~80 |
| `tests/mcp_servers/test_colbert_server.py` | 新 | colbert server.py 协议层单测(mock index_manager) | ~60 |
| `tests/mcp_servers/test_colbert_via_client.py` | 新 | 集成测试(mark slow);真起 colbert-mcp 进程跑 build+search | ~80 |

### 不动

- `paperpilot/core/*.py` —— 一行不改
- `paperpilot/agent/loop.py` —— 一行不改
- `paperpilot/tools/mcp_client.py` 的接口/结构 —— 仅改超时常量值

### 布局决策

1. **colbert-mcp 内部分两层** (`server.py` / `index_manager.py`)
   - `server.py` 不 import PyLate,纯协议层;单测 mock `index_manager` 即可,跑得飞快
   - `index_manager.py` 唯一接触 PyLate,日后换检索后端只改这一文件
2. **download_paper 直接放进 `arxiv/server.py`,不另开 `pdf_extractor.py`** —— 下载+pymupdf 总共 ~30 行,YAGNI;后续若 graph/vlm 也要 PDF→text 再抽公共模块(rule of three)
3. **索引根路径硬编码 `data/colbert_index/`(绝对路径)** —— 不让 PyLate 默认散到 cwd

---

## 5. Tool 签名

### `arxiv.search_papers`(Day 5 已有,不动)

### `arxiv.download_paper`(Day 6 新增)

```
Input:
  arxiv_id: str        # 形如 "2401.12345" 或带版本 "2401.12345v2"

Output:
  {
    "paper_id": str,   # 与输入一致
    "text": str        # pymupdf 提的纯 text;空白规范化但不 chunk
  }

行为:
  1. 若 data/papers/<arxiv_id>.txt 存在 → 读盘返回(跳过下载+解析)
  2. 否则 urllib 下 https://arxiv.org/pdf/<arxiv_id>.pdf
     → pymupdf.open() 提 text → 落盘 → 返回

注: 不返回 title。LLM 调 download_paper 之前必然已通过 search_papers
拿到该 paper 的 title;让 download 再调一次 arxiv metadata API 填 title
是不必要的网络往返。LLM 在最终引用回答时从 search 结果取 title。
```

### `colbert.build_index`(Day 6 新增)

```
Input:
  documents: list[{"paper_id": str, "text": str}]

Output:
  {
    "indexed_count": int,
    "index_name": "paperpilot_current"  # 硬编码,作为对 LLM 的明示
  }

行为:
  1. 校验 documents 非空、每个 dict 含 paper_id + text 字段(空 → ValueError)
  2. 对每篇 paper 用 self._model.tokenizer 切 256/overlap=32 token 滑窗
       chunk_id 命名 f"{paper_id}::chunk_{i}"
       text 为空 / 全空白的 paper 跳过(不产生 chunk,不抛错)
  3. self._model.encode(all_chunks, is_query=False) → 多向量 embedding
  4. pylate.indexes.PLAID(index_folder=str(INDEX_ROOT),
                          index_name="paperpilot_current",
                          override=True)
     index.add_documents(documents_ids=all_chunk_ids,
                         documents_embeddings=embs)
  5. 原子替换 self._index = index;self._chunk_texts = dict(zip(ids, texts))
  6. 返回 {"indexed_count": len(documents),  # 按 paper 数,不是 chunk 数
           "index_name": "paperpilot_current"}
```

### `colbert.search`(Day 6 新增)

```
Input:
  query: str
  top_k: int = 5

Output:
  list[{"paper_id": str, "chunk_text": str, "score": float}]

行为:
  1. 若 self._index is None 或 data/colbert_index/paperpilot_current/ 不存在
     → raise IndexNotFoundError("must call build_index first")
  2. q_emb = self._model.encode([query], is_query=True)
  3. retr = pylate.retrieve.ColBERT(index=self._index)
     scores = retr.retrieve(queries_embeddings=q_emb, k=top_k)
       # shape: list[list[{id, score}]] — 外层 query (len=1),内层 top-k
  4. 每条 r:
       paper_id = r["id"].split("::", 1)[0]
       chunk_text = self._chunk_texts[r["id"]]   # 用 [] 不掩盖 KeyError
       score = float(r["score"])
     返回 list[{paper_id, chunk_text, score}]
```

---

## 6. 数据流(端到端时序)

```
用户: "RAG eval 最近啥新方法? RAGAS 在 NQ 上 faithfulness 多少?"

[1] agent_loop → Claude API(prompt + 4 个 tool schema)
[2] Claude → tool_use(arxiv__search_papers, q="RAG evaluation 2025")
[3] mcp_client → arxiv-mcp 进程 → 30 篇 metadata
[4] Claude 看 abstract → 5 个 tool_use(arxiv__download_paper, id=...) ×5
[5] mcp_client 串行执行:
      每次 download_paper 内部:
        缓存 miss → urllib 下 PDF → pymupdf 提 text → 落盘 → 返回
        (5 次平均 ~15-30s 总耗时)
[6] Claude → tool_use(colbert__build_index, documents=[5 份 {id, text}])
[7] mcp_client → colbert-mcp:
      index_manager.build():
        chunk + self._model.encode + indexes.PLAID(override=True)
        chunking + embedding ~30-90s
      返回 {indexed_count: 5, index_name: "paperpilot_current"}
[8] Claude → tool_use(colbert__search, q="RAGAS faithfulness NQ", top_k=5)
[9] mcp_client → colbert-mcp:
      index_manager.search() → 5 个 chunks (毫秒级)
[10] Claude 看 chunks → text block(带引用)
[11] agent_loop 没 tool_use,结束;输出给用户
```

**两个关键点**:
- 第 [5] 步 5 次下载**串行**(不并发),与 Day 5 mcp_client 顺序执行 tool_use blocks 对齐
- 第 [7] 步 build_index **30-90s 同步阻塞** —— 触发了把全局超时拉到 180s 的需求(§7)

---

## 7. 错误处理

按 Day 5 红线分三类:

### A. 启动期 hard-fail(进程起不来 → mcp_client startup 失败 → main 异常退出)

| 失败 | 触发 |
|---|---|
| `import pylate` / `import fitz` 失败 | 依赖装漏 |
| `data/colbert_index/` 不可写 | 权限/路径错 |
| `pylate.models.ColBERT(model_name_or_path="lightonai/colbertv2.0")` 失败 | HF cache miss + 网络断 |

**启动期还会做(不是 fail,是 setup)**:
- `rm -rf data/colbert_index/paperpilot_current/`(Q6 决策);若该路径不存在则跳过;`shutil.rmtree(..., ignore_errors=False)`,**清理本身失败时 hard-fail**(权限错说明环境坏)

理由:配置 bug,LLM 看到也无能为力;立即吵醒。

### B. 运行时 soft-fail(tool raise → mcp_client 转 `is_error: true` → LLM 决策)

| 失败点 | 异常类型 | LLM 看到啥 |
|---|---|---|
| download_paper: arxiv 404 / 无效 id | `ArxivNotFoundError` | "paper 不存在" |
| download_paper: 网络超时 | `urllib.error.URLError` | "网络挂了" |
| download_paper: pymupdf 解析挂 | `PDFParseError`(自定义,wrap fitz 异常) | "PDF 坏了" |
| build_index: documents 空或字段缺失 | `ValueError` | "输入有问题" |
| build_index: 内部 PyLate 抛 | 透传 | LLM 决定重试或换 |
| search: index 目录不存在 | `IndexNotFoundError` | "你没 build_index 就 search" |
| search: 索引损坏 | 透传 PyLate 异常 | LLM 决定 rebuild |

**全部不做**: 重试、退避、降级、fallback。

### C. 超时调整

Day 5 mcp_client 锁了 60s tool 超时,但 build_index 真实耗时 30-90s 中位数 ~60s,**会撞超时**。

**解**: `paperpilot/tools/mcp_client.py` 全局超时 60s → **180s**。

理由: Day 5 锁的是 60s 这个具体数,不是 60s 这个抽象;改常量值不破坏 client 接口/抽象。Day 9 vlm-mcp 的 Qwen-VL 推理也慢,顺带受益。不做 per-server 超时(破坏 manifest 与 Claude Desktop schema 对齐)。不做 build_index 异步化(过度设计)。

---

## 8. 测试策略

### 层 1: 单元测试(快,CI 跑,秒级)

| 文件 | 测什么 | 怎么测 |
|---|---|---|
| `test_arxiv_download.py::test_parse_real_pdf` | download_paper 提 text | mock urllib 返回 fixture PDF bytes;真跑 pymupdf;断言 text 含已知字符串 |
| `test_arxiv_download.py::test_cache_hit` | 缓存命中跳下载 | 两次调同 id;第二次断言 urllib 没被调(spy) |
| `test_arxiv_download.py::test_arxiv_404` | 错误传播 | mock urllib raise HTTPError(404);断言 server 抛 `ArxivNotFoundError` |
| `test_arxiv_download.py::test_pdf_corrupt` | PDF 解析错误 | 喂坏字节;断言抛 `PDFParseError` |
| `test_colbert_server.py::test_build_routes_to_manager` | 协议层 build 路由 | mock `index_manager.build`;断言被调用 + 参数正确 |
| `test_colbert_server.py::test_search_index_missing` | 索引不存在错误 | mock index_manager 让 path 不存在;断言抛 `IndexNotFoundError` |
| `test_colbert_server.py::test_build_empty_documents` | 输入校验 | 传 `[]`;断言抛 `ValueError` |

**Fixture**: `tests/fixtures/sample_paper.pdf` —— 一篇真实 arxiv 老 paper(如 BERT, ~10 页)。**不用** mock 字节流,因为 pymupdf 行为对真实排版才有意义。

### 层 2: 集成测试(中速,~30-60s/case,标 `@pytest.mark.slow`)

| 文件 | 测什么 |
|---|---|
| `test_colbert_via_client.py::test_build_and_search` | 真起 mcp_client + colbert-mcp;3 篇短 dummy text 建索引 + search;断言命中关键词 |
| `test_colbert_via_client.py::test_startup_clears_stale_index` | 预先在 paperpilot_current/ 路径放假目录;启动 colbert-mcp;断言路径被清空(Q6) |
| `test_colbert_via_client.py::test_search_without_build_fails` | 启动后不调 build_index 直接 search;断言抛 `IndexNotFoundError`(Q6 副作用验证) |
| `test_colbert_via_client.py::test_startup_hard_fail` | data/colbert_index/ 设只读;断言 client startup 抛 `MCPStartupError` |

**为啥用 dummy 短文本而非真 PDF**: 测 colbert-mcp 自己,不绑 arxiv 下载链路。隔离失败维度。

**Slow 标记**: 本地 `pytest -m slow`;CI 默认跳;首次跑要拉 400MB 模型。

### 层 3: Day 6 smoke(`scripts/day6_smoke.py`,端到端真实 LLM)

完工标志。

```
prompt: "搜一下 attention 相关的 paper, 下 1 篇全文, 在里面查 multi-head 是怎么定义的"

期望路径:
  arxiv__search_papers → arxiv__download_paper(1篇)
  → colbert__build_index → colbert__search
  → LLM 输出含 multi-head attention 定义

完工: 退出码 0 + stdout 含 "✅ Day 6 smoke PASSED"
```

顺手做:
- `sys.stdout.reconfigure(encoding="utf-8")` 加到 day5/day6 smoke 顶部
- README 加一行预警: "首次跑 colbert 会拉 ~400MB 模型"

---

## 9. 工作量预估 + 完工标志

| 阶段 | 预估 |
|---|---|
| 写 plan | ~30 min |
| Task 1: arxiv-mcp 扩 download_paper + 单测 | ~45 min |
| Task 2: colbert-mcp 骨架 (server.py + manifest + mcp_servers.json + 协议层单测) | ~60 min |
| Task 3: colbert index_manager + 模型启动加载 + 集成测试 | ~60 min |
| Task 4: mcp_client 超时常量改 60s→180s + 验证 Day 5 测试不回归 | ~10 min |
| Task 5: scripts/day6_smoke.py + Day 5 smoke 加 utf-8 reconfigure | ~30 min |
| Task 6: 端到端联调(首次模型下载 + 真实 arxiv 调用 + 全链路 smoke 跑通) | ~45 min |
| **合计** | **~5 小时**(预算 6 小时,留 1 小时 buffer 处理 ColBERT 首次踩坑) |

**完工标志**:
1. `pytest tests/` 全绿(slow 测试本地手跑)
2. `python scripts/day5_smoke.py` 无回归
3. `python scripts/day6_smoke.py` 退出 0 + 打印 PASSED
4. `data/colbert_index/paperpilot_current/` 存在且 ~50-150MB
5. `data/papers/` 下有至少 1 个 .txt
6. 所有改动按合理粒度分 commit(参考 Day 5 教训: 每 task 开始前 `git status` 确认 index 干净)

---

## 10. 不在范围 / 推迟到未来的事

| 项 | 推迟到何时触发 |
|---|---|
| 个人长期 papers 库 ETL pipeline | 用户明确提需求且本 session-local 不够用时 |
| 多 session 并存索引(命名空间) | 单机出现真实并发 main loop 需求时(Week 3 subagent 也是多进程实例,不在 loop 内) |
| build_index 异步化 / 进度上报 | 真出现 LLM 体感"等太久 timeout 后悔"模式时 |
| `get_index_stats()` / `delete_index()` tool | LLM 真在 loop 里需要这些信息做决策时(目前不需要) |
| chunk_id / chunk 偏移返回 | 真实需求:LLM 要"在原文里高亮 chunk"或"取相邻段" |
| ColBERT 模型微调 / 换 multilingual 模型 | 实测中文/多语 paper 检索质量明显不行时 |
| PDF→text 抽公共模块 `pdf_extractor.py` | graph/vlm server 也要做 PDF→text 时(rule of three) |
| `data/papers/` LRU/TTL 清理 | 实测盘空间真出问题时 |

---

## 附: 与 Day 5 spec 的衔接点

- **复用 Day 5 mcp_client 抽象** —— 新 server 只在 `mcp_servers.json` 加 entry,client 启动逻辑零改动(超时常量值变化不算抽象改动)
- **复用 Day 5 manifest schema**(`command/args/env`)—— colbert entry 与 arxiv entry 同形态
- **复用 Day 5 错误传播路径**(handler raise → is_error tool_result → LLM 决策)—— colbert 所有运行时错误走同一通道
- **复用 Day 5 命名约定**(`mcp__<server>__<tool>`)—— `mcp__colbert__build_index` / `mcp__colbert__search` / `mcp__arxiv__download_paper`
