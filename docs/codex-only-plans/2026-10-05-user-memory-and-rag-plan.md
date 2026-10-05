# 用户记忆与 RAG 增强三阶段计划（A → B → C）

> Codex only。本计划基于《AI Agents in Depth》第 3 章「用户记忆和知识库」的讨论，
> 用户已确认分阶段路线。分支：`rag`（自 main 合并 `codex/context-management` 后切出）。

## 背景与目标

PaperPilot 已完成第二章对应的单会话分层上下文管理（`deep_reading/context_management/`，
`turn_archives` / `context_artifacts` 表）。第三章讨论的两个尺度在当前代码中处于空白：

- **跨会话用户记忆**：conversation 之间相互隔离，用户的研究方向、偏好、已读论文
  在新会话中全部丢失。
- **RAG 管道质量**：`mcp_servers/colbert` 只有单路稠密检索，分块是固定 token 窗口
  （`_chunk()`），无元数据、无上下文前缀、无稀疏检索与融合。

目标：按书中的技术路径分三阶段补齐，每阶段独立可验收、可回退。

## 已确认决策（2026-10-05 与用户讨论）

1. 路线：**A（RAG 管道增强）→ B（跨会话用户记忆）→ C（双层记忆架构）**。
2. B 阶段记忆读取形态：**Agent 工具（智能体化记忆）**，即 `search_user_memory`
   工具由 agent 按需多轮检索，而非自动注入。
3. git：`codex/context-management` 已 fast-forward 合并回 `main`；新工作在 `rag` 分支。

## 约束条件

- 不引入新依赖：BM25 自实现（`requirements.txt` 无检索库、无中文分词库）。
- 现有 MCP 工具返回结构向后兼容：只增字段不改已有字段含义。
- 旧索引（chunks.json v1）必须继续可读可搜。
- 模型依赖（pylate/torch）不进单元测试：沿用 `tests/mcp_servers/test_index_manager.py`
  的 MagicMock 模式。
- 遵守 `AGENTS.md` 代码组织：算法层放 `paperpilot/retrieval/`（一模块一职责），
  服务封装在 `mcp_servers/colbert`，测试镜像源码结构。

## 三阶段总览

| 阶段 | 主题 | 对应书中章节 | 状态 |
|---|---|---|---|
| A | 结构感知分块 + 上下文前缀 + BM25/RRF 混合检索 + 评测集 | 3.2.1 / 3.2.3 / 3.2.4 / 3.3.5 | 本计划详细展开 |
| B | 跨会话用户记忆：表 + 提取管线 + `search_user_memory` 工具 + 三层次评估 | 3.1 / 3.3.4 | 设计要点见下，开工前单独细化 |
| C | 双层记忆架构：研究画像卡片常驻 + 会话历史上下文感知检索 + 定期整理 | 3.3.5 / 3.3.3 | 概述 |

---

## A 阶段：分步骤执行计划

### A1 结构感知分块 + chunk 元数据

**文件**：`paperpilot/mcp_servers/colbert/index_manager.py`

- `_chunk(text)` 替换为结构感知的 `chunk_document(text) -> list[dict]`：
  1. 按空行分段（PDF 提取文本以空行为段界），连续非空行聚合；
  2. 逐段用 `tokenizer.encode(piece, add_special_tokens=False)` 计数，贪心聚合
     相邻段至 ≤ `CHUNK_SIZE`（256）；
  3. 超长段降级到句子边界（`. ! ? 。！？` + 空白）切分，仍超长再按 token 硬切；
  4. 重叠：相邻块共享边界处最后一句/段（约 ≤ `OVERLAP`）；
  5. heading 启发式：短行（< 80 字符）匹配编号标题模式（`^\d+(\.\d+)*\s+\S`、
     `Abstract`、全大写行）时更新 `section_hint`，传播到后续 chunk 元数据。
- 每个产出记录：`text`（原文）、`char_span`、`section_hint`、`chunk_index`。
- `chunks.json` 升级 v2：`{"version": 2, "chunks": {id: {"text", "meta"}}}`；
  读取时兼容 v1（`dict[str, str]` 视为无 meta）。
- 现有测试 `test_index_manager.py` 的 fixture 需让 fake tokenizer 的 encode 返回
  随输入长度变化的 token 列表（现在是固定 300）。

### A2 索引期上下文前缀（Contextual Retrieval 模板版）

**文件**：`index_manager.py`

- 前缀模板：`[source: paper {paper_id}{; section: {section_hint}}; part {i+1}/{n}]`。
- **编码文本 = 前缀 + "\n" + 原文**，同时用于 ColBERT 文档编码与 BM25 语料
  （前缀给稠密注入语义背景、给稀疏提供可精确匹配的关键词，对应书中 3.3.5）。
- 返回结构：`chunk_text` 保持**原文**（证据溯源展示干净），新增
  `context_prefix` 字段；消费方（evidence_pool / verifier / deep_reading 工具）
  读 `chunk_text` 的行为不变。
- `prefix_provider` 钩子：`IndexManager.__init__(prefix_provider=...)`，
  默认模板实现；B/C 阶段记忆检索可复用同机制（LLM 前缀版后续按需）。

### A3 BM25 稀疏检索 + RRF 混合融合

**新文件**（算法层，零模型依赖）：

- `paperpilot/retrieval/bm25.py`：`tokenize()`（lowercase + `[a-z0-9]+` 英文词 +
  CJK bigram，应对中文 query）、`BM25Index`（k1=1.5, b=0.75，IDF 加 1 平滑），
  `search(query, top_k)`。
- `paperpilot/retrieval/fusion.py`：`rrf_fuse(rankings, k=60, top_k)`
  —— score = Σ 1/(k + rank)。

**集成**（`index_manager.py`）：

- `_IndexState` 增加 `bm25` 与 `chunk_meta`；懒加载与冷建时从 chunks.json（含前缀
  文本）构建 BM25，内存态，无需重建 PLAID。
- `search(query, paper_id, top_k, mode="hybrid")`：
  - `dense`：现行为不变（score = ColBERT 分）；
  - `hybrid`（默认）：稠密、稀疏各取 `max(4*top_k, 20)` 候选 → RRF → 截断 top_k；
    返回条目增加 `dense_rank` / `sparse_rank` / `fused_rank`。
- `server.py`：`search` MCP 工具增加可选 `mode` 参数（默认 `hybrid`）；
  `planned_retrieval` 经既有 search 回调自动获得 hybrid，不改签名。
- deep_reading 的 `retrieve_paper_evidence` 工具面**暂不**暴露 mode（默认走
  hybrid），减少行为面变化。

### A4 检索评测集（recall@k / MRR）

**新文件**：

- `paperpilot/retrieval/metrics.py`：`recall_at_k`、`mrr`、`ndcg`。
- `tests/fixtures/retrieval_eval/corpus.jsonl`：约 15–20 个合成论文 chunk，
  刻意包含上下文歧义场景（「该模型」「本节」等指代脱离前缀后无法消解）。
- `tests/fixtures/retrieval_eval/queries.jsonl`：query → 相关 chunk id 标注。
- `tests/retrieval/test_retrieval_quality.py`：
  - BM25-only recall@5 / MRR 达标（合成语料阈值取高值）；
  - **前缀消融**：同一批歧义 query，无前缀 BM25 vs 有前缀 BM25 的 recall 对比
    （前缀版 ≥ 无前缀版，验证上下文感知检索价值，不依赖 ColBERT 模型）；
  - RRF 融合：受控 fake 稠密排名 + 真实 BM25 → 融合排序合理性。
- ColBERT 真模型端到端检索评测挂现有 smoke 体系（需模型环境，不在单测阻塞）。

### A 阶段文件清单

```
paperpilot/retrieval/bm25.py            新增
paperpilot/retrieval/fusion.py          新增
paperpilot/retrieval/metrics.py         新增
paperpilot/retrieval/__init__.py        导出
paperpilot/mcp_servers/colbert/index_manager.py  改（分块/前缀/混合/持久化v2）
paperpilot/mcp_servers/colbert/server.py         改（mode 参数）
tests/retrieval/test_bm25.py            新增
tests/retrieval/test_fusion.py          新增
tests/retrieval/test_metrics.py         新增
tests/retrieval/test_retrieval_quality.py 新增
tests/fixtures/retrieval_eval/*.jsonl   新增
tests/mcp_servers/test_index_manager.py 改（v2 断言 + 新行为）
```

---

## B 阶段设计要点（开工前单独出细化计划并确认）

- **表** `user_memories`：append-only（Mem0 v3 思路，只 ADD 不物理删），
  `user_id` 隔离（FK users）、`kind`（preference / fact / project / paper_note…）、
  `content` + 结构化 `context` JSON、来源三元组（conversation/task/message）、
  `status`（active/archived）、`superseded_by`（软引用版本链）、时间戳。
- **提取管线**：发布成功 / turn archive narrative 完成后投递 Celery 异步任务：
  LLM 按「选择性 / 抽象化 / 结构化」三规则提取候选 → 规则核验（候选必须携带
  原文支撑 span，无则丢弃）→ 只 ADD 写入；同 kind 高重叠旧记忆标 archived。
- **读取（已确认：Agent 工具）**：`research_agent` 注册 `search_user_memory(query)`
  工具，服务端混合检索（复用 A3 的 BM25 + 时间衰减信号）返回带时间戳与来源的
  记忆条目；工具描述引导 agent 在涉及用户历史/偏好时调用。记忆内容按数据对待
  （来源标记），不作为指令。
- **评估**：三层次评估集（基础回忆 / 多会话检索 / 主动服务）挂 smoke，
  起步先建 layer1 + layer2 用例。
- **风险**：涉及 migration 与发布链路行为变化 + 额外 LLM 成本，须先确认再动。

## C 阶段概述

- 用户研究画像卡片（kind=profile 的聚合视图，定期整理任务生成）常驻注入
  system prompt（上限约 300 token）——「概览层」。
- 会话历史上下文感知检索（复用 A2 前缀机制索引历史 turn archive）——「细节层」。
- 定期整理任务：去重合并、回到原始证据核查、冲突场景限定（对应书 3.3.3）。

## 验证方式

- A 阶段：`pytest tests/retrieval tests/mcp_servers -x`，随后全量 `pytest`；
  评测集指标（recall@5 / MRR / 前缀消融对比）在 CI 可跑、不依赖模型。
- 未覆盖项：ColBERT 真模型下的混合检索端到端效果（需模型环境，挂 smoke 或
  手动验证，A 阶段收尾时明确标注）。

## 风险与待确认项

1. **默认 hybrid 改变线上检索排序**：保留 `mode="dense"` 回退；靠 A4 评测 +
   全量测试护栏。用户已选方案 A（含混检），视为已确认方向，默认值在收尾汇报
   中显式说明。
2. **BM25 中文分词**：无分词库，CJK bigram 起步，不引新依赖；中文 query 的
   关键词召回可用，语义召回仍靠稠密路。
3. **旧 v1 索引**：继续可读可搜（无前缀、无 meta，BM25 从原文构建）；新构建
   才产 v2。不主动迁移、不删除旧索引目录。
4. **fake tokenizer 行为调整**：A1 需改现有测试 fixture 的 encode 返回策略，
   属测试基建调整，随 A1 一并提交并说明。
