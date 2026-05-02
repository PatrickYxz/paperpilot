# PaperPilot graph-mcp 设计 (Day 8)

| 项 | 值 |
|---|---|
| 日期 | 2026-05-02 (方案 C Day 7 完工后, 设计 Day 8 graph-mcp) |
| 范围 | (1) 新建第三个 MCP server `graph-mcp` (NetworkX 引用关系图); (2) 接 Semantic Scholar Graph API 拉 references / citations; (3) 暴露 4 个 tool: `build_graph` / `get_neighbors` / `get_shortest_path` / `get_common_citations`; (4) pickle 持久化到 `data/graph/citation_graph.pkl`; (5) `mcp_servers.json` + 依赖 + smoke |
| 不在范围 | 个人长期 papers 库 ETL; PDF 内 reference 抽取; centrality / find_papers / export_subgraph tool; 增量 refresh 已存节点; HTTP cache 层; 多 graph 命名空间; 前端可视化 |
| 状态 | Draft, 待用户 review |

---

## 1. 目标

让 LLM 在 main loop 里能做"基于引用拓扑的学术导航":

1. 用 `arxiv.search_papers` + `arxiv.download_paper` 撒网下载一批 paper
2. 用 `graph.build_graph(arxiv_ids=[...])` 一次性把这批 paper 的引用关系拉进 NetworkX 图 (含一跳邻居: 双向 references + citations)
3. 用 `graph.get_neighbors(arxiv_id, direction)` 查某 paper 引了什么 / 被谁引
4. 用 `graph.get_shortest_path(from_id, to_id)` 找两篇 paper 之间的引用桥梁
5. 用 `graph.get_common_citations(arxiv_ids=[...])` 找一批 paper 共同引用的 ground-truth 经典文献

这是 colbert (语义) 和 arxiv (关键词) 都做不到的能力 —— graph-mcp 的存在理由。

同时严守方案 C / Day 5 锁定的红线:

- **L1 基座层 + Day 5 mcp_client 零改动** —— 不动接口, 不动超时常量
- **server 之间不互通信** —— graph-mcp 不知道 arxiv-mcp / colbert-mcp 存在; 串联只在 LLM 那一层
- **决策由 LLM 做** —— graph-mcp 不内置重试 / 退避 / 降级 / 路由分支; LLM 看到错误自己决定下一步
- **不做推测性抽象** —— 不留多 graph 口子, 不预留 SQLite 后端, 不抽 SS client 公共类
- **复用 Day 7 trip wire** —— 4 个 tool 全返 `list[dict]` 与 colbert.search 风格一致, 多 JSON 对象拼接监控不松

---

## 2. 关键设计决策 (Q1-Q7)

| ID | 决策 | 选项 | 主要理由 |
|---|---|---|---|
| Q1 | 做哪个 server | **graph-mcp** (vs personal-papers / system prompt) | 5 server 架构里差异化最强 (引用拓扑); personal-papers 与 colbert 重合; system prompt 是 1h 内的 trial-and-error 不该走 spec 流程 |
| Q2 | 引用关系数据源 | **Semantic Scholar Graph API** | arxiv_id 直接映射 (`references[].externalIds.ArXiv`), 与 PaperPilot 现有 ID 体系无缝对齐; 学界主流 SS / OpenAlex 选 SS; PDF 抽取 (GROBID/refextract) 准确率坑深, 4-6h 预算扛不住 |
| Q3 | graph 范围 | **已下载 paper + 一跳双向邻居** (轻量节点, references + citations) | a 仅下载 paper 节点稀疏, centrality / shortest_path 退化; b 一跳邻居 ~1.5k 节点让图算法真有意义; c 仅 references 失去"谁引了我读的" 学术工作流; d 完全 dynamic 失去 NetworkX 价值 |
| Q4 | build 触发 | **`build_graph(arxiv_ids: list)` 批量 tool, 增量语义** (重复 id 跳过) | 解耦 (arxiv-mcp 不知 graph-mcp 存在); LLM 决策权 (与 memory feedback 一致); SS `paper/batch` POST endpoint 单次 500 paper, 比单 paper 散调省 N 倍 latency |
| Q5 | tool 集 | **4 个: build_graph + get_neighbors + get_shortest_path + get_common_citations** | shortest_path 是 graph-mcp 差异化卖点; common_citations 对学术综述场景价值最高; centrality / find_papers / export 第一版用不上 (YAGNI) |
| Q6 | 持久化 | **pickle on every build_graph** at `data/graph/citation_graph.pkl`, atomic rename, 启动 load | 跨 session 复用图; build_graph 是 expensive (网络 5s+) 不该重跑; 1.5k 节点 pickle <2MB / write <100ms; SQLite 对单 graph 单进程过设计 |
| Q7 | API key + 失败处理 | **`SEMANTIC_SCHOLAR_API_KEY` env (可空)**, 5 类失败规则 (见 §7) | key 空也工作 (unauth shared pool 100req/5min, demo 50 paper 一次 batch 远低于阈值); 限速/超时/5xx exp backoff retry 3 次; missing paper / 无 arxiv_id 邻居 skip + 报告 |

### 隐含决策 (已锁)

| 项 | 值 | 备注 |
|---|---|---|
| 邻居节点元数据 | `{arxiv_id, title, year, authors}` | 不存 abstract / pdf_url, 节点轻量; 内存 < 5MB |
| 节点角色 (`node_type` 属性) | `'downloaded'` (build_graph 主节点) 或 `'neighbor'` (一跳邻居; 仅由其它主节点的 SS 返回引出) | 区分"我读过的" vs "拓扑里出现的相邻文献"; 决定 build_graph 增量策略 (见 §5 行为 2) |
| 节点 ID 体系 | arxiv_id 作为 NetworkX node key | 与 PaperPilot 全局 ID 一致; SS 返回的 reference 没 arxiv_id 则 skip |
| 边方向 | 有向: `A → B` 表示 "A cites B" | NetworkX `DiGraph` |
| SS endpoint | `POST /graph/v1/paper/batch` | 单次 500 paper; fields=`references.externalIds,references.title,references.year,references.authors,citations.externalIds,citations.title,citations.year,citations.authors,title,year,authors,externalIds` |
| HTTP client | `httpx` (sync) | requests 也行, 但 PaperPilot 后续可能用 async, httpx 更通用; 不引入 async pipeline (YAGNI) |
| Graph mutation 并发 | 单线程同步, 无锁 | MCP server 单进程同步处理 tool call, 没有并发 mutation |

---

## 3. 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  Main Loop  (agent_loop.py — Day 4 落地, Day 8 不动)                 │
│      │                                                               │
│      │ tool_use blocks                                               │
│      ▼                                                               │
│  MCPClient  (Day 5 落地, Day 8 不动 — 含 Day 6 改的 180s 全局超时)   │
│      │                                                               │
│      ├──► arxiv-mcp 进程   (Day 5/6 已建)                            │
│      │       └── search_papers / download_paper                      │
│      │                                                               │
│      ├──► colbert-mcp 进程  (Day 6 已建)                             │
│      │       └── build_index / search                                │
│      │                                                               │
│      └──► graph-mcp 进程   ← Day 8 新建                              │
│              └── build_graph         ← 拉 SS Graph API               │
│              └── get_neighbors       ← NetworkX successors/predecessors │
│              └── get_shortest_path   ← NetworkX shortest_path        │
│              └── get_common_citations ← 集合交集 + 度数排序          │
│              (内部: NetworkX DiGraph + httpx → Semantic Scholar)     │
│              (持久化: data/graph/citation_graph.pkl)                 │
└──────────────────────────────────────────────────────────────────────┘

  data/graph/citation_graph.pkl    ← pickle, 每次 build_graph 完原子 rename
```

**三个 server 进程互不通信**。串联只在 LLM 那一层 (LLM 拿 arxiv search → 决定下载 → 决定 build_graph → 决定查 neighbors)。这是 MCP 架构核心边界。

---

## 4. 文件布局

### 新增 / 改动

| 路径 | 状态 | 作用 | 预估行数 |
|---|---|---|---|
| `paperpilot/mcp_servers/graph/__init__.py` | 新 | 空文件 | 0 |
| `paperpilot/mcp_servers/graph/server.py` | 新 | MCP 协议层 (FastMCP); 4 个 tool handler; **不接触 NetworkX / httpx** | ~100 |
| `paperpilot/mcp_servers/graph/graph_manager.py` | 新 | 唯一接触 NetworkX 的地方; `build()` / `get_neighbors()` / `get_shortest_path()` / `get_common_citations()` / `_load()` / `_save()` | ~140 |
| `paperpilot/mcp_servers/graph/ss_client.py` | 新 | 唯一接触 httpx + Semantic Scholar 的地方; `fetch_papers_batch(arxiv_ids: list) -> list[dict]` + retry/backoff | ~80 |
| `paperpilot/mcp_servers.json` | 改 | + graph entry | +5 |
| `requirements.txt` | 改 | + `networkx>=3.2`, `httpx>=0.27` | +2 |
| `tests/fixtures/ss_attention_bert.json` | 新 | Attention + BERT 两 paper 的 SS batch API response 快照 (一次手抓, mock 用; 单测取 list 索引即可分别测两 paper 场景) | ~150 |
| `tests/mcp_servers/test_graph_server.py` | 新 | server.py 协议层单测 (mock graph_manager) | ~100 |
| `tests/mcp_servers/test_graph_manager.py` | 新 | graph_manager 单测 (mock ss_client; 真跑 NetworkX) | ~120 |
| `tests/mcp_servers/test_ss_client.py` | 新 | ss_client 单测 (mock httpx; 测 retry/backoff/missing 行为) | ~100 |
| `tests/mcp_servers/test_graph_via_client.py` | 新 | 集成测试 (mark slow); 真起 graph-mcp 进程; mock SS via env var (见 §8) | ~80 |
| `scripts/day8_smoke.py` | 新 | 端到端冒烟 (search → download 3 篇 → build_graph → query); 顶部 `sys.stdout.reconfigure(encoding="utf-8")` | ~60 |

### 不动

- `paperpilot/core/*.py` — 一行不改
- `paperpilot/agent/loop.py` — 一行不改
- `paperpilot/tools/mcp_client.py` — 一行不改 (Day 6 已设 180s 全局超时, build_graph SS batch <30s 不撞)
- 现有 arxiv-mcp / colbert-mcp 代码 — 一行不改

### 布局决策

1. **graph-mcp 内部分三层** (`server.py` / `graph_manager.py` / `ss_client.py`)
   - `server.py` 不 import networkx / httpx, 纯协议层; 单测 mock `graph_manager` 即可, 跑得飞快
   - `graph_manager.py` 唯一接触 NetworkX, 唯一管 pickle 持久化
   - `ss_client.py` 唯一接触 httpx + SS API, retry/backoff 全在这一层
   - 单测可以分别 mock 上层依赖, 隔离失败维度
2. **不做 manifest.json 单文件** —— 与 colbert-mcp / arxiv-mcp 保持一致 (这两个 server 实际代码也只用 `mcp_servers.json` 配置, manifest.json 早期 spec 写了但代码没建)
3. **`data/graph/` 路径与 `data/colbert_index/` / `data/papers/` 同级** —— 命名一致, 一眼看出归属
4. **fixture 用真实 SS API response 快照** —— 一次性手抓 (curl + jq pretty-print 落 .json), 之后单测 mock httpx 直接返 fixture; 比构造假数据更可信, 避免"测我的假数据" 反模式

---

## 5. Tool 签名

### `graph.build_graph` (新增)

```
Input:
  arxiv_ids: list[str]   # 形如 ["2401.12345", "2403.67890"]; 不接受带版本后缀

Output:
  {
    "added_count": int,        # 本次因 input 新建为主节点的数量 (含从 neighbor 升格 + 全新)
    "skipped_count": int,      # 因已是 downloaded 主节点而跳过的数量
    "missing": list[str],      # SS 查不到的 arxiv_id (LLM 看见后可决策, 例如不去 download)
    "total_nodes": int,        # build 后图的总节点数 (含一跳邻居)
    "total_edges": int         # build 后图的总边数
  }

行为:
  1. 校验 arxiv_ids 非空、每项 str (空 list / 非 str → ValueError)
  2. 分流 input ids:
       a. 已在图中且 node_type='downloaded' → 加入 skipped 列表, 不调 SS (重复 build 无意义)
       b. 已在图中但 node_type='neighbor' → 视作"升格请求", 与新 ids 一起调 SS;
          之所以不 skip: LLM 常见 workflow 是"先看 get_neighbors 找到 interesting 邻居 → 再 build_graph 深挖", 这时该 id 在图里是 neighbor 但应升格为 downloaded
       c. 不在图中的新 id → 与 b 一起调 SS
  3. 上述 b + c 调 ss_client.fetch_papers_batch(ids)
       SS POST /graph/v1/paper/batch 单次最多 500, 这里 ids 一般 <50 一次过
       SS 对未找到的 paper 在返回 list 对应位置返 null
  4. 每个非 null 返回的 paper:
       a. 加 / 升格主节点 (node_type='downloaded', arxiv_id, title, year, authors;
          若该 id 之前是 'neighbor' → 改 type 不删边, 已存边保留)
       b. 遍历 references[]:
            ref_arxiv_id = r["externalIds"].get("ArXiv")
            if ref_arxiv_id 且不在图中: 加邻居节点 (node_type='neighbor', ...)
            if ref_arxiv_id: add_edge(主, ref) — 已存边自动幂等不重复
            else: skip 该 reference
       c. 同理处理 citations[] (邻居节点 + 边 cite_paper→主)
  5. SS 返回 null 的位置对应的 input id (paper 不在 SS 库) → 加入 missing
  6. 持久化: pickle.dump(G, "citation_graph.pkl.tmp") + os.replace 到正式名
  7. 返回统计 (added_count 仅计 type 真从无到有的主节点; 升格的算 added_count, skipped_count 仅计 type='downloaded' 重复)
```

### `graph.get_neighbors` (新增)

```
Input:
  arxiv_id: str
  direction: str = "both"     # "references" | "citations" | "both"
  limit: int = 10

Output:
  list[{"arxiv_id": str, "title": str, "year": int|null, "authors": list[str], "edge": str}]
    edge: "references" 或 "citations" (告诉 LLM 这个邻居与 input 的关系方向)

行为:
  1. 若 arxiv_id not in self._graph → raise NodeNotFoundError("must call build_graph first or paper not in graph")
  2. direction=='references': successors (out-edges)
     direction=='citations':  predecessors (in-edges)
     direction=='both': successors + predecessors
  3. 按节点 year 倒序 (新 paper 优先), tie-break 按 arxiv_id 字典序; 取前 limit
  4. 每个邻居拼 dict 返回; edge 字段标方向
```

### `graph.get_shortest_path` (新增)

```
Input:
  from_id: str
  to_id: str

Output:
  {
    "path": list[{"arxiv_id": str, "title": str}],   # 从 from 到 to 的引用链, 含两端
    "length": int                                    # 边数 = len(path) - 1
  }
  或当无路径:
  {"path": [], "length": -1}

行为:
  1. 若 from_id 或 to_id 不在图 → raise NodeNotFoundError
  2. nx.shortest_path(self._graph, source=from_id, target=to_id)
     无路径捕 nx.NetworkXNoPath → 返 {"path": [], "length": -1} (不抛错; LLM 自己解读)
  3. 把每个节点拼 dict (含 title); 按路径顺序返回
```

### `graph.get_common_citations` (新增)

```
Input:
  arxiv_ids: list[str]   # 一批 paper, 求共同引用的下游
  top_k: int = 10

Output:
  list[{"arxiv_id": str, "title": str, "year": int|null, "authors": list[str], "cited_by_count": int}]
    cited_by_count: 在 input arxiv_ids 中, 有多少篇引用了它 (1..len(arxiv_ids))

行为:
  1. 校验 arxiv_ids 非空且 >=2 (单 paper 算共同没意义 → ValueError)
  2. 对每个 input id:
       若不在图 → skip 该 id (不抛错; 当作没这篇 paper 的引用信息)
       否则取 successors (它 references 的所有节点)
  3. 统计所有 successors 的出现频次 (Counter)
  4. 过滤 cited_by_count >= 2 (至少被 input 中 2 篇 paper 共引才有意义)
  5. 按 cited_by_count 倒序, tie-break 按 year 倒序; 取前 top_k
  6. 拼 dict 返回
```

### 输出统一约定 (复用 Day 7 trip wire)

所有 4 个 tool 的输出**全为 list[dict] 或含 list[dict] 字段的 dict** —— 与 `colbert.search` 风格一致, FastMCP 多 JSON 对象拼接监控同套校准生效。spec §10 trip wire 章节明确把 graph-mcp 也纳入监控范围。

---

## 6. 数据流 (端到端时序)

```
用户: "RAG 评估这块, attention 论文跟最新 retrieval-augmented 工作有引用关系吗?"

[1] agent_loop → Claude API (prompt + 8 个 tool schema: arxiv x2 + colbert x2 + graph x4)
[2] Claude → tool_use(arxiv__search_papers, q="attention is all you need")
      → 返 metadata, 拿到 1706.03762 (Attention)
[3] Claude → tool_use(arxiv__search_papers, q="retrieval augmented generation 2024")
      → 拿到 2-3 个最新 RAG 论文 arxiv_id
[4] Claude → tool_use(arxiv__download_paper) ×3-4
      → mcp_client 串行下载, ~30s
[5] Claude → tool_use(graph__build_graph, arxiv_ids=["1706.03762", <RAG ids>])
[6] mcp_client → graph-mcp:
      ss_client.fetch_papers_batch(4 个 ids) → 1 次 SS POST batch (~3-5s)
      graph_manager.build():
        加 4 个主节点 + ~120 个一跳邻居节点 + 边
        pickle.dump → atomic rename
      返 {added_count: 4, skipped_count: 0, missing: [], total_nodes: 124, total_edges: 240}
[7] Claude → tool_use(graph__get_shortest_path, from_id="1706.03762", to_id=<某 RAG paper>)
      → 几毫秒, 返 path: [Attention, intermediate paper, RAG paper]
[8] Claude → tool_use(graph__get_common_citations, arxiv_ids=[<3 个 RAG ids>], top_k=5)
      → 返 5 篇被这批 RAG 论文共同引用的 ground-truth 经典文献
[9] Claude 看引用链 + 共引经典 → 给用户结构化回答
[10] agent_loop 没 tool_use, 结束
```

**两个关键点**:
- 第 [6] 步 SS batch API 一次 ~3-5s, 远低于 mcp_client 180s 全局超时 (Day 6 已设)
- 第 [7]-[8] 步图查询毫秒级, 网络成本仅在 build_graph 那一次

---

## 7. 错误处理

按 Day 5 红线分三类:

### A. 启动期 hard-fail (进程起不来 → mcp_client startup 失败 → main 异常退出)

| 失败 | 触发 |
|---|---|
| `import networkx` / `import httpx` 失败 | 依赖装漏 |
| `data/graph/` 不可写 (权限错) | 启动时尝试 `mkdir -p` 失败 |
| 已存 `citation_graph.pkl` corrupt (pickle.load 抛) | 文件损坏 |

**启动期还会做 (不是 fail, 是 setup)**:
- `mkdir -p data/graph/`
- 尝试 `pickle.load("citation_graph.pkl")`; 不存在 → 空图开始 (info log, 正常); corrupt → hard-fail (要求用户手动删文件再重启, 而非静默丢弃数据)

理由: 配置 / 环境 bug, LLM 看到也无能为力; corrupt 静默重置图等于丢数据, 比报错更糟。

### B. 运行时 soft-fail (tool raise → mcp_client 转 `is_error: true` → LLM 决策)

| 失败点 | 异常类型 | LLM 看到啥 |
|---|---|---|
| build_graph: arxiv_ids 空 / 非 str | `ValueError` | "输入有问题" |
| build_graph: SS 网络挂 (urllib/httpx error) | `SSAPIError` (自定义, wrap httpx.HTTPError) | "SS 服务挂了, 我等会再 build_graph" |
| build_graph: SS 限速 + 重试耗尽 | `SSRateLimitError` | "SS 限速了, 退一退" |
| build_graph: SS 5xx + 重试耗尽 | `SSAPIError` | "SS 服务故障" |
| get_neighbors / get_shortest_path: arxiv_id 不在图 | `NodeNotFoundError` | "你没 build 这个 paper, 先去 build_graph" |
| get_common_citations: <2 个 input | `ValueError` | "输入有问题" |
| get_common_citations: 所有 input 都不在图 | 返 `[]` (不抛错) | LLM 看到空列表自己解读 |

**全部不做**: 重试无穷次、无脑 fallback 到其它数据源、降级到 PDF 抽取。SS 抓不到就让 LLM 知道并决策。

### C. SS retry 策略 (ss_client 内部, 唯一例外的"重试")

| 触发 | 行为 |
|---|---|
| HTTP 429 (rate limit) | exp backoff: 2s / 4s / 8s, max 3 次; 仍失败 → `SSRateLimitError` |
| HTTP 5xx (服务端) | 同 429 | exp backoff retry 3 次; 仍失败 → `SSAPIError` |
| Network timeout (单 request 30s) | 同 429 | retry 3 次 |
| HTTP 4xx 非 429 (e.g. 404 整个 batch 不该出现, 真出现说明请求构造错) | 不 retry, 立即抛 `SSAPIError` (这是 bug, 不是网络抖) |

理由: retry 在 ss_client 内部封, **graph-mcp 业务层不感知**。LLM 看到的是"SS 服务真挂了" (重试 3 次后), 不是"我自己代码 bug"。这与 Day 5 红线"决策由 LLM 做"不冲突 —— 这是网络层的标准抗抖动, 不是业务层的"自动找替代源"。

### D. SS 数据级别的"软" miss (不算错误)

| 场景 | 处理 |
|---|---|
| 某 input arxiv_id 不在 SS 数据库 | build_graph 返回值 `missing` 列表里列出, 不抛错 |
| SS 返回的 reference / citation 节点没 arxiv_id (只有 SS internal id 或 DOI) | skip 该邻居节点, 不抛错; graph 只装有 arxiv_id 的节点 |

理由: "paper 不在 SS" 是数据现实, 不是错误; 让 LLM 看到 missing 列表自己决定要不要换 paper。"邻居没 arxiv_id" 是 SS 数据噪音, skip 维持 PaperPilot 全局 ID 体系一致。

---

## 8. 测试策略

### 层 1: 单元测试 (快, CI 跑, 秒级)

| 文件 | 测什么 | 怎么测 |
|---|---|---|
| `test_ss_client.py::test_fetch_papers_batch_happy` | 正常返 SS response | mock `httpx.Client.post` 返 fixture json; 断言解析正确 |
| `test_ss_client.py::test_fetch_429_retry_then_success` | 限速重试成功 | mock 第一次 429, 第二次 200; 断言总调用 2 次, sleep 2s |
| `test_ss_client.py::test_fetch_429_retry_exhaust` | 重试耗尽 | mock 一直 429; 断言抛 `SSRateLimitError`, 总调用 3 次 |
| `test_ss_client.py::test_fetch_5xx_retry` | 5xx 重试 | mock 第一次 503, 第二次 200; 断言成功 |
| `test_ss_client.py::test_fetch_4xx_no_retry` | 4xx 立即抛 | mock 400; 断言总调用 1 次 (无 retry) |
| `test_ss_client.py::test_fetch_no_api_key_uses_unauth` | 无 key 行为 | env 不设 SEMANTIC_SCHOLAR_API_KEY; 断言 request 不带 x-api-key header |
| `test_ss_client.py::test_fetch_with_api_key_sets_header` | 带 key | env 设 fake key; 断言 request 带 x-api-key header |
| `test_graph_manager.py::test_build_adds_main_and_neighbors` | build 主+邻 | mock ss_client 返 2 papers (含 references / citations); 断言图含 2 主 + N 邻居 + 边 |
| `test_graph_manager.py::test_build_skips_existing` | 增量 | 连续 build 同 id 两次; 断言第二次 skipped_count 含该 id |
| `test_graph_manager.py::test_build_persists_pickle` | 持久化 | tmpdir; build 后断言 pickle 文件存在; load 回来节点数一致 |
| `test_graph_manager.py::test_build_pickle_atomic` | atomic | mock os.replace 失败前先确认 .tmp 存在; 断言正式名未被半写覆盖 |
| `test_graph_manager.py::test_load_corrupt_raises` | 损坏文件 | 写入坏 bytes 到 pickle; 断言 GraphManager init 抛 `GraphCorruptError` (hard-fail) |
| `test_graph_manager.py::test_get_neighbors_directions` | 邻居方向 | 建 1 主 + 2 ref + 2 cite; 测 references / citations / both 各返回正确节点 |
| `test_graph_manager.py::test_get_neighbors_node_missing` | 节点不存在 | 查不在图的 id; 断言抛 `NodeNotFoundError` |
| `test_graph_manager.py::test_get_neighbors_limit_year_sort` | 排序 | 5 邻居 year 不同; 断言按 year 倒序取前 limit |
| `test_graph_manager.py::test_shortest_path_happy` | 路径 | A→B→C 链; 查 A→C 返 [A, B, C], length=2 |
| `test_graph_manager.py::test_shortest_path_no_path` | 无路径 | 两个连通分量; 断言返 {path: [], length: -1} |
| `test_graph_manager.py::test_common_citations` | 共引 | 3 paper 共引 1 ground-truth; 断言 cited_by_count=3 |
| `test_graph_manager.py::test_common_citations_min_2` | <2 input | 单 paper 输入; 断言抛 ValueError |
| `test_graph_manager.py::test_common_citations_all_missing` | 全部 input 不在图 | 断言返空列表 (不抛错) |
| `test_graph_server.py::test_build_routes_to_manager` | 协议层 build 路由 | mock graph_manager.build; 断言被调用 + 参数正确 |
| `test_graph_server.py::test_build_empty_input_validation` | 协议层校验 | 传 [] 或 [123]; 断言抛 ValueError |
| `test_graph_server.py::test_get_neighbors_routes` | 协议层 neighbors 路由 | mock graph_manager; 断言路由 |
| `test_graph_server.py::test_get_shortest_path_routes` | 协议层 path 路由 | 同上 |
| `test_graph_server.py::test_get_common_citations_routes` | 协议层共引路由 | 同上 |

**Fixture**: `tests/fixtures/ss_paper_attention.json` + `ss_paper_bert.json`
- 一次性手抓 (实际 fields 列表见 §2 隐含决策):
  ```
  curl -X POST \
    -H "Content-Type: application/json" \
    -d '{"ids":["ARXIV:1706.03762","ARXIV:1810.04805"]}' \
    "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,year,authors,externalIds,references.externalIds,references.title,references.year,references.authors,citations.externalIds,citations.title,citations.year,citations.authors" \
    | python -m json.tool > tests/fixtures/ss_attention_bert.json
  ```
  (注: SS batch 一次返多 paper, 实际可一个文件存两 paper response, 测试时按 index 取)
- 单测 mock httpx 直接返 fixture, **不真发 SS 请求** (CI 不能依赖外网 + SS 限速)

### 层 2: 集成测试 (中速, ~5-10s/case, 标 `@pytest.mark.slow`)

| 文件 | 测什么 |
|---|---|
| `test_graph_via_client.py::test_build_and_query` | 真起 mcp_client + graph-mcp; mock SS via `PAPERPILOT_SS_FIXTURE_DIR` env (graph-mcp 启动时若该 env 设, ss_client 改读 fixture 不发真 HTTP); build_graph 4 papers; 验证 get_neighbors / shortest_path / common_citations 全跑通 |
| `test_graph_via_client.py::test_pickle_persists_across_restart` | 同进程内 build → close → 重起 graph-mcp; 断言图 load 回来, get_neighbors 仍能查到 |
| `test_graph_via_client.py::test_corrupt_pickle_startup_fail` | 预写坏 pickle 到 data/graph/citation_graph.pkl; 启动 graph-mcp; 断言 client startup 抛错 |

**为啥不真发 SS 请求**: CI 限速 + 网络抖动 + SS API 演化都会让集成测试不稳。fixture mode 让"集成测试"测的是"server 进程 + MCP 协议 + graph_manager 全链路", 而非"SS API 还活着"。后者交给 day8_smoke 端到端验证。

**为啥用 env 切换 fixture mode 而非 monkeypatch**: graph-mcp 是子进程, monkeypatch 不跨进程。env 是进程间唯一干净通道。

**Slow 标记**: 本地 `pytest -m slow`; CI 默认跳。

### 层 3: Day 8 smoke (`scripts/day8_smoke.py`, 端到端真实 LLM + 真实 SS)

完工标志。

```
prompt: "搜 attention 和 BERT 这两篇经典 paper, 下载, 用 graph 工具看它们之间有没有
        引用关系, 顺便看 BERT 引用了哪些其它 paper"

期望路径:
  arxiv__search_papers ×2 → arxiv__download_paper ×2
  → graph__build_graph(arxiv_ids=[<attention>, <BERT>])
  → graph__get_shortest_path(from=BERT, to=attention) — BERT 引 attention, 期望 length=1
  → graph__get_neighbors(arxiv_id=BERT, direction="references", limit=10)
  → LLM 输出含 "BERT cites Attention is all you need" 之类的引用关系陈述

完工: 退出码 0 + stdout 含 "✅ Day 8 smoke PASSED"
```

顺手做:
- `sys.stdout.reconfigure(encoding="utf-8")` 加到 day8_smoke 顶部
- README 加一行: "graph-mcp 需要 SEMANTIC_SCHOLAR_API_KEY env (可空, unauth 也工作)"

---

## 9. 工作量预估 + 完工标志

| 阶段 | 预估 |
|---|---|
| 写 plan | ~30 min |
| Task 1: ss_client.py + fixture 抓 + 单测 (含 retry/backoff/auth) | ~60 min |
| Task 2: graph_manager.py + 单测 (build/persist/queries) | ~75 min |
| Task 3: server.py 协议层 + manifest + mcp_servers.json + 协议单测 | ~45 min |
| Task 4: 集成测试 (test_graph_via_client.py + fixture mode env 切换) | ~30 min |
| Task 5: scripts/day8_smoke.py + 端到端联调 (含真 SS API + 真 LLM) | ~40 min |
| **合计** | **~4.5 小时** (预算 6 小时, 留 1.5 小时 buffer 给首次 SS API 接入踩坑 + LLM tool decision) |

**完工标志 (Definition of Done)**:
1. `pytest tests/` 全绿 (含 graph 三个新 test 文件; slow 测试本地手跑)
2. `python scripts/day5_smoke.py` 无回归
3. `python scripts/day6_smoke.py` 无回归
4. `python scripts/day8_smoke.py` 退出 0 + 打印 PASSED
5. `data/graph/citation_graph.pkl` 存在 (smoke 跑后)
6. `git grep "TODO\|FIXME" paperpilot/mcp_servers/graph/` 空
7. 所有改动按合理粒度分 commit

---

## 10. trip wire (Day 7 监控复用)

graph-mcp 4 个 query tool 输出全为 list[dict] 或含 list[dict], 与 colbert.search 同结构。Day 7 锁的 trip wire **同套生效**, 但**新增以下监控条件**:

| 触发条件 | 行动 |
|---|---|
| 换 LLM 主模型 (DeepSeek → 其它) | 重跑 `scripts/day7_fastmcp_repro.py` (现有, 测 colbert.search 多 JSON 拼接); graph 端复用同样 trip wire 假设, 触发时若发现 graph 输出真有解析问题再写 `day8_graph_repro.py` |
| 单次 graph query 返 >10 邻居 | 同上 |
| 已下载 paper 数 >50 / 一跳邻居总节点 >5000 | 同上 + 检查 pickle 大小是否 >50MB (内存压力) |

第一版**不写** day8_graph_repro 脚本 —— smoke 通过即视为 baseline 稳定; 只保留触发条件作为"未来某天换模型 / 数据量级跳变" 的提醒钉子, 真触发时再写。

---

## 11. 不在范围 / 推迟到未来的事

| 项 | 推迟到何时触发 |
|---|---|
| 个人长期 papers 库 ETL pipeline (personal-papers-mcp) | Day 9+ 单独 brainstorm |
| PDF 内 reference 抽取 (GROBID / refextract) 作为 fallback 数据源 | SS 真不够用且实测 < 60% 命中率时 |
| centrality / find_papers / export_subgraph tool | LLM 真在 loop 里需要这些信息做决策时 |
| 增量 refresh 已存节点 (paper 引用关系变化重抓) | 真出现"某 paper SS 数据更新而我们用旧数据" 误导 LLM 案例时 |
| HTTP cache 层 (cachetools / requests-cache) | pickle 落盘后 graph 自身就是缓存; 真出现重复 build 浪费 SS 配额时再加 |
| 多 graph 命名空间 (并存多个 citation_graph) | 单机出现多任务并发需求时 |
| OpenAlex 作为备选数据源 | SS 服务不稳或政策变化时 |
| 前端可视化 (D3.js / Cytoscape) | PaperPilot 真出现 web UI 时 |
| 与 colbert-mcp 联动 (graph 查到的邻居 paper 自动喂 colbert 索引) | LLM workflow 真涌现这个模式时; 第一版让 LLM 自己 download → build_index 串联 |

---

## 附: 与已有 spec 的衔接点

- **复用 Day 5 mcp_client 抽象** —— 新 server 只在 `mcp_servers.json` 加 entry, client 启动逻辑零改动 (含 Day 6 已设的 180s 全局超时, build_graph 单次 SS batch <30s 不撞)
- **复用 Day 5 manifest schema** (`command/args`) —— graph entry 与 arxiv/colbert entry 同形态
- **复用 Day 5 错误传播路径** (handler raise → is_error tool_result → LLM 决策) —— graph-mcp 所有运行时错误 (NodeNotFoundError / SSAPIError / ValueError) 走同一通道
- **复用 Day 5 命名约定** (`mcp__<server>__<tool>`) —— `mcp__graph__build_graph` / `mcp__graph__get_neighbors` / `mcp__graph__get_shortest_path` / `mcp__graph__get_common_citations`
- **复用 Day 6 colbert-mcp 三层布局** (server / manager / 外部依赖封装) —— graph-mcp 同样三层 (server / graph_manager / ss_client)
- **复用 Day 6 fixture-based 单测** —— ss_client 用真 SS API response 快照做 mock, 与 colbert 用真实 PDF fixture 同模式
- **复用 Day 7 trip wire** —— 多 JSON 对象拼接监控同套生效 (§10), 不松绑
