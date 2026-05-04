# Day 12 设计:ColBERT per-paper 索引隔离 + compare-papers skill

> Day 11 用 `THREAD_POOL_SIZE = 1` 把 `paper_deep_read` 子 agent 强制串行,根因是
> ColBERT MCP server 用单一全局索引 `paperpilot_current/`,并行 build 会互相覆写。
> Day 12 把 ColBERT 改成 per-paper 隔离索引,撕掉串行限制,并新增 `compare-papers`
> skill 让 LLM 在多论文对比场景自动触发 `paper_deep_read`,把 Day 11 的架构闭环。

## 目标与范围

- **B 层(架构)**:ColBERT 索引按 paper_id 隔离,每篇一套独立 PLAID 索引;`build_index`
  命中即跳过 + 跨 session 持久化;`search` 加必填入参 `paper_id`;`paper_deep_read`
  的 `THREAD_POOL_SIZE` 提到 `min(3, len(paper_ids))`,撕掉 Day 11 保守注释。
- **A 层(业务)**:新增 `compare-papers` skill,极简版——给定 3-8 个 paper_id,
  调一次 `paper_deep_read` 后综合对比。
- **不做**:候选论文发现 / research_todo 集成 / pdf-parse / VLM / 任何
  `loop.py`、`adapter.py`、`mcp_client.py` 的改动。

## 关键决策(brainstorm 已敲定)

| 项 | 结论 | 理由 |
|---|---|---|
| 索引隔离粒度 | per-paper 目录 `data/colbert_index/<paper_key>/` | paper_id 是稳定逻辑 key,但旧 arXiv ID 可含 `/`;磁盘目录必须经 `_paper_key(paper_id)` 转义,跨 session 可缓存,真并发零锁 |
| `search` 接口 | `paper_id: str`(单值,必填) | 当前 0 个调用点需要多 paper 检索;按"不要推测性抽象"红线选单值 |
| `build_index` 重复行为 | 命中即跳过 | 重复 build 同一 paper 跳过 ColBERT encode(~30-90s),回访同一 paper 演示/调试场景显著省时 |
| 启动期清理 | 不清,持久化 | 配合命中跳过,跨 session 复用 |
| `THREAD_POOL_SIZE` | `min(3, len(paper_ids))` | Day 12 smoke 只需 3 篇即可证明真并发;先降低本地 ColBERT/CPU 压力,稳定后再考虑提到 4 |
| `compare-papers` skill 范围 | 极简(只管 N 篇综合) | 候选发现交给 LLM 自决;skill 越聚焦触发越准;不与 find-classics 重叠 |

## 组件改动

### `paperpilot/mcp_servers/colbert/index_manager.py`(主要重构)

**文件布局**

```
data/colbert_index/
  <paper_key_A>/
    paperpilot_current/      # PLAID 索引目录(沿用 INDEX_NAME 常量)
    chunks.json              # {chunk_id: chunk_text}, search 时回填
  <paper_key_B>/
    ...
```

`INDEX_NAME = "paperpilot_current"` 保留为 PLAID 内部目录名;`INDEX_ROOT = Path("data/colbert_index")` 不变;
新加 helper `_paper_key(paper_id)` 把 `/`、`\`、`:` 等路径字符转成 filesystem-safe key,
例如旧式 arXiv ID `cs/0501001` 不能直接作为目录名;`chunks.json` 内仍保留原始 chunk id / paper_id。
再加 helper `_paper_root(paper_id) -> INDEX_ROOT / _paper_key(paper_id)`、
`_index_path(paper_id) -> _paper_root(paper_id) / INDEX_NAME`、
`_chunks_path(paper_id) -> _paper_root(paper_id) / "chunks.json"`。

**内部状态**

```python
@dataclass
class _IndexState:
    index: indexes.PLAID
    chunk_texts: dict[str, str]

class IndexManager:
    _states: dict[str, _IndexState]
```

**`__init__` 简化**

- 删 `_clear_stale_index`(跨 session 持久化)
- `INDEX_ROOT.mkdir(parents=True, exist_ok=True)`
- 加载 `models.ColBERT`
- `self._states = {}`

**`build(documents)` 三路径(按 paper_id 循环)**

```
for d in documents:
    pid = d["paper_id"]
    if pid in self._states:
        # memory-hit
        emit cached=True for this paper
        continue
    if self._index_path(pid).exists() and self._chunks_path(pid).exists():
        try lazy_load(pid)
        if success:
            emit cached=True for this paper
            continue
        else:
            # fallback 到冷启动(chunks.json 损坏等)
    cold_build(pid, d["text"])
    emit cached=False for this paper
return {"indexed_count": len(documents), "index_name": INDEX_NAME, "cached_papers": [...], "fresh_papers": [...]}
```

`cold_build`:
1. chunk 文本(沿用 256/32 滑窗)
2. encode chunk_texts
3. 构造 `indexes.PLAID(index_folder=str(_paper_root(pid)), index_name=INDEX_NAME, override=True)`(不再多 paper 共用一个 folder,override=True 保护本 paper 旧索引被新版顶掉;`_paper_root` 必须使用 `_paper_key`)
4. `add_documents`
5. dump `chunks.json` 到 `_chunks_path(pid)`(JSON,utf-8)
6. `self._states[pid] = _IndexState(index, chunk_texts)`

`lazy_load`:
1. `indexes.PLAID(index_folder=..., index_name=..., override=False)` 读盘
2. `json.loads(_chunks_path(pid).read_text("utf-8"))` 还原 chunk_texts
3. 写入 `self._states`
4. 任一步失败 → return False(让上层走冷启动 fallback)

**`search(query, paper_id, top_k)`**

1. `state = self._states.get(paper_id)`
2. 没缓存 → 试 lazy_load;还失败 → raise `IndexNotFoundError("no index for <paper_id>; call build_index first")`
3. 用 `state.index` 检索,从 `state.chunk_texts` 回填 chunk_text
4. 返回结构不变(`paper_id` / `chunk_text` / `score`)

**`_release_current_index`**:删除(per-paper 不再需要 override 同一路径)。

### `paperpilot/mcp_servers/colbert/server.py`(MCP schema 调整)

- `search` tool input_schema 加 `paper_id: {"type": "string", "minLength": 1}`,加入 `required`
- `build_index` schema 不变,返回 dict 多 `cached_papers` / `fresh_papers` 字段(纯日志观测用,LLM 不一定看)
- tool description 同步反映"per-paper 索引,跨 session 缓存"

### `paperpilot/builtin_tools/subagent.py`(撕保守限制)

```python
THREAD_POOL_SIZE = 3    # was 1
```

`SUBAGENT_SYSTEM` workflow 第 3 步改为:
```
3. Call mcp__colbert__search(query="...", paper_id="<paper_id>", top_k=5)
```

模块顶部 docstring 删 "Day 11 conservative version" 与 "single global index" 整段,
改成简短"each paper read by an isolated subagent in parallel"。

handler 内 `max_workers = min(THREAD_POOL_SIZE, len(paper_ids))` 已就位,无需改。

工具 description 删 "Day 11 runs workers serially..."句。

为 day12 smoke 提供稳定并发证据,`paper_deep_read` 还需要在每个 subagent worker 的生命周期 emit 两个事件:
- `subagent_start`
- `subagent_done`

这两个事件同样经过现有 `make_sub_emit(paper_id)` 包装,因此 payload 会带 `subagent_paper_id`。smoke 用 subagent 生命周期窗口重叠证明并发,不要用 `mcp__colbert__search` tool_call/tool_result 窗口证明并发,因为 MCP stdio/server 端可能串行化 tool call。

### `paperpilot/skills/compare-papers.md`(新建)

```yaml
---
name: compare-papers
description: 多论文深读对比:用 paper_deep_read 并发精读 3-8 篇,综合对比方法/发现/适用场景
when_to_use: 用户给定 3-8 个 arxiv id 或论文,要求对比、综合或并列分析
---
```

body 步骤:
1. 调 `paper_deep_read(paper_ids=[...], user_query="<原问题>")` 一次。每篇被独立 subagent 精读。
2. 拿到 markdown 结果(含 `### <paper_id>` 多 section)后,综合对比:相同点 / 不同点 / 各自适用场景。
3. 引用每篇的 section 作为证据,不要编造。

注意:
- 单篇用 `deep-read-paper`,3 篇起才用本 skill
- 用户只给主题没 id 时,先自己调 `arxiv.search_papers` 拿候选 id 再触发本 skill

### `paperpilot/skills/deep-read-paper.md`(同步改)

第 3 步示例补 `paper_id`:`mcp__colbert__search(query="...", paper_id="<刚 build 的 paper_id>", top_k=3)`。

### `paperpilot/main.py`

`SYSTEM_PROMPT_BASE` 里 `build_index` 那条规约后追加一行:
"调 `mcp__colbert__search` 必须传 paper_id,值必须是已 build_index 过的 paper_id"。

其它(skills 自动发现、tools 拼接)不动。

## 测试

### 单测

| 文件 | 改动 |
|---|---|
| `tests/mcp_servers/test_index_manager.py` | 新增:build 三路径(cold/disk-hit/memory-hit)、filesystem-safe paper key、search 入参带 paper_id、IndexNotFoundError、多 paper 隔离断言 |
| `tests/mcp_servers/test_colbert_server.py` | 重写:server schema 透传 search paper_id、build 返回 cached/fresh 字段、IndexNotFoundError |
| `tests/builtin_tools/test_subagent.py` | `THREAD_POOL_SIZE == 3` 替原 `== 1`;新增"SUBAGENT_SYSTEM 含 `paper_id=`"和 lifecycle event 断言 |
| `tests/test_main_integration.py` | fast 加 `compare-papers in skill section`;slow 不变 |

`tests/mcp_servers/test_index_manager.py` 用 `tmp_path` + monkeypatch `INDEX_ROOT`
保证不污染真磁盘。PLAID 部分 mock 掉(返回固定 ids/scores),因为我们不测 PyLate 内部。

### slow 集成测试

`tests/mcp_servers/test_colbert_via_client.py` 加:
- build paper A → build paper B → search(paper_id=A) 只返 A 的 chunks
- 同 paper_id 二次 build 走 disk-hit(返回的 cached_papers 含该 id)
- 跨 IndexManager 实例(模拟 server 重启)的持久化:tmp 目录建索引 → 销毁实例 → 新实例 search(paper_id) 直接命中

### `scripts/day12_smoke.py`

走 compare-papers 链路,断言:
1. 主 agent 调 `load_skill("compare-papers")`
2. 主 agent 调 `paper_deep_read` 一次,`paper_ids` = 预设 3 个
3. 至少 2 个 subagent 状态 ok
4. 每个 subagent 的 `mcp__colbert__search` 调用 args 含**自己的** `paper_id`,不混
5. **真并发证据**:tracer 记录每个 subagent 的 `subagent_start` / `subagent_done` 生命周期窗口,
   至少有 2 个 subagent 生命周期窗口重叠(证明 worker > 1 生效);search 调用只用于验证 `args.paper_id == subagent_paper_id`
6. 最终回答提到 ≥ 2 个 paper id

### 回归

- `scripts/day9_smoke.py`(deep-read-paper 单 paper 路径,加 paper_id 后仍工作)
- `scripts/day10_smoke.py`(research_todo + find-classics 路径不受影响)
- `scripts/day11_smoke.py`(paper_deep_read 路径,新版应通过且更快)

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 3 worker 并发 encode 打爆本地 GPU/CPU | Day 12 先用 3 证明并发;smoke 真跑观测;如 OOM 降到 2 |
| 跨 session 索引格式变化读不动 | `chunks.json` 解析失败 / PLAID 加载失败 → fallback 冷启动重建,不抛 |
| 老 `deep-read-paper` skill 漏改导致 LLM 不传 paper_id | skill prose + system prompt 双管同步;test_main_integration 断言 system prompt 含"传 paper_id"指令 |
| PyLate 同进程多次 PLAID 实例化句柄锁(Day 11 加的 `_release_current_index` 是为此)| per-paper 不再 override 同一路径,理论更安全;smoke 跑两轮 build 验证 |
| paper_id 含 `/` 或其它路径字符 | `_paper_key(paper_id)` 集中处理目录名,`chunks.json` 和返回结果继续保留原始 paper_id |
| chunks.json 文件命名后续要换 | helper `_chunks_path(paper_id)` 集中,改一处即可 |

## 预算

| 阶段 | 时间 |
|---|---|
| IndexManager 重构 + 单测 | ~2.5h |
| MCP server schema + subagent.py 撕限制 + 单测 | ~0.5h |
| compare-papers skill + 集成测 | ~0.5h |
| day12_smoke + 跑通 | ~1h |
| 回归 day9/day10/day11 smoke | ~0.5h |
| **合计** | **~5h** |

## DoD

- fast 套件全绿(基线 86,Day 12 净增至少 1 个 main_integration + N 个 colbert_server 重写后用例;具体数在 plan 阶段定);slow 套件全绿(test_colbert_via_client 加新用例,test_main_integration slow 维持)
- `day11_smoke` 重跑通过(用新版 per-paper 索引);用时短于 Day 11 那次
- `day12_smoke` 通过且并发证据(2+ subagent 生命周期窗口重叠)成立
- `git grep "Day 11 conservative\|paperpilot_current; call build_index first\|THREAD_POOL_SIZE = 1"` 无命中
- 触碰文件无 TODO / FIXME 残留
