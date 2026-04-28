# FastMCP `list[dict]` 序列化隐性 bug — 复现 + 条件 fix 设计

**Day 7 / 2026-04-28** · brainstorm 锁 Q1-Q5 + 自审

## 背景

Day 6 Task B 集成测试时发现：`colbert.search` 返回 `list[dict]` 时，FastMCP 把每个 dict 序列化为独立 `TextContent` block，`mcp_client._make_handler` 用 `"\n".join(...)` 拼接得到 **多个连续 JSON 对象**（不是 JSON 数组），`json.loads()` 直接抛 `Extra data`。测试里用 `JSONDecoder().raw_decode()` 循环 workaround。

Day 6 smoke 因 index 仅 1 篇 paper、命中 1 chunk，**没真正触发多对象拼接**，端到端 PASSED 没暴露问题。Day 7+ 用户场景下载 ≥3 篇 paper、search 返多 chunks 时，LLM 可能只解析第一个对象，丢失其余检索结果。

**结构层 bug 已确认**（test workaround 就是证据）。本 spec 解决的剩余未知是：**LLM 拿到拼接 JSON 时是否实际丢信息**——这才是要复现的。

## 目标 & 范围

**目标**：先复现验证 LLM 是否受影响；条件触发最小修复。

**范围内**：
- 写复现脚本 `scripts/day7_fastmcp_repro.py`（3 篇 arxiv paper × top_k=5）
- 在 `mcp_client._make_handler` 加临时 `[REPRO]` 日志
- 三信号判定（C1 结构 / C2 LLM 引用 / C3 LLM 抱怨）
- **条件 fix**（仅当 LLM 受影响时）：
  - `colbert.search` 协议层 wrap 返 `{"results": [...]}`（单 dict，1 个 text block）
  - 删 `test_colbert_via_client.py` 的 `raw_decode` workaround
  - `colbert/server.py` 模块 docstring + `mcp_client._make_handler` 上方注释 各加一句"MCP tool 必须返单 dict"约定
- 复现脚本 commit 留档作回归 sentinel
- spec 写完进 git，无论 fix 是否触发都留下"复现 + 判定 + 决策"档案

**范围外**：
- Client 侧通用 multi-block JSON merge 逻辑（违反反推测抽象，仅一个 server 触发）
- 改 `build_index`（已是单 dict，不受影响）
- 改其他 MCP server（当前只有 arxiv-mcp，已检查不返裸 list）
- 独立 AGENTS.md 全局约定文档（spec + 两处内联 docstring 即可）

## 复现 smoke 设计

### 脚本结构

`scripts/day7_fastmcp_repro.py`，~50 行，参考 `scripts/day6_smoke.py`。

步骤：
1. `arxiv-mcp.search_papers(query="multi-head attention transformer", max_results=3)`
2. 对 3 个结果各调 `arxiv-mcp.download_paper`
3. `colbert.build_index(documents=[...3 docs...])`
4. `colbert.search(query="How does multi-head attention compute the output?", top_k=5)`
5. 把 search 的 tool_result 喂给 `agent_loop`，system prompt 明确要求"在答案中标注 paper_id"
6. 末尾打印判定结果

### 临时日志

在 `paperpilot/tools/mcp_client.py` 的 `_make_handler` 内 join 之后、return 之前加：
```python
import sys
print(f"[REPRO] tool={server}.{tool} joined text:\n{text}\n[REPRO END]",
      file=sys.stderr, flush=True)
```
**Fix 流程结束后必须删除**（DoD checklist 强制 grep）。

### 三信号 + 判定矩阵

- **C1（结构）**：joined text 是否含多个 top-level JSON 对象。脚本扫 `[REPRO]` 块用 `JSONDecoder().raw_decode()` 数对象数；>1 即 C1=true。
  - 预期必 true（结构事实，已知）
- **C2（LLM 行为）**：LLM 答案里**显式引用的 unique paper_id 数** vs **tool_result 里 unique paper_id 数**。前者 < 后者即 C2=false（信息丢失）。
- **C3（LLM 抱怨）**：扫 LLM `text` 内容，匹配 `(?i)(json|parse|format|invalid|malformed)`。命中即 C3=false。

判定逻辑（脚本末尾打印）：
```
触发 fix = NOT (C2 AND C3)
🔴 FIX REQUIRED   if 触发
🟢 NO FIX NEEDED  otherwise
```

## Fix 设计（条件触发）

仅当 `🔴 FIX REQUIRED` 时执行。

### A. 代码改动（1 处）

`paperpilot/mcp_servers/colbert/server.py` 的 `search`：

```python
@mcp.tool()
def search(query: str, top_k: int = 5) -> dict:
    """在当前 ColBERT 索引上查询 top-k 段落。

    Returns:
        dict 含 results 字段:
        {"results": [{"paper_id": str, "chunk_text": str, "score": float}, ...]}
        若索引不存在(未先调 build_index)则抛 IndexNotFoundError。
    """
    return {"results": _search_impl(query, top_k)}
```

`IndexManager.search` 不动，仍返 `list[dict]`。Wrap 留在协议层，符合现有"协议层做输入校验 + 路由，索引层接触 PyLate"分工。

### B. 测试改动

`tests/mcp_servers/test_colbert_via_client.py` 中 `test_build_and_search` 内的 `raw_decode` 循环段（FastMCP 序列化注释 + while 循环）整段删除，换成：
```python
payload = json.loads(search_result)
results = payload["results"]
```

### C. 约定 docstring（2 处内联）

`paperpilot/mcp_servers/colbert/server.py` 模块 docstring 顶部加：
> 返回约定：tool 必须返单 dict。FastMCP 把 `list[dict]` 序列化为 N 个 text block，
> 经 mcp_client `"\n".join` 后变成多 JSON 对象拼接，LLM 解析易丢失数据。

`paperpilot/tools/mcp_client.py` 的 `_make_handler` 上方注释：
> 约定：所有 MCP tool 返单 dict（不返裸 list）。FastMCP 拆 list 为 N text block，
> 拼接后非合法 JSON。详见 docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md

### D. Fix 后回归

重跑 `day7_fastmcp_repro.py`，期望：
- C1=false（单 JSON 对象）
- C2=true、C3=true
- 末尾打印 `🟢 FIX VERIFIED`

## 工作流 & 决策树

```
1. 写 scripts/day7_fastmcp_repro.py + 临时给 _make_handler 加 [REPRO] print
2. 跑 1 次 (~3 min)
3. 看末尾打印：
   ├─ 🟢 NO FIX NEEDED → 删临时 print + commit 脚本 + commit spec
   │                     spec 末尾追加 "baseline 已记录 + 重测条件"
   └─ 🔴 FIX REQUIRED  → 执行 §A/B/C
                       → 重跑脚本验证 🟢 FIX VERIFIED
                       → 删临时 print + commit fix + 测试更新 + spec
4. 无论分支，scripts/day7_fastmcp_repro.py 都 commit 留档（回归 sentinel）
```

**预算**：
- 无 fix 路径：~30 min（脚本 15 + 跑 3 + spec 更新 10）
- 含 fix 路径：~60 min（脚本 15 + 跑 3 + fix 10 + 测试 5 + 重跑 3 + spec 更新 15 + commits 10）

## 错误处理 & 边界

- **arxiv 下载失败**（SSL / 网络）：复现脚本不重试，直接抛。Day 6 已修 SSL 证书；如再失败说明真问题不应掩盖。
- **ColBERT 模型加载失败**：直接抛。Day 6 已验证可用。
- **LLM API 失败**：复现脚本不吞错，直接抛。LLM 不是被测对象，失败本身污染判定。
- **临时 print 残留**：DoD 包含 `git grep "\[REPRO\]" paperpilot/` 必须空。
- **判定假阴**（LLM 当前 OK 但结构 bug 在）：spec 留"重测条件"——换 LLM model / paper 数 >10 / top_k >10 任一变更必须重跑 `day7_fastmcp_repro.py`。

## 测试

- 复现阶段：`scripts/day7_fastmcp_repro.py` 自我判定，输出 🔴 / 🟢 + C1/C2/C3 三值
- 单元测试不动（`tests/mcp_servers/test_colbert_server.py` 测的是 `_search_impl` 内部，不经 FastMCP）
- 集成测试：`tests/mcp_servers/test_colbert_via_client.py` 仅在 fix 触发时改 `test_build_and_search` 的 parse 段

## Definition of Done

无论分支共有：
- [ ] `scripts/day7_fastmcp_repro.py` 提交 + 跑过一次 + 末尾打印 🔴 或 🟢 + 三信号值
- [ ] spec 文档（本文件）末尾"决策记录"小节填好实测三信号 + 决策（fix 或 no-fix）+ 时间戳
- [ ] `git grep "\[REPRO\]" paperpilot/` 空
- [ ] `pytest tests/` 仍 12 passed（unit）
- [ ] commit 历史干净，无 AI 署名

仅 fix 分支额外：
- [ ] `colbert/server.py` `search` 返 `{"results": [...]}`
- [ ] `test_colbert_via_client.py::test_build_and_search` 用 `json.loads(...)["results"]` 解析
- [ ] `pytest -m slow tests/mcp_servers/test_colbert_via_client.py` 3 passed 1 skipped
- [ ] `colbert/server.py` 模块 docstring + `mcp_client._make_handler` 注释 各含约定句
- [ ] `day7_fastmcp_repro.py` 重跑打印 🟢 FIX VERIFIED

## 决策记录

（实施时填）
- 三信号实测：C1=?, C2=?, C3=?
- 决策：fix / no-fix
- 时间戳：?
- 重测条件触发后须重跑此脚本
