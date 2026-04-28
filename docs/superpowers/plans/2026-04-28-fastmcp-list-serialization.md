# FastMCP `list[dict]` 序列化 bug — 复现 + 条件 fix 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 复现 `colbert.search` 经 FastMCP 序列化为多 JSON 对象拼接对 LLM 的实际影响，条件触发最小修复（server 协议层 wrap 单 dict）。

**Architecture:** 写一次性复现脚本 `scripts/day7_fastmcp_repro.py`，临时给 `mcp_client._make_handler` 加 `[REPRO]` stderr 日志输出 raw joined text。脚本调三步流（arxiv 搜+下 3 篇 → colbert build/search → agent_loop 回答），末尾分析 C1（结构事实必 true）+ C2（LLM 引用 paper 数）+ C3（LLM 抱怨 JSON），打印 `🔴 FIX REQUIRED` 或 `🟢 NO FIX NEEDED`。若需 fix，TDD 改 `colbert.search` 返 `{"results": [...]}` 单 dict，删测试 raw_decode workaround，server.py 模块 docstring + mcp_client._make_handler 注释各加约定句。

**Tech Stack:** Python 3.12、FastMCP、PyLate（ColBERT v2）、anthropic SDK + DeepSeek 兼容端点、pytest

**Spec:** `docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md`

---

## File Structure

| 文件 | 责任 | 操作 |
|---|---|---|
| `scripts/day7_fastmcp_repro.py` | 复现脚本 + 三信号判定 + 打印 verdict；commit 留档作回归 sentinel | 新建 |
| `paperpilot/tools/mcp_client.py` | Task 1 加临时 `[REPRO]` print；fix 触发时加约定注释；Task 5 删 print | 改 |
| `paperpilot/mcp_servers/colbert/server.py` | fix 触发时 `search` 返 `{"results": [...]}` + 模块 docstring 加约定 | 条件改 |
| `tests/mcp_servers/test_colbert_via_client.py` | fix 触发时 parse 段从 `raw_decode` 循环改为 `json.loads(...)["results"]` | 条件改 |
| `docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md` | 末尾"决策记录"小节填实测三信号 + 分支决策 + 时间戳 | 改 |

---

## Task 1: 写复现脚本 + 加临时 [REPRO] 日志

**Files:**
- Create: `scripts/day7_fastmcp_repro.py`
- Modify: `paperpilot/tools/mcp_client.py:132-150` (`_make_handler` 内 join 后加 stderr print)

**目的:** 准备复现工具链。本任务不跑脚本（在 Task 2）。

- [ ] **Step 1.1: `mcp_client._make_handler` 加 `[REPRO]` stderr 日志**

打开 `paperpilot/tools/mcp_client.py`，找到 `_make_handler` 方法（约 132-150 行）。在 `text = "\n".join(...)` 那行之后、`if getattr(result, "isError", ...)` 之前插入：

```python
            text = "\n".join(
                b.text for b in result.content if hasattr(b, "text")
            )
            # [REPRO] Day 7 临时诊断日志，Task 5 必须删除
            import sys
            print(
                f"[REPRO] tool=mcp__{server}__{tool} blocks={len(result.content)} "
                f"joined_text:\n{text}\n[REPRO END]",
                file=sys.stderr, flush=True,
            )
            if getattr(result, "isError", False):
```

- [ ] **Step 1.2: 写复现脚本 `scripts/day7_fastmcp_repro.py`**

新建文件，全文如下（请逐字复制，不要省略）：

```python
"""Day 7 复现: FastMCP `list[dict]` 序列化对 LLM 的实际影响。

设计 spec: docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md

跑法: python scripts/day7_fastmcp_repro.py 2> repro.stderr
脚本本身往 stdout 打人类摘要 + verdict; mcp_client 的 [REPRO] 块走 stderr。
分析时 stderr 里会有 ≥1 个 [REPRO] 块, Task 2 用 grep 取出。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

# arxiv id 正则: 2401.12345 / 2401.12345v2 等
ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
COMPLAINT_RE = re.compile(
    r"(?i)(invalid json|malformed json|json (parse|format) error|"
    r"unable to parse|extra data|cannot parse)"
)


def main() -> None:
    tool_result_paper_ids: set[str] = set()
    answer_text_chunks: list[str] = []

    def tracer(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            print(f"  → {payload['name']}({list(payload.get('arguments', {}).keys())})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            if payload["name"] == "mcp__colbert__search" and isinstance(content, str):
                # 把 search 工具结果里所有 paper_id 抓出来
                for m in ARXIV_ID_RE.finditer(content):
                    tool_result_paper_ids.add(m.group(0))
            preview = content[:120] if isinstance(content, str) else str(content)[:120]
            print(f"  ← {payload['name']}: {preview}...")
        elif kind == "turn":
            txt = payload.get("text") or ""
            if txt:
                answer_text_chunks.append(txt)

    messages = run(
        "搜 3 篇 multi-head attention transformer 相关的 arxiv 论文(最近一两年),"
        "全部下载全文,然后用 colbert.search 在这些论文里查 "
        "'How does multi-head attention compute the output?',"
        "用一段话回答我。回答必须基于 colbert.search 返回的具体段落,"
        "并在答案里**显式标注每段引用的 paper arxiv id**(如 2401.12345)。",
        max_iter=12,
        on_event=tracer,
    )

    full_answer = "\n".join(answer_text_chunks)
    print("\n=== FINAL ANSWER ===")
    print(full_answer)

    # 信号分析
    cited_ids = set(ARXIV_ID_RE.findall(full_answer))
    c1_unknown = "(C1 走 stderr 分析, 见下文)"
    c2 = bool(tool_result_paper_ids) and cited_ids >= tool_result_paper_ids
    c3 = COMPLAINT_RE.search(full_answer) is None

    print("\n=== 三信号 ===")
    print(f"  tool_result paper_ids: {sorted(tool_result_paper_ids)}")
    print(f"  answer cited paper_ids: {sorted(cited_ids)}")
    print(f"  C1 (结构, 多 JSON 对象拼接): {c1_unknown}")
    print(f"  C2 (LLM 引用全 vs 缺失):   {'PASS' if c2 else 'FAIL'}")
    print(f"  C3 (LLM 不抱怨 JSON):       {'PASS' if c3 else 'FAIL'}")

    if c2 and c3:
        print("\n🟢 NO FIX NEEDED — LLM 实际未受影响 (但需在 spec 记 baseline)")
    else:
        print("\n🔴 FIX REQUIRED — LLM 表现失常, 触发 §3 fix")

    # exit 0 即可,不靠 exit code 判定 (留给人 + spec 决策记录)


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.3: 验证脚本能 import (不真跑)**

Run: `python -c "import ast; ast.parse(open('scripts/day7_fastmcp_repro.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

Run: `python -c "from paperpilot.main import run; print('import OK')"`
Expected: `import OK`（确认 main 模块仍可 import，没被 mcp_client 改坏）

- [ ] **Step 1.4: Commit**

```bash
git add scripts/day7_fastmcp_repro.py paperpilot/tools/mcp_client.py
git commit -m "Day 7 Task 1: FastMCP list 序列化复现脚本 + mcp_client 临时 [REPRO] 日志"
```

---

## Task 2: 跑复现 + 记录三信号 + 决策分支

**Files:**
- Modify: `docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md`（"决策记录"小节）

**目的:** 真跑复现脚本，分析三信号，把结果写进 spec 决策记录，决定走 fix 分支还是 no-fix 分支。

- [ ] **Step 2.1: 跑复现脚本，stderr 重定向到 repro.stderr**

Run: `python scripts/day7_fastmcp_repro.py 2> repro.stderr`
Expected:
- 进程退出码 0
- stdout 末尾出现 `🟢 NO FIX NEEDED` 或 `🔴 FIX REQUIRED`
- `repro.stderr` 文件存在且含至少一个 `[REPRO]` 块
- 脚本运行约 60-180s（3 篇 paper 下载 + ColBERT build + ≥1 次 search + ≤12 轮 LLM）

**网络/API 失败时**：直接报错，不重试。Day 6 已修 SSL；如再失败，先解决根因再继续，不跳过本步。

- [ ] **Step 2.2: 分析 C1（结构信号）**

Run: `grep -c "^\[REPRO\] tool=mcp__colbert__search" repro.stderr`
Expected: 1 或更多（即 colbert.search 至少被调一次）

把每个 colbert.search 的 `[REPRO]` 块 joined_text 部分用 `JSONDecoder().raw_decode()` 数顶层 JSON 对象数。**手动判定**或用一行 Python：

Run:
```bash
python -c "
import json, re, sys
data = open('repro.stderr', encoding='utf-8', errors='replace').read()
blocks = re.findall(r'\[REPRO\] tool=mcp__colbert__search.*?joined_text:\n(.*?)\n\[REPRO END\]', data, re.S)
for i, b in enumerate(blocks):
    s = b.strip()
    dec, count, idx = json.JSONDecoder(), 0, 0
    while idx < len(s):
        try:
            _, end = dec.raw_decode(s, idx)
            count += 1
            idx = end
            while idx < len(s) and s[idx] in ' \n\r\t': idx += 1
        except json.JSONDecodeError:
            break
    print(f'block {i}: {count} top-level JSON object(s)')
"
```

Expected: 至少有一个 block 报 ≥2 顶层对象。**如果所有 block 都是 1 对象**：说明 LLM 没真的让 search 返 ≥2 chunk（top_k 设置或检索结果不够），改 day7_fastmcp_repro.py 让 prompt 里强制 `top_k=5` 或扩大 paper 数量后**回到 Step 2.1 重跑**。

C1 = (任一 block 报 ≥2 对象)。预期 C1=true。

- [ ] **Step 2.3: 把 C1/C2/C3 + 决策填进 spec**

打开 `docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md`，找到末尾 "## 决策记录" 小节。把内容替换为（保留小节标题）：

```markdown
## 决策记录

- **运行时间**: <从命令行 date 抓的 ISO8601 时间, 如 2026-04-28T15:23:00+08:00>
- **三信号实测**:
  - C1 (结构, 多 JSON 对象拼接): <true / false> — 实测 search 调用 N 次, 每次 block 数: <如 [3, 5]>
  - C2 (LLM 引用全 vs 缺失): <PASS / FAIL> — tool_result paper_ids: <list>; answer cited: <list>
  - C3 (LLM 不抱怨 JSON): <PASS / FAIL> — <若 FAIL 引用 LLM 抱怨片段>
- **决策**: <fix / no-fix>
- **理由**: <一句话>
- **重测条件**（fix 与否都适用）：换 LLM model / paper 数 >10 / top_k >10 任一变更必须重跑 `scripts/day7_fastmcp_repro.py`
```

把 `<...>` 全部替换为实测值。**不要留任何尖括号 placeholder**。

- [ ] **Step 2.4: 决策分支**

如果决策是 **no-fix**（C2 PASS && C3 PASS）：
- 跳过 Task 3 + Task 4
- 直接进 Task 5

如果决策是 **fix**（C2 FAIL || C3 FAIL）：
- 进 Task 3

- [ ] **Step 2.5: Commit 复现结果**

```bash
git add docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md
git commit -m "Day 7 Task 2: 跑复现脚本, 三信号实测填 spec 决策记录"
```

注意：`repro.stderr` 不要 commit，加进 `.gitignore`（如已忽略可跳过）：

Run: `grep -q "^repro.stderr$" .gitignore || echo "repro.stderr" >> .gitignore`

如果 .gitignore 被改了：
```bash
git add .gitignore
git commit -m "Day 7: ignore repro.stderr (Task 2 临时产物)"
```

---

## Task 3 (条件): TDD 改 colbert.search 返单 dict + 删测试 workaround + 加约定 docstring

**仅当 Task 2 决策为 fix 时执行；no-fix 路径跳到 Task 5。**

**Files:**
- Modify: `tests/mcp_servers/test_colbert_via_client.py:58-70`（test_build_and_search 内 parse 段）
- Modify: `paperpilot/mcp_servers/colbert/server.py:51-63`（search 函数）+ 文件顶部模块 docstring
- Modify: `paperpilot/tools/mcp_client.py:132`（`_make_handler` 上方注释）

- [ ] **Step 3.1: TDD red — 改 test_build_and_search 用单 dict + results 解析**

打开 `tests/mcp_servers/test_colbert_via_client.py`，找到 `test_build_and_search`，把第 58-70 行的整段 parse 逻辑（`# FastMCP serializes...` 注释 + while 循环）替换为：

```python
        # 协议层 wrap 单 dict, 单 text block, 单段合法 JSON
        if isinstance(search_result, str):
            payload = json.loads(search_result)
            results = payload["results"]
        else:
            results = search_result["results"]
```

注意保留缩进（8 空格）。其余 assert 行（`len(results) >= 1`、`any(r["paper_id"] == "p3" ...)` 等）不动。

- [ ] **Step 3.2: 跑测试确认 RED**

Run: `pytest -m slow tests/mcp_servers/test_colbert_via_client.py::test_build_and_search -v`
Expected: **FAIL**，错误信息含 `KeyError: 'results'` 或 `JSONDecodeError`（因为 server 还没改，仍返多对象拼接，`json.loads` 抛 `Extra data`）。

如果意外 PASS：说明 ColBERT search 这次只返 1 个 chunk，FastMCP 序列化为单 block 单 JSON 对象，恰好能被 `json.loads` 解析得到一个 `dict`，然后 `["results"]` 抛 `KeyError`——但这仍应是 FAIL。如果真 PASS，回到 Task 2 检查复现是否真触发了多 chunk 场景。

- [ ] **Step 3.3: TDD green — 改 colbert/server.py 的 search**

打开 `paperpilot/mcp_servers/colbert/server.py`，把 `search` 函数（51-63 行附近）替换为：

```python
@mcp.tool()
def search(query: str, top_k: int = 5) -> dict:
    """在当前 ColBERT 索引上查询 top-k 段落。

    Args:
        query: 自然语言查询。
        top_k: 返回的段落数,默认 5。

    Returns:
        dict 含 results 字段:
        {"results": [{"paper_id": str, "chunk_text": str, "score": float}, ...]}
        若索引不存在(未先调 build_index)则抛 IndexNotFoundError。
    """
    return {"results": _search_impl(query, top_k)}
```

`_search_impl` 函数本身（66-68 行）不动，仍返 `list[dict]`。

- [ ] **Step 3.4: 跑测试确认 GREEN**

Run: `pytest -m slow tests/mcp_servers/test_colbert_via_client.py::test_build_and_search -v`
Expected: **PASS**

跑完整 slow 集成确保其他用例不回归：
Run: `pytest -m slow tests/mcp_servers/test_colbert_via_client.py -v`
Expected: 3 passed, 1 skipped

跑默认 unit suite 也确认不回归：
Run: `pytest tests/`
Expected: 12 passed, 4 deselected

- [ ] **Step 3.5: 加 server.py 模块 docstring 约定**

打开 `paperpilot/mcp_servers/colbert/server.py`，把文件顶部 docstring（第 1-7 行）改为：

```python
"""colbert-mcp: ColBERT 段落级语义检索 server。Day 6 起。

启动:python -m paperpilot.mcp_servers.colbert.server
通过 stdio 被 paperpilot.tools.mcp_client 拉起,manifest 见 paperpilot/mcp_servers.json。

协议层职责: 输入校验 + 路由到 IndexManager。**不接触 PyLate**。

返回约定: tool 必须返单 dict。FastMCP 把 list[dict] 序列化为 N 个 text block,
经 mcp_client "\\n".join 后变成多 JSON 对象拼接,LLM 解析易丢失数据。
详见 docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md
"""
```

- [ ] **Step 3.6: 加 mcp_client._make_handler 约定注释**

打开 `paperpilot/tools/mcp_client.py`，找到 `_make_handler` 方法的 `def handler(args: dict) -> str:` 那行（约 133 行）。在 `def _make_handler(self, server: str, tool: str):` 那行**上方**插入一行注释：

```python
    # MCP tool 返回约定: 必须返单 dict, 不返裸 list。
    # FastMCP 把 list 拆成 N 个 text block, "\n".join 后变多 JSON 对象拼接,
    # 不是合法 JSON, LLM 解析易丢失。详见
    # docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md
    def _make_handler(self, server: str, tool: str):
```

(注意此处仍保留之前 Task 1 加的 `[REPRO]` 临时 print，Task 5 才删。)

- [ ] **Step 3.7: 再跑一次单测和集成确认 docstring 改动没破坏**

Run: `pytest tests/`
Expected: 12 passed, 4 deselected

- [ ] **Step 3.8: Commit**

```bash
git add paperpilot/mcp_servers/colbert/server.py paperpilot/tools/mcp_client.py tests/mcp_servers/test_colbert_via_client.py
git commit -m "Day 7 Task 3: colbert.search 返 {results: [...]} 单 dict + 删测试 raw_decode workaround + 加返单 dict 约定 docstring"
```

---

## Task 4 (条件): 重跑复现验证 fix + 更新 spec

**仅当 Task 3 执行后才进入此 Task。**

**Files:**
- Modify: `docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md`（决策记录追加 fix 验证段）

- [ ] **Step 4.1: 重跑复现脚本**

Run: `python scripts/day7_fastmcp_repro.py 2> repro_after_fix.stderr`
Expected:
- stdout 末尾 `🟢 NO FIX NEEDED`（C2 + C3 都 PASS，因为 LLM 现在拿到的是合法单 JSON）
- 进程退出 0

- [ ] **Step 4.2: 验证 C1 现在变 false（单 JSON 对象）**

Run:
```bash
python -c "
import json, re
data = open('repro_after_fix.stderr', encoding='utf-8', errors='replace').read()
blocks = re.findall(r'\[REPRO\] tool=mcp__colbert__search.*?joined_text:\n(.*?)\n\[REPRO END\]', data, re.S)
for i, b in enumerate(blocks):
    s = b.strip()
    dec, count, idx = json.JSONDecoder(), 0, 0
    while idx < len(s):
        try:
            _, end = dec.raw_decode(s, idx)
            count += 1
            idx = end
            while idx < len(s) and s[idx] in ' \n\r\t': idx += 1
        except json.JSONDecodeError:
            break
    print(f'block {i}: {count} top-level JSON object(s)')
"
```

Expected: 每个 block 都报 `1 top-level JSON object(s)`。

- [ ] **Step 4.3: spec 决策记录追加 fix 验证段**

在 `docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md` 的 "## 决策记录" 小节末尾追加：

```markdown
- **Fix 验证时间**: <ISO8601>
- **Fix 后三信号**:
  - C1 (结构): false — 所有 block 单 JSON 对象（实测 N=...）
  - C2: PASS — tool_result paper_ids: <list>; answer cited: <list>
  - C3: PASS
- **🟢 FIX VERIFIED**
```

替换 `<...>` 为实测值。

- [ ] **Step 4.4: Commit spec 更新**

```bash
git add docs/superpowers/specs/2026-04-28-fastmcp-list-serialization-design.md
git commit -m "Day 7 Task 4: fix 后重跑复现, C1 变 false, C2/C3 PASS, spec 验证段追加"
```

---

## Task 5: 清理 [REPRO] print + 全套回归 + 终段 commit

**所有分支都执行此任务。**

**Files:**
- Modify: `paperpilot/tools/mcp_client.py`（删 Task 1 加的 [REPRO] print）

- [ ] **Step 5.1: 删 mcp_client 里的 [REPRO] print**

打开 `paperpilot/tools/mcp_client.py`，找到 Task 1 Step 1.1 加的那 7 行（`# [REPRO] Day 7 临时诊断日志...` 注释 + `import sys` + `print(...)` + 闭括号）。**整段 7 行删除**，保留前后的 `text = "\n".join(...)` 和 `if getattr(result, "isError", False):` 不动。

(Task 3 在此函数上方加的"MCP tool 返回约定"注释不属于临时日志，**不要删**。)

- [ ] **Step 5.2: 验证临时日志已无残余**

Run: `git grep "\[REPRO\]" paperpilot/`
Expected: 空（exit code 1，无输出）

Run: `git grep "\[REPRO\]" scripts/`
Expected: 仍命中（`scripts/day7_fastmcp_repro.py` 注释里有，**这是预期保留的**）

- [ ] **Step 5.3: 全套回归**

Run: `pytest tests/`
Expected: 12 passed, 4 deselected

Run: `pytest -m slow tests/mcp_servers/test_colbert_via_client.py -v`
Expected: 3 passed, 1 skipped（如果 Task 3 跳过——即 no-fix 分支——这里也仍应 3 passed 1 skipped；Day 6 baseline 已稳定）

Run: `python scripts/day5_smoke.py`
Expected: 进程退出 0，无 stack trace

- [ ] **Step 5.4: 删 repro.stderr (临时产物)**

Run: `rm -f repro.stderr repro_after_fix.stderr`

- [ ] **Step 5.5: 终段 commit**

```bash
git add paperpilot/tools/mcp_client.py
git commit -m "Day 7 Task 5: 删 mcp_client 临时 [REPRO] 诊断日志, 全套回归通过"
```

- [ ] **Step 5.6: 检查 git log 干净 + DoD**

Run: `git log --oneline -10`
Expected:
- Task 1, 2, 5 的 commit 必有
- 若 fix 路径，Task 3, 4 的 commit 也在
- 无任何 AI 署名（`Co-Authored-By: Claude`、`🤖 Generated with` 等）

DoD 对账：
- [ ] `scripts/day7_fastmcp_repro.py` 已 commit + 已跑过 + 末尾打印 🔴 或 🟢
- [ ] spec 决策记录三信号 + 决策 + 时间戳 已填
- [ ] `git grep "\[REPRO\]" paperpilot/` 空
- [ ] `pytest tests/` 12 passed
- [ ] commits 无 AI 署名

仅 fix 分支额外 DoD：
- [ ] `colbert.server.py` `search` 返 `{"results": [...]}`
- [ ] `test_colbert_via_client.py::test_build_and_search` 用 `json.loads(...)["results"]` 解析
- [ ] `pytest -m slow tests/mcp_servers/test_colbert_via_client.py` 3 passed 1 skipped
- [ ] server.py 模块 docstring + mcp_client `_make_handler` 上方注释 各含约定句
- [ ] day7_fastmcp_repro.py 重跑 spec 中已记 🟢 FIX VERIFIED
