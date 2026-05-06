# Day 15: `write-research-report` 主题综述 skill 设计

**目标:** 顶层产出能力。用户给一个研究方向,agent 自主走 arxiv→colbert→(可选 graph)→subagent 并发精读→产出一篇半固定骨架的 markdown 综述。**纯新增 1 个 skill prose,不引入任何代码 / tool / MCP**;落盘交给 caller(smoke 脚本)。

**Why:** Day 14 末已建齐 4 MCP server + 4 内嵌 tool + 5 skill。需要一个"压轴" skill 把所有能力串起来,产出可演示的 markdown,作为后续 Day 16 LLM-as-Judge 评估的输入和 Day 28 投简历的 demo 产物。

**Scope:** 1 个新 skill md + 1 个 smoke 脚本 + 1 行 main_integration test 断言 + 一个 `data/reports/` 目录。零代码改动,红线全守恒。

---

## §1 设计决策(brainstorm 沉淀)

| Q | 选择 | 理由 |
|---|---|---|
| Q1: 使用场景 | 1 主题综述型 | 一句话 → 一篇报告,面试杀手锏;强制走通 arxiv→colbert→graph→subagent 全链路;对比型/外扩型已被 compare-papers / explore-citations 部分覆盖 |
| Q2: 输出结构 | 2 半固定骨架 | 完全自由不利 LLM-as-Judge 稳定打分;完全死板违反"LLM 自决"精神;半固定 = 推荐骨架 + 主题适配 |
| Q3: 单次规模 | 1 轻量 5-8 篇候选 / 3-5 篇精读 | demo 现场跑得起(2-3 分钟);评估批跑成本可控(30 query × 几万 token);后续要加深改 prose 数字即可 |
| Q4: 落盘机制 | 2 agent message 即报告,caller 落盘 | 不引入新 tool,红线零风险;单一职责;评估器直接拿 text 不读文件;CLI/Streamlit 复用同套消息流 |
| Q5: 实现方案 | A 纯 skill + smoke | DoD 是"能产出报告",不是"漂亮 CLI";Day 17 一并写 CLI 多子命令更顺手 |

**红线守恒(zero diff):**
- `paperpilot/main.py` / `paperpilot/core/loop.py` / `paperpilot/core/adapter.py` / `paperpilot/builtin_tools/*.py`
- 现有 5 个 skill / 4 个 MCP server / `mcp_servers.json`

**唯一改动**:`tests/test_main_integration.py` 加 1 行 `assert "write-research-report" in prompt`(skill 自动发现的回归断言,跟 Day 14 同 pattern)。

---

## §2 文件结构

| 路径 | 动作 | 责任 |
|---|---|---|
| `paperpilot/skills/write-research-report.md` | 新建 | 综述工作流 prose + 半固定输出骨架 + 质量准则 |
| `scripts/day15_smoke.py` | 新建 | 真 LLM 跑一篇 + tracer 断言 + 落盘 + 报告内容断言 |
| `data/reports/.gitkeep` | 新建,空 | 让产物目录入仓 |
| `tests/test_main_integration.py` | 改 1 行 | 加 `assert "write-research-report" in prompt` |

---

## §3 Skill prose 骨架(`write-research-report.md`)

```markdown
---
name: write-research-report
description: 给定一个研究方向,产出一篇结构化综述 markdown
when_to_use: 用户给一个主题/方向(不是单篇 paper / 不是对比指定几篇),希望"写一篇综述"或"了解 XX 领域近年进展"
---

# Write Research Report

## 适用场景
- "总结一下 long-context retrieval 2024 后的进展"
- "我想入门 X 方向,给我一份 reading list 加每篇要点"
- 不适用: 用户给定具体 paper(用 deep-read-paper / compare-papers)

## 工作流(推荐,非强制)
1. **规划**: `research_todo` 写下 plan(搜索关键词 / 候选数 / 精读数)
2. **粗搜**: `mcp__arxiv__search(query=..., max_results=8)`,标题摘要先筛
3. **细排**: 把候选喂 `mcp__colbert__build_index` → `mcp__colbert__search(query=主题问题, k=5)` 做语义重排
4. **扩展(可选)**: 若主题有明确"奠基 paper",用 `mcp__graph__cited_by / cites` 1 跳找经典或最新工作
5. **精读**: 选 3-5 篇代表作,`paper_deep_read(papers=[...], n_workers=3-5)` 并发精读
6. **下笔**: 用半固定骨架组织 markdown 输出(见下)

## 输出骨架(半固定,可按主题适配)
1. **Title** — 综述标题
2. **TL;DR** — 3-5 句话核心结论
3. **Background** — 问题定义、为什么重要
4. **Key Methods / Approaches** — 主流路线分类(每条带 1-3 篇代表作 + arxiv_id)
5. **Recent Trends (2024-)** — 最新进展
6. **Open Problems** — 未解、争议、benchmark gap
7. **Reading List** — 推荐阅读顺序(5-8 篇,每篇一句话点评 + arxiv_id 链接)

## 质量准则
- 每篇引用必须带 arxiv_id(写成 `[Title (arxiv:2401.12345)]`),用户能点开
- 至少 3 篇 paper 来自 paper_deep_read 的精读结果(不能全靠 abstract)
- 报告 ≥ 800 字
- 主题方法学 vs 应用偏向不同,骨架可砍可加(纯方法论可砍"应用",偏应用可并掉"Open Problems")

## 注意
- 不要把所有候选都精读 — 3-5 篇够;其余靠 abstract + colbert chunk 补
- compose 阶段先写完整 markdown 再输出,不要边搜边写
```

---

## §4 Smoke 脚本设计

**Demo 主题**: `"long-context retrieval beyond 100k tokens"`
- arxiv 上覆盖足、有 RAG/sparse/dense 多条路线适合"半固定骨架"展开
- 2024+ 文章多,"Recent Trends"段不会空

**调用形式**:
```python
from paperpilot.main import run

prompt = (
    "请为我写一篇综述,主题:long-context retrieval beyond 100k tokens。"
    "按 write-research-report skill 操作: 先 load_skill,然后跟 prose 走"
    "(arxiv 搜索 → colbert 重排 → 选 3-5 篇 paper_deep_read 并发精读 → 半固定骨架 markdown)。"
)
messages = run(prompt, max_iter=30, on_event=tracer)
```

**追踪断言(tracer 累积 tool_calls)**:
- `load_skill` 至少 1 次,`name == "write-research-report"`
- `mcp__arxiv__search` 至少 1 次
- `mcp__colbert__build_index` 至少 1 次
- `mcp__colbert__search` 至少 1 次
- `paper_deep_read` 至少 1 次,且 `n_workers >= 3`
- `research_todo` 至少 1 次(plan 步)
- **不断言** `mcp__graph__*`(skill 标了"可选",强行断言违反 LLM 自决)

**最终报告断言(final assistant message text)**:
- 长度 ≥ 800 字符(中英文混排,字符级足够)
- ≥ 3 个不同 arxiv_id 命中(regex `arxiv[:/ ]\d{4}\.\d{4,5}`,`set()` 去重 ≥ 3)
- 7 个 section 标题里至少 4 个命中字面量: `TL;DR / Background / Key Methods / Recent Trends / Open Problems / Reading List` 任意 4 个

**落盘(脚本视角,非 agent 视角)**:
```python
from datetime import datetime
from pathlib import Path

slug = "long-context-retrieval"
ts = datetime.now().strftime("%Y%m%d-%H%M%S")
out = Path(f"data/reports/{slug}-{ts}.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(final_text, encoding="utf-8")
print(f"Report saved: {out}")
```

**max_iter**: `30`
- 估算: plan(1) + arxiv(1) + colbert build(1) + colbert search(1-2) + 可选 graph(0-2) + paper_deep_read(1) + compose(1) ≈ 8-12 工具调用
- 加纠错重试余量,30 稳;走完只用 15 也无所谓

---

## §5 失败排查(注释在脚本里,不写排查代码)

| 症状 | 排查方向 |
|---|---|
| 没 load_skill | 检查 `_build_system_prompt` 是否真把 skill 列进去(已被 main_integration test 覆盖,理论上不会) |
| 没 paper_deep_read | 调强 prompt,加一句"必须并发精读 3-5 篇" |
| 报告 < 800 字 | 提高 max_iter 到 40,或 prompt 强调"≥800 字" |
| arxiv_id 数量不够 | prose"质量准则"里的"至少 3 篇 deep_read"已强调,若仍不够提高 prompt 权重 |

**绝不:** 不要为了让 smoke 过加 if-else 分支 / 加引导短语绕过 LLM 决策。skill prose 调整 OK,loop 内 nudge 文本不动。

---

## §6 DoD 验证

- [ ] `pytest tests -q` 全绿,`test_main_integration` 含 write-research-report 断言
- [ ] `python scripts/day15_smoke.py` 终端打印 tool_calls 链路 + `Day 15 smoke PASSED`
- [ ] `data/reports/long-context-retrieval-<timestamp>.md` 落盘,人眼读起来像"一篇综述"(不是 bullet 拼盘)
- [ ] `git diff main -- paperpilot/main.py paperpilot/core/loop.py paperpilot/core/adapter.py paperpilot/builtin_tools/subagent.py` 空(红线守恒)
- [ ] `git log --oneline | head -3` 含 Day 15 spec 与后续 plan/impl commit