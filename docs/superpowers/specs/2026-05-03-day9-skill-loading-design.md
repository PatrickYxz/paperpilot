# PaperPilot Day 9 设计：`load_skill` 内嵌 tool + 第一批 skill

| 项 | 值 |
|---|---|
| 日期 | 2026-05-03 (Day 8 graph-mcp 完工后, 设计 Day 9 skill loading) |
| 范围 | (1) 新增 L2 内嵌 tool `load_skill`; (2) 新增 `paperpilot/builtin_tools/skill_loader.py` 实现 SkillRegistry; (3) 启动时扫 `paperpilot/skills/*.md` 把 frontmatter 注入 system prompt; (4) 第一批写 3 个 skill: `deep-read-paper` / `explore-citations` / `find-classics`; (5) 单测 + day9 smoke |
| 不在范围 | `research_todo` / `paper_deep_read` / `compact_context` 三个内嵌 tool; pdf-parse-mcp; vlm-mcp; skill 优先级/分级机制; skill 内 sub-skill 引用; skill body 缓存 / 去重; skill hot reload; tools_hint 字段 |
| 状态 | Draft, 待用户 review |

---

## 1. 目标

让 LLM 在 main loop 里能用"按需加载 skill prose"的方式选择检索策略, 彻底替代 if-else / switch-case 的 query-aware planner:

1. 启动时 LLM 通过 system prompt 知道有哪些 skill (name + description + when_to_use)
2. 看到合适场景时 LLM 主动调 `load_skill(name="...")` 拿完整 markdown 步骤说明
3. tool_result 把 skill body 自然带进 conversation history, 后续轮次 LLM 按 skill prose 调底层 mcp tool

这是 PaperPilot 第一个 L2 内嵌 tool, 也是"决策由 LLM 做"红线最直接的产物 —— skill 只是 prompt 里的"先验知识", 不是路由器, 不是宏, 不是 tool 屏蔽。

同时严守方案 C / Day 5-8 锁定的红线:

- **L1 基座层 + Day 5 mcp_client 零改动** —— 内嵌 tool 与 MCP tool 在 agent_loop 里同等对待
- **server 之间不互通信** —— skill_loader 不知道任何 mcp server 存在, 它只读本地 .md 文件
- **决策由 LLM 做** —— skill 文本是 prose, 不是路由分支; 加载哪个 skill / 加载之后要不要执行步骤 / 跳过某步, 全由 LLM 决定
- **不做推测性抽象** —— 不留 skill 优先级 / 多版本 / hot reload / tools_hint / skill 内嵌 skill 引用等口子
- **复用 Day 7 trip wire** —— `load_skill` tool_result 是 str, 不是 list[dict], 不撞 FastMCP 多 JSON 拼接监控

---

## 2. 关键设计决策 (Q1-Q6)

| ID | 决策 | 选项 | 主要理由 |
|---|---|---|---|
| Q1 | Day 9 做哪一块 | **`load_skill` + 第一批 skill** (vs research_todo / pdf-parse / paper_deep_read / vlm) | "反 if-else"红线的核心证据; 一旦有 load_skill, 后面 paper_deep_read / research_todo 都可以包成 skill, 架构上更干净; 现有 3 个 MCP server 已能跑, 正好演示"main loop 用 load_skill 选检索策略"链路 |
| Q2 | skill 本质 | **纯 prose 注入** (vs prose+宏 / prose+tool 子集屏蔽) | 与 Claude Code s05 完全一致; 决策权 100% 在 LLM; 实现最简单; 简历讲解金句一致 |
| Q3 | 发现机制 | **启动时 name+description 注入 system prompt** (vs list_skills tool / 混合分级) | LLM 必须看到 description 才能正确决策何时 load; 5-8 个 skill × 一行 desc < 300 tokens, 成本可忽略; 与 Claude Code 完全对齐 |
| Q4 | body 抵达 LLM 通路 | **作为 load_skill 的 tool_result 字符串** (vs system prompt mutation / 合成 user message) | 零额外抽象; conversation history 自然持有; 与现有 agent_loop 完全兼容; 这是 Claude Code 实际做法 |
| Q5 | 文件格式 + 位置 | **`paperpilot/skills/*.md`, YAML frontmatter (name/description/when_to_use) + markdown body** | 包内放方便测试 import path; 三字段够用 (无 tools_hint, YAGNI); load 时返完整文件含 frontmatter, 让 LLM 看 when_to_use 自校 |
| Q6 | 第一批 skill | **`deep-read-paper` + `explore-citations` + `find-classics`** (vs +survey / +compare) | 三个 skill 分别覆盖 colbert 全流程 / graph 全流程 / 跨 server 协同 (arxiv+graph 共引); survey 等 paper_deep_read subagent 做完更合理; compare 等 pdf-parse-mcp 做完能对比图表更香 |

### 隐含决策 (已锁)

| 项 | 值 | 备注 |
|---|---|---|
| 内嵌 tool 模块路径 | `paperpilot/builtin_tools/` | 与 `paperpilot/mcp_servers/` 区分; 后续 research_todo / compact_context 同目录 |
| frontmatter 解析 | 手写 stdlib (无 PyYAML 依赖) | 只取 3 个固定字段, 不需要 YAML 全功能; 如果未来字段变多再考虑 PyYAML |
| frontmatter 字段 | `name` / `description` / `when_to_use` | 三字段必填, 缺失启动期 hard-fail |
| `load_skill` 返回 | 完整 .md 文件内容 (含 frontmatter) | LLM 看 when_to_use 自校; 文件本身不大, 不剥离 frontmatter |
| 启动期 hard-fail 触发 | skills 目录不存在 / frontmatter 格式错 / 字段缺失 | 不静默降级 —— 让用户立刻发现 skill 文件坏了 |
| 重复 load 行为 | 不去重, 正常返回相同内容 | LLM 想再看一遍是它自己的事; 实现上零状态 |
| 未知 skill name | tool error (is_error=true), content 含可用 skill 列表 | LLM 看到列表自己换一个名字 |
| skill 数量上限 | 不设硬限 | 第一版 3 个, 远低于 token 预算 |
| skill 内引用其它 skill | 不支持, prose 写就行 | 第一版没必要做 transclusion |

---

## 3. 架构

```
┌────────────────────────────────────────────────────────────────────┐
│  main.py 启动流程                                                  │
│    1. SkillRegistry(paperpilot/skills/).scan()                    │
│       a. 扫目录, 读每个 .md                                        │
│       b. 解析 frontmatter (失败 → hard-fail)                       │
│       c. 内存里存 {name: {description, when_to_use, body, path}}   │
│    2. system prompt 拼接:                                         │
│       SYSTEM_PROMPT + "\n\n## 可用 skill\n" + 列表渲染            │
│    3. tools = [...mcp tool, load_skill_tool(registry)]            │
│    4. agent_loop(messages, system, tools, ...)                    │
└────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────────────┐
│  agent_loop 运行时 (loop.py 不改)                                  │
│    LLM 看到 system 里有 skill 清单                                 │
│      → 调 load_skill(name="deep-read-paper")                      │
│      → handler = registry.load(name)                              │
│      → tool_result.content = 完整 .md 文件字符串                   │
│      → conversation history 自然持有 skill body                    │
│      → LLM 下一轮按 skill prose 调 mcp__arxiv__download_paper /    │
│         mcp__colbert__build_index / mcp__colbert__search           │
└────────────────────────────────────────────────────────────────────┘
```

**关键边界**:
- `load_skill` 是 L2 内嵌 tool (Python 函数 handler), **不走 MCP 协议** —— 与 Claude Code skill loading 一致, 与未来 `research_todo` / `compact_context` 同层级
- 内嵌 tool 与 MCP tool 在 agent_loop 里**同等对待** (统一走 `Tool` dataclass + handler dispatch)
- skill body **不预加载到 system prompt** —— 否则失去懒加载意义, token 浪费

---

## 4. 文件布局

### 新增 / 改动

| 路径 | 状态 | 作用 | 预估行数 |
|---|---|---|---|
| `paperpilot/builtin_tools/__init__.py` | 新 | 模块导出 | ~5 |
| `paperpilot/builtin_tools/skill_loader.py` | 新 | `SkillRegistry` + `load_skill_tool(registry)` 工厂; frontmatter 解析; 目录扫描 | ~80 |
| `paperpilot/skills/deep-read-paper.md` | 新 | colbert 单篇深读流程 prose | ~40 |
| `paperpilot/skills/explore-citations.md` | 新 | graph 引用拓扑探索 prose | ~40 |
| `paperpilot/skills/find-classics.md` | 新 | arxiv + graph 共引经典文献 prose | ~40 |
| `paperpilot/main.py` | 改 | 启动时初始化 SkillRegistry; system prompt 拼接 skill 列表; 注册 load_skill tool | +20 |
| `tests/builtin_tools/__init__.py` | 新 | 空文件 | 0 |
| `tests/builtin_tools/test_skill_loader.py` | 新 | SkillRegistry 单测 | ~80 |
| `scripts/day9_smoke.py` | 新 | 端到端: 让 LLM 自主 load_skill + 跑通 deep-read 链路 | ~50 |

### 不动

- `paperpilot/core/loop.py` —— 一行不改 (内嵌 tool 已经统一走 `Tool` 抽象)
- `paperpilot/core/adapter.py` —— 一行不改
- `paperpilot/core/guardrail.py` —— 一行不改
- `paperpilot/tools/mcp_client.py` —— 一行不改
- 现有 mcp_servers (arxiv/colbert/graph) —— 一行不改

### 布局决策

1. **`paperpilot/builtin_tools/` 与 `paperpilot/mcp_servers/` 平级** —— 区分"L2 内嵌 tool" vs "L3 业务 MCP server"; 与方案 C 三层架构一致
2. **skill 文件直接放 `paperpilot/skills/`** —— 不再分子目录 (3 个 skill 不需要分类); 后续若 >10 个再分
3. **frontmatter 解析手写** —— 不引 PyYAML 依赖; 只解析 `---` 之间的 3 个固定 key:value 行; 解析逻辑严格但简单 (~25 行)
4. **`load_skill_tool(registry)` 工厂返 `Tool` 对象** —— 与 mcp_client 注册 tool 同接口; main.py 一行 `tools.append(load_skill_tool(registry))` 完事

---

## 5. 数据契约

### skill 文件格式

```markdown
---
name: deep-read-paper
description: 深读单篇 arxiv 论文:下载 → 索引 → 多轮检索 → 综合回答
when_to_use: 用户问"详细讲讲 XXX 论文"、"YYY 论文里 ZZZ 是怎么做的"、需要基于全文回答细节问题
---

# Deep Read Paper

## 适用场景
- 用户给定 arxiv id 或论文标题, 要求详细讲解
- 用户问某篇具体论文里某个概念怎么定义 / 某个 method 怎么做的
- 需要基于论文段落给出有据可查的回答

## 步骤

1. **拿全文**: 调 `mcp__arxiv__download_paper(arxiv_id="...")`
2. **建索引**: 调 `mcp__colbert__build_index(documents=[download_paper 返回值])`
   注意 documents 是 list, 元素是 download 返回的对象 (含 paper_id 和 text)
3. **多轮检索**: 针对用户问题里的关键概念, 调 `mcp__colbert__search(query="...", top_k=3)` 取 top 3 chunk
   如果一个 query 不够, 拆成多个 query 多调几次
4. **综合回答**: 基于 colbert.search 返回的具体段落综合回答, 引用段落原文

## 注意
- 不要只看 abstract 就回答细节问题
- 不要编造段落, 只用 colbert.search 真实返回的内容
```

### frontmatter 解析规则

```python
# 文件以 "---\n" 起始, 第二个 "---\n" 之间是 frontmatter
# frontmatter 内每行格式: "<key>: <value>"
# 必填 key: name, description, when_to_use
# value 不支持引号 / 换行 / YAML 嵌套 / 列表
# value 是 strip 后的字符串
# 不符合上述任一条 → ValueError("skill <path> frontmatter invalid: <reason>")
```

### `SkillRegistry` 接口

```python
class SkillRegistry:
    def __init__(self, skills_dir: Path):
        """扫描目录, 解析所有 .md 的 frontmatter, 失败抛 ValueError."""

    def list_metadata(self) -> list[dict]:
        """返 [{'name', 'description', 'when_to_use'}, ...], 按 name 字典序."""

    def load(self, name: str) -> str:
        """返完整 .md 文件内容 (含 frontmatter)。
        name 不存在 → raise SkillNotFoundError(含 available 列表)."""

class SkillNotFoundError(LookupError):
    """LLM 调 load_skill 传错 name 时抛; 由 agent_loop 转 is_error=True tool_result."""
```

### `load_skill` tool schema

```python
def load_skill_tool(registry: SkillRegistry) -> Tool:
    return Tool(
        name="load_skill",
        description=(
            "加载一个 skill 的完整说明书。skill 列表见 system prompt。"
            "tool_result 是该 skill 的 markdown 全文; 你照着步骤调 mcp tool 即可。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "skill name, 例如 deep-read-paper"}
            },
            "required": ["name"],
        },
        handler=lambda args: registry.load(args["name"]),
    )
```

### system prompt 拼接

```python
SYSTEM_PROMPT_BASE = """你是 PaperPilot,一个学术论文研究助手。
- 有 tool 可用时优先调 tool;不要自己编造论文标题、作者或 arxiv id
- 一次只解决用户问的事,不主动扩展任务范围
- tool 报错时,根据错误信息决定:重试(换参数) / 换工具 / 告诉用户失败原因
- 调 tool 时必须按 schema 传完整必填参数;如果错误提示缺字段,下一轮必须补齐字段,不要重复同一个空参数
- 调 mcp__colbert__build_index 时,documents 必须是非空列表,每项包含 paper_id 和 text;通常直接使用 mcp__arxiv__download_paper 返回的对象组成 documents=[download_result]
"""

def render_skill_section(metadata: list[dict]) -> str:
    if not metadata:
        return ""
    lines = ["", "## 可用 skill (按需调 load_skill 加载完整步骤)", ""]
    for m in metadata:
        lines.append(f"- **{m['name']}**: {m['description']}")
        lines.append(f"  适用: {m['when_to_use']}")
    lines.append("")
    lines.append("看到适合场景时调 load_skill(name=\"...\") 拿完整步骤, 再按步骤调 mcp tool。")
    return "\n".join(lines)

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + render_skill_section(registry.list_metadata())
```

---

## 6. 三个 skill 内容草案

### `deep-read-paper.md`

```
---
name: deep-read-paper
description: 深读单篇 arxiv 论文:下载 → 索引 → 多轮检索 → 综合回答
when_to_use: 用户给定 arxiv id 或论文标题要求详细讲解、问某篇具体论文里某概念定义或 method 细节
---

# Deep Read Paper

## 适用场景
- 用户给定 arxiv id 或论文标题, 要求详细讲解
- 用户问某篇具体论文里某个概念怎么定义 / 某个 method 怎么做的
- 需要基于论文段落给出有据可查的回答

## 步骤
1. 拿全文: mcp__arxiv__download_paper(arxiv_id="...")
2. 建索引: mcp__colbert__build_index(documents=[download_paper 返回值])
3. 多轮检索: 针对用户问题里的关键概念, mcp__colbert__search(query="...", top_k=3)
   一个 query 不够就拆多个
4. 综合回答: 基于 colbert.search 返回段落回答, 引用原文

## 注意
- 不要只看 abstract 回答细节问题
- 不要编造段落, 只用 colbert.search 真实返回的内容
```

### `explore-citations.md`

```
---
name: explore-citations
description: 引用拓扑探索:构图 → 看邻居 → 找路径,理解一篇 paper 的学术上下文
when_to_use: 用户问"XXX 论文引了哪些工作"、"XXX 和 YYY 之间引用关系"、"XXX 的 follow-up 工作"
---

# Explore Citations

## 适用场景
- 用户想知道某 paper 引了哪些前作 / 被哪些后续 paper 引用
- 想看两篇 paper 之间是否存在直接或间接引用路径
- 想理解某 paper 在学术脉络中的位置

## 步骤
1. 构图: mcp__graph__build_graph(arxiv_ids=[感兴趣的 paper id 列表])
   返回 missing 列表; SS 查不到就跳过
2. 看引用 / 被引: mcp__graph__get_neighbors(arxiv_id="...", direction="references" 或 "citations" 或 "both", limit=10)
3. 看路径 (可选): mcp__graph__get_shortest_path(from_id="...", to_id="...")
   length=-1 表示无路径
4. 综合回答: 基于 neighbors / path 返回的 title + year + authors 给学术脉络说明

## 注意
- get_neighbors / shortest_path 之前必须先 build_graph 把目标 paper 拉进图
- direction 默认 both, 如果用户明确问"它引了什么"就用 references, 问"谁引了它"就用 citations
```

### `find-classics.md`

```
---
name: find-classics
description: 找经典文献:对一组相关 paper 求共同引用,找出领域内被反复引用的 ground-truth 工作
when_to_use: 用户想入门某领域、想找"读这个领域必读的几篇"、想看一批新论文共同的理论基础
---

# Find Classics

## 适用场景
- 用户想入门某领域, 问"必读哪几篇"
- 用户想看一批新论文背后共同的理论基础
- 需要从一组当代 paper 反推领域内经典

## 步骤
1. 找当代 paper: mcp__arxiv__search_papers(query="该领域关键词", max_results=5)
2. 把这批 paper 拉进引用图: mcp__graph__build_graph(arxiv_ids=[选 3-5 个有代表性的])
3. 求共引: mcp__graph__get_common_citations(arxiv_ids=[同上], top_k=10)
   返回 cited_by_count >= 2 的经典文献
4. 综合回答: 把 top 共引文献按 cited_by_count + year 列出, 解释为什么是该领域经典

## 注意
- 至少需要 2 篇 input 才能求共引 (单篇没意义)
- 如果 build_graph 的 missing 列表非空, 说明部分 paper SS 查不到, 用剩下的继续
- common_citations 返回的是引用关系, 不一定每篇都"广义经典", 但被多篇 input 共引说明在该 cluster 里有 ground-truth 地位
```

---

## 7. 错误处理

按 Day 5 红线分两类:

### A. 启动期 hard-fail (进程起不来 → main 异常退出)

| 失败 | 触发 | 异常类型 |
|---|---|---|
| `paperpilot/skills/` 目录不存在 | 环境坏了 | `FileNotFoundError` |
| 某 .md 文件读失败 (权限 / IO error) | OS 错 | 原生 OSError |
| frontmatter 缺 `---` 起止分隔 | skill 文件写坏 | `ValueError("skill <path> frontmatter invalid: missing '---' delimiters")` |
| frontmatter 缺必填字段 (name/description/when_to_use) | skill 文件写坏 | `ValueError("skill <path> frontmatter missing required field: <key>")` |
| 多 skill 文件 name 冲突 | 两个 .md frontmatter name 相同 | `ValueError("duplicate skill name: <name> in <path1> and <path2>")` |

理由: skill loading 是 Day 9 起的核心机制, 任何一个 skill 文件坏掉都该让用户立刻发现, 而不是 demo 时静默降级。

### B. 运行时 soft-fail (tool raise → agent_loop 转 is_error=true → LLM 决策)

| 失败 | 触发 | LLM 看到 |
|---|---|---|
| `load_skill` 传错 name | 不存在 | `SkillNotFoundError("skill 'X' not found; available: [a, b, c]")` → "我打错名字了, 换 a/b/c 之一" |
| `load_skill` 传空 name / 缺字段 | LLM 漏字段 | handler 取 `args["name"]` 抛 KeyError; agent_loop 转 is_error=True, content="Error: KeyError: 'name'" → LLM 看到自己补上 name 重试 |

**全部不做**: 自动 fuzzy match name / 自动 fallback 到默认 skill / 重试。LLM 看到错误自己决定下一步。

---

## 8. 测试策略

### 层 1: 单元测试 (`tests/builtin_tools/test_skill_loader.py`, 秒级, CI 跑)

| 测试 | 测什么 |
|---|---|
| `test_registry_scans_skills_dir` | tmp dir 写 3 个合法 .md; 断言 `list_metadata()` 返 3 项, 按 name 字典序 |
| `test_frontmatter_parse_happy` | 单 .md 三字段齐全; 断言解析正确 |
| `test_frontmatter_missing_delimiter_raises` | 文件不以 `---` 起; 断言 ValueError |
| `test_frontmatter_missing_required_field_raises` | 缺 description; 断言 ValueError 含字段名 |
| `test_frontmatter_extra_field_ignored` | 多余字段; 断言不抛错, 多余字段不出现在 metadata |
| `test_load_returns_full_content_with_frontmatter` | load 后字符串以 `---\n` 起 + 含 body |
| `test_load_unknown_skill_raises` | 未知 name; 断言 `SkillNotFoundError`, message 含 available 列表 |
| `test_duplicate_skill_name_raises` | 两 .md 同 name; 断言 ValueError 含两个 path |
| `test_empty_dir_returns_empty_metadata` | 空目录; 断言 `list_metadata() == []`, system prompt 拼接段为空字符串 |
| `test_render_skill_section_format` | 给定 3 个 metadata; 断言渲染含每个 name + description + when_to_use 标签 |
| `test_load_skill_tool_routes_to_registry` | mock SkillRegistry; 断言 `load_skill_tool(registry)` 返 Tool 对象, handler 调用路由到 registry.load |
| `test_load_skill_tool_unknown_name_returns_error` | handler 调用未知 name; 断言抛 SkillNotFoundError (由 agent_loop 转 is_error) |

### 层 2: main.py 集成单测 (`tests/test_main_skill_integration.py`, 秒级)

| 测试 | 测什么 |
|---|---|
| `test_main_injects_skill_list_into_system_prompt` | 启动 main 路径, mock LLMClient.call 截获 system 参数; 断言 system 含 3 个 skill name |
| `test_main_registers_load_skill_tool` | 截获 tools 参数; 断言 tools 列表含 name="load_skill" 的 Tool |

(若 main.py 当前结构难注入 mock, 改成测 `build_system_prompt()` 和 `build_tools()` 这两个新拆函数)

### 层 3: Day 9 smoke (`scripts/day9_smoke.py`, 端到端真 LLM + 真 MCP)

```
prompt: "帮我深读 arxiv 论文 1706.03762, 重点讲 multi-head attention 是怎么定义的。"

期望 tool 序列必含:
  load_skill(name="deep-read-paper")
  → mcp__arxiv__download_paper(arxiv_id="1706.03762")
  → mcp__colbert__build_index(documents=[...])
  → mcp__colbert__search(query="multi-head attention definition", ...)

完工: 退出码 0 + stdout 含 "Day 9 smoke PASSED"
       tracer 必须抓到上述 4 个 tool name
       最终回答非空且基于 colbert.search 返回段落
```

**smoke prompt 强约束程度**: prompt 里**不**显式说"先调 load_skill", 让 LLM 自主决定 —— 这是 skill loading 机制存在的意义; 如果 LLM 不主动 load 而直接调 mcp tool 也算技术上"通", 但 smoke 应该 fail (否则 skill loading 没起作用)。

verification 必须包含 `assert "load_skill" in saw, "skill loading 没起作用"`。

**编码** (沿用 Day 5/6/8 模式): smoke 顶部 `sys.stdout.reconfigure(encoding="utf-8")`。

---

## 9. 工作量预估 + 完工标志

| 阶段 | 预估 |
|---|---|
| 写 plan | ~30 min |
| Task 1: SkillRegistry + frontmatter 解析 + 12 个单测 | ~60 min |
| Task 2: 写 3 个 skill markdown | ~60 min |
| Task 3: main.py 集成 (system prompt 拼接 + tool 注册 + 集成单测) | ~45 min |
| Task 4: scripts/day9_smoke.py + 真 LLM 联调 | ~45 min |
| **合计** | **~3.5 小时** (预算 6-7h, 留 ~3h buffer 给 LLM 真不主动 load_skill 时调 prompt) |

**完工标志 (Definition of Done)**:
1. `pytest tests/` 全绿 (含 `test_skill_loader.py` 12 个新测; main 集成测)
2. `python scripts/day5_smoke.py` 无回归
3. `python scripts/day6_smoke.py` 无回归
4. `python scripts/day8_smoke.py` 无回归
5. `python scripts/day9_smoke.py` 退出 0 + 打印 PASSED
6. day9 smoke tracer 必须抓到 `load_skill` 调用 (验证机制真起作用)
7. `git grep "TODO\|FIXME" paperpilot/builtin_tools/ paperpilot/skills/` 空
8. 所有改动按合理粒度分 commit

---

## 10. trip wire (Day 7 监控复用)

`load_skill` tool_result 是 **str (markdown 文本)**, 不是 list[dict] —— 与 Day 7 锁的 FastMCP 多 JSON 对象拼接监控不冲突 (那是针对 list[dict] 输出的 tool 的)。

新增以下监控条件:

| 触发条件 | 行动 |
|---|---|
| 单个 skill .md 文件 > 5KB | 检查是否塞了不必要的内容; skill 应 < 1KB 才符合"prompt 片段"定位 |
| skill 总数 > 8 | system prompt 拼接段开始膨胀, 考虑分级或子目录 |
| 同一 session 内 LLM 反复 load 同一 skill > 3 次 | LLM 没正确利用 conversation history; 可能是 compact_context (Day 9+ 之后) 把 skill body 截断了 |

第一版**不写**监控脚本 —— 上述条件作为"未来某天数据量级跳变"的提醒钉子。

---

## 11. 不在范围 / 推迟到未来的事

| 项 | 推迟到何时触发 |
|---|---|
| `research_todo` 内嵌 tool (TodoWrite 风格) | Day 10 或与 paper_deep_read 同期 |
| `paper_deep_read` subagent (杀手锏) | Week 3 (原方案 Day 18) |
| `compact_context` 内嵌 tool | 长 PDF 真撞 token 上限时 |
| `pdf-parse-mcp` (第 4 个 MCP server) | Day 10+ (原方案 Day 6 已推迟) |
| `vlm-mcp` + Qwen-VL | Week 3 (原方案 Day 15-17) |
| skill 优先级 / 分级 | skill 总数 > 8 时 |
| skill 内 sub-skill transclusion | 出现重复 prose 真痛时 |
| skill hot reload | 开发体验真痛时 (重启 demo 慢的话) |
| skill 执行可观测性 (哪个 skill 跑通哪些 tool 链) | 简历 / 面试需要数据支撑时 |
| skill body 缓存 / 去重 (避免 LLM 反复 load 浪费 token) | 真出现这种现象时 |
| `tools_hint` 字段 (skill 声明它主要用哪些 tool) | LLM 在多 skill 同时 load 后混淆 tool 时 |

---

## 附: 与已有 spec / 架构的衔接点

- **复用 Day 4 agent_loop** —— 内嵌 tool 与 mcp tool 走同一 `Tool` dataclass + handler dispatch, agent_loop 一行不改
- **复用 Day 5 mcp_client 启动** —— skill 注册不影响 mcp_client; main.py 启动顺序: mcp_client.startup() → SkillRegistry.scan() → tools 合并 → agent_loop
- **复用 Day 6 build_index 参数 repair** —— 不冲突; deep-read-paper skill 的 prose 与 main.py SYSTEM_PROMPT 里 build_index 强约束一致, 双保险
- **复用 Day 7 trip wire** —— `load_skill` 不是 list[dict] 输出, 不撞 FastMCP 监控
- **复用 Day 8 graph-mcp** —— `explore-citations` 和 `find-classics` 两个 skill 直接 prose 调 graph 4 个 tool, 验证 Day 8 工作真能被 LLM 串起来
- **不与 nanobot 相关** —— 方案 C 已砍 nanobot, skill loading 机制完全自写, 与 Claude Code s05 对齐
