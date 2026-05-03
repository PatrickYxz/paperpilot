# Day 9 Skill Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 PaperPilot 第一个 L2 内嵌 tool `load_skill`,让 LLM 在启动时通过 system prompt 看到 skill 清单,按需调 `load_skill(name="...")` 拿完整 markdown 步骤说明,再按 prose 调底层 mcp tool。

**Architecture:** 新建 `paperpilot/builtin_tools/skill_loader.py`,实现 `SkillRegistry`(扫 `paperpilot/skills/*.md`,解析 frontmatter)和 `load_skill_tool(registry)`(返回 `Tool` 对象)。`main.py` 启动时初始化 registry,把 metadata 渲染成 system prompt 末尾的"可用 skill"列表段,把 `load_skill` 与 mcp tools 一起塞进 agent_loop。skill body 通过 tool_result 字符串自然进 conversation history,LLM 后续轮次按 prose 调 mcp tool。

**Tech Stack:** Python stdlib(无 PyYAML 依赖,frontmatter 手写解析); `paperpilot.core.adapter.Tool` dataclass; pytest。

**Spec:** `docs/superpowers/specs/2026-05-03-day9-skill-loading-design.md`

---

## File Structure

| 路径 | 状态 | 责任 |
|---|---|---|
| `paperpilot/builtin_tools/__init__.py` | 新 | 模块导出(空 / 简单 re-export) |
| `paperpilot/builtin_tools/skill_loader.py` | 新 | `SkillRegistry` + `SkillNotFoundError` + `load_skill_tool(registry)` 工厂 + `render_skill_section(metadata)` |
| `paperpilot/skills/deep-read-paper.md` | 新 | colbert 单篇深读 prose |
| `paperpilot/skills/explore-citations.md` | 新 | graph 引用拓扑探索 prose |
| `paperpilot/skills/find-classics.md` | 新 | arxiv + graph 共引经典 prose |
| `paperpilot/main.py` | 改 | 拆 `_build_system_prompt()` / `_build_tools()`;`run()` 初始化 SkillRegistry 并合并 tools |
| `tests/builtin_tools/__init__.py` | 新 | 空 |
| `tests/builtin_tools/test_skill_loader.py` | 新 | SkillRegistry / frontmatter / load_skill_tool / render 单测 |
| `tests/test_main_integration.py` | 新 | main 启动后 system prompt 含 skill 列表 + tools 含 load_skill 集成测 |
| `scripts/day9_smoke.py` | 新 | 端到端真 LLM:load_skill → arxiv → colbert 全链路 |

---

## Task 1: SkillRegistry 数据结构与 frontmatter 解析

**Files:**
- Create: `paperpilot/builtin_tools/__init__.py`
- Create: `paperpilot/builtin_tools/skill_loader.py`
- Create: `tests/builtin_tools/__init__.py`
- Create: `tests/builtin_tools/test_skill_loader.py`

- [ ] **Step 1.1: Create empty package init files**

```bash
mkdir -p paperpilot/builtin_tools tests/builtin_tools
```

Write `paperpilot/builtin_tools/__init__.py`:
```python
"""L2 内嵌 tool:与 mcp_servers 同级,跑在 agent 进程内。"""
```

Write `tests/builtin_tools/__init__.py`:
```python
```

- [ ] **Step 1.2: Write failing tests for frontmatter parsing**

Write `tests/builtin_tools/test_skill_loader.py`:
```python
"""SkillRegistry / frontmatter 解析 / load_skill_tool 单测。"""
from __future__ import annotations

import pytest

from paperpilot.builtin_tools.skill_loader import (
    SkillNotFoundError,
    SkillRegistry,
    load_skill_tool,
    render_skill_section,
)


SKILL_OK = """\
---
name: deep-read-paper
description: 深读单篇 arxiv 论文
when_to_use: 用户给定 arxiv id 要求详细讲解
---

# Deep Read Paper

正文内容。
"""

SKILL_OK_2 = """\
---
name: explore-citations
description: 引用拓扑探索
when_to_use: 用户问引用关系
---

正文。
"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_registry_scans_skills_dir(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    _write(tmp_path, "explore-citations.md", SKILL_OK_2)
    reg = SkillRegistry(tmp_path)
    md = reg.list_metadata()
    assert [m["name"] for m in md] == ["deep-read-paper", "explore-citations"]
    assert md[0]["description"] == "深读单篇 arxiv 论文"
    assert md[0]["when_to_use"] == "用户给定 arxiv id 要求详细讲解"


def test_frontmatter_missing_delimiter_raises(tmp_path):
    _write(tmp_path, "bad.md", "no frontmatter here\n")
    with pytest.raises(ValueError, match="frontmatter"):
        SkillRegistry(tmp_path)


def test_frontmatter_missing_required_field_raises(tmp_path):
    _write(
        tmp_path,
        "bad.md",
        "---\nname: x\ndescription: y\n---\nbody\n",
    )
    with pytest.raises(ValueError, match="when_to_use"):
        SkillRegistry(tmp_path)


def test_frontmatter_extra_field_ignored(tmp_path):
    _write(
        tmp_path,
        "ok.md",
        (
            "---\nname: x\ndescription: y\nwhen_to_use: z\nextra: ignored\n"
            "---\nbody\n"
        ),
    )
    reg = SkillRegistry(tmp_path)
    md = reg.list_metadata()
    assert md == [{"name": "x", "description": "y", "when_to_use": "z"}]


def test_load_returns_full_content_with_frontmatter(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    reg = SkillRegistry(tmp_path)
    body = reg.load("deep-read-paper")
    assert body.startswith("---\n")
    assert "Deep Read Paper" in body
    assert "正文内容" in body


def test_load_unknown_skill_raises(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    reg = SkillRegistry(tmp_path)
    with pytest.raises(SkillNotFoundError) as exc:
        reg.load("nonexistent")
    msg = str(exc.value)
    assert "nonexistent" in msg
    assert "deep-read-paper" in msg


def test_duplicate_skill_name_raises(tmp_path):
    _write(tmp_path, "a.md", SKILL_OK)
    _write(tmp_path, "b.md", SKILL_OK)
    with pytest.raises(ValueError, match="duplicate"):
        SkillRegistry(tmp_path)


def test_empty_dir_returns_empty_metadata(tmp_path):
    reg = SkillRegistry(tmp_path)
    assert reg.list_metadata() == []


def test_skills_dir_not_found_raises(tmp_path):
    missing = tmp_path / "no-such-dir"
    with pytest.raises(FileNotFoundError):
        SkillRegistry(missing)


def test_render_skill_section_format():
    md = [
        {"name": "a", "description": "desc-a", "when_to_use": "use-a"},
        {"name": "b", "description": "desc-b", "when_to_use": "use-b"},
    ]
    section = render_skill_section(md)
    assert "## 可用 skill" in section
    assert "**a**" in section
    assert "desc-a" in section
    assert "use-a" in section
    assert "**b**" in section
    assert "load_skill" in section


def test_render_skill_section_empty():
    assert render_skill_section([]) == ""


def test_load_skill_tool_routes_to_registry(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    reg = SkillRegistry(tmp_path)
    tool = load_skill_tool(reg)
    assert tool.name == "load_skill"
    assert "name" in tool.input_schema["properties"]
    body = tool.handler({"name": "deep-read-paper"})
    assert "Deep Read Paper" in body


def test_load_skill_tool_unknown_name_raises(tmp_path):
    _write(tmp_path, "deep-read-paper.md", SKILL_OK)
    reg = SkillRegistry(tmp_path)
    tool = load_skill_tool(reg)
    with pytest.raises(SkillNotFoundError):
        tool.handler({"name": "ghost"})
```

- [ ] **Step 1.3: Run tests to verify they fail**

Run: `python -m pytest tests/builtin_tools/test_skill_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'paperpilot.builtin_tools.skill_loader'`

- [ ] **Step 1.4: Implement skill_loader.py**

Write `paperpilot/builtin_tools/skill_loader.py`:
```python
"""SkillRegistry + load_skill 内嵌 tool。

skill = paperpilot/skills/*.md, YAML frontmatter (name/description/when_to_use)
+ markdown body。启动时全部扫一遍,frontmatter 注入 system prompt;body 由
load_skill tool 按需返。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from paperpilot.core.adapter import Tool

REQUIRED_FIELDS = ("name", "description", "when_to_use")


class SkillNotFoundError(LookupError):
    """LLM 调 load_skill 传错 name 时抛;由 agent_loop 转 is_error=True。"""


class SkillRegistry:
    def __init__(self, skills_dir: Path):
        if not skills_dir.exists():
            raise FileNotFoundError(f"skills dir not found: {skills_dir}")
        self._dir = skills_dir
        self._skills: dict[str, dict] = {}
        self._scan()

    def list_metadata(self) -> list[dict]:
        return [
            {
                "name": s["name"],
                "description": s["description"],
                "when_to_use": s["when_to_use"],
            }
            for s in self._skills.values()
        ]

    def load(self, name: str) -> str:
        skill = self._skills.get(name)
        if skill is None:
            available = sorted(self._skills.keys())
            raise SkillNotFoundError(
                f"skill '{name}' not found; available: {available}"
            )
        return skill["path"].read_text(encoding="utf-8")

    def _scan(self) -> None:
        paths = sorted(self._dir.glob("*.md"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            meta = _parse_frontmatter(text, path)
            name = meta["name"]
            if name in self._skills:
                first = self._skills[name]["path"]
                raise ValueError(
                    f"duplicate skill name: {name} in {first} and {path}"
                )
            self._skills[name] = {**meta, "path": path}


def _parse_frontmatter(text: str, path: Path) -> dict:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(
            f"skill {path} frontmatter invalid: missing leading '---' delimiter"
        )
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end == -1:
        raise ValueError(
            f"skill {path} frontmatter invalid: missing closing '---' delimiter"
        )
    meta: dict = {}
    for raw in lines[1:end]:
        if not raw.strip():
            continue
        if ":" not in raw:
            raise ValueError(
                f"skill {path} frontmatter invalid: line not 'key: value': {raw!r}"
            )
        key, _, value = raw.partition(":")
        meta[key.strip()] = value.strip()
    for field in REQUIRED_FIELDS:
        if field not in meta or not meta[field]:
            raise ValueError(
                f"skill {path} frontmatter missing required field: {field}"
            )
    return {field: meta[field] for field in REQUIRED_FIELDS}


def render_skill_section(metadata: list[dict]) -> str:
    if not metadata:
        return ""
    lines = ["", "## 可用 skill (按需调 load_skill 加载完整步骤)", ""]
    for m in metadata:
        lines.append(f"- **{m['name']}**: {m['description']}")
        lines.append(f"  适用: {m['when_to_use']}")
    lines.append("")
    lines.append(
        "看到适合场景时调 load_skill(name=\"...\") 拿完整步骤,再按步骤调 mcp tool。"
    )
    return "\n".join(lines)


def load_skill_tool(registry: SkillRegistry) -> Tool:
    def _handler(args: dict) -> Any:
        return registry.load(args["name"])

    return Tool(
        name="load_skill",
        description=(
            "加载一个 skill 的完整说明书。skill 列表见 system prompt 的"
            " '可用 skill' 段。tool_result 是该 skill 的 markdown 全文;"
            "你照着步骤调底层 mcp tool 即可。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "skill name, 例如 deep-read-paper",
                },
            },
            "required": ["name"],
        },
        handler=_handler,
    )
```

- [ ] **Step 1.5: Run tests to verify they pass**

Run: `python -m pytest tests/builtin_tools/test_skill_loader.py -v`
Expected: PASS, 13 tests passed

- [ ] **Step 1.6: Commit**

```bash
git add paperpilot/builtin_tools tests/builtin_tools
git commit -m "Day 9 Task 1: SkillRegistry + frontmatter 解析 + load_skill_tool"
```

---

## Task 2: 写 3 个 skill markdown

**Files:**
- Create: `paperpilot/skills/deep-read-paper.md`
- Create: `paperpilot/skills/explore-citations.md`
- Create: `paperpilot/skills/find-classics.md`

- [ ] **Step 2.1: Create skills directory and deep-read-paper.md**

```bash
mkdir -p paperpilot/skills
```

Write `paperpilot/skills/deep-read-paper.md`:
```markdown
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
1. 拿全文: `mcp__arxiv__download_paper(arxiv_id="...")`
2. 建索引: `mcp__colbert__build_index(documents=[download_paper 返回值])`
   - documents 必须是非空 list, 每项含 paper_id 和 text
   - 直接传 download_paper 的返回对象即可
3. 多轮检索: 针对用户问题里的关键概念, 调 `mcp__colbert__search(query="...", top_k=3)`
   - 一个 query 不够就拆成多个 query 多调几次
4. 综合回答: 基于 colbert.search 返回的具体段落综合回答, 引用段落原文

## 注意
- 不要只看 abstract 回答细节问题
- 不要编造段落, 只用 colbert.search 真实返回的内容
- 如果 search 返回与问题无关, 换一个 query 再 search,而不是凭印象回答
```

- [ ] **Step 2.2: Create explore-citations.md**

Write `paperpilot/skills/explore-citations.md`:
```markdown
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
1. 构图: `mcp__graph__build_graph(arxiv_ids=[感兴趣的 paper id 列表])`
   - 返回值含 missing 列表, SS 查不到的 id 跳过即可
2. 看引用 / 被引: `mcp__graph__get_neighbors(arxiv_id="...", direction="...", limit=10)`
   - direction 可选 "references"(它引谁) / "citations"(谁引它) / "both"(两者都看)
3. 看路径(可选): `mcp__graph__get_shortest_path(from_id="...", to_id="...")`
   - length=-1 表示无路径; 有路径时 path 列表给出中间节点
4. 综合回答: 基于 neighbors / path 返回的 title + year + authors 给学术脉络说明

## 注意
- get_neighbors / shortest_path 之前必须先 build_graph 把目标 paper 拉进图
- direction 默认 both, 用户明确问"它引了什么"用 references, 问"谁引了它"用 citations
- 不要编造 title 或 year, 只用 graph tool 真实返回的字段
```

- [ ] **Step 2.3: Create find-classics.md**

Write `paperpilot/skills/find-classics.md`:
```markdown
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
1. 找当代 paper: `mcp__arxiv__search_papers(query="该领域关键词", max_results=5)`
2. 把这批 paper 拉进引用图: `mcp__graph__build_graph(arxiv_ids=[选 3-5 篇有代表性的])`
3. 求共引: `mcp__graph__get_common_citations(arxiv_ids=[同上], top_k=10)`
   - 返回 cited_by_count >= 2 的经典文献
4. 综合回答: 把 top 共引文献按 cited_by_count + year 列出, 解释为什么是该领域经典

## 注意
- 至少需要 2 篇 input 才能求共引(单篇没意义)
- 如果 build_graph 的 missing 列表非空, 用剩下的 id 继续, 不要中断
- common_citations 返回的是引用关系, 不一定每篇都"广义经典", 但被多篇 input 共引说明在该 cluster 里有 ground-truth 地位
```

- [ ] **Step 2.4: Verify skills load correctly**

Run quick sanity check via Python:
```bash
python -c "from pathlib import Path; from paperpilot.builtin_tools.skill_loader import SkillRegistry; r = SkillRegistry(Path('paperpilot/skills')); print([m['name'] for m in r.list_metadata()])"
```
Expected output: `['deep-read-paper', 'explore-citations', 'find-classics']`

- [ ] **Step 2.5: Commit**

```bash
git add paperpilot/skills
git commit -m "Day 9 Task 2: 第一批 3 个 skill (deep-read / explore-citations / find-classics)"
```

---

## Task 3: main.py 集成 + 集成测试

**Files:**
- Modify: `paperpilot/main.py`
- Create: `tests/test_main_integration.py`

- [ ] **Step 3.1: Write failing integration test**

Write `tests/test_main_integration.py`:
```python
"""main 启动后 system prompt 含 skill 列表 + tools 含 load_skill。"""
from __future__ import annotations

import pytest

from paperpilot.main import _build_system_prompt, _build_tools


def test_build_system_prompt_includes_skills():
    """system prompt 必须包含三个 skill 的 name 与 description (无外部依赖, fast)。"""
    prompt = _build_system_prompt()
    assert "## 可用 skill" in prompt
    assert "deep-read-paper" in prompt
    assert "explore-citations" in prompt
    assert "find-classics" in prompt
    # 工作原则段不能丢
    assert "PaperPilot" in prompt
    assert "build_index" in prompt


@pytest.mark.slow
def test_build_tools_contains_load_skill_and_mcp_tools():
    """tools 必须包含 load_skill, 且与 mcp tool 同列表。

    标 slow: 真启动 3 个 mcp server 子进程, ~5s。
    """
    tools, mcp = _build_tools()
    try:
        names = [t.name for t in tools]
        assert "load_skill" in names
        # 至少应该有一个 mcp__ 前缀的 tool (来自 mcp_servers.json)
        assert any(n.startswith("mcp__") for n in names)
    finally:
        mcp.close()
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `python -m pytest tests/test_main_integration.py -v`
Expected: FAIL with `ImportError: cannot import name '_build_system_prompt'`

- [ ] **Step 3.3: Refactor main.py — extract `_build_system_prompt` and `_build_tools`**

Replace contents of `paperpilot/main.py`:
```python
"""PaperPilot 顶层入口。Day 5 起;Day 9 加 L2 内嵌 tool load_skill。

CLI:python -m paperpilot.main --query "..."
库:from paperpilot.main import run; run(query, max_iter=8)
"""
from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path

from dotenv import load_dotenv

from paperpilot.builtin_tools.skill_loader import (
    SkillRegistry,
    load_skill_tool,
    render_skill_section,
)
from paperpilot.core import Guardrail, LLMClient, agent_loop
from paperpilot.core.adapter import Tool
from paperpilot.tools.mcp_client import MCPClient

MANIFEST_PATH = Path(__file__).parent / "mcp_servers.json"
SKILLS_DIR = Path(__file__).parent / "skills"

SYSTEM_PROMPT_BASE = """你是 PaperPilot,一个学术论文研究助手。

工作原则:
- 有 tool 可用时优先调 tool;不要自己编造论文标题、作者或 arxiv id
- 一次只解决用户问的事,不主动扩展任务范围
- tool 报错时,根据错误信息决定:重试(换参数) / 换工具 / 告诉用户失败原因
- 调 tool 时必须按 schema 传完整必填参数;如果错误提示缺字段,下一轮必须补齐字段,不要重复同一个空参数
- 调 mcp__colbert__build_index 时,documents 必须是非空列表,每项包含 paper_id 和 text;通常直接使用 mcp__arxiv__download_paper 返回的对象组成 documents=[download_result]
""".strip()


def _build_system_prompt() -> str:
    registry = SkillRegistry(SKILLS_DIR)
    return SYSTEM_PROMPT_BASE + render_skill_section(registry.list_metadata())


def _build_tools() -> tuple[list[Tool], MCPClient]:
    """返 (tools, mcp_client);调用方负责 mcp_client.close()。"""
    registry = SkillRegistry(SKILLS_DIR)

    mcp = MCPClient(MANIFEST_PATH)
    try:
        mcp.start()
        tools: list[Tool] = [load_skill_tool(registry), *mcp.list_tools()]
        return tools, mcp
    except Exception:
        mcp.close()
        raise


def _default_logger(kind: str, payload: dict) -> None:
    if kind == "tool_call":
        print(f"  → {payload['name']}({payload['arguments']})")
    elif kind == "tool_result":
        print(f"  ← {payload['name']}: {payload['content'][:120]}...")
    elif kind == "guardrail_stop":
        print(f"  ⚠ guardrail: {payload['reason']}")


def run(query: str, *, max_iter: int = 8, on_event=None) -> list[dict]:
    """跑一次完整 agent 会话,返回最终 messages。"""
    load_dotenv()

    tools, mcp = _build_tools()
    atexit.register(mcp.close)

    messages = [{"role": "user", "content": query}]
    return agent_loop(
        messages,
        system=_build_system_prompt(),
        tools=tools,
        client=LLMClient(),
        guardrail=Guardrail(
            max_iterations=max_iter,
            budget_tokens=int(os.environ.get("BUDGET_TOKENS", 50_000)),
        ),
        on_event=on_event or _default_logger,
    )


def main() -> None:
    p = argparse.ArgumentParser(prog="paperpilot")
    p.add_argument("--query", required=True)
    p.add_argument("--max-iter", type=int, default=8)
    args = p.parse_args()
    messages = run(args.query, max_iter=args.max_iter)
    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for b in last:
            if hasattr(b, "text"):
                print(b.text)
    else:
        print(last)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4: Run test to verify it passes**

Run默认(快测): `python -m pytest tests/test_main_integration.py -v`
Expected: 1 passed, 1 deselected (slow 测默认跳过)

Run slow 测: `python -m pytest tests/test_main_integration.py -v -m slow`
Expected: 1 passed (真启 mcp servers, ~5s)

- [ ] **Step 3.5: Run full default test suite to check for regressions**

Run: `python -m pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py`
Expected: PASS, all default tests green (原 43 + Task 1 新 13 + Task 3 新 1 fast = 57 个; slow 5 个 deselected)

- [ ] **Step 3.6: Commit**

```bash
git add paperpilot/main.py tests/test_main_integration.py
git commit -m "Day 9 Task 3: main 集成 load_skill + skill list 注入 system prompt"
```

---

## Task 4: Day 9 端到端 smoke

**Files:**
- Create: `scripts/day9_smoke.py`

- [ ] **Step 4.1: Write day9_smoke.py**

Write `scripts/day9_smoke.py`:
```python
"""Day 9 smoke: Main Loop -> load_skill -> arxiv-mcp + colbert-mcp -> LLM 答案。

让 LLM 自主选择 load_skill (deep-read-paper),不在 prompt 里显式说"先 load_skill"。
若 LLM 没主动 load skill 直接调 mcp tool, smoke fail (skill loading 没起作用)。
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperpilot.main import run

EXPECT_TOOLS = {
    "load_skill",
    "mcp__arxiv__download_paper",
    "mcp__colbert__build_index",
    "mcp__colbert__search",
}


def main() -> None:
    saw: set[str] = set()

    def tracer(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            saw.add(payload["name"])
            print(f"  -> {payload['name']}({_preview(payload.get('arguments', {}))})")
        elif kind == "tool_result":
            content = payload.get("content", "")
            preview = content[:160] if isinstance(content, str) else str(content)[:160]
            print(f"  <- {payload['name']}: {preview}...")
        elif kind == "guardrail_stop":
            print(f"  !! guardrail: {payload['reason']}")
        elif kind == "tool_arg_repair":
            print(f"  ~ repaired {payload['name']}: {payload['repaired']}")

    messages = run(
        "帮我深读 arxiv 论文 1706.03762, 重点讲 multi-head attention 是怎么定义的。"
        "回答必须基于论文具体段落, 不要只看 abstract。",
        max_iter=10,
        on_event=tracer,
    )

    print("\n=== FINAL ===")
    last = messages[-1].get("content")
    if isinstance(last, list):
        for b in last:
            if hasattr(b, "text"):
                print(b.text)
    else:
        print(last)

    missing = EXPECT_TOOLS - saw
    assert not missing, f"FAIL: expected tools missing: {missing}; saw {saw}"
    assert "load_skill" in saw, "FAIL: skill loading 没起作用 (LLM 没主动 load_skill)"
    print("\nDay 9 smoke PASSED")


def _preview(args: dict) -> dict:
    if "documents" not in args:
        return args
    docs = args.get("documents") or []
    out = []
    for doc in docs:
        if isinstance(doc, dict):
            out.append({"paper_id": doc.get("paper_id"), "text_len": len(doc.get("text", ""))})
    return {**args, "documents": out}


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Run smoke (real LLM + real MCP servers)**

Run: `python scripts/day9_smoke.py`
Expected:
- tracer 应抓到 4 个 tool name(`load_skill` / `mcp__arxiv__download_paper` / `mcp__colbert__build_index` / `mcp__colbert__search`)
- 退出码 0,stdout 末尾 `Day 9 smoke PASSED`
- 最终回答含基于段落的 multi-head attention 定义说明

如果失败,可能原因:
- LLM 没主动 load skill → 检查 `_build_system_prompt()` 输出,确认 skill list 段拼进去了
- arxiv 429 → 等几分钟重跑
- 链路断在 colbert → 检查 Day 6 自动 repair 还在 (loop.py 应有 `_should_repair_build_index_args`)

- [ ] **Step 4.3: Verify regression on prior smoke scripts (optional but recommended)**

```bash
python scripts/day5_smoke.py
python scripts/day6_smoke.py
python scripts/day8_smoke.py
```
Expected: 三个全 PASSED

(若 arxiv 429 或外部服务抖,记录但不阻塞 commit;day9 smoke 通过即视为完工。)

- [ ] **Step 4.4: Commit**

```bash
git add scripts/day9_smoke.py
git commit -m "Day 9 Task 4: day9_smoke 端到端 (load_skill -> arxiv -> colbert -> LLM 答案)"
```

---

## Definition of Done

1. `python -m pytest tests -q --ignore=tests/mcp_servers/test_graph_via_client.py` 全绿(原 43 + Task 1 新 13 + Task 3 新 1 fast = 57 个;slow 5 个 deselected)
2. `python scripts/day5_smoke.py` 无回归(可选,arxiv 抖动允许跳过)
3. `python scripts/day6_smoke.py` 无回归(可选)
4. `python scripts/day8_smoke.py` 无回归(可选)
5. `python scripts/day9_smoke.py` 退出 0 + 打印 `Day 9 smoke PASSED`
6. day9 smoke tracer 必须抓到 `load_skill` 调用(验证机制真起作用)
7. `git grep -E "TODO|FIXME" paperpilot/builtin_tools paperpilot/skills` 空
8. 4 个 commit:Task 1 / Task 2 / Task 3 / Task 4
