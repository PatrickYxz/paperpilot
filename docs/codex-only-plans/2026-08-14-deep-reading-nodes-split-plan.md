# Deep Reading LangGraph Nodes Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `paperpilot/deep_reading/nodes.py` 拆成 `nodes/` 包，并保证 6 个 LangGraph 节点各自位于独立文件且运行行为不变。

**Architecture:** `nodes/__init__.py` 维持现有包级导入；6 个同名节点模块只承载各自阶段，`context.py`、`binding.py`、`validation.py` 承载明确的跨节点依赖。迁移使用当前工作区中的 `nodes.py` 为唯一来源，保留尚未提交的 `initialize_turn -> Command(goto=...)` 实现。

**Tech Stack:** Python 3.12、LangGraph 1.2、LangChain 1.3、Pydantic 2、pytest。

## Global Constraints

- 不改变节点名称、Graph 边、执行顺序、State 字段、异常类型、异常消息、Prompt、MCP 工具调用、数据库事务或 checkpoint 行为。
- 不修改公开导入路径 `paperpilot.deep_reading.nodes`，也不修改 `paperpilot.deep_reading` 的导出集合。
- 不引入新依赖，不拆分 `tests/deep_reading/test_nodes.py`，不重构 Graph 之外的运行链路。
- 迁移必须读取当前工作区版本，不能从 `HEAD` 恢复 `nodes.py`；当前未提交的 `Command` 路由实现及其测试必须保留。
- 工作区存在与本任务重叠的用户修改，未经用户明确授权不得创建包含这些修改的实现提交。
- 设计依据：`docs/codex-only-plans/2026-08-14-deep-reading-nodes-split-design.md`。

## Target File Map

| 文件 | 唯一职责 | 从当前 `nodes.py` 迁移的符号 |
| --- | --- | --- |
| `nodes/__init__.py` | 稳定公开导入 | 只再导出已批准的 8 个公开符号 |
| `nodes/context.py` | 每次 Graph 运行的可信依赖 | `PaperSearch`, `EventSink`, `DeepReadingContext` |
| `nodes/binding.py` | State 与业务对象绑定校验 | `_validate_runtime_binding`, `_required_binding_text` |
| `nodes/validation.py` | 跨节点结构和引用校验 | `_required_text`, `_validated_research_result`, `_validated_answer_draft`, `_validate_answer_citations` |
| `nodes/initialize_turn.py` | 清理本轮字段并路由 | `initialize_turn` |
| `nodes/summarize_history.py` | 摘要与历史窗口 | `needs_summary`, `summarize_history`, `_estimated_tokens`, `_message_character_count`, `_recent_turns` |
| `nodes/prepare_primary_paper.py` | 下载并索引主论文 | `_DOWNLOAD_TOOL`, `_BUILD_TOOL`, `prepare_primary_paper`, `_canonical_arxiv_id`, `_call_prepare_mcp_json`, `_safe_prepare_event_arguments`, `_indexed_paper_ids` |
| `nodes/research_evidence.py` | 调用受限 Research Agent | `research_evidence` |
| `nodes/write_answer.py` | 生成并校验结构化答案 | `write_answer`, `_paper_metadata` |
| `nodes/publish_result.py` | 幂等发布最终结果 | `publish_result`, `_publication_metadata`, `_read_authoritative_publication`, `_used_paper_inputs` |

---

### Task 1: 锁定迁移前行为和“每节点一文件”结构契约

**Files:**
- Modify: `tests/deep_reading/test_nodes.py`

**Interfaces:**
- Consumes: 当前包级节点函数 `initialize_turn`, `summarize_history`, `prepare_primary_paper`, `research_evidence`, `write_answer`, `publish_result`。
- Produces: 结构测试 `test_each_graph_node_lives_in_its_dedicated_module`，后续实现只有在模块边界正确且包级再导出指向同一函数对象时才能通过。

- [ ] **Step 1: 记录当前重叠修改并运行迁移前基线**

Run:

```bash
git status --short
git diff -- paperpilot/deep_reading/__init__.py paperpilot/deep_reading/graph.py paperpilot/deep_reading/nodes.py tests/deep_reading/test_graph.py tests/deep_reading/test_nodes.py
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/deep_reading/test_nodes.py tests/deep_reading/test_graph.py -q
```

Expected: Git diff 仍包含用户当前的 `Command` 路由改动；两份测试在开始结构迁移前通过。如果基线失败，先判断失败是否来自当前未提交工作，不能用回退用户修改的方式制造绿色基线。

- [ ] **Step 2: 写入节点模块结构测试**

在 `tests/deep_reading/test_nodes.py` 的标准库导入区增加：

```python
from importlib import import_module
```

在节点状态基础测试附近增加：

```python
@pytest.mark.parametrize(
    ("node_name", "module_name"),
    [
        ("initialize_turn", "paperpilot.deep_reading.nodes.initialize_turn"),
        ("summarize_history", "paperpilot.deep_reading.nodes.summarize_history"),
        (
            "prepare_primary_paper",
            "paperpilot.deep_reading.nodes.prepare_primary_paper",
        ),
        ("research_evidence", "paperpilot.deep_reading.nodes.research_evidence"),
        ("write_answer", "paperpilot.deep_reading.nodes.write_answer"),
        ("publish_result", "paperpilot.deep_reading.nodes.publish_result"),
    ],
)
def test_each_graph_node_lives_in_its_dedicated_module(
    node_name: str,
    module_name: str,
) -> None:
    module = import_module(module_name)
    node = getattr(nodes_module, node_name)

    assert getattr(module, node_name) is node
    assert node.__module__ == module_name
```

- [ ] **Step 3: 运行结构测试并确认它按预期失败**

Run:

```bash
.venv/bin/python -m pytest tests/deep_reading/test_nodes.py::test_each_graph_node_lives_in_its_dedicated_module -q
```

Expected: 6 个参数用例因为 `paperpilot.deep_reading.nodes` 仍是单文件模块、不能导入子模块而失败；失败原因不能是缺少项目依赖或测试收集错误。

---

### Task 2: 将当前节点实现机械迁移到职责明确的包

**Files:**
- Delete: `paperpilot/deep_reading/nodes.py`
- Create: `paperpilot/deep_reading/nodes/__init__.py`
- Create: `paperpilot/deep_reading/nodes/context.py`
- Create: `paperpilot/deep_reading/nodes/binding.py`
- Create: `paperpilot/deep_reading/nodes/validation.py`
- Create: `paperpilot/deep_reading/nodes/initialize_turn.py`
- Create: `paperpilot/deep_reading/nodes/summarize_history.py`
- Create: `paperpilot/deep_reading/nodes/prepare_primary_paper.py`
- Create: `paperpilot/deep_reading/nodes/research_evidence.py`
- Create: `paperpilot/deep_reading/nodes/write_answer.py`
- Create: `paperpilot/deep_reading/nodes/publish_result.py`
- Modify: `tests/deep_reading/test_nodes.py`

**Interfaces:**
- Consumes: `DeepReadingState`, `Runtime[DeepReadingContext]`, `ResearchResult`, `AnswerDraft`、当前 `research_agent.py` 错误类型和 `run_research_agent`。
- Produces: 与旧模块相同的 8 个公开符号；每个节点函数的签名、返回值和副作用保持不变。

- [ ] **Step 1: 创建支持模块并逐字迁移对应实现**

按 Target File Map 剪切符号，保持函数体、错误字符串和 dataclass 默认值不变。支持模块的依赖方向必须是：

```text
context.py     -> papers + tools.types + web.task_store
binding.py     -> context.py + state.py + research_agent errors + web.task_store records
validation.py  -> schemas.py + research_agent errors + pydantic
```

`context.py` 保留以下公开形状：

```python
PaperSearch = Callable[[str, int], list[PaperCandidate]]
EventSink = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class DeepReadingContext:
    user_id: str
    conversation_id: str
    task_id: str
    current_user_message_id: str
    base_checkpoint_id: str | None
    task_store: TaskStore
    model: Any
    mcp_tools: Mapping[str, Tool]
    paper_search: PaperSearch
    event_sink: EventSink
    summary_token_threshold: int = 32_000
    summary_recent_turns: int = 6
    research_recursion_limit: int = 24
    research_model_call_limit: int = 8
    research_tool_call_limit: int = 12
    research_max_output_tokens: int = 4096
    research_model_retries: int = 1
```

必须把当前 `DeepReadingContext.__post_init__` 原样放在该类中，继续执行正数、最少两次结构化输出尝试和非负重试校验。

- [ ] **Step 2: 创建 6 个节点模块并迁移节点私有逻辑**

按 Target File Map 迁移节点与辅助符号；函数签名保持为：

```python
def initialize_turn(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> Command[Literal["summarize_history", "prepare_primary_paper"]]:

def summarize_history(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:

def prepare_primary_paper(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:

def research_evidence(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:

def write_answer(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:

def publish_result(
    state: DeepReadingState,
    runtime: Runtime[DeepReadingContext],
) -> DeepReadingState:
```

使用下列明确依赖，不复制共享函数：

```text
initialize_turn.py       -> summarize_history.needs_summary
summarize_history.py     -> binding._validate_runtime_binding
prepare_primary_paper.py -> binding._validate_runtime_binding + validation._required_text
research_evidence.py     -> binding._validate_runtime_binding + research_agent.run_research_agent
write_answer.py          -> binding._validate_runtime_binding + validation result/citation helpers + summarize_history._recent_turns
publish_result.py        -> binding._validate_runtime_binding + validation result/draft/citation helpers
```

`initialize_turn` 必须保留当前工作区中的 `Command(update=..., goto=...)` 实现；不得恢复已移除的 `route_after_initialize`。

- [ ] **Step 3: 建立兼容的包级公开入口**

`paperpilot/deep_reading/nodes/__init__.py` 使用以下明确导出：

```python
"""LangGraph nodes for the deep-reading workflow."""

from .context import DeepReadingContext
from .initialize_turn import initialize_turn
from .prepare_primary_paper import prepare_primary_paper
from .publish_result import publish_result
from .research_evidence import research_evidence
from .summarize_history import needs_summary, summarize_history
from .write_answer import write_answer

__all__ = [
    "DeepReadingContext",
    "initialize_turn",
    "needs_summary",
    "prepare_primary_paper",
    "publish_result",
    "research_evidence",
    "summarize_history",
    "write_answer",
]
```

`paperpilot/deep_reading/graph.py`、`runner.py`、`research_agent.py` 和包根 `__init__.py` 继续使用原导入路径；除非测试暴露真实循环导入，否则不修改这些调用方。

- [ ] **Step 4: 更新 Research Agent 的测试替换点**

在当前 `test_research_evidence_writes_complete_json_research_result` 中，把对聚合包偶然属性的 patch：

```python
monkeypatch.setattr(nodes_module, "run_research_agent", fake_run)
```

替换为：

```python
research_evidence_module = import_module(
    "paperpilot.deep_reading.nodes.research_evidence"
)
monkeypatch.setattr(research_evidence_module, "run_research_agent", fake_run)
```

不要从 `nodes/__init__.py` 再导出 `run_research_agent`；它不是节点包的公开接口。

- [ ] **Step 5: 删除旧单文件并运行节点级测试**

确认 Target File Map 中每个符号都只有一个实现后删除 `paperpilot/deep_reading/nodes.py`，然后运行：

```bash
.venv/bin/python -m pytest tests/deep_reading/test_nodes.py::test_each_graph_node_lives_in_its_dedicated_module -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/deep_reading/test_nodes.py tests/deep_reading/test_graph.py -q
```

Expected: 结构测试 6 个参数用例通过；节点和 Graph 测试全部通过。`initialize_turn` 的两个路由分支继续返回当前 `Command`。

---

### Task 3: 执行深度阅读回归与仓库级检查

**Files:**
- Modify only if required by a demonstrated regression: files listed in Task 2
- Test: `tests/deep_reading/test_runner.py`
- Test: `tests/deep_reading/test_research_agent.py`
- Test: `tests/architecture/test_repository_allowlist.py`

**Interfaces:**
- Consumes: Task 2 完成后的节点包。
- Produces: Graph、Runner、Research Agent 和仓库约束均未回归的验证证据。

- [ ] **Step 1: 运行深度阅读完整回归**

Run:

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/deep_reading -q
```

Expected: 全部通过。若失败，先确认是导入位置、循环依赖或 monkeypatch 目标变化，再做范围内最小修复；不得借机改变节点业务逻辑。

- [ ] **Step 2: 运行架构白名单与完整测试集**

Run:

```bash
.venv/bin/python -m pytest tests/architecture/test_repository_allowlist.py -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests -q
```

Expected: 两条命令全部通过；完整测试不能访问付费模型或外部服务。

- [ ] **Step 3: 检查最终 diff 和格式**

Run:

```bash
git diff --check
git status --short
git diff --stat
git diff -- paperpilot/deep_reading tests/deep_reading tests/architecture/test_repository_allowlist.py
```

Expected: 无空白错误；旧 `nodes.py` 的有效内容完整分布在新包；没有无关运行时文件变化；用户原有的 `Command` 路由改动仍存在。

- [ ] **Step 4: 在用户明确授权前保留实现为未提交状态**

因为 `nodes.py`、`graph.py` 和对应测试在任务开始前已经存在彼此关联的未提交修改，拆分后的实现无法安全地与这些修改分开提交。完成验证后汇报文件、测试结果和未提交状态；只有用户明确要求提交时，才暂存经确认的完整范围并创建提交。
