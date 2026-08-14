# Deep Reading LangGraph 节点拆分设计

> Codex only：本文记录 2026-08-14 已确认的节点模块拆分方案。此次重构只调整代码组织，不改变 LangGraph 节点名称、执行路径、State 更新、异常语义或持久化行为。

## 1. 背景与目标

`paperpilot/deep_reading/nodes.py` 当前同时包含 6 个 LangGraph 节点、运行时上下文、业务绑定校验、结构化结果校验和多个节点私有辅助函数。文件已经超过 700 行，阅读一个节点时需要同时理解其他阶段的实现。

本次重构的目标是：

1. 将每个 LangGraph 节点放入 `paperpilot/deep_reading/nodes/` 下的独立模块。
2. 将真正跨节点复用的运行时上下文、绑定校验和结果校验提取为职责明确的支持模块。
3. 通过 `nodes/__init__.py` 保持当前包级导入接口兼容。
4. 保留工作区中尚未提交的 `initialize_turn` 使用 `Command(goto=...)` 完成路由的改动。
5. 通过现有深度阅读测试证明本次迁移没有改变运行行为。

## 2. 范围与约束

### 2.1 本次包含

- 用 `paperpilot/deep_reading/nodes/` 包替换单文件 `paperpilot/deep_reading/nodes.py`。
- 将 6 个节点分别迁移到同名模块。
- 提取 `DeepReadingContext`、运行时绑定校验和跨节点结构校验。
- 更新节点测试中依赖旧模块内部符号位置的 monkeypatch。
- 保持 `paperpilot.deep_reading.nodes` 的现有公开导出。
- 更新仓库文档白名单以登记本文和后续实施计划。

### 2.2 本次不包含

- 不调整 Graph 的节点名称、边或执行顺序。
- 不修改 Prompt、工具调用、预算、数据库事务或 checkpoint 行为。
- 不修改 State、Pydantic Schema 或公开 API。
- 不拆分 `tests/deep_reading/test_nodes.py`；测试文件重组不属于本次目标。
- 不重构 `research_agent.py`、`runner.py` 或 Web 运行链路。
- 不引入新依赖。

## 3. 目标文件结构

```text
paperpilot/deep_reading/nodes/
├── __init__.py
├── context.py
├── binding.py
├── validation.py
├── initialize_turn.py
├── summarize_history.py
├── prepare_primary_paper.py
├── research_evidence.py
├── write_answer.py
└── publish_result.py
```

### 3.1 公开入口

`nodes/__init__.py` 继续导出：

- `DeepReadingContext`
- `initialize_turn`
- `needs_summary`
- `summarize_history`
- `prepare_primary_paper`
- `research_evidence`
- `write_answer`
- `publish_result`

因此以下现有调用方不需要改变导入路径：

- `paperpilot/deep_reading/graph.py`
- `paperpilot/deep_reading/runner.py`
- `paperpilot/deep_reading/research_agent.py` 中的类型检查导入
- `paperpilot/deep_reading/__init__.py`
- 现有测试中的包级节点导入

### 3.2 支持模块职责

- `context.py`：定义 `PaperSearch`、`EventSink` 和 `DeepReadingContext`，包括现有预算参数校验。
- `binding.py`：负责校验 Runtime context、Research Task、Conversation、当前 User Message 与 Graph State 的绑定关系。
- `validation.py`：负责解析并校验跨节点传递的 `ResearchResult`、`AnswerDraft` 和引用关系。

支持模块只放置至少被两个节点使用的逻辑。只服务单个节点的辅助函数与常量保留在对应节点模块，避免形成新的大型 `_shared.py`。

### 3.3 节点模块职责

- `initialize_turn.py`：重置本轮字段，并通过现有 `Command` 选择是否进入摘要节点。
- `summarize_history.py`：判断是否需要摘要、估算上下文、保留最近轮次并生成结构化摘要。
- `prepare_primary_paper.py`：校验主论文身份，调用下载与索引 MCP 工具并更新活跃论文。
- `research_evidence.py`：校验运行绑定、调用受限 Research Agent，并保存完整研究结果。
- `write_answer.py`：从可信研究结果和论文元数据生成、校验结构化答案。
- `publish_result.py`：幂等发布最终消息、元数据和实际使用的论文。

## 4. 依赖与数据流

```text
graph.py
  -> nodes/__init__.py
      -> initialize_turn.py -> summarize_history.needs_summary
      -> summarize_history.py -> context.py + binding.py
      -> prepare_primary_paper.py -> context.py + binding.py
      -> research_evidence.py -> context.py + binding.py + research_agent.py
      -> write_answer.py -> context.py + binding.py + validation.py
      -> publish_result.py -> context.py + binding.py + validation.py
```

节点仍然接收 `DeepReadingState` 与 `Runtime[DeepReadingContext]`，并返回当前实现相同的 State 局部更新或 `Command`。支持模块不能注册节点、修改 Graph 拓扑或持有新的可变全局状态。

## 5. 兼容性与异常处理

- `from paperpilot.deep_reading.nodes import ...` 必须继续工作。
- `build_deep_reading_graph()` 继续通过同样的 6 个函数对象注册节点。
- 所有 `ResearchContractError`、`TaskBindingError` 和 `FinalCheckpointError` 的触发条件与消息保持不变。
- MCP 工具事件、数据库写入顺序和发布幂等流程保持不变。
- 测试如果需要替换 `run_research_agent`，应 patch 实际使用该符号的 `research_evidence` 模块，而不是依赖聚合包的偶然内部属性。
- Python 中同一路径不能同时存在 `nodes.py` 和 `nodes/` 包，因此迁移完成后删除旧 `nodes.py`；其全部有效实现必须先迁入新包。

## 6. 验证标准

至少执行以下验证：

1. 节点包的公开导入兼容性测试通过。
2. `tests/deep_reading/test_nodes.py` 全部通过。
3. `tests/deep_reading/test_graph.py` 全部通过，覆盖可选摘要路径与当前 `Command` 路由。
4. `tests/deep_reading/test_runner.py` 全部通过，确认 Runner 与 checkpoint 恢复边界未变化。
5. `tests/deep_reading/test_research_agent.py` 全部通过，确认类型依赖与节点调用关系没有循环导入。
6. `tests/architecture/test_repository_allowlist.py` 通过。
7. 对本次涉及的 Python 文件运行项目现有静态检查，并运行 `git diff --check`。

## 7. 风险控制

- **循环导入风险**：`DeepReadingContext` 迁入无节点依赖的 `context.py`；`research_agent.py` 继续仅在类型检查阶段导入它。
- **monkeypatch 失效风险**：测试 patch 节点实现模块中的符号，不依赖 `nodes/__init__.py` 的再导出绑定。
- **辅助模块再次膨胀**：共享文件按“运行绑定”和“结构校验”分开，单节点逻辑不进入共享模块。
- **现有未提交改动丢失**：以当前工作区版本为迁移源，不从 `HEAD` 重建或回退 `initialize_turn`、`graph.py` 和相关测试。
- **行为漂移风险**：迁移阶段以机械移动和导入调整为主，不顺带重写业务逻辑。
