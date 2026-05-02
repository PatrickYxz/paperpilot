# Day 6 smoke 稳定化计划

Codex only. Claude Code 不需要参考本文件。

## 背景与目标

Day 8 实施后默认测试、graph slow 集成、Day 5 smoke、Day 8 smoke 均已通过；但 `scripts/day6_smoke.py` 两次未稳定通过。

失败信号有两个：

1. arXiv API 返回 HTTP 429，导致 LLM 多轮重复 `search_papers`，无法进入 download / colbert。
2. LLM 在已拿到 `download_paper` 结果后，连续调用 `build_index({})`，漏传必填 `documents` 字段，触发 FastMCP 参数校验失败。

目标是让 Day 6 smoke 稳定验证“arxiv download -> colbert build_index -> colbert search -> 基于 chunk 回答”这条链路，而不是依赖模型在宽泛搜索任务中随机学会正确参数。

## 约束条件

- 不改 `MCPClient`，保持 Day 5/Day 8 的 server 解耦边界。
- 不让 `colbert-mcp` 隐式读取 arXiv 或本地缓存来补参，避免 server 之间产生隐藏耦合。
- 不把 smoke 伪装成通过；仍要真实调用 `download_paper`、`build_index`、`search`。
- 优先小改，避免扩大到 agent 架构重构。

## 方案

### 推荐方案：system prompt + Day 6 smoke prompt 双重收紧

1. 在 `paperpilot/main.py` 的 `SYSTEM_PROMPT` 增加 tool 参数完整性约束：
   - 调 tool 时必须按 schema 传完整必填参数。
   - `build_index` 必须传 `documents=[download_paper 的返回对象]`，不能传 `{}`。
   - 如果 tool 报缺字段，下一轮必须修正参数，不能重复同一空参数。
2. 在 `scripts/day6_smoke.py` 中把任务改成固定 `1706.03762`：
   - 仍要求调用 `search_papers` 以覆盖 arXiv search。
   - 明确下载 `1706.03762`。
   - 明确 `build_index(documents=[download_paper 返回值])`。
   - 明确再调用 `colbert.search` 查询 `multi-head attention definition`。

优点：保留真实 agent loop 和真实 MCP 工具调用，最小化实现改动。
风险：仍依赖 LLM，但提示约束更强；arXiv 429 仍可能偶发。

### 备选方案：写确定性 Day 6 tool-chain smoke

新增一个不经过 LLM 的脚本，直接用 `MCPClient` 调 arxiv / colbert 工具。

优点：最稳定，适合 CI。
风险：失去“LLM 会正确串联工具”的验证价值，与当前 day6_smoke 目标不同。

### 暂不推荐：让 build_index 自动补最近下载文档

当 `documents` 缺失时，从 `data/papers` 读取最近文件自动建索引。

优点：能掩盖 LLM 漏参。
风险：隐藏状态强、行为不透明，还会让 colbert server 间接依赖 arxiv 下载产物，不符合当前架构边界。

## 执行步骤

1. 修改 `SYSTEM_PROMPT`，加入工具参数完整性和 build_index 明确示例。
2. 修改 `scripts/day6_smoke.py` prompt，固定论文 id 并明确工具调用参数。
3. 若实测 LLM 仍传 `build_index({})`，在 `agent_loop` 增加当前会话内的参数修复：
   - 记录最近一次 `download_paper` 返回的 `{paper_id, text}`。
   - 当后续 `build_index` 漏传或传空 `documents` 时，在调用 tool 前补成 `documents=[最近下载文档]`。
   - 发出 `tool_arg_repair` 事件，保证行为可追踪。
4. 运行默认测试：`pytest tests -v`。
5. 运行 `scripts/day6_smoke.py`，必要时因 arXiv 429 等待后重跑一次。
6. 若通过，提交一个 Day 6 smoke 稳定化 commit。

## 执行记录

- 首轮只改 `SYSTEM_PROMPT` 和 `day6_smoke.py` prompt 后，Day 6 smoke 仍失败：
  - `search_papers('1706.03762')` 成功。
  - `download_paper('1706.03762')` 成功。
  - LLM 连续两次调用 `build_index({})`，随后触发 repeated-call guardrail。
- 结论：问题不是提示词可完全解决，而是让 LLM 复制整篇全文作为下一次 tool 参数本身不稳定。
- 已转向 agent loop 级参数修复方案。

## 验证方式

- `pytest tests -v` 通过。
- `scripts/day6_smoke.py` 退出 0 并打印 `Day 6 smoke PASSED`。
- tracer 中必须出现：
  - `mcp__arxiv__search_papers`
  - `mcp__arxiv__download_paper`
  - `mcp__colbert__build_index`
  - `mcp__colbert__search`

## 风险与待确认项

- system prompt 是行为变更，需要用户确认。
- 如果 arXiv 429 持续，Day 6 smoke 仍可能因外部服务失败；这时应考虑增加一个独立的 deterministic smoke，而不是继续让 LLM 重试搜索。
