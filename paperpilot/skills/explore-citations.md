---
name: explore-citations
description: 引用拓扑探索:构图 -> 看邻居 -> 找路径 -> 理解一篇 paper 的学术上下文
when_to_use: 用户问某篇论文引用了哪些工作、被哪些后续工作引用、两篇论文之间有什么引用关系或 follow-up 工作
---

# Explore Citations

## 适用场景
- 用户想知道某篇 paper 引用了哪些前作,或被哪些后续 paper 引用。
- 用户想看两篇 paper 之间是否存在直接或间接引用路径。
- 用户想理解某篇 paper 在学术网络中的位置。

## 步骤
1. 构图: 调 `mcp__graph__build_graph(arxiv_ids=[目标 paper id 列表])`。
   - 返回值包含 `missing`; Semantic Scholar 查不到的 id 跳过即可。
2. 看引用或被引: 调 `mcp__graph__get_neighbors(arxiv_id="...", direction="...", limit=10)`。
   - `direction="references"` 表示它引用了谁。
   - `direction="citations"` 表示谁引用了它。
   - `direction="both"` 表示两者都看。
3. 看路径: 用户问两篇论文关系时,调 `mcp__graph__get_shortest_path(from_id="...", to_id="...")`。
   - `length=-1` 表示无路径。
   - 有路径时,`path` 列表给出中间节点。
4. 综合回答: 基于 graph tool 返回的 `title`、`year`、`authors`、`arxiv_id` 说明引用上下文。

## 注意
- `get_neighbors` 和 `get_shortest_path` 之前必须先 `build_graph` 把目标 paper 拉进图。
- 用户明确问“它引用了什么”时用 `references`;问“谁引用了它”时用 `citations`。
- 不要编造 title、year 或 authors,只使用 graph tool 的真实返回字段。

