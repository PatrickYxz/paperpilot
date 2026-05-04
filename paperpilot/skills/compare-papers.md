---
name: compare-papers
description: 多论文深读对比:用 paper_deep_read 并发精读 3-8 篇,综合对比方法/发现/适用场景
when_to_use: 用户给定 3-8 个 arxiv id 或论文,要求对比、综合或并列分析
---

# Compare Papers

## 适用场景
- 用户明确给出 3-8 个 arxiv id 或论文标题,要求对比/综合/并列分析。
- 用户问"这几篇 paper 的 X 设计有什么异同"。

## 步骤
1. 调一次 `paper_deep_read(paper_ids=[...], user_query="<原问题>")`。
   - 每篇会被一个独立子 agent 精读,用 download + colbert.build_index + colbert.search 流程。
   - 工具返回 markdown,每篇一个 `### <paper_id>` section,含核心方法 / 关键发现 / 与查询相关性。
2. 综合对比:相同点 / 不同点 / 各自适用场景。
3. 引用每篇的 section 作为证据,不要编造段落或结论。

## 注意
- 单篇用 `deep-read-paper`,3 篇起才用本 skill。
- 用户只给主题没具体 id 时,先自己调 `mcp__arxiv__search_papers` 拿候选 id 再触发本 skill。
- 不要为了凑数把不相关的 paper 塞进去;少而准比多而散更有用。
