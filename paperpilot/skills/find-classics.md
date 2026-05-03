---
name: find-classics
description: 找经典文献:搜索当代相关 paper -> 构建引用图 -> 求共同引用 -> 找领域内被反复引用的基础工作
when_to_use: 用户想入门某领域、想找必读论文、或想看一批新论文共同依赖的理论基础
---

# Find Classics

## 适用场景
- 用户想入门某个领域,问“必读哪几篇”。
- 用户想看一批新论文背后共同依赖的理论基础。
- 用户给出一个主题,希望从相关论文反推领域内的经典工作。

## 步骤
1. 找当代 paper: 调 `mcp__arxiv__search_papers(query="领域关键词", max_results=5)`。
2. 从搜索结果文本中提取 3-5 个有代表性的 `arxiv_id`。
3. 构建引用图: 调 `mcp__graph__build_graph(arxiv_ids=[提取出的 arxiv_id])`。
   - 如果返回 `missing`,用剩下的 id 继续,不要中断。
4. 求共引: 调 `mcp__graph__get_common_citations(arxiv_ids=[同一批可用 id], top_k=10)`。
   - 返回 `cited_by_count >= 2` 的共同引用文献。
5. 综合回答: 按 `cited_by_count`、年份和主题相关性列出 top 文献,解释为什么它们可能是该 cluster 的基础工作。

## 注意
- 至少需要 2 篇 input paper 才能求共同引用;单篇没有共引意义。
- common citations 说明“被这一组论文共同引用”,不必然等于广义经典。
- 不要把 `search_papers` 的整段文本直接传给 graph;必须先提取其中的 `arxiv_id`。

