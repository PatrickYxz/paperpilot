---
name: write-research-report
description: 给定一个研究方向,产出一篇结构化综述 markdown
when_to_use: 用户给一个主题/方向(不是单篇 paper / 不是对比指定几篇),希望"写一篇综述"或"了解 XX 领域近年进展"
---

# Write Research Report

## 适用场景
- "总结一下 long-context retrieval 2024 后的进展"
- "我想入门 X 方向,给我一份 reading list 加每篇要点"
- 不适用:用户给定具体 paper(用 deep-read-paper / compare-papers)

## 工作流(推荐,非强制)
1. **规划**:`research_todo` 写下 plan(搜索关键词 / 候选数 / 精读数)
2. **粗搜**:`mcp__arxiv__search_papers(query=..., max_results=8)`,标题摘要先筛
3. **下载 + 细排**:从候选里选 3-5 篇,逐篇 `mcp__arxiv__download_paper(arxiv_id=...)`;再把 download 返回的 `{paper_id,text}` 喂 `mcp__colbert__build_index(documents=[...])` -> `mcp__colbert__search(query=主题问题, paper_id=..., top_k=5)` 做语义重排
4. **扩展(可选)**:若主题有明确"奠基 paper",用 `mcp__graph__build_graph` + `mcp__graph__get_neighbors(direction="references"/"citations")` 1 跳找经典或最新工作
5. **精读**:选 3-5 篇代表作,`paper_deep_read(paper_ids=[...], user_query="...")` 并发精读(并发数由工具内部固定线程池控制)
6. **下笔**:用半固定骨架组织 markdown 输出(见下)

## 输出骨架(半固定,可按主题适配)
1. **Title** - 综述标题
2. **TL;DR** - 3-5 句话核心结论
3. **Background** - 问题定义、为什么重要
4. **Key Methods / Approaches** - 主流路线分类(每条带 1-3 篇代表作 + arxiv_id)
5. **Recent Trends (2024-)** - 最新进展
6. **Open Problems** - 未解、争议、benchmark gap
7. **Reading List** - 推荐阅读顺序(5-8 篇,每篇一句话点评 + arxiv_id 链接)

## 质量准则
- 每篇引用必须带 arxiv_id(写成 `[Title (arxiv:2401.12345)]`),用户能点开
- 至少 3 篇 paper 来自 paper_deep_read 的精读结果(不能全靠 abstract)
- 报告 >= 800 字
- 最终回答只能输出报告 markdown 正文,不要输出任务执行总结、状态表或"已完成"说明
- 主题方法学 vs 应用偏向不同,骨架可砍可加(纯方法论可砍"应用",偏应用可并掉"Open Problems")

## 注意
- 不要把所有候选都精读;3-5 篇够,其余靠 abstract + colbert chunk 补
- compose 阶段先写完整 markdown 再输出,不要边搜边写
- `paper_deep_read` 返回后直接写报告,不要继续扩展工具调用
- 除非用户明确问图表/table/architecture diagram,不要调用 VLM
