---
name: analyze-figures
description: 看论文里的 figure / table / architecture diagram:先文本定位再调 VLM
when_to_use: 用户问某篇 paper 的 figure / table / 架构图含义,或要解释画的是什么
---

# Analyze Figures

## 适用场景
- 用户给定 arxiv_id 问 "Figure N 画的是什么 / Table M 数据怎么读 / 架构图怎么理解"
- 用户问"这篇 paper 的 X 模块结构"且文本不足以说清

## 步骤
1. 先用 colbert 文本定位:
   - `mcp__arxiv__download_paper(arxiv_id="...")`
   - `mcp__colbert__build_index(documents=[<download 返回值>])`
   - `mcp__colbert__search(query="Figure N", paper_id="...")` 找到 caption / 引用上下文,从中读出页码
2. 调 `mcp__vlm__understand_paper_page(arxiv_id="...", page_num=<step1 拿到的页>, query="describe Figure N in detail")`
3. 综合 VLM 描述 + colbert 上下文回答

## 注意
- 不要不经文本定位就乱调 VLM,VLM 比 colbert 贵
- 一次 understand_paper_page 只看一页;跨页或对比多页就分多次调
- 用户只问 paper 整体内容时用 deep-read-paper,不要用本 skill
