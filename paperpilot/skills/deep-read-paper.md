---
name: deep-read-paper
description: 深读单篇 arxiv 论文:下载全文 -> 建索引 -> 多轮检索 -> 综合回答
when_to_use: 用户给定 arxiv id 或论文标题要求详细讲解,或追问某篇论文里的概念定义和 method 细节
---

# Deep Read Paper

## 适用场景
- 用户给定 arxiv id 或论文标题,要求详细讲解。
- 用户问某篇具体论文里的概念定义、method 细节、实验设置或结论依据。
- 回答需要基于论文全文里的具体段落,不能只看 abstract。

## 步骤
1. 拿全文: 调 `mcp__arxiv__download_paper(arxiv_id="...")`。
2. 建索引: 调 `mcp__colbert__build_index(documents=[download_paper 返回值])`。
   - `documents` 必须是非空 list。
   - 每个元素必须包含 `paper_id` 和 `text`。
   - 通常直接把 `download_paper` 返回对象作为 list 里的唯一元素。
3. 多轮检索: 针对用户问题里的关键概念调 `mcp__colbert__search(query="...", paper_id="<step 2 build 的 paper_id>", top_k=3)`。
   - paper_id 必传,值与 step 2 build_index 时的 paper_id 一致。
   - 一个 query 不够时,拆成多个具体 query 多搜几次。
   - query 应围绕用户真正关心的术语,例如 definition、architecture、experiment、ablation。
4. 综合回答: 只基于 `colbert.search` 返回的具体段落回答,并引用或转述段落里的证据。

## 注意
- 不要只看 abstract 回答细节问题。
- 不要编造段落、标题、作者或结论。
- 如果 search 返回内容与问题无关,换 query 继续搜,而不是凭印象回答。
