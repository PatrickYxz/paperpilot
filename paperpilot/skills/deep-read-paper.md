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
   - 默认至少做 3 次差异化 search;除非工具错误或已到迭代上限,不要只搜 1-2 次就进入最终回答。
   - 第 1 次搜用户原问题或最接近的英文问题。
   - 第 2 次搜关键术语 / 同义词 / 缩写,例如 dataset、benchmark、corpus、definition、metric、baseline。
   - 第 3 次搜答案所在位置的线索,例如 experiment setup、table、appendix、method、evaluation、ablation。
   - 如果前 3 次结果仍不相关,换 query 继续搜,而不是凭印象回答。
4. 综合回答: 只基于 `colbert.search` 返回的具体段落回答,并引用或转述段落里的证据。
   - 最终回答先写 `Short answer:` 一句话,直接给出最短答案或关键 span。
   - 然后写 `Evidence:` 说明来自哪些检索段落,不要先写长篇背景。
   - 最后写必要解释;如果没找到证据,明确说未在检索结果中找到,不要泛泛总结。

## 注意
- 不要只看 abstract 回答细节问题。
- 不要编造段落、标题、作者或结论。
- 问题问数据集、数值、方法名、baseline、指标时,最终答案必须优先输出这些原子事实。
- 如果 search 返回内容与问题无关,换 query 继续搜,而不是凭印象回答。
