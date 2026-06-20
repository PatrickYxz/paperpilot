---
name: deep-read-paper
description: 深读单篇 arXiv 论文：下载全文 -> 建索引 -> 计划检索 -> 基于证据回答
when_to_use: 用户给定 arXiv id 或论文标题，要求回答某篇论文里的定义、方法、实验设置、数据集、指标、结果或结论依据
---

# Deep Read Paper

## 适用场景

- 用户给定 arXiv id 或论文标题，要求详细理解单篇论文。
- 用户追问某篇论文里的概念定义、method 细节、实验设置、数据集、指标、baseline 或结论依据。
- 回答必须基于论文全文里的具体段落，不能只看 abstract。

## Workflow

1. 下载全文：调用 `mcp__arxiv__download_paper(arxiv_id="...")`。
2. 建索引：调用 `mcp__colbert__build_index(documents=[download_paper_result])`。
   - `documents` 必须是非空 list。
   - 每个元素必须包含 `paper_id` 和 `text`。
   - 通常直接把 `download_paper` 返回对象作为 list 里的唯一元素。
3. 计划检索：优先调用 `mcp__colbert__planned_retrieval(question="<用户问题>", paper_id="<indexed_paper_id>", paper_title="<title if known>", abstract="<abstract if known>", top_k_each=5, summary_k=8)`。
   - `paper_id` 必须传，且必须与 build_index 时的 `paper_id` 一致。
   - planned_retrieval 会先生成 query plan，再强制执行多条计划 query，并返回 `summary_text`、`evidence_pool`、`missing_requirements` 和 `query_plan_meta`。
   - 优先使用 `summary_text` 和 `evidence_pool.summary_items` 中的证据回答。
   - 如果 `missing_requirements` 非空，不能把缺失部分编成答案；在最终回答的 `Notes:` 里说明证据不足或范围限制。
4. 可选补充检索：只有当 planned retrieval 的证据不足、`missing_requirements` 未覆盖，或用户问题需要进一步澄清时，才调用 `mcp__colbert__search(query="...", paper_id="<indexed_paper_id>", top_k=3)`。
   - 补充 search 必须针对具体缺口，例如缺少 dataset、metric、baseline、table/figure evidence、negative/contrastive evidence。
   - 如果补充 search 仍不相关，继续换 query 搜索缺口；不要凭印象回答。
5. 综合回答：只基于 `planned_retrieval` 或补充 `colbert.search` 返回的具体段落回答。

## Final Answer Contract

最终回答必须只输出面向用户的答案，不要输出执行过程。

Required sections:

```text
Short answer: <one direct answer to the question>

Evidence: <retrieved evidence that directly supports the short answer>

Notes: <optional caveats only when needed>
```

Rules:

- `Short answer:` 的第一句必须直接回答问题，不要先写背景。
- `Evidence:` 只能放直接支持 short answer 的检索证据；不要把 related work、额外 baseline、额外数据集、额外指标混入证据。
- `Notes:` 可省略；只有在证据不足、问题有歧义或需要限定范围时才写。
- 不要在最终回答里输出 `Step 4`、tool progress、chain-of-thought style narration、`Answer span candidates`、`Now I have enough evidence` 或类似过程痕迹。
- 不要把内部候选 span、搜索计划、执行状态写给用户。
- 问什么答什么。不要因为段落里有相关内容，就把未被问题询问的方法、数据集、指标、baseline 或实验细节加入 direct answer。
- 如果问题问列表，例如 methods / datasets / metrics / baselines / components，先确认每个列出的 item 都被检索证据直接支持；不要加入只是在同一段落附近出现但不属于问题范围的 item。
- 如果问题问数值、百分比、分数、数据集名、corpus 名、method 名、metric 名，`Short answer:` 里写出的关键数字或实体必须也出现在 `Evidence:` 中。
- 如果检索证据不足以回答，`Short answer:` 直接说证据不足，不要泛泛总结或猜测。

## Common Failure Modes To Avoid

- Overbroad scope: 问四个 clustering methods，却把 related-work methods、baseline variants 或 unrelated neural variants 全部列入答案。
- Missing required part: 问多个 components / methods / datasets，只回答其中一部分。
- Wrong numeric or entity: 问 improvement / score / dataset，使用相似段落里的错误数字或错误数据集名。
- Format noise: 最终答案残留 `Step 4`、`Answer span candidates`、搜索过程或长篇解释。

## Notes

- 不要只看 abstract 回答细节问题。
- 不要编造段落、标题、作者、数字或结论。
- 问 patterns / findings / observations 时，优先复制或紧贴论文里的总体发现句，再做必要解释。
- 如果 search 返回内容与问题无关，换 query 继续搜，而不是凭印象回答。
