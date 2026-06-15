# QASPER 原始任务和 PaperPilot 当前评测口径

Date: 2026-06-10

## 这份文档要解决什么问题

PaperPilot 当前 README 里写着一个很醒目的数字：

```text
paperpilot: 99 / 150 = 66.0%
```

这个数字有价值，但它不能被简单理解成“PaperPilot 真实论文问答准确率
就是 66%”。原因是：QASPER 原始任务、QASPER 官方数据结构、以及
PaperPilot 当前评测脚本使用的口径并不完全一样。

这份文档先把三件事分清楚：

1. QASPER 原本想评测什么。
2. QASPER 数据里到底标了哪些东西。
3. PaperPilot 当前评测把它简化成了什么。

只有先弄清楚这些，后面讨论 LLM judge、semantic audit、规则诊断、
evidence judge 才不会跑偏。

## QASPER 原本是什么任务

QASPER 是一个面向科学论文的 question answering 数据集。它的基本任务
可以理解成：

```text
输入：一篇完整 NLP 论文 + 一个关于这篇论文的问题
输出：基于论文内容回答这个问题
```

这里最重要的是“完整论文”。QASPER 不是让模型只读 abstract，也不是让模型
只看某个短片段。它想模拟的是：一个读者看到论文标题和摘要之后，产生一些
想继续从全文里确认的问题。

官方论文摘要里描述了这个数据构造方式：

- 问题由 NLP practitioner 写出。
- 提问者只读 title 和 abstract。
- 问题寻找的是 full text 里的信息。
- 答案由另一批 NLP practitioner 标注。
- 答案标注者还会提供 supporting evidence。

这说明 QASPER 的问题不是普通 trivia，也不只是“这句话里有没有关键词”。
它更接近：

```text
paper-grounded information seeking QA
```

也就是“基于论文全文的信息寻求式问答”。

## QASPER 问题是否都需要全文理解

不能简单说“每一道题都需要整篇论文的深度理解”，也不能说“每一道题只要读
一个局部句子”。

更准确的说法是：

```text
问题面向全文提出；
答案通常由一个或多个 evidence 段落支撑；
有些题是局部事实抽取，有些题需要跨段综合或理解论文设置。
```

比如下面几类问题通常更接近局部事实抽取：

```text
What dataset do they use?
How many comments were used?
Which languages are used?
What labels are available?
```

这类题经常可以在实验设置、数据集描述、表格附近找到答案。

但下面这些问题就更像论文级理解：

```text
How are relations used to propagate polarity?
What are the results?
How do the authors measure user engagement?
What patterns do they observe?
```

这类题可能需要理解方法、实验、指标、结果，甚至多个段落之间的关系。

因此，如果后面做 LLM judge，不能默认“oracle span 附近几句话”就足够
代表标准答案语境。对部分题可以，对另一部分题不一定。

## QASPER 数据里有什么

QASPER 数据结构比 PaperPilot 当前使用的字段更丰富。一个 QA 标注里通常
会有：

```text
question
answers
answer.extractive_spans
answer.free_form_answer
answer.yes_no
answer.unanswerable
answer.evidence
answer.highlighted_evidence
```

这些字段的含义大致如下：

- `question`：问题。
- `extractive_spans`：可以直接从论文中抽取出来的答案片段。
- `free_form_answer`：人工写出的自由文本答案。
- `yes_no`：是/否答案。
- `unanswerable`：论文里无法回答。
- `evidence`：支撑答案的段落、表格或图片文本。
- `highlighted_evidence`：标注者选出的更细粒度 textual evidence。

所以 QASPER 原始标注不是只有 oracle span。它同时关心：

```text
答案是什么
答案从哪里来
答案是否有证据支持
```

Hugging Face 数据卡也把 QASPER 的 supported tasks 写成两类：

```text
question-answering
evidence-selection
```

这对 PaperPilot 很重要。因为 PaperPilot 本身是 RAG / agent 系统，只看最终
答案是否包含一个短 span，会丢掉“是否找到了正确证据”这个关键维度。

## PaperPilot 当前是怎么评测的

当前 PaperPilot 的评测入口主要在这些文件：

```text
paperpilot/eval/qasper_loader.py
paperpilot/eval/scorer.py
paperpilot/eval/baselines.py
scripts/day16_prepare_eval.py
scripts/day16_run_eval.py
scripts/day16_summarize.py
```

当前 loader 做了一个很明确的简化：

```python
def _answer_spans(answer_record: dict[str, Any]) -> list[str]:
    inner = answer_record.get("answer")
    if not isinstance(inner, dict):
        inner = answer_record
    spans = inner.get("extractive_spans") if isinstance(inner, dict) else None
    if not spans:
        return []
    return [s for s in spans if isinstance(s, str) and s.strip()]
```

也就是说，当前评测只收集 `extractive_spans`，然后丢掉或暂时不用这些信息：

```text
free_form_answer
yes_no
unanswerable
evidence
highlighted_evidence
answer annotation metadata
```

当前 `EvalCase` 只保留：

```python
case_id
arxiv_id
paper_title
abstract
full_text
question
oracle_spans
```

当前 scorer 更简单：

```python
def is_pass(predicted: str, oracle_spans: list[str] | tuple[str, ...]) -> bool:
    pred = _normalize(predicted)
    for span in oracle_spans:
        if not span or not span.strip():
            continue
        norm_span = _normalize(span)
        if norm_span and norm_span in pred:
            return True
    return False
```

换句话说：

```text
只要 predicted answer 里包含任意一个 oracle extractive span，就算 pass。
```

因此，当前 66.0% 更准确的名字应该是：

```text
QASPER extractive-span substring recall on a 150-case subset
```

而不是：

```text
完整 QASPER 官方准确率
```

也不是：

```text
PaperPilot 真实科研问答准确率
```

## 当前口径的优点

这个简化不是没有价值。它有几个明显优点：

1. 可复现。

   同一个 predicted answer 和同一组 oracle spans 永远得到同一个 pass/fail。
   不会因为 LLM judge 状态变化而漂。

2. 便宜。

   不需要额外模型调用，可以快速跑 baseline。

3. 对短答案事实题有效。

   对数据集名、方法名、标签、数字、范围、指标等题型，要求答案包含原文
   关键短语是合理的。

4. 适合做历史硬指标。

   Day 16、Day 18、后续改动可以沿用同一把尺子比较。

所以这个指标应该保留。它适合作为：

```text
strict exact-span baseline metric
```

## 当前口径的主要偏差

### 1. 会低估语义正确但表述不同的答案

比如 oracle 是：

```text
from 50K to 4.8M
```

模型回答：

```text
The number of comments ranges from 50,000 to 4.8 million.
```

语义可能正确，但 substring 不一定命中。

这类是 false negative 风险。

### 2. 会高估只提到 oracle span 但逻辑不对的答案

比如 oracle 是：

```text
positive
negative
```

模型回答：

```text
The paper discusses positive and negative examples, but those are not the
dataset labels.
```

如果 scorer 只看 span 出现，它可能判 pass。但这个回答未必真正回答了问题。

这类是 false positive 风险。

### 3. 任意 oracle 命中就 pass，可能放过不完整答案

如果标准答案有多个 span：

```text
overall rating
mean number of turns
```

模型只答出一个：

```text
They measure user engagement with overall rating.
```

当前 scorer 会 pass，因为命中了 `overall rating`。但从问题角度看，这可能
只是 partial answer。

### 4. 丢掉了 evidence selection 维度

QASPER 原本也关心 evidence。PaperPilot 作为 RAG 系统，也应该关心：

```text
有没有找到正确证据
答案是否被证据支持
最终回答是否正确使用证据
```

当前 pass/fail 只看最终输出，不看 evidence 是否对。

这会让我们很难判断失败来自哪里：

```text
检索没找到？
找到了但没抽取？
抽取了但表述不匹配？
回答碰巧包含了关键词但逻辑不成立？
```

## 为什么不能直接用 oracle span 附近窗口做 judge

之前讨论过一个口径：

```text
question + oracle span + oracle span 附近上下文 + predicted answer
```

这个比单纯 oracle span 强，但仍然不够稳。原因是：

1. QASPER 的问题是全文信息寻求问题，有些答案可能由多个段落支撑。
2. 短 span 附近窗口不一定表达完整命题。
3. 对 `positive`、`negative`、`English`、`accuracy` 这类短词，窗口大小会显著
   影响 judge 判断。
4. 如果答案涉及表格、图、多个实验设置，单个 span window 很可能漏信息。

所以如果要做 semantic judge，优先级应该是：

```text
gold evidence / highlighted evidence > oracle span window > oracle span only
```

也就是说，judge 应该尽量基于 QASPER 原始 evidence，而不是只基于 span 附近
字符串。

## 更合理的后续评测分层

后续 PaperPilot 评测可以拆成几层，而不是试图用一个数字解决所有问题。

### 第一层：strict span score

保留当前指标：

```text
predicted answer contains any oracle extractive span
```

用途：

- 历史对比；
- 快速回归；
- 严格短答案命中率；
- README 里继续保留，但名字要讲清楚。

### 第二层：answer semantic audit

输入：

```text
question
gold answer fields
gold evidence / highlighted evidence
predicted answer
```

输出：

```text
correct
partial
incorrect
contradictory
unverifiable
judge_uncertain
```

用途：

- 判断 strict fail 里有多少 false negative；
- 判断 strict pass 里有多少 false positive；
- 看 66.0% 是低估还是高估；
- 建立更接近真实问答质量的指标。

### 第三层：evidence selection audit

输入：

```text
QASPER gold evidence
PaperPilot retrieved evidence / trace
predicted answer
```

输出：

```text
evidence_found
evidence_missing
evidence_irrelevant
answer_supported
answer_unsupported
```

用途：

- 判断失败是不是 retrieval 问题；
- 判断 ColBERT 是否找到了正确证据；
- 判断 agent 是否正确使用了证据。

### 第四层：system-level research QA audit

这层不再只服务 QASPER，而是评估真实 PaperPilot 使用场景：

```text
多论文比较
证据综合
报告写作
引用可靠性
不确定性表达
```

这更接近真实产品质量，但不适合作为第一步。

## 对当前 66.0% 的正确解释

当前最稳妥的说法是：

```text
在一个 150-case 的 QASPER extractive QA 子集上，PaperPilot 的最终答案有
66.0% 命中了至少一个 gold extractive span。这个 strict substring 指标
可复现，适合历史对比，但它不是完整 QASPER 官方评测，也不能完全代表真实
科研问答准确率。
```

更适合 README 或面试表达的说法是：

```text
PaperPilot 当前用 QASPER 的 extractive QA 子集做了一个严格的 span-recall
评测：150 个问题里有 99 个最终答案包含人工标注答案 span，达到 66.0%，
明显高于 full-text baseline 的 40.7%。这个结果证明检索和 deep-read 流程
有增益，但剩余质量还需要通过 semantic audit 和 evidence audit 继续拆解。
```

## 下一步建议

不要马上改 PaperPilot 生成逻辑，也不要马上接 LLM judge。更稳的下一步是：

1. 升级 `EvalCase` 或新增 `EvalCaseV2`，保留 QASPER 原始答案 metadata。
2. 重新准备一个 enriched subset，包含：

   ```text
   extractive_spans
   free_form_answer
   yes_no
   unanswerable
   evidence
   highlighted_evidence
   ```

3. 写一个离线分析脚本，先统计 150 个样本的问题类型和答案类型。
4. 再决定 semantic judge 的输入格式。

这样做的好处是：后面的评测升级会以 QASPER 原始设计为基础，而不是在当前
substring scorer 上继续叠补丁。

## 参考来源

- QASPER paper: https://arxiv.org/abs/2105.03011
- QASPER dataset card: https://huggingface.co/datasets/allenai/qasper
- Current PaperPilot loader: `paperpilot/eval/qasper_loader.py`
- Current PaperPilot scorer: `paperpilot/eval/scorer.py`
- Current PaperPilot eval runner: `scripts/day16_run_eval.py`
