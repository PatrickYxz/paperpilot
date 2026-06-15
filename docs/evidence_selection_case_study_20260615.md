# Evidence Selection Case Study

Date: 2026-06-15

Calibration source: `data\eval\semantic_calibration_candidates_20260614.jsonl`

This report inspects three representative QASPER failures before changing PaperPilot behavior again.

Important trace limitation: current JSONL traces store truncated tool-result content. The report can inspect search queries and snippet heads, but not the full retrieved top-k ranking.

## Summary

| case_id | issue | likely failed layer | recommended next check |
|---|---|---|---|
| `qasper-1910.04601-q1` | WikiHop vs HotpotQA | retrieval/evidence-selection miss | For dataset/entity questions, evidence selection should require the evidence sentence to express the relation asked by the question and should not inject a candidate entity into the search query before verifying alternatives. |
| `qasper-1701.00185-q1` | Overbroad clustering-method list | scope-control failure after relevant evidence | For list questions, final synthesis needs a scope boundary step: identify the noun phrase being asked for, then include only items governed by that phrase. |
| `qasper-1910.07181-q0` | Wrong improvement percentages | numeric/entity verification failure | For numeric questions, evidence selection should prefer sentences that directly answer the comparative relation in the question and should verify that the exact numbers in the short answer occur in that selected sentence. |

## qasper-1910.04601-q1: WikiHop vs HotpotQA

- Question: What dataset was used in the experiment?
- Review decision: `downgrade_to_incorrect`
- Manual note: Gold says WikiHop; prediction says HotpotQA.
- Likely failed layer: `retrieval/evidence-selection miss`
- Trace used: `data\traces\qasper-1910.04601-q1.jsonl`

### Gold Evidence

> Our study uses WikiHop BIBREF0, as it is an entity-based multi-hop QA dataset and has been actively used. We randomly sampled 10,000 instances from 43,738 training instances and 2,000 instances from 5,129 validation instances (i.e. 36,000 annotation tasks were published on AMT). We manually converted structured WikiHop question-answer pairs (e.g. locatedIn(Macchu Picchu, Peru)) into natural language statements (Macchu Picchu is located in Peru) using a simple conversion dictionary.

### Search Queries And Snippet Heads

**Search 1**

- Query: `What dataset was used in the experiment`
- Snippet head: > { "paper_id": "1910.04601", "chunk_text": "##set, the ﬁrst, quality - assured dataset consisting of 4. 6k questions, each of which is annotated with 3 reference derivations ( i. e. 13. 8k derivati

**Search 2**

- Query: `dataset corpus annotation source used in experiments`
- Snippet head: > { "paper_id": "1910.04601", "chunk_text": "reference derivations. the most closest work to our dataset is the worldtree corpus ( jansen et al., 2018 ), the largest qa dataset annotated with explan

**Search 3**

- Query: `experiment setup benchmark dataset HotpotQA`
- Snippet head: > { "paper_id": "1910.04601", "chunk_text": ", pontus stenetorp, and sebastian riedel. 2018. constructing datasets for multi - hop reading comprehension across documents. trans. of acl, 6 : 287 – 30

### Final Answer Excerpt

> 证据已足够充分。现在综合回答。 Short answer: 实验中使用的数据集是 **HotpotQA**（Yang et al., 2018）。 Evidence: 论文 §3.3 Dataset 明确指出： > "Our study uses **HotpotQA** (Yang et al., 2018), one of the most actively used multi-hop QA datasets." 该论文从 HotpotQA 的 90,564 个训练实例中采样了 3,000 个，从 7,405 个开发实例中采样了 3,000 个，用于标注推理（derivation）。最终构建的 R4C 数据集包含 **4,588 个来自 HotpotQA 的问题**，每个问题标注了 3 个参考推理（共 13,764 个推理）。

### Diagnosis

The available latest trace searched generic dataset queries and an explicit `HotpotQA` query. The truncated retrieved snippets and final answer center on HotpotQA, while the calibration gold evidence says WikiHop. This means the answer repair loop can clean the output, but it cannot recover the gold answer because the selected evidence already points to the wrong dataset.

### Recommended Next Action

For dataset/entity questions, evidence selection should require the evidence sentence to express the relation asked by the question and should not inject a candidate entity into the search query before verifying alternatives.


## qasper-1701.00185-q1: Overbroad clustering-method list

- Question: Which popular clustering methods did they experiment with?
- Review decision: `keep_partial`
- Manual note: Gold asks for four popular clustering methods; answer includes them but adds separate baseline/neural methods that blur scope.
- Likely failed layer: `scope-control failure after relevant evidence`
- Trace used: `data\traces_day16_before_day18\qasper-1701.00185-q1.jsonl`

### Gold Evidence

> In our experiment, some widely used text clustering methods are compared with our approach. Besides K-means, Skip-thought Vectors, Recursive Neural Network and Paragraph Vector based clustering methods, four baseline clustering methods are directly based on the popular unsupervised dimensionality reduction methods as described in Section SECREF11 . We further compare our approach with some other non-biased neural networks, such as bidirectional RNN. More details are listed as follows:

### Search Queries And Snippet Heads

**Search 1**

- Query: `What popular clustering methods were compared as baselines? comparison methods baselines experiments`
- Snippet head: > { "paper_id": "1701.00185", "chunk_text": "of our proposed methods and three clustering methods on three datasets. for recnn ( top ), k - means is conducted on the learned vectors of the top tree

**Search 2**

- Query: `baselines: Skip-thought Vectors Recursive Neural Network Paragraph Vector Average Embedding LSA LPI LE spectral clustering`
- Snippet head: > { "paper_id": "1701.00185", "chunk_text": "datasets, we can get a solid conclusion that our proposed approaches is an [UNK] approaches to get useful semantic features for short text clustering. 5.

### Final Answer Excerpt

> 现在信息已经足够，可以综合回答。 --- 根据论文 **Section 4.3 Comparisons**（实验部分）的描述，该论文实验中所对比的流行聚类方法可以分为以下几大类： ### 1️⃣ 传统/基线方法 | 方法 | 说明 | |------|------| | **K-means (TF)** | 在原始词袋特征（词频 TF 加权）上直接运行 K-means | | **K-means (TF-IDF)** | 在原始词袋特征（TF-IDF 加权）上直接运行 K-means | ### 2️⃣ 基于深度/神经网络的文本表示方法 | 方法 | 说明 | |------|------| | **Skip-thought Vectors (SkipVec)** | 使用预训练的 Skip-thought 模型生成句子向量（含 Uni / Bi / Combine 三种模式），然后在其上运行 K-means | | **Recursive Neural Network (RecNN)** | 使用递归自编码器学习文本向量，分 **Top**（仅用顶层节点）、**Ave.**（所有节点平均）、**Top+Ave.** 三种变体，然后在其上运行 K-means | | **Paragraph Vector (Para2vec)** | 使用 Doc2Vec / Paragraph Vector 学习固定长度向量，然后在其上运行 K-means | ### 3️⃣ 基于传统无监督降维的方法（作为 STC2 框架中伪标签来源的对照基线） | 方法 | 说明 | |------|------| | **Average Embedding (AE)** | 对词嵌入向量取加权平均（分别以 TF 和 TF-IDF 加权），然后运行 K-means | | **Latent Semantic Analysis (LSA)** | 使用 SVD 对词-文档矩阵做矩阵分解降维，然后运行 K-means | | **Laplacian Eigenmaps (L ...[truncated]

### Diagnosis

The gold evidence itself contains the narrow answer clause followed by broader context about baseline dimensionality-reduction methods and non-biased neural networks. PaperPilot promoted the broader neighboring context into the direct answer, so this is not primarily a retrieval miss.

### Recommended Next Action

For list questions, final synthesis needs a scope boundary step: identify the noun phrase being asked for, then include only items governed by that phrase.


## qasper-1910.07181-q0: Wrong improvement percentages

- Question: How much is representaton improved for rare/medum frequency words compared to standalone BERT and previous work?
- Review decision: `downgrade_to_incorrect`
- Manual note: Question asks how much improvement; the central percentages differ from the gold answer.
- Likely failed layer: `numeric/entity verification failure`
- Trace used: `data\traces_day16_before_day18\qasper-1910.07181-q0.jsonl`

### Gold Evidence

> Results on WNLaMPro rare and medium are shown in Table TABREF34, where the mean reciprocal rank (MRR) is reported for BERT, Attentive Mimicking and Bertram. As can be seen, supplementing BERT with any of the proposed relearning methods results in noticeable improvements for the rare subset, with add clearly outperforming replace. Moreover, the add and add-gated variants of Bertram perform surprisingly well for more frequent words, improving the score for WNLaMPro-medium by 50% compared to BERT$_\text{base}$ and 31% compared to Attentive Mimicking. This makes sense considering that compared to Attentive Mimicking, the key enhancement of Bertram lies in improving context representations and interconnection of form and context; naturally, the more contexts are given, the more this comes into play. Noticeably, despite being both based on and integrated into a BERT$_\text{base} ...[truncated]

### Search Queries And Snippet Heads

**Search 1**

- Query: `WNLaMPro results rare medium frequency words MRR improvement BERTRAM`
- Snippet head: > { "paper_id": "1910.07181", "chunk_text": "and frequent ( all remaining words ). for our evaluation, we compare the performance of a standalone bert ( or roberta ) model with one that uses bertram

**Search 2**

- Query: `Table 1 MRR BERT base BERTRAM ADD rare medium 0.269 0.367`
- Snippet head: > { "paper_id": "1910.07181", "chunk_text": "2020 ) 0. 251 0. 267 + bertram - shallow 0. 250 0. 246 + bertram - replace 0. 155 0. 216 + bertram - add 0. 269 0. 367 bert ( large ) 0. 143 0. 264 rober

**Search 3**

- Query: `improving the score for WNLaMPro-MEDIUM by 58% compared to BERTbase and 37% compared to Attentive Mimicking`
- Snippet head: > { "paper_id": "1910.07181", "chunk_text": "to bertbase and 37 % compared to attentive mimicking. this makes sense con - sidering that the key enhancement of bertram over am lies in improving conte

**Search 4**

- Query: `BERTRAM downstream task accuracy improvement over BERT baseline rarified datasets MNLI AG News DBPedia`
- Snippet head: > { "paper_id": "1910.07181", "chunk_text": "results for our baselines on the rariﬁed datasets, we use the original mimick implementa - tion of pinter et al. ( 2017 ), the a la carte imple - mentati

### Final Answer Excerpt

> 现在我有足够的信息来综合回答了。下面是完整的答案。 --- ## 回答：BERTRAM 对 Rare/Medium 频率词的表示改进 ### 评估框架：WNLaMPro 数据集 论文使用 **WNLaMPro**（Schick & Schütze, 2020）数据集来探测模型对稀有词的理解能力。该数据集按照关键词在 WWC（Westbury Wikipedia Corpus）中的出现频率划分为三个子集： | 子集 | 频率范围 | |------|----------| | **RARE（稀有词）** | 出现 < 10 次 | | **MEDIUM（中等频率词）** | 出现 10–100 次 | | FREQUENT（高频词） | 其余（> 100 次） | 论文只评估了 RARE 和 MEDIUM 子集，评估指标为 **MRR（Mean Reciprocal Rank）**。 --- ### 核心结果：WNLaMPro 上的 MRR（Table 1） | 模型 | RARE (MRR) | MEDIUM (MRR) | |------|:----------:|:------------:| | **BERT (base)** — 基线 | 0.112 | 0.234 | | + AM（Attentive Mimicking，之前最好方法） | **0.251** | **0.267** | | + BERTRAM-SHALLOW | 0.250 | 0.246 | | + BERTRAM-REPLACE | 0.155 | 0.216 | | + **BERTRAM-ADD 🏆** | **0.269** | **0.367** | | BERT (large) | 0.143 | 0.264 | | RoBERTa (large) | 0.270 | 0.275 | | + **BERTRAM-ADD** (on RoBERTa) | **0.306** | **0.323** | --- ### 定量改进分析 #### 对 ** ...[truncated]

### Diagnosis

The answer selected MRR table values and a percentage statement from retrieved snippets, but calibration says the central percentages should be 50% and 31%, not 58% and 37%. This needs exact numeric relation verification against the gold-like evidence sentence, not more formatting cleanup.

### Recommended Next Action

For numeric questions, evidence selection should prefer sentences that directly answer the comparative relation in the question and should verify that the exact numbers in the short answer occur in that selected sentence.

## Proposed Evidence Selection V1 Direction

The next implementation should not expand the repair loop. It should add a narrow evidence-selection or verification step before final synthesis.

Recommended first slice:

1. Classify question type into `list`, `numeric`, `entity/dataset/method`, or `general`.
2. For `entity/dataset/method` questions, require a selected evidence sentence to contain both the candidate entity and the relation asked by the question.
3. For `numeric` questions, require the exact number in `Short answer` to appear in the selected evidence sentence.
4. For `list` questions, require the final list to stay inside the requested category phrase.

Before implementing this, improve trace capture so full retrieved top-k chunks are available for analysis.
