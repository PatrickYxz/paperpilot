# Semantic Calibration Candidates

Date: 2026-06-14

This file mirrors `data/eval/semantic_calibration_candidates_20260614.jsonl`.
Use it to fill manual review decisions, then copy those decisions back into the JSONL.

Allowed `review_decision` values:

- `TODO`
- `accept_as_correct`
- `keep_partial`
- `downgrade_to_incorrect`
- `judge_error`

## Summary

- Total candidates: 31
- partial_high: 11
- partial_medium: 14
- strict_pass_semantic_bad: 6

## partial_high

### 1. `qasper-1611.01400-q0`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `high`

**Question**

what were the baselines?

**Oracle spans**

- `Rank by the number of times a citation is mentioned in the document`
- `Rank by the number of times the citation is cited in the literature (citation impact).`
- `Rank using Google Scholar Related Articles.`
- `Rank by the TF*IDF weighted cosine similarity.`
- `ank using a learning-to-rank model trained on text similarity rankings`
- `(1) Rank by the number of times a citation is mentioned in the document.`
- `(2) Rank by the number of times the citation is cited in the literature (citation impact).`
- `(3) Rank using Google Scholar Related Articles.`
- `(4) Rank by the TF*IDF weighted cosine similarity.`
- `(5) Rank using a learning-to-rank model trained on text similarity rankings.`

**PaperPilot predicted excerpt**

```text
检索充分，所有关键信息已获取。现在进行 **Step 4: 综合回答**。

---

## 回答

**Answer span candidates:**
- `(1) Rank by the number of times a citation is mentioned in the document. (2) Rank by the number of times the citation is cited in the literature (citation impact). (3) Rank using Google Scholar Related Articles. (4) Rank by the TF*IDF weighted cosine similarity. (5) Rank using a learning-to-rank model trained on text similarity rankings.`
- `random baseline`
- `state-of-the-art baseline systems`

**Short answer:**  
The baselines were: **(1) rank by citation mention count in the document, (2) rank by citation impact (times cited in the literature), (3) rank using Google Scholar Related Articles, (4) rank by TF*IDF weighted cosine similarity, (5) rank using a learning-to-rank model trained on text similarity rankings** — plus a **random baseline** for reference.

**Evidence:**  
论文在 "Baseline Systems" 一节中明确列出：

> "We compare our system to a variety of baselines. (1) Rank by the number of times a citation is mentioned in the document. (2) Rank by the number of times the citation is cited in the literature (citation impact). (3) Rank using Google Scholar Related Articles. (4) Rank by the TF*IDF weighted cosine similarity. (5) Rank using a learning-to-rank model trained on text similarity rankings."

同时实验结果中还包括一个随机基线（random baseline）作为参考：`"A random baseline is provided for reference."`

**解释：**

- **Baseline (1)** — 按文献内被提及次数排序：假设文中引用次数越多的文献越重要。
- **Baseline (2)** — 按引文影响力排序（即该文献被其他文献引用的总次数）：被引越多越重要。
- **Baseline (3)** — 使用 Google Scholar 的 "Related Articles" 功能：将 Google Scholar 给出的
...[truncated]
```

**Judge reason**

The predicted answer correctly lists the five baselines from the gold evidence and adds 'random baseline' and 'state-of-the-art baseline systems' which are not present in the gold evidence. However, the core five baselines are semantically equivalent to the gold answer. The extra items are unsupported; the gold answer only mentions those five specific baselines. Since the gold question asks for 'the baselines' and the gold evidence does not include a random baseline or state-of-the-art baseline systems, the prediction includes an extra unsupported element, making it partially correct.

**Manual notes**

- Core five baselines are complete; random baseline is extra but not harmful for product use.

### 2. `qasper-1701.00185-q1`

- Review decision: `keep_partial`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `high`

**Question**

Which popular clustering methods did they experiment with?

**Oracle spans**

- `K-means, Skip-thought Vectors, Recursive Neural Network and Paragraph Vector based clustering methods`

**PaperPilot predicted excerpt**

```text
现在信息充足了，进行综合回答。

**Answer span candidates:**
- "K-means, Skip-thought Vectors, Recursive Neural Network and Paragraph Vector based clustering methods"
- "four baseline clustering methods are directly based on the popular unsupervised dimensionality reduction methods" (AE, LSA, LE, LPI)
- "some other non-biased neural networks, such as bidirectional RNN" (bi-LSTM, bi-GRU)

**Short answer:**  
The paper experimented with the following popular clustering methods: **K-means (with TF and TF-IDF), Skip-thought Vectors (SkipVec), Recursive Neural Network (RecNN), Paragraph Vector (Para2vec), Average Embedding (AE), Latent Semantic Analysis (LSA), Laplacian Eigenmaps (LE), Locality Preserving Indexing (LPI), and bidirectional RNN variants (bi-LSTM and bi-GRU).**

**Evidence:**  
Section 4.3 "Comparisons" explicitly lists these methods:

> *"In our experiment, some widely used text clustering methods are compared with our approach. Besides **K-means, Skip-thought Vectors, Recursive Neural Network and Paragraph Vector** based clustering methods, four baseline clustering methods are directly based on the popular unsupervised dimensionality reduction methods as described in Section 3.2" — namely **AE, LSA, LE, LPI** — "We further compare our approach with some other non-biased neural networks, such as **bidirectional RNN**" — i.e. **bi-LSTM and bi-GRU**.*

**Explanation:**  
论文将对比方法分为三大类：

| 类别 | 具体方法 |
|------|----------|
| **传统方法** | K-means (TF), K-means (TF-IDF) — 直接在原始词特征上做 K-means |
| **表示学习方法** | Skip-thought Vectors (SkipVec Uni/Bi/Combine)、Recursive Neural Netwo
...[truncated]
```

**Judge reason**

The predicted answer correctly includes K-means, Skip-thought Vectors, Recursive Neural Network, and Paragraph Vector methods from the gold extractive spans. However, the gold answer identifies only those four methods as 'popular clustering methods' (which is the specific category asked about), whereas the predicted answer also adds multiple other methods (AE, LSA, LE, LPI, bi-LSTM, bi-GRU) that are not described as 'popular clustering methods' in the gold answer/evidence; the evidence groups those additional methods separately as baseline or non-biased neural networks. Therefore, the core answer is correct for the explicit list, but the inclusion of extra methods goes beyond what the question asks and may introduce confusion, making it only partially aligned with the intended scope of the question.

**Manual notes**

- Gold asks for four popular clustering methods; answer includes them but adds separate baseline/neural methods that blur scope.

### 3. `qasper-1807.03367-q1`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `high`

**Question**

What evaluation metrics did the authors look at?

**Oracle spans**

- `localization accuracy`

**PaperPilot predicted excerpt**

```text
Now I have enough evidence. Let me synthesize the answer.

**Step 4: 综合回答**

## Answer span candidates:

- "localization accuracy" (百分比形式，如 88.33%, 76.74%)
- "human success rate" / "76.74% of all finished tasks"
- "task ends when the evaluation is successful — i.e., when (x, y) = (x, y)_tgt — or otherwise continues until a total of three failed attempts"
- "#steps" (平均步数，如 ~15 steps for humans, ~34 for best model)
- "accuracy of localization models on full task, using evaluation protocol defined in Algorithm 1"

---

**Short answer:** The authors used **localization accuracy (%)** as the primary evaluation metric for the tourist localization sub-task, and **task success rate (%)** (percentage of successfully completed navigation tasks) along with **number of actions/steps** for the full task evaluation.

**Evidence:**

从检索到段落中可以看到：

1. **任务成功标准 (Full task)**: "the task ends when the evaluation is successful — i.e., when (x, y) = (x, y)_tgt — or otherwise continues until a total of three failed attempts" — 即 agent 能否将 tourist 导航到正确目标位置。

2. **Human success rate**: "Turkers successfully completed 76.74% of all finished tasks (we use this statistic as the human success rate)."

3. **Localization accuracy**: 论文各结果表（Table 2, 3, 4, 9）统一用 **accuracy (%)** 来衡量 localization 模型预测 tourist 位置的正确率。例如 Table 4 显示 "Best Cont. 88.33% on test set" 和 "Human 76.74%"。

4. **Number of steps (#steps)**: 在 full task 评估（Table 4, Table 9）中，还报告了完成任务所需的平均步数，例如 human annotators 平均约 **15.05 steps**，而最好的 continuous 模型平均约 **34.47 steps**。

5. **Upper bound (Bayes error rate / irreducible er
...[truncated]
```

**Judge reason**

The predicted answer correctly identifies 'localization accuracy' as a key metric, which matches the gold extractive spans. However, it also introduces additional evaluation metrics (e.g., task success rate, number of steps) that are not specified in the gold answer. Since the gold answer only includes 'localization accuracy', the predicted answer partially matches but goes beyond the scope.

**Manual notes**

- Localization accuracy is covered; extra success-rate/step metrics are relevant broader evaluation context.

### 4. `qasper-1807.03367-q2`

- Review decision: `accept_as_correct`
- Strict pass: `false`
- Semantic label: `partial`
- Confidence: `high`

**Question**

What data did they use?

**Oracle spans**

- `dataset on Mechanical Turk involving human perception, action and communication`

**PaperPilot predicted excerpt**

```text
Now I have comprehensive information. Let me synthesize the answer.

---

**Answer span candidates:**
- "over 10k successful dialogues"
- "five neighborhoods: Hell's Kitchen, East Village, the Financial District, Williamsburg and the Upper East Side"
- "307 different landmarks divided among nine classes: Bar, Playfield, Bank, Hotel, Shop, Subway, Coffee Shop, Restaurant, Theater"
- "crowd-sourced on Amazon Mechanical Turk (MTurk)"
- "vocabulary size of more than 10K"
- "more than 62 acts (utterances and actions) per dialogue"

**Short answer:** The Talk The Walk dataset consists of **over 10k successful dialogues** crowd-sourced on **Amazon Mechanical Turk (MTurk)**, set in **five NYC neighborhoods** (Hell's Kitchen, East Village, Financial District, Williamsburg, Upper East Side), with **307 landmarks across nine categories** (Bar, Playfield, Bank, Hotel, Shop, Subway, Coffee Shop, Restaurant, Theater), using **~10x10 grids per neighborhood** with **4x4 subgrids** per task.

**Evidence:**
From Section 2 ("Talk The Walk"): "we capture parts of Hell's Kitchen, East Village, the Financial District, Williamsburg and the Upper East Side… For each neighborhood, we choose an approximately 5x5 grid and capture a 360 view on all four corners of each intersection, leading to a grid-size of roughly 10x10 per neighborhood."

From Section 2.2 ("Data Collection"): "We crowd-sourced the collection of the dataset on Amazon Mechanical Turk (MTurk). We use the MTurk interface of ParlAI… We paired Turkers at random and let them alternate between the tourist and guide role acro
...[truncated]
```

**Judge reason**

The predicted answer provides a rich set of details about the dataset, but the core gold definition is 'dataset on Mechanical Turk involving human perception, action and communication.' The prediction correctly identifies MTurk and the dataset size, which suggests it is at least partially correct. However, the gold answer and evidence do not require the additional specifics (neighborhoods, landmarks, grid sizes) that the prediction includes; therefore the prediction goes beyond the necessary core. But since it does capture the essential 'MTurk' aspect and the general nature of the dataset, it is more than incorrect. It is not fully correct because the gold answer is more general and the prediction adds extra, unsupported specifics that may not be needed. Hence 'partial' is most appropriate.

**Manual notes**

- Identifies the MTurk Talk The Walk dataset and gives specific dataset details; core gold answer is covered.

### 5. `qasper-1905.11901-q1`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `high`

**Question**

what were their experimental results in the low-resource dataset?

**Oracle spans**

- `10.37 BLEU`

**PaperPilot predicted excerpt**

```text
现在我已经收集到所有关键数据。让我综合回答。

**Answer span candidates:**  
- **DE→EN 100k words (ultra-low):** NMT baseline 0.00 BLEU → optimized (system 8) **16.57 ± 0.26** BLEU; PBSMT (system 1) **15.87 ± 0.19** BLEU  
- **DE→EN full 3.2M words:** NMT baseline 25.70 → optimized (system 7) **33.60 ± 0.14** BLEU; PBSMT 26.60  
- **"adaptations to the ultra-low data setting yield 9.4 BLEU (7.2→16.6)"**  
- **KO→EN:** NMT baseline 2.93 → optimized (system 8) **10.37 ± 0.29** BLEU; previous SOTA (Gu et al. 2018b) 5.97  
- **"optimized NMT system can outperform PBSMT with as little as 100 000 words of parallel training data"**

**Short answer:** On IWSLT14 DE→EN with 100k words (ultra-low), their optimized NMT achieved **16.57 ± 0.26 BLEU** (vs NMT baseline 0.00 and PBSMT 15.87); on full 3.2M words it reached **33.60 ± 0.14 BLEU** (best system 7). On KO→EN low-resource, optimized NMT achieved **10.37 ± 0.29 BLEU**, surpassing previous best (5.97) by ~4 BLEU.

**Evidence:**

1. **Table 2 (DE→EN ablation, 100k vs 3.2M words):** The paper reports the full ablation in Table 2:
   - System 1 (PBSMT): 15.87 ± 0.19 (100k) / 26.60 (3.2M)
   - System 2 (NMT baseline): 0.00 ± 0.00 (100k) / 25.70 ± 0.33 (3.2M)
   - System 3 (+mainstream improvements): 7.20 ± 0.62 / 31.93 ± 0.05
   - System 4 (+reduce BPE vocab 14k→2k): 12.10 ± 0.16 / —
   - System 5 (+reduce batch size 4k→1k): 12.40 ± 0.08 / 31.97 ± 0.26
   - System 6 (+lexical model): 13.03 ± 0.49 / 31.80 ± 0.22
   - System 7 (+aggressive word dropout): **15.87 ± 0.09 / 33.60 ± 0.14**
   - System 8 (+other hyperparameter tuning): **16.57 ± 0.2
...[truncated]
```

**Judge reason**

The predicted answer includes the gold extractive span '10.37 BLEU' for the KO→EN direction, which is the core answer from the gold evidence. However, the question asks for 'their experimental results in the low-resource dataset' (singular), and the gold answer specifically refers to the KO→EN low-resource dataset, not multiple datasets. The predicted answer adds extensive results on DE→EN (100k and 3.2M words), which are from a different experiment and not part of the gold answer for this question. Since the gold answer defines the core answer as the KO→EN BLEU score, the extra DE→EN results represent additional, unrequested information that does not contradict but makes the answer more specific and includes experiments not asked about. As the core claim (KO→EN 10.37 BLEU) is present and correct, the answer is partially correct: it addresses the gold answer but goes beyon ...[truncated]

**Manual notes**

- Includes the gold KO-EN 10.37 BLEU result; extra low-resource DE-EN results are related context.

### 6. `qasper-1910.11471-q0`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `high`

**Question**

What additional techniques are incorporated?

**Oracle spans**

- `incorporating coding syntax tree model`

**PaperPilot predicted excerpt**

```text
检索充分。现在综合回答。

---

**Answer span candidates:**

1. "coding syntax tree model"
2. "phrase-based word embedding"
3. "Abstract Syntax Tree (AST)"
4. "more extensive set of data (100k+ in such scenario)"

**Short answer:** The paper mentions three additional techniques that can be incorporated: (1) **coding syntax tree model** / **Abstract Syntax Tree (AST)** for more accurate target code per line, (2) **phrase-based word embedding** for improved vocabulary mapping, and (3) a **more extensive set of data (100k+)** to build a more accurate LSTM model.

**Evidence:**

From the "Result Analysis" and "Conclusion & Future Works" sections (retrieved chunks):

1. **Coding syntax tree model / AST** — The paper states: *"the incoherence can be resolved by incorporating coding syntax tree model in future"* and in the Conclusion: *"to get more accurate target code for each line, abstract syntax tree (AST) can be beneficial."*

2. **Phrase-based word embedding** — The Conclusion states: *"In later phase, phrase-based word embedding can be incorporated for improved vocabulary mapping."*

3. **Larger training data** — The Results section explains: *"LSTM generally requires a more extensive set of data (100k+ in such scenario) to build a more accurate model."*

**Explanation:**

The paper achieved 74.40% accuracy with an LSTM-based seq2seq model using 18,805 parallel text-code pairs. The authors explicitly acknowledge that the model's output was often incoherent and missing noun phrases (parameter/function names). To address this, they propose three concrete improvements: (1) i
...[truncated]
```

**Judge reason**

The predicted answer lists three techniques: coding syntax tree model/AST, phrase-based word embedding, and larger training data. The gold answer indicates that Question 1's answer is unanswerable (Answer 1) and Answer 2's core answer is only 'coding syntax tree model'. The prediction includes that core part correctly, but also adds 'phrase-based word embedding' and 'larger training data' as extra techniques. Since the gold evidence only supports 'coding syntax tree model' for this specific question, the extra additions make the answer partially correct but not fully aligned with the gold answer. The core correct technique is included, but the prediction is broader than the gold answer, so it is labeled partial.

**Manual notes**

- Includes coding syntax tree/AST, the gold technique; extra future-work techniques are supported context.

### 7. `qasper-1911.03385-q1`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `high`

**Question**

What metrics are used for automatic evaluation?

**Oracle spans**

- `classification accuracy`
- `BLEU scores`
- `model perplexities of the reconstruction`

**PaperPilot predicted excerpt**

```text
现在我已经收集到了所有必要的信息。让我综合回答。

**Answer span candidates:**
- "BLEU scores" (Table 4: Baseline 25.07, StyleEQ 30.04)
- "perplexity" (Table 4: Baseline 4.60, StyleEQ 3.33)
- "Ablated NVA classifier accuracy" (Table 6, oracle method: StyleEQ up to 84.1%)
- "Exact, Direction, Atomic" control change rates (Table 5)
- "length normalized log-likelihood"

**Short answer:** The paper uses **BLEU score**, **perplexity** (in nats), **Ablated NVA classifier accuracy**, and **Exact/Direction/Atomic feature control change rates** as its automatic evaluation metrics.

**Evidence:**

1. **BLEU & Perplexity (Section 4.1):** "In Table 4 we report BLEU scores for the reconstruction of test set sentences from their content and feature representations, as well as the model perplexities of the reconstruction." Table 4 shows: Baseline BLEU=**25.07**, Perplexity=**4.60**; StyleEQ BLEU=**30.04**, Perplexity=**3.33** (in nats).

2. **Feature Control Rates (Section 4.2, Table 5):** Three metrics — **Exact** (percentage of generated texts that realize the exact number of control features specified), **Direction** (percentage where the change goes in the correct direction though not exact), and **Atomic** (percentage where only that specific control changes while others remain constant). For example, conjunction has Exact=93.56%, Direction=98.75%, Atomic=11.43%.

3. **Ablated NVA Classifier Accuracy (Section 4.3, Table 6):** The most content-blind classifier (ablating nouns, verbs, adjectives) is used to measure whether transferred sentences are classified into the intended style. Results are
...[truncated]
```

**Judge reason**

The gold answer specifies three metrics: classification accuracy, BLEU scores, and model perplexities of the reconstruction. The predicted answer includes BLEU scores and perplexity (core metrics) but also adds 'Ablated NVA classifier accuracy' and 'Exact/Direction/Atomic feature control change rates', which are not part of the gold answer. The core answer covers two of the three gold metrics, missing 'classification accuracy', and includes extra details. Thus, it is partially correct.

**Manual notes**

- Covers BLEU, perplexity, and classifier accuracy; judge was too strict about the classification accuracy wording.

### 8. `qasper-1911.11750-q2`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `high`

**Question**

How do they evaluate knowledge extraction performance?

**Oracle spans**

- `SRCC`

**PaperPilot predicted excerpt**

```text
所有检索结果已经足够回答问题了。现在进入**Step 4: 综合回答**。

---

## Answer

**Answer span candidates:**

- "dataset of 14 TDs" / "subjects: (aliens, stories, law, news)"
- "compared the similarity values produced by each of the similarity measures CS, SRCC and PCC" / "Table II"
- "SRCC does not perform worse than any other of these similarity measures"
- "nonlinear association between some of the TDs" / "Figure 2"
- "our future work will concentrate on the implementation of machine learning techniques, such as clustering and subsequent classification of textual data"

**Short answer:** They evaluate knowledge extraction performance by **comparing the similarity values produced by CS, SRCC and PCC** on **a dataset of 14 TDs** covering subjects (aliens, stories, law, news), presenting results in **Table II** and visually in **Figure 1**, and additionally demonstrating **nonlinear associations** in **Figure 2**.

---

**Evidence:**

从 Section V (EXPERIMENTS) 的检索段落可知，论文通过以下方式评估知识提取（KE）性能：

1. **数据集**：使用了 **14 个文本文档 (TDs)**，内容涵盖 **aliens, stories, law, news** 四个主题（引用自 [16]）。

2. **比较指标**：将提出的 **SRCC** 与 **余弦相似度 (CS)** 和 **皮尔逊相关系数 (PCC)** 进行对比。在 Table II 中列出了多个文档对的相似度数值，例如：
   - d1 (领导力) vs d5 (家庭医疗法案)：CS=0.36, PCC=0.05, **SRCC=0.0018**——作者认为 SRCC 的 0.0018 更真实地反映了二者语义不相关的事实，而 CS 的 0.36 则是"very unreal"。
   - d8 vs d9（外星人/天使话题相近）：CS=0.97, PCC=0.97, SRCC=0.95——在直白相似的情况下 SRCC 性能与 CS、PCC 相当。

3. **定性分析**：论文对每个文档对的内容进行了人工语义审查，判断哪个相似度值最符合真实语义关联。结论是 **"SRCC does not perform worse than any other of these similarity measures"** 且 **"it indeed does perform at least equally good as other measures"
...[truncated]
```

**Judge reason**

The predicted answer lists multiple evaluation methods (comparing similarity measures, using a dataset of 14 TDs, tables, etc.), but the gold answer (extractive spans: 'SRCC') and evidence indicate that the evaluation uses the SRCC method itself to show performance, focusing on SRCC's capability. The prediction includes SRCC as one of the measures compared but adds extra details (dataset, subjects, tables) not explicitly required by the gold answer. It correctly captures that comparing similarity measures involving SRCC is the evaluation approach, which aligns with the gold answer's core; however, the prediction is overly broad and includes many elements not part of the concise gold answer, making it partially correct but not a direct match to the core claim.

**Manual notes**

- SRCC is included and the answer explains the evaluation comparison setup; useful answer to a how-question.

### 9. `qasper-1912.03234-q0`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `high`

**Question**

What evaluation metrics were used?

**Oracle spans**

- `AUC-ROC`

**PaperPilot predicted excerpt**

```text
信息已经非常充分了，可以进入最终回答。

---

## 最终回答

**Answer span candidates:**
- `AUC-ROC`
- `Overall Accuracy` / `O.A.`
- `top-1 accuracy`
- `user interruption rate`, `reuse of this and other VVA skills`, `number of active dialogs`

**Short answer:** The evaluation metrics used were **AUC-ROC**, **Overall Accuracy**, **top-1 accuracy**, and online user-satisfaction metrics including **user interruption rate**, **reuse of this and other VVA skills**, and **number of active dialogs**.

**Evidence:**

1. **AUC-ROC** — 论文中明确说明："hyperparameters were optimized to maximize the **AUC-ROC** on the validation set"；Table 4 标题为 "Relative change w.r.t popularity model of **AUC-ROC** and **Overall Accuracy**"；"We start by evaluating the models using **AUC-ROC**"。

2. **Overall Accuracy (O.A.)** — Table 4 中列出的第二列指标 "R. Ch. O.A."（即 Relative Change of Overall Accuracy）；文中指出 "Similar conclusions can be reached regarding **overall accuracy**. However, given the class imbalance, accuracy is not necessarily the best metric to consider."

3. **top-1 accuracy** — "to further evaluate the ranking capabilities of the proposed methods, we use **top-1 accuracy**. Additional positions in the ranking are not considered because only the top ranked joke is presented to the customer."

4. **Online user-satisfaction metrics** — 在线 A/B 测试中监测："Several **user-satisfaction metrics** such as **user interruption rate**, **reuse of this and other VVA skills**, and **number of active dialogs** are monitored during the tests."

5. **Statistical significance** — 在线测试中还测量了统计显著性："The **statistical significance** is me
...[truncated]
```

**Judge reason**

The gold answer only mentions AUC-ROC. The predicted answer includes AUC-ROC (correct part) but also adds multiple other metrics that are not present in the gold extractive spans or evidence. Since the gold answer is single-faceted (only AUC-ROC), listing additional metrics beyond AUC-ROC makes the prediction partially correct but not fully matching the gold answer's scope. The prediction correctly contains AUC-ROC, satisfying the core requirement, but the extra metrics are not part of the gold answer, so it is not a complete match. According to the guidelines, when the gold answer is specific (only AUC-ROC), a prediction that adds extra metrics is considered partial because it goes beyond what was asked and verified.

**Manual notes**

- AUC-ROC is present; additional metrics are relevant evaluation context and do not change the core answer.

### 10. `qasper-1912.13109-q2`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `high`

**Question**

How big is the dataset?

**Oracle spans**

- `3189 rows of text messages`

**PaperPilot predicted excerpt**

```text
检索结果非常充分。现在进行 **Step 4：综合回答**。

---

**Answer span candidates:**
- "3189 rows of text messages"
- "total 3189" (Table 1, Hinglish dataset)
- "14509" (Table 1, English dataset total)
- "total 7934" (Table 3, augmented training set)
- "700 tweets/messages were held out for testing" (test set)
- "average length of 116 words and with a range of 1, 1295"

**Short answer:** The Hinglish labeled dataset has **3,189 rows** (messages/tweets), while the parallel English dataset has **14,509** messages. After data augmentation on the training portion (2,489 original messages), the augmented training set becomes **7,934 messages**, with 700 messages held out for testing. The average message length is 116 words (range 1–1,295).

**Evidence:**

From Section 1.1 (Modeling challenges) and Section 3 (Dataset and Features):
- "only available labelled dataset had **3189 rows** of text messages of average length of **116 words** and with a range of **1, 1295**"
- Table 1 shows the Hinglish dataset total is **3,189** (Non-Offensive 1121 + Offensive 303 + Hate Inducing 1765), and the parallel English dataset total is **14,509**
- "a train-test split of 78%-22% was done from the original, cleansed data set. Thus, **700 tweets/messages** were held out for testing ... The training set of **2489 messages** were however sent to an offline pipeline for augmenting the data. The resulting training dataset was thus **7934 messages**."
- Table 3 confirms: Train total = **7,934**, Test total = **700**

**Explanation:**

论文使用了两个数据集：(1) **HEOT (Hinglish) 数据集** — 原始标注为 **3,189 条**推文/消息，来自 Mathu
...[truncated]
```

**Judge reason**

The predicted answer includes the core fact from Gold Answer 1 (3,189 rows) but also includes additional numbers (14,509 English dataset, 7,934 augmented training set, 700 test set) which correspond to Gold Answer 2. The question 'How big is the dataset?' is ambiguous, but the gold answers separately provide the original dataset size (3,189 rows) and the augmented training/test set sizes (7,934/700). The prediction covers both parts, making it partially correct but overly broad, and the extra numbers for the English dataset are unsupported by the gold evidence for this question.

**Manual notes**

- Includes 3,189 rows, the core dataset size; extra split/augmented sizes are related context.

### 11. `qasper-2002.11893-q1`

- Review decision: `accept_as_correct`
- Strict pass: `false`
- Semantic label: `partial`
- Confidence: `high`

**Question**

What are the benchmark models?

**Oracle spans**

- `BERTNLU from ConvLab-2`
- `a rule-based model (RuleDST)`
- `TRADE (Transferable Dialogue State Generator)`
- `a vanilla policy trained in a supervised fashion from ConvLab-2 (SL policy)`

**PaperPilot predicted excerpt**

```text
信息已足够充分。现在进行综合回答。

## Answer span candidates:

- **Natural Language Understanding (NLU)**: **BERTNLU** (adapted from ConvLab-2, using Chinese pre-trained BERT-wwm-ext)
- **Dialogue State Tracking (DST)**: **RuleDST** (rule-based model) and **TRADE** (Transferable Dialogue State Generator)
- **Dialogue Policy Learning**: **SL policy** (supervised learning policy from ConvLab-2)
- **Natural Language Generation (NLG)**: **TemplateNLG** (template-based) and **SC-LSTM** (Semantically Conditioned LSTM)
- **User Simulator**: **Rule-based user simulator** (dialogue act level)
- **System-level evaluation configurations**: **DA Sim** (dialogue act level), **NL Sim (Template)** (using TemplateNLG), **NL Sim (SC-LSTM)** (using SC-LSTM)

---

## Short answer:

The benchmark models are: **BERTNLU** (NLU), **RuleDST & TRADE** (DST), **SL policy** (policy learning), **TemplateNLG & SC-LSTM** (NLG), and a **rule-based user simulator**, all implemented using **ConvLab-2**.

## Evidence:

从 Section 6 (Benchmark and Analysis) 的检索段落中可以明确提取：

1. **Section 6 开头**: "we provided benchmark models for different components of a pipelined task-oriented dialogue system, including natural language understanding (NLU), dialogue state tracking (DST), dialogue policy learning, and natural language generation (NLG). These models are implemented using ConvLab-2."

2. **Section 6.1 – NLU**: "We adapted **BERTNLU** from ConvLab-2… We use Chinese pre-trained BERT (Cui et al., 2019) for initialization."

3. **Section 6.2 – DST**: "We implemented a rule-based model (**RuleDST**) and adapted **TRADE*
...[truncated]
```

**Judge reason**

The predicted answer correctly lists the four gold benchmark models (BERTNLU, RuleDST, TRADE, SL policy), which matches the gold answer. However, it additionally includes models and components (TemplateNLG, SC-LSTM, rule-based user simulator, DA Sim, NL Sim configurations) that are not part of the gold answer's list of benchmark models. While extra detail is acceptable per the rules, the predicted answer frames these as part of the benchmark models, which goes beyond the gold answer and could mislead. Since the core required answer (the four models) is present and correct, the label is partial due to the inclusion of extraneous elements presented as benchmarks.

**Manual notes**

- BERTNLU, RuleDST, TRADE, and SL policy are all present; extra component models are broader but supported.


## partial_medium

### 1. `qasper-1705.09665-q0`

- Review decision: `accept_as_correct`
- Strict pass: `false`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

What patterns do they observe about how user engagement varies with the characteristics of a community?

**Oracle spans**

- `communities that are characterized by specialized, constantly-updating content have higher user retention rates, but also exhibit larger linguistic gaps that separate newcomers from established members`
- `within distinctive communities, established users have an increased propensity to engage with the community's specialized content, compared to newcomers`

**PaperPilot predicted excerpt**

```text
Now I have rich evidence from 4 searches. Let me synthesize the answer.

**Answer span candidates:**
- "communities with distinctive and highly dynamic identities are more likely to retain their users"
- "dynamic communities ... have substantially higher rates of monthly user retention than more stable communities (Spearman's ρ = 0.70, p < 0.001)"
- "more distinctive communities ... exhibit moderately higher monthly retention rates (Spearman's ρ = 0.33, p < 0.001)"
- "no significant relationship between distinctiveness and long-term engagement (Spearman's ρ = 0.03, p = 0.77)"
- "niche communities also exhibit much larger acculturation gaps between existing users and newcomers"
- "highly distinctive communities have far higher acculturation gaps, while dynamicity exhibits a non-linear relationship"
- "while in distinctive communities active members have a higher affinity for both community-specific content and for highly volatile content, the opposite is true for generic communities, where it is the outsiders who engage more with volatile content"
- "temporally ephemeral content might serve as an entry point into such a community, without necessarily engaging users in the long term"

---

**Short answer:** The paper finds that **"communities with distinctive and highly dynamic identities are more likely to retain their users"**; dynamicity strongly drives both short-term (ρ=0.70) and long-term retention (ρ=0.41), while distinctiveness only moderately predicts monthly retention (ρ=0.33) and does not predict long-term tenure (ρ=0.03, n.s.); distinctive communiti
...[truncated]
```

**Judge reason**

The predicted answer lists multiple specific observations (retention rates correlated with distinctiveness/dynamicity, acculturation gaps, content engagement patterns) but does not directly and concisely answer the question as a single coherent statement. The gold answer provides a clear, integrated summary: specialized constantly-updating content leads to higher retention but larger linguistic gaps, and in distinctive communities established users engage more with specialized content than newcomers. The prediction enumerates many true findings but misses the explicit causal pairing and the specific comparison of established vs. newcomers in distinctive communities. It covers part of the gold answer—retention and linguistic gaps—but the core insight about relative engagement of established users vs. newcomers is not stated as a distinct pattern. Hence partial.

**Manual notes**

- Covers retention, acculturation gaps, and established-vs-newcomer engagement patterns; judge penalized verbosity more than correctness.

### 2. `qasper-1808.05902-q0`

- Review decision: `keep_partial`
- Strict pass: `false`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

what are the advantages of the proposed model?

**Oracle spans**

- `he proposed model outperforms all the baselines, being the svi version the one that performs best.`
- `the svi version converges much faster to higher values of the log marginal likelihood when compared to the batch version, which reflects the efficiency of the svi algorithm.`

**PaperPilot predicted excerpt**

```text
现在我有足够的证据来综合回答这个问题了。

---

**Answer span candidates:**

1. "account for the different reliabilities of multiple annotators and correct their biases"
2. "the proposed models are able to outperform other state-of-the-art approaches in both classification and regression tasks"
3. "it differs significantly by using a fully Bayesian approach for estimating the reliabilities and biases of the different annotators"
4. "we take this idea one step further by also considering a per-annotator bias parameter"
5. "an efficient stochastic variational inference algorithm that is able to scale to very large datasets"

**Short answer:** The proposed model's core advantages are: (a) it **accounts for the heterogeneity and biases among different annotators** via per-annotator confusion matrices (classification) and per-annotator bias + precision parameters (regression); (b) it **jointly models topic discovery and multiple-annotator noise** in a single fully generative framework; (c) it uses a **fully Bayesian approach** (Dirichlet priors on annotator reliability) rather than point estimates; (d) the regression variant introduces a **per-annotator bias parameter** beyond what prior works did; and (e) it provides an **efficient stochastic variational inference algorithm** that scales to large datasets — all leading to **outperforming state-of-the-art baselines** on real-world classification and regression tasks.

**Evidence:**

1. **Abstract & Introduction** — "we propose two supervised topic models, one for classification and another for regression problems, which account for th
...[truncated]
```

**Judge reason**

The gold answer emphasizes that the proposed model outperforms all baselines and that the SVI version is the best and converges faster. The predicted answer mentions outperforming state-of-the-art approaches, which is consistent, and also lists other advantages (accounting for annotator heterogeneity, joint modeling, fully Bayesian approach, per-annotator bias parameter, efficient SVI), most of which are supported by evidence in the paper (e.g., accounting for different reliabilities and biases, fully Bayesian approach, efficient SVI algorithm). However, the gold answer's specific emphasis on the SVI version being the best and converging faster is only partially captured. The prediction does not explicitly state that the SVI version performs best or converges faster, but it does mention 'efficient SVI algorithm' which implies computational advantage. The prediction is more ...[truncated]

**Manual notes**

- Mentions outperforming baselines and efficient SVI, but misses the specific SVI-best/converges-faster emphasis.

### 3. `qasper-1808.05902-q1`

- Review decision: `keep_partial`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

what are the state of the art approaches?

**Oracle spans**

- `Bosch 2006 (mv)`
- `LDA + LogReg (mv)`
- `LDA + Raykar`
- `LDA + Rodrigues`
- `Blei 2003 (mv)`
- `sLDA (mv)`

**PaperPilot predicted excerpt**

```text
现在我有足够的信息来综合回答。让我整理答案。

---

## Answer span candidates:

- **"supervised LDA (sLDA)"**
- **"DiscLDA"**, **"Labeled-LDA"**
- **"MedLDA"** (max-margin supervised topic model)
- **"Dirichlet-multinomial regression (DMR)"**
- **"inverse regression topic model (IRTM)"**
- **"Raykar et al."** / **"LDA+Raykar"**
- **"Rodrigues et al."** / **"LDA+Rodrigues"**
- **"Blei 2003 (mv)"** (LDA + SVM with majority voting)
- **"sLDA (mv)"**
- **"Bosch 2006 (mv)"** (pLSA + kNN)
- **"Gaussian process classifiers"** for multiple annotators [23], [24]
- **"LDA + LinReg (mean)"** and **"sLDA (mean)"** for regression

## Short answer:

The paper identifies **two major categories of state-of-the-art approaches**: **(1) Supervised topic models** (sLDA, DiscLDA, Labeled-LDA, MedLDA, DMR, IRTM) and **(2) Learning-from-multiple-annotators methods** (Raykar et al. [22], Rodrigues et al. [10], Gaussian process approaches [23][24], Dawid & Skene [20]). The paper's own proposed models — **MA-sLDAc** (classification) and **MA-sLDAr** (regression) — are shown to **outperform** these state-of-the-art approaches on both simulated and real crowdsourced datasets.

## Evidence:

From Section 2 (State of the Art), the paper reviews:

1. **Supervised topic models:**
   - **sLDA** (Mcauliffe & Blei, 2008) — extends LDA with a response variable linearly dependent on mean topic-assignments; later extended to classification with softmax [3].
   - **DiscLDA** (Lacoste-Julien et al., 2009) — introduces class-dependent linear transformations on topic mixture proportions.
   - **Labeled-LDA** (Ramage et al.
...[truncated]
```

**Judge reason**

The predicted answer lists many approaches not mentioned in the gold evidence (e.g., DiscLDA, Labeled-LDA, MedLDA, DMR, IRTM, Gaussian process classifiers, Dawid & Skene), and organizes them into categories not present in the gold answer. However, it does include the core gold approaches (Bosch 2006 (mv), LDA + LogReg (mv), LDA + Raykar, LDA + Rodrigues, Blei 2003 (mv), sLDA (mv)) as part of its list, and these are directly stated. The gold answer is a specific list of six approaches from the paper's baselines; the prediction covers these but adds many unrelated ones. Since the core gold approaches are present, the answer is partially correct but includes extraneous content that deviates from the specific set requested.

**Manual notes**

- Includes the six gold approaches but mixes in many related-work methods, making the scope noisy.

### 4. `qasper-1808.05902-q2`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

what datasets were used?

**Oracle spans**

- `Reuters-21578 BIBREF30`
- `LabelMe BIBREF31`
- `20-Newsgroups benchmark corpus BIBREF29`
- `20-Newsgroups benchmark corpus`
- `Reuters-21578`
- `LabelMe`

**PaperPilot predicted excerpt**

```text
检索结果已经非常充分。现在综合回答。

---

**Answer span candidates:**
- 20 Newsgroups
- Reuters-21578
- LabelMe
- we8there
- movie reviews dataset

**Short answer:** The paper used **5 datasets**: **20 Newsgroups**, **Reuters-21578**, **LabelMe** (for classification); **we8there** and the **movie review dataset** from Pang & Lee (for regression).

---

**Evidence:** 来自论文 Sec. 5 Experiments 以及 Table 2 和 Table 4 中的详细记录。

**详细说明：**

### 分类任务（Classification）使用的数据集

| 数据集 | 类别数 | 训练/测试规模 | 标注来源 | 详情 |
|--------|--------|-------------|---------|------|
| **20 Newsgroups** | 4 | 11,536 / 3,846 | 模拟标注者 | 选取了4个最大超类："computers"、"science"、"politics"、"recreative"。模拟了5个标注者，准确率分别为 0.737, 0.468, 0.284, 0.278, 0.260，平均准确率 0.405。 |
| **Reuters-21578** | 8 | 1,800 / 5,216 | Amazon Mechanical Turk | ModApte 划分，仅保留单标签文档，共 7,016 篇文档。1,800 篇提交到 AMT 标注，平均每个文档约 3 个答案，工人平均准确率 56.8%，多数投票后 71.0%。 |
| **LabelMe** | 8 | 1,000 / 1,688 | Amazon Mechanical Turk | 图像分类数据集，8 个场景类别（highway, inside city, tall building, street, forest, coast, mountain, open country）。2,688 张图片，1,000 张提交 AMT，平均每个图像 2.547 个标注，工人平均准确率 69.2%，多数投票后 76.9%。图像用 SIFT 描述子 + k-means（k=200）得到的视觉词袋表示。 |

### 回归任务（Regression）使用的数据集

| 数据集 | 训练/测试规模 | 标注来源 | 详情 |
|--------|-------------|---------|------|
| **we8there** | 4,624 / 1,542 | 模拟标注者 | 来自网站 we8there.com 的餐厅评论数据集，共 6,260 条评论。目标是从评论文本预测总体体验评分（五星制）。模拟了 5 个标注者，设定了不同的偏差和精度参数。 |
| **Movie Reviews** | 1,500 / 3,506 | Amazon Mechanical Turk | 来自 Pang & Lee 的电影评论数据集，共 5,006 条评论，star rating 1-10。1,500 条提交 AMT 标注，平均每个评论 4.96 个答案，每个工人约标注 55 条。均值答案估计的 R² 达 0.830。 |

**总结：** 论文在分类任务中使用了一个模拟标注数据集（20
...[truncated]
```

**Judge reason**

The predicated answer correctly lists Reuters-21578, LabelMe, and 20 Newsgroups (20-Newsgroups benchmark corpus) which match the gold answer. However, it also includes we8there and movie reviews dataset as additional datasets from the paper (for regression) which go beyond the gold extractive spans and evidence that only mention the three classification datasets. The question 'what datasets were used?' is broad and the gold answer specifically only includes the classification datasets; the predicted answer adds regression datasets that are not part of the gold answer evidence. Since the gold answer is limited to those three, the prediction over-answers by including extra datasets, making it partly correct but incomplete relative to the intended gold scope.

**Manual notes**

- Gold datasets are all included; regression datasets are extra but the broad question asks what datasets were used.

### 5. `qasper-1809.04960-q0`

- Review decision: `keep_partial`
- Strict pass: `false`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

Which paired corpora did they use in the other experiment?

**Oracle spans**

- `dataset that contains article-comment parallel contents INLINEFORM0 , and an unpaired dataset that contains the documents (articles or comments) INLINEFORM1`
- `Chinese dataset BIBREF0`

**PaperPilot predicted excerpt**

```text
现在我对论文内容有了全面的理解。让我综合回答您的问题。

根据论文中的内容，问题中提到的"the other experiment"是指**半监督实验（semi-supervised experiment）**。在该实验中，作者使用了不同大小的 paired data（即 article-comment 平行语料）。

---

**Answer span candidates:**

1. "50K" paired data
2. "4.8M" paired data
3. "from 50K to 4.8M"

**Short answer:** In the semi-supervised experiment (Figure 1), they used paired data sizes of **50K** and **4.8M** article-comment pairs, ranging from 50K to 4.8M.

**Evidence:**

- 论文 Analysis and Discussion 部分明确指出: *"We train the supervised IR model with different numbers of paired data. … we further combine the supervised IR with our unsupervised model, which is trained with full unpaired data (4.8M) and different number of paired data (from 50K to 4.8M)."*
- Table 2 和 Table 3 的 "Semi-supervised" 部分也列得很清楚：**Proposed+S2S1** 和 **Proposed+IR1** 使用了 **50K** paired data；**Proposed+S2S2** 和 **Proposed+IR2** 使用了 **4.8M** paired data。

**解释：** 这里的 "paired corpora" 是指从腾讯新闻（Tencent News）收集的 article-comment 平行数据集（Qin et al. 2018 的数据集）。在半监督实验中，作者分别使用了 **50,000 (50K)** 条 article-comment 对 和 **4,800,000 (4.8M)** 条 article-comment 对作为有监督的 paired data，同时搭配 4.8M 条无监督的 unpaired data 来训练模型。Fig. 1 展示了随着 paired data 从 50K 增加到 4.8M 时 Recall@1 性能的变化曲线。
```

**Judge reason**

The gold answer asks 'Which paired corpora did they use in the other experiment?' and the gold evidence identifies two paired corpora: (1) a paired dataset that contains article-comment parallel contents INLINEFORM0, and an unpaired dataset that contains the documents (articles or comments) INLINEFORM1; (2) Chinese dataset BIBREF0. The predicted answer correctly identifies the paired data sizes (50K and 4.8M) used in the semi-supervised experiment and mentions the Tencent News dataset, but it does not explicitly mention the first paired corpus described in the gold evidence (i.e., the paired dataset that contains article-comment parallel contents INLINEFORM0). Thus, it captures a core part of the answer but misses one required gold corpus, making it partial.

**Manual notes**

- Answers paired-data sizes and Tencent data but does not clearly name both gold corpus descriptions.

### 6. `qasper-1907.02030-q0`

- Review decision: `keep_partial`
- Strict pass: `false`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

How is the accuracy of the system measured?

**Oracle spans**

- `F1 score of 0.71 for this task without any specific training, simply by choosing a threshold below which all sentence pairs are considered duplicates`
- `distances between duplicate and non-duplicate questions using different embedding systems`

**PaperPilot predicted excerpt**

```text
很好，我已经收集到了充足的检索结果。现在进行**Step 4: 综合回答**。

---

## Answer

**Answer span candidates:**
- `P=0.90, R=0.89, F1=0.89` (Table 2, Google USE Large claim detection)
- `60.74%` (Percentage of claims in majority clusters, Google USE Large)
- `7.02%` (Percentage of claims in clusters of one story, Google USE Large)
- `A × Pos + B × Pcc × (C × Nc)` (clustering correctness formula)

**Short answer:**
The system's accuracy is measured via **two separate evaluations**: (1) **Claim Detection** using **Precision (P), Recall (R), and F1-score** — Google USE Large achieved **P=0.90, R=0.89, F1=0.89** (Table 2); (2) **Claim Clustering** using a custom formula `A × Pos + B × Pcc × (C × Nc)` and the percentages of claims in majority clusters (Google USE Large: **60.74%**) and in clusters of one story (**7.02%**) (Table 3).

**Evidence:**
- **Table 2** (from search results): *"Table 2: Claim Detection Results. Embedding Method P R F1 / Google USE Large 0.90 0.89 0.89 / Full Fact (not on the same data) 0.88 0.80 0.83 / TFIDF (Baseline) 0.84 0.84 0.84"*
- **Table 3** (from search results): *"Table 3: Comparing Sentence Embeddings for Clustering News Claims ... Google USE Large ... Percentage of claims in majority clusters 60.74% / Percentage of claims in clusters of one story 7.02%"*
- **Custom formula** (from search results): *"we developed an equation to judge how well our corpus of sentences was clustered, rewarding clustering which matches the article clustering and the total number of claims clustered. The precise formula is given below, where Pos is the proportion of claims in
...[truncated]
```

**Judge reason**

The predicted answer correctly identifies F1 score and discusses both claim detection and clustering accuracy, which are relevant. However, the gold evidence emphasizes F1 score of 0.71 and euclidean distances between duplicate/non-duplicate questions. The predicted answer's F1=0.89 and clustering formula are not reflected in the gold evidence, making the core answer incomplete and partially aligned with the gold answer.

**Manual notes**

- Identifies F1/accuracy evaluation but uses different numbers and adds clustering formula not aligned with gold.

### 7. `qasper-1910.03042-q2`

- Review decision: `keep_partial`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

What the system designs introduced?

**Oracle spans**

- `Amazon Conversational Bot Toolkit`
- `natural language understanding (NLU) (nlu) module`
- `dialog manager`
- `knowledge bases`
- `natural language generation (NLG) (nlg) module`
- `text to speech (TTS) (tts)`

**PaperPilot predicted excerpt**

```text
现在我有足够的信息来综合回答。以下是分析结果。

---

## 综合回答

### Answer span candidates

- "multi-step language understanding modules"
- "novel dialog act scheme" / "MIDAS, a human-machine dialog act scheme with 23 tags"
- "fact/opinion interleaving strategy"
- "extensive persona database" / "Gunrock Persona Backstory database, consisting of over 1,000 responses to possible questions"
- "hierarchical dialog manager, consisting of a high level and low level DMs"
- "modular finite state transducers"
- "ASR correction according to the context" / "The Double Metaphone Search Algorithm"
- "coreference resolution on entities...across multiple turns"
- "template manager" / "response templates dynamically composed"
- "rule-based system to add interjections, specifically Alexa Speechcons, and fillers"

---

### Short answer:
**"multi-step language understanding modules", "novel dialog act scheme (MIDAS)", "fact/opinion interleaving strategy", "extensive persona database (Gunrock Persona Backstory database)", "hierarchical dialog manager (high-level + low-level DMs)", "ASR correction via Double Metaphone", "coreference resolution", "template-based NLG", and "TTS enhancement with interjections/fillers".**

---

### Evidence:

1. **Multi-step NLU + MIDAS dialog act scheme** — 论文指出: *"Gunrock's multi-step language understanding modules enable the system to provide more useful information to the dialog manager, including a novel dialog act scheme"* 以及 *"we designed MIDAS, a human-machine dialog act scheme with 23 tags and implemented a multi-label dialog act classification model using contextua
...[truncated]
```

**Judge reason**

The predicted answer lists many specific design elements, which include some of the gold systems (e.g., natural language understanding module, dialog manager, knowledge bases, natural language generation, and text-to-speech implicitly via 'TTS enhancement with interjections/fillers'). However, it omits the explicit mention of the 'Amazon Conversational Bot Toolkit' as a system design component, and it adds many unsupported details that are not part of the gold answer. The core answer covers several but not all of the gold extractive spans, making it incomplete.

**Manual notes**

- Covers several system components but misses Amazon Conversational Bot Toolkit and some exact gold components.

### 8. `qasper-1910.07181-q0`

- Review decision: `downgrade_to_incorrect`
- Strict pass: `false`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

How much is representaton improved for rare/medum frequency words compared to standalone BERT and previous work?

**Oracle spans**

- `improving the score for WNLaMPro-medium by 50% compared to BERT$_\text{base}$ and 31% compared to Attentive Mimicking`

**PaperPilot predicted excerpt**

```text
现在我有足够的材料来综合回答了。下面整理结果。

---

## Answer span candidates:
1. **Table 1 MRR values**: BERTbase RARE=0.112, MEDIUM=0.234; +BERTRAM-ADD RARE=0.269, MEDIUM=0.367; +AM RARE=0.251, MEDIUM=0.267
2. **"improving the score for WNLaMPro-MEDIUM by 58% compared to BERTbase and 37% compared to Attentive Mimicking"**
3. **"resulting in an absolute improvement of up to 25% over BERT"** (on rarified downstream tasks)
4. **Table 3 downstream accuracy**: e.g., BERTbase on DBPedia All=49.3 → +BERTRAM=64.2 → +BERTRAM-SLASH=65.7 → +BERTRAM-SLASH+INDOMAIN=74.2

## Short answer:
On WNLaMPro, BERTRAM-ADD **improves MRR** from **0.112→0.269** (RARE) and **0.234→0.367** (MEDIUM) vs. standalone BERTbase — a **58% improvement** for MEDIUM frequency words over BERTbase and **37% over Attentive Mimicking**. On rarified downstream tasks, BERTRAM yields **up to 25% absolute accuracy improvement** over BERT. For RoBERTa-large, BERTRAM-ADD on WNLaMPro improves RARE from 0.270→0.306 and MEDIUM from 0.275→0.323.

## Evidence:

### (1) WNLaMPro (Rare Word Probing) — Table 1

Table 1 shows MRR on WNLaMPro:

| Model | RARE | MEDIUM |
|-------|------|--------|
| BERT (base) | 0.112 | 0.234 |
| + AM (Schick & Schütze, 2020) | 0.251 | 0.267 |
| + BERTRAM-SHALLOW | 0.250 | 0.246 |
| + BERTRAM-REPLACE | 0.155 | 0.216 |
| **+ BERTRAM-ADD** | **0.269** | **0.367** |
| BERT (large) | 0.143 | 0.264 |
| RoBERTa (large) | 0.270 | 0.275 |
| **+ BERTRAM-ADD (RoBERTa)** | **0.306** | **0.323** |

论文原文指出：**"improving the score for WNLaMPro-MEDIUM by 58% compared to BERTbase and 37% compared to Attentive Mimicking
...[truncated]
```

**Judge reason**

The core answer correctly identifies improvement for medium-frequency words (WNLaMPro-medium), but provides incorrect percentages (58% and 37% instead of gold's 50% and 31%). The rare/medium coverage is broader than gold (includes specific MRR values and RoBERTa), but the central quantitative claim is wrong. The predicted answer also mixes in downstream accuracy numbers not required by the question. Labeling 'partial' because the key numbers are wrong, making the answer only partly correct.

**Manual notes**

- Question asks how much improvement; the central percentages differ from the gold answer.

### 9. `qasper-1911.03385-q0`

- Review decision: `keep_partial`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

How they perform manual evaluation, what is criteria?

**Oracle spans**

- `accuracy`

**PaperPilot predicted excerpt**

```text
检索结果充分覆盖了 human evaluation 的详细信息。现在进行综合回答。

---

## Answer span candidates:

- "three English Literature PhD candidates, all of whom had passed qualifying exams that included determining both genre and era of various literary texts"
- "Fluency Evaluation: score reference sentences, reconstructed sentences, and transferred sentences on a 0-5 scale, where 0 was incoherent and 5 was a well-written human sentence"
- "Human Classification: which-of-3 (simply marked which style they thought a sentence was from) and which-of-2 (given the original style and marked which style they thought the sentence was transferred into)"
- "each annotator marked 180 sentences: 90 from each model, with an even split across the three genres. Annotators were presented the sentences in a random order, without information about the models. In total, each marked 270 sentences"

## Short answer:

The paper performs **two human evaluation tasks**: (1) **Fluency Evaluation** — annotators scored reference, reconstructed, and transferred sentences on a **0–5 scale, where 0 was incoherent and 5 was a well-written human sentence**; (2) **Human Classification** — two sub-tasks: **which-of-3** (annotators simply marked which of 3 styles—gothic/sci-fi/philosophy—a sentence was from) and **which-of-2** (given the original style, annotators marked which style they thought the sentence was transferred into). **Annotators**: three English Literature PhD candidates, all passed qualifying exams on determining genre and era.

## Evidence:

From Section 5 (Human Evaluation):

- **Participants recruited**
...[truncated]
```

**Judge reason**

The predicted answer provides a detailed description of the human evaluation procedures and criteria (fluency evaluation and human classification), but the gold answer indicates that the core criterion is 'accuracy' (specifically, accuracy on classifying styles). The predicted answer does not explicitly mention 'accuracy' as the criteria for manual evaluation. While it covers broader aspects of the evaluation, it misses the essential gold-answer criterion for manual evaluation, making it only partially correct.

**Manual notes**

- Describes manual evaluation procedure, but does not state the gold criterion accuracy clearly.

### 10. `qasper-1911.03894-q1`

- Review decision: `downgrade_to_incorrect`
- Strict pass: `false`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

Which tasks does CamemBERT not improve on?

**Oracle spans**

- `its performance still lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa`

**PaperPilot predicted excerpt**

```text
我已经找到了关键证据。现在进行综合回答。

---

**Answer span candidates:**

1. "except for POS tagging on Sequoia and POS tagging on Spoken, where CamemBERT lags by 0.03% and 0.14% UPOS respectively"
2. "CamemBERT improves the state of the art for the 4 downstream tasks considered"
3. "CamemBERT reaches state-of-the-art scores on all treebanks and metrics in both scenarios"

**Short answer:** **POS tagging on Sequoia** and **POS tagging on Spoken** — where CamemBERT lags by **0.03%** and **0.14% UPOS** respectively compared to UDPipe Future + mBERT + Flair.

**Evidence:**

论文 Section 5 "POS tagging and dependency parsing" 中明确写道：

> "CamemBERT achieves overall slightly better results than the previous state-of-the-art and task-specific architecture UDPipe Future+mBERT+Flair, **except for POS tagging on Sequoia and POS tagging on Spoken, where CamemBERT lags by 0.03% and 0.14% UPOS respectively**."

论文同样明确指出，CamemBERT 在 **四个主要任务（POS tagging、dependency parsing、NER、NLI）的总体 state-of-the-art** 上都取得了改进或持平，但**在上述两个具体的细分评测集/指标上，CamemBERT 未超越之前的 SOTA（UDPipe Future + mBERT + Flair）**。

**解释：**

论文在 Table 2 中对比了多个模型在4个法语句法树库（GSD、Sequoia、Spoken、ParTUT）上的 UPOS（词性标注准确率）和 LAS（依存分析标签准确率）。CamemBERT 整体表现最佳，但在 **Sequoia 树库的 UPOS** 和 **Spoken 树库的 UPOS** 这两个指标上，其得分略低于 UDPipe Future + mBERT + Flair 模型。论文认为这可能是因为 Flair 使用的 contextualized character-level word embeddings 能更好地处理拼写错误和子词结构，而 CamemBERT 在 capitalized data 上训练，对于 Spoken（无大小写、无标点）数据的处理稍逊一筹。

因此，**CamemBERT 在"任务"层面全面超越或达到 SOTA，但在 Sequoia 和 Spoken 两个树库的 UPOS（词性标注）指标上，没有超越 UDPipe Future + mBERT + Flair 这个特定的强基线模型。**
```

**Judge reason**

The predicted answer correctly identifies specific POS tagging subtasks (on Sequoia and Spoken) where CamemBERT's performance lags relative to a previous state-of-the-art model, which is a valid interpretation of 'tasks it does not improve on.' However, the gold answer/evidence explicitly mentions only the XNLI TRANSLATE-TEST setting as a failure to improve (lagging behind models trained on original English training set). The predicted answer focuses on different tasks (POS tagging subtrees) not mentioned in the gold evidence, making it partially correct in addressing the question's intent but not matching the gold evidence's core claim. The strict scorer failed, and the gold evidence does not support the POS tagging subtask details as the intended answer.

**Manual notes**

- Gold answer is XNLI TRANSLATE-TEST lag; prediction answers different POS subtasks.

### 11. `qasper-1911.03894-q2`

- Review decision: `keep_partial`
- Strict pass: `false`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

How much better was results of CamemBERT than previous results on these tasks?

**Oracle spans**

- `2.36 point increase in the F1 score with respect to the best SEM architecture`
- `on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 for XLM)`
- `lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa`
- `For POS tagging, we observe error reductions of respectively 0.71% for GSD, 0.81% for Sequoia, 0.7% for Spoken and 0.28% for ParTUT`
- `For parsing, we observe error reductions in LAS of 2.96% for GSD, 3.33% for Sequoia, 1.70% for Spoken and 1.65% for ParTUT`

**PaperPilot predicted excerpt**

```text
现在我已经收集了足够多的检索结果，可以综合回答。

## Answer span candidates

**Task-wise improvements:**

- **POS & Dependency Parsing (Table 2):** CamemBERT (fine-tuned) vs UDify: `+4.15% LAS on Sequoia`, `+5.37 LAS on ParTUT`; CamemBERT outperforms mBERT and XLMMLM-TLM on all treebanks; overall "reaches state-of-the-art scores on all treebanks and metrics"
- **NER (Table 3):** CamemBERT fine-tuned `89.08` F1, CamemBERT embeddings `89.55` F1, vs mBERT fine-tuned `87.35` F1, vs LSTM-CRF (Dupont) `85.57` F1, vs SEM (CRF) `85.02` F1
- **NLI / XNLI (Table 4):** CamemBERT (BASE) `82.5` accuracy vs mBERT `76.9` → `+5.6%`; vs XLMMLM-TLM `80.2` → `+2.3`; vs XLM-RBASE `80.1` → `+2.4`; CamemBERTLARGE `85.7` vs XLM-RLARGE `85.2`
- **Summary:** "CamemBERT improves on the state of the art in all four tasks compared to previous monolingual and multilingual approaches including mBERT, XLM and XLM-R"

---

## Short answer:

CamemBERT achieved **state-of-the-art** results across all four tasks, with concrete improvements: **+4.15% to +5.37% LAS** over UDify in dependency parsing; **+1.73 to +4.06 F1** over mBERT/previous baselines in NER; **+5.6% accuracy** over mBERT and **+2.3–2.4%** over XLM-based models in NLI; and the LARGE variant reached **85.7%** on XNLI, **surpassing XLM-RLARGE (85.2%)** with fewer parameters.

## Evidence:

**1. POS tagging & Dependency Parsing (Table 2):**
- "fine-tuning CamemBERT outperforms UDify on all treebanks and sometimes by a large margin (e.g. `+4.15% LAS on Sequoia` and `+5.37 LAS on ParTUT`)"
- CamemBERT "also reaches better performance than other multilingual
...[truncated]
```

**Judge reason**

The predicted answer provides some correct specific improvements (e.g., +2.36 F1 for NER, comparative figures for XNLI, error reductions for POS/parsing) that align with the gold evidence. However, it omits the key point that CamemBERT lagged behind RoBERTa on the TRANSLATE-TEST setting, which is part of the answer the question expects about how much better CamemBERT was than previous results. The gold spans and evidence explicitly mention this limitation, so the answer is incomplete. Moreover, the prediction includes numbers (like +5.6% accuracy over mBERT, +2.3-2.4% over XLM) that are not present in the gold spans—these extra details are neither contradicted nor supported by gold evidence but are not the core issue. The core answer fails to cover all required aspects of performance comparison, hence partial.

**Manual notes**

- Includes several correct improvements but misses the RoBERTa TRANSLATE-TEST lag required by gold.

### 12. `qasper-1912.00871-q0`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

What neural configurations are explored?

**Oracle spans**

- `tried many configurations of our network models, but report results with only three configurations`
- `Transformer Type 1`
- `Transformer Type 2`
- `Transformer Type 3`

**PaperPilot predicted excerpt**

```text
现在我已经有了充分的信息来回答这个问题。让我综合给出答案。

---

## Answer span candidates:

- **Transformer Type 1**: "4 Transformer layers, 8 attention heads with a depth of 512 and a feed-forward depth of 1024"
- **Transformer Type 2**: "2 Transformer layers, 8 attention heads with a depth of 256 and a feed-forward depth of 1024"
- **Transformer Type 3**: "1 Transformer layer, 8 attention heads with a depth of 256 and a feed-forward depth of 512"
- Three **representation formats**: "Prefix-Transformer, Postfix-Transformer, and Infix-Transformer"
- Two **pre-training conditions**: "pre-trained" vs "non-pre-trained"

## Short answer:

The paper explores **3 Transformer architectures** (Type 1 / Type 2 / Type 3) × **3 representation formats** (Prefix / Postfix / Infix) × **2 pre-training conditions** (pre-trained on IMDb / non-pre-trained), yielding **18 neural configurations** in total.

## Evidence:

From the paper's Section III.D (Method: Training and Testing):

1. **3 Transformer types**:
   - **Transformer Type 1**: "small to medium-sized network consisting of 4 Transformer layers. Each layer utilizes 8 attention heads with a depth of 512 and a feed-forward depth of 1024."
   - **Transformer Type 2**: "small in size, using 2 Transformer layers. The layers utilize 8 attention heads with a depth of 256 and a feed-forward depth of 1024."
   - **Transformer Type 3**: "minimal, using only 1 Transformer layer. This network utilizes 8 attention heads with a depth of 256 and a feed-forward depth of 512."

2. **3 representation formats**: "Three representation models are trained and tested s
...[truncated]
```

**Judge reason**

The predicted answer correctly identifies the three Transformer architectures (Type 1, 2, 3) and their configurations, matching the gold extractive spans. However, it additionally claims a combinatorial total of 18 neural configurations by multiplying with representation formats and pre-training conditions, which is not supported by the gold evidence that only mentions 'three configurations of Transformers'. The core answer is correct, but the unsupported extra detail makes it partly correct.

**Manual notes**

- The three Transformer Type configurations are present; representation/pretraining combinations are extra context.

### 13. `qasper-2001.06286-q1`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

What is the state of the art?

**Oracle spans**

- `BERTje BIBREF8`
- `an ULMFiT model (Universal Language Model Fine-tuning for Text Classification model) BIBREF19.`
- `mBERT`

**PaperPilot predicted excerpt**

```text
现在信息收集充分了，我整理回答。

---

## Answer span candidates:

- **Sentiment Analysis (DBRD, full dataset):** RobBERT v2 **95.144%** accuracy; previous SOTA ULMFiT **93.8%**; BERTje **93.0%**; BERT-NL **84.0%** F1
- **Die/Dat disambiguation (Europarl, full, fine-tuned):** RobBERT v2 **99.232%** accuracy; previous SOTA baseline (Allein et al.) **75.03%**; mBERT **98.285%**; BERTje **98.268%**
- **Die/Dat zero-shot:** RobBERT v2 **98.75%** accuracy; BERTje **94.94%**; mBERT **90.21%**
- **POS tagging (Lassy UD):** mBERT **96.5%**; RobBERT v2 **96.4%**; BERTje **96.3%** (no statistically significant difference)
- **NER (CoNLL-2002):** Best SOTA is mBERT (Wu & Dredze) **90.94%** F1; RobBERT v2 **89.08%**; BERTje **88.3%**; BERT-NL **89.7%**
- **Key claim:** "RobBERT improves state-of-the-art results for various tasks, and especially significantly outperforms other models when dealing with smaller datasets"

---

## Short answer:
The state-of-the-art (SOTA) results in this paper refer to the previous best-performing models on **four Dutch NLP tasks**, which **RobBERT** (especially v2) improves upon in most cases: Sentiment Analysis on DBRD (previous SOTA ULMFiT **93.8%** → RobBERT v2 **95.144%**), Die/Dat disambiguation on Europarl (previous SOTA baseline **75.03%** → RobBERT v2 **99.232%**), and zero-shot Die/Dat prediction (previous best BERTje **94.94%** → RobBERT v2 **98.75%**). On POS tagging, RobBERT v2 (**96.4%**) is comparable to mBERT (**96.5%**) with no significant difference. On NER, the best SOTA is mBERT from Wu & Dredze (**90.94%** F1), which RobBERT v2 (**89.08
...[truncated]
```

**Judge reason**

The predicted answer provides detailed performance numbers and identifies the specific models (BERTje, ULMFiT, mBERT) that are the state of the art, which matches the gold extractive spans. However, the gold answer lists three examples (BERTje, ULMFiT, mBERT) without granularity, and the prediction adds extensive extra detail (e.g., zero-shot, POS tagging, NER) that is not part of the gold answer. The core claim—that the state of the art consists of these models—is correctly conveyed, but the prediction goes beyond the scope of the gold answer by listing many specific tasks and numbers, making it a partial match because the gold answer only expects the core list of models. The prediction does not contradict the gold evidence, and the extra details do not invalidate the core answer, but because the question asks 'What is the state of the art?' and the gold answer is a simpl ...[truncated]

**Manual notes**

- BERTje, ULMFiT, and mBERT are identified; task-specific scores are extra but not harmful.

### 14. `qasper-2001.09899-q1`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `partial`
- Confidence: `medium`

**Question**

What controversial topics are experimented with?

**Oracle spans**

- `political events such as elections, corruption cases or justice decisions`

**PaperPilot predicted excerpt**

```text
现在我已经获取到了完整的信息。让我综合回答。

---

## Answer span candidates:

- **Controversial topics (top group of Table 1):** `#netanyahuspeech`, `#ukraine`, `@mauriciomacri` (multiple periods), `Kavanaugh`, `Bolsonaro`, `Lula`, `Dilma`
- **Non-controversial topics (bottom group of Table 1):** `#sxsw`, `#1dfamheretostay` (OneDirection), `#germanwings`, `#mothersday`, `#nepal`, `#ultralive`, `#kingjacksonday`, `#Wrestlemania`, `Notredam`, `Nintendo`, `Halsey`, `#Bigil`, `#VanduMuruganAJITH`, `Messi`, `#Area51`
- **12-topic testbed — controversial:** `(netanyahu, ukraine, @mauriciomacri 1-11 Jan, Kavanaugh 3 Oct, @mauriciomacri 11-18 Mar, Bolsonaro 27 Oct)`
- **12-topic testbed — non-controversial:** `(sxsw, germanwings, onedirection, ultralive, nepal, mothersday)`

## Short answer:
The controversial topics experimented with are: **#netanyahuspeech** (Netanyahu at U.S. Congress), **#ukraine** (Ukraine conflict), **@mauriciomacri** (multiple periods, mentions to Argentine president), **Kavanaugh** (nomination to US Supreme Court), **Bolsonaro** (Brazilian presidential elections), **Lula** (mentions to Lula on the day of Moro chats news), and **Dilma** (Dilma Rousseff impeachment).

## Evidence:
From **Table 1 in Appendix A** (retrieved via colbert search):

> **Controversial topics:** `#netanyahuspeech` (124,704 tweets, Netanyahu at U.S. Congress, Mar 3-5, 2015), `#ukraine` (145,794 tweets, Ukraine conflict, Feb 27-Mar 2, 2015), `@mauriciomacri` (multiple intervals: Jan 11, Mar 11-18, Mar 20-27, Apr 05-11, May 05-11, 2018 — mentions to Argentine president), `Kavanaugh` (three tim
...[truncated]
```

**Judge reason**

The predicted answer lists specific hashtags and mentions (e.g., #netanyahuspeech, #ukraine, @mauriciomacri, Kavanaugh, Bolsonaro, Lula, Dilma) which are more specific instances of the gold answer's general category 'political events such as elections, corruption cases or justice decisions'. The gold answer defines the core as a general category, and the prediction provides specific examples from the same domain. However, the gold evidence explicitly lists only examples like 'elections, corruption cases, or justice decisions', and the prediction's specific hashtags (e.g., #netanyahuspeech, #ukraine, Lula, Dilma) go beyond what is directly stated in the gold answer. While extra specific examples do not contradict the general gold definition, the gold answer's core is a general statement, and the prediction's specific list is more granular but semantically consistent. Howeve ...[truncated]

**Manual notes**

- Specific controversial hashtags are valid instances of the political-event gold category.


## strict_pass_semantic_bad

### 1. `qasper-1804.10686-q1`

- Review decision: `downgrade_to_incorrect`
- Strict pass: `true`
- Semantic label: `incorrect`
- Confidence: `medium`

**Question**

Which corpus of synsets are used?

**Oracle spans**

- `Wiktionary`

**PaperPilot predicted excerpt**

```text
信息已经非常充分了。现在进入**Step 4: 综合回答**。

---

**Answer span candidates:**
1. "WATLINK, a word sense network constructed automatically. It uses the synsets induced in an unsupervised way by the WATSET [CWnolog, MCL] method"
2. "RuThes, a large-scale lexical ontology for Russian created by a group of expert lexicographers"
3. "RuWordNet, a semi-automatic conversion of the RuThes lexical ontology into a WordNet-like structure"

**Short answer:** Three synset corpora (sense inventories) were used: **WATLINK**, **RuThes**, and **RuWordNet**.

**Evidence:** Section 4.2 (Dataset) of the paper explicitly states:

> *"The following different sense inventories have been used during the evaluation:*
> - *WATLINK, a word sense network constructed automatically. It uses the synsets induced in an unsupervised way by the WATSET [CWnolog, MCL] method (Ustalov et al., 2017) and the semantic relations from such dictionaries as Wiktionary referred as Joint+Exp+SWN in Ustalov (2017). This is the only automatically built inventory we use in the evaluation.*
> - *RuThes, a large-scale lexical ontology for Russian created by a group of expert lexicographers (Loukachevitch, 2011).*
> - *RuWordNet, a semi-automatic conversion of the RuThes lexical ontology into a WordNet-like structure (Loukachevitch et al., 2016)."*

**解释：** 这篇论文对 Watasense 系统进行了评估，使用了三种不同的 sense inventory（即 synset 语料库）：

1. **WATLINK** — 自动构建的词义网络，基于 WATSET 方法从同义词图中无监督地归纳出 synsets，并结合了 Wiktionary 等词典的语义关系，是三种中唯一自动构建的。
2. **RuThes** — 由专家词典编纂者手工创建的大规模俄语词汇本体。
3. **RuWordNet** — 将 RuThes 半自动转换为 WordNet 风格结构的产物。

此外，系统还用到了 **R
...[truncated]
```

**Judge reason**

The gold answer states only 'Wiktionary' as the corpus of synsets used, but the predicted answer lists three different sense inventories (WATLINK, RuThes, RuWordNet) without mentioning Wiktionary. Therefore the core claim is factually wrong regarding which specific corpus is used.

**Manual notes**

- Gold answer is Wiktionary; prediction answers WATLINK/RuThes/RuWordNet instead.

### 2. `qasper-1902.06843-q2`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `contradictory`
- Confidence: `low`

**Question**

What types of features are used from each data type?

**Oracle spans**

- `facial presence`
- `Facial Expression`
- `General Image Features`
- `textual content`
- `analytical thinking`
- `clout`
- `authenticity`
- `emotional tone`
- `Sixltr`
- `informal language markers`
- `1st person singular pronouns`

**PaperPilot predicted excerpt**

```text
所有证据已经足够充分。现在综合回答。

---

## Answer span candidates:
- "aesthetic features from posted images (colorfulness, hue variance, sharpness, brightness, blurriness, naturalness)"
- "choice of profile picture (for gender, age, and facial expression)"
- "the screen name"
- "language features from both textual content and profile's description (n-gram, emotion, sentiment)"
- "sociability from ego-network, and user engagement"
- "facial presence, emotion from facial expression, and demographic features from profile/posted images"
- "Reciprocity, Prestige Ratio, Graph Density, Clustering Coefficient, Embeddedness, Ego components and Size of two-hop neighborhood"
- "LIWC" / "first person pronouns" / "negative emotions and anger words"
- "normalized red, green, blue and the mean of original colors, and brightness and contrast"
- "mean and variance for saturation and hue"
- "Colorfulness" / "Naturalness"
- "friends_count, followers_cnt, status_cnt, avg_fav_cnt, avg_retw_cnt, favourites_cnt"

---

**Short answer:** The paper uses three data modalities: **(1) Visual features** (aesthetic features from posted images: colorfulness, hue variance, sharpness, brightness, blurriness, naturalness; profile picture choice for gender, age, and facial expression; facial presence and facial expression based on Ekman's six emotions; OCR-extracted text sentiment from images); **(2) Textual features** (n-gram, emotion, sentiment, LIWC psycholinguistic cues, language style markers — analytical thinking, clout, authenticity, self-references [first person pronouns], informal language [swear, ne
...[truncated]
```

**Judge reason**

The predicted answer lists features from the paper but does not match the gold answer's required structure of listing what types of features are used from each data type. The gold extractive spans and evidence specify features like facial presence, facial expression, general image features, textual content, and specific LIWC language variables. The prediction includes unrelated features (e.g., friends_count, followers_cnt, sociability from ego-network) that are not in the gold answer or evidence for this question. This introduces contradictions by suggesting these are the features used from each data type when they are not part of the gold answer.

**Manual notes**

- Prediction covers the visual/textual gold feature types; low-confidence contradictory judge label looks too harsh.

### 3. `qasper-1910.04601-q1`

- Review decision: `downgrade_to_incorrect`
- Strict pass: `true`
- Semantic label: `incorrect`
- Confidence: `high`

**Question**

What dataset was used in the experiment?

**Oracle spans**

- `WikiHop`

**PaperPilot predicted excerpt**

```text
检索结果非常明确。现在综合回答。

**Step 4: 综合回答**

Answer span candidates:
- **HotpotQA** (Yang et al., 2018)
- **4,588 questions** from HotpotQA

**Short answer:** The R4C dataset was built from **HotpotQA** (Yang et al., 2018).

**Evidence:** Section 3.3 ("Dataset") explicitly states: *"There are a large number of choices of RC datasets that meet the criteria described in §3.1 including SQuAD (Rajpurkar et al., 2016) and WikiHop (Welbl et al., 2018). Our study uses HotpotQA (Yang et al., 2018), one of the most actively used multi-hop QA datasets."* And Section 4.2 states: *"The final R4C dataset includes 4,588 questions from HotpotQA (see Table 1), each of which is annotated with 3 reference derivations (i.e. 13,764 derivations)."*

**解释：** 论文作者从 HotpotQA 数据集中采样了 3,000 个训练实例和 3,000 个开发实例，经过质量筛选（排除错误答案和"neither"答案的提交，仅保留恰好有 3 条推导标注的实例）后，最终得到包含 **4,588 个问题**的 R4C 数据集（训练集 2,379 个 + 开发集 2,209 个），每个问题配有 3 条参考推导（共计 13,764 条推导）。因此，实验所用的基础数据集是 **HotpotQA**。
```

**Judge reason**

The gold answer and evidence explicitly state that the experiment used WikiHop. The predicted answer claims HotpotQA was used, which directly contradicts the gold answer and evidence; thus it is incorrect.

**Manual notes**

- Gold says WikiHop; prediction says HotpotQA.

### 4. `qasper-2001.06286-q2`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `incorrect`
- Confidence: `high`

**Question**

What language tasks did they experiment on?

**Oracle spans**

- `sentiment analysis`
- `the disambiguation of demonstrative pronouns,`

**PaperPilot predicted excerpt**

```text
现在已经收集到了充分的证据。让我综合回答。

---

## Answer

**Answer span candidates:**
- `sentiment analysis`, `demonstrative and relative pronoun prediction` (i.e., die/dat disambiguation), `part-of-speech (POS) tagging`, `named entity recognition (NER)`
- `Dutch Book Reviews Dataset (DBRD)`, `Europarl corpus`, `Lassy UD (Universal Dependencies)`, `CoNLL-2002 dataset`
- `"For testing text classification, we evaluate on sentiment analysis and on demonstrative and relative pronoun prediction"`, `"For testing RobBERT's token tagging capabilities, we used both part-of-speech (POS) tagging and named entity recognition (NER) tasks"`

**Short answer:** They experimented on **4 language tasks**: **Sentiment Analysis** (on Dutch Book Reviews Dataset / DBRD), **Die/Dat Disambiguation** (on Europarl corpus), **Part-of-Speech (POS) Tagging** (on Lassy UD dataset), and **Named Entity Recognition (NER)** (on CoNLL-2002 dataset).

**Evidence:**

Section 4 (Evaluation) of the paper states:

> *"For testing text classification, we evaluate on sentiment analysis and on demonstrative and relative pronoun prediction. [...] For testing RobBERT's token tagging capabilities, we used both part-of-speech (POS) tagging and named entity recognition (NER) tasks."*

The four tasks and their datasets are detailed in subsections 4.1–4.4:

| Task | Type | Dataset |
|------|------|---------|
| **Sentiment Analysis** (§4.1) | Text classification | Dutch Book Reviews Dataset (**DBRD**) — 22,252 labeled positive/negative reviews |
| **Die/Dat Disambiguation** (§4.2) | Text classification / zero-shot coreference |
...[truncated]
```

**Judge reason**

The predicted answer lists sentiment analysis, disambiguation of demonstrative pronouns, part-of-speech tagging, and named entity recognition. However, the gold answer only includes sentiment analysis and disambiguation of demonstrative pronouns. The gold evidence explicitly mentions only these two tasks as the ones they experimented on. The predicted answer thus adds two tasks not present in the gold answer and evidence, making it incorrect.

**Manual notes**

- Includes the two gold tasks and adds POS/NER tasks that are supported broader evaluation tasks.

### 5. `qasper-2001.09899-q0`

- Review decision: `keep_partial`
- Strict pass: `true`
- Semantic label: `incorrect`
- Confidence: `high`

**Question**

What are the state of the art measures?

**Oracle spans**

- `Randomwalk`
- `Walktrap`
- `Louvain clustering`

**PaperPilot predicted excerpt**

```text
现在我有足够信息来回答问题了。

**Answer span candidates:**

- "Garimella et al. [17] ... Randomwalk"
- "Matakos et al. [28] develop a polarization index"
- "state-of-the-art structure-based measures"
- "best method Randomwalk"
- "AUC ROC ... 0.935"

---

**Short answer:** The state of the art measures are **structure-based measures**, specifically **Garimella et al.'s Randomwalk** (a measure based on random walks over the endorsement graph) and **Matakos et al.'s polarization index** (which captures opinions concentration in network communities).

**Evidence:**

From the Related Work section (Section 2):
> *"The main characteristic of those works is that the measures proposed are based on the structural characteristics of the underlying graph. Among them, we highlight the work of Garimella et al. [17] that presents an extensive comparison of controversy measures, different graph-building approaches, and data sources, achieving the best performance of all. In their research they propose different metrics to measure polarization on Twitter. Their techniques based on the structure of the endorsement graph can successfully detect whether a discussion is controversial or not... Matakos et al. [28] develop a polarization index. Their measure captures the tendency of opinions to concentrate in network communities, creating echo-chambers."*

From the Results section (Section 4.3):
> *"As Garimella et al. [17] have made their code public, we reproduced their best method Randomwalk on our datasets and measured the AUC ROC, obtaining a score of 0.935."*

From Abstract:
> *"We find th
...[truncated]
```

**Judge reason**

The gold answer specifies three extractive spans: Randomwalk, Walktrap, Louvain clustering. The predicted answer mentions only Randomwalk, incorrectly adds Matakos et al.'s polarization index, and omits Walktrap and Louvain clustering entirely. Therefore it does not directly and completely answer the question.

**Manual notes**

- Mentions Randomwalk but omits Walktrap and Louvain clustering, so it is incomplete rather than fully correct.

### 6. `qasper-2004.03685-q1`

- Review decision: `accept_as_correct`
- Strict pass: `true`
- Semantic label: `incorrect`
- Confidence: `high`

**Question**

What faithfulness criteria does they propose?

**Oracle spans**

- `Across models and tasks: The degree (as grayscale) of faithfulness at the level of specific models and tasks.`
- `Across input space: The degree of faithfulness at the level of subspaces of the input space, such as neighborhoods of similar inputs, or singular inputs themselves.`

**PaperPilot predicted excerpt**

```text
Now I have all the information needed. Let me synthesize the answer.

---

## Answer span candidates:

1. "Assumption 1 (The Model Assumption). Two models will make the same predictions if and only if they use the same reasoning process."
2. "Assumption 2 (The Prediction Assumption). On similar inputs, the model makes similar decisions if and only if its reasoning is similar."
3. "Assumption 3 (The Linearity Assumption). Certain parts of the input are more important to the model reasoning than others. Moreover, the contributions of different parts of the input are independent from each other."
4. "plausibility and faithfulness" — "two particularly notable criteria"
5. "a graded criteria that measures the extent and likelihood of an interpretation to be faithful"
6. "the degree (as grayscale) of faithfulness at the level of specific models and tasks"
7. "the degree of faithfulness at the level of subspaces of the input space"

---

**Short answer:** The paper proposes **three assumptions** that encapsulate the current working definitions of faithfulness used by the community: **the Model Assumption**, **the Prediction Assumption**, and **the Linearity Assumption**. Additionally, it proposes replacing the binary notion of faithfulness with a **graded (grayscale) criteria** across two dimensions: (1) **across models and tasks**, and (2) **across input space**.

**Evidence:** From §6 ("Defining Faithfulness"), the paper states: "These assumptions, to our knowledge, encapsulate the current working definitions of faithfulness used by the research community." It the
...[truncated]
```

**Judge reason**

The gold answer defines two faithfulness criteria: 'across models and tasks' and 'across input space'. The predicted answer describes three assumptions (Model, Prediction, Linearity) and mentions 'plausibility and faithfulness', which are not the proposed criteria. The core claim does not match the gold answer's two criteria.

**Manual notes**

- The two gold faithfulness criteria are included; extra assumptions are background rather than a core contradiction.
