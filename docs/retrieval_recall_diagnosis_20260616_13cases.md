# Retrieval Recall Diagnosis

Date: 2026-06-16

Enriched source: `data\eval\qasper_subset_enriched.jsonl`

This report checks whether representative QASPER gold/oracle evidence is present in the eval full text, recalled by current PaperPilot traces, and retrievable with gold-oriented diagnostic queries.

Gold-oriented probe queries are diagnostic only. They must not be used by runtime PaperPilot.

The probe builds a separate diagnostic ColBERT index from QASPER enriched `full_text`, keyed by a content hash, so it does not reuse the runtime arXiv/PaperPilot index.

## Summary

| case_id | full text has gold | current trace recalls gold | probe retrieves gold | interpretation |
|---|---:|---:|---:|---|
| `qasper-1701.00185-q1` | yes | yes | no | `gold_evidence_recalled_in_current_trace` |
| `qasper-1808.05902-q0` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1808.05902-q1` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1809.04960-q0` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1907.02030-q0` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1910.03042-q2` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1910.07181-q0` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1911.03385-q0` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1911.03894-q1` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1911.03894-q2` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1804.10686-q1` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |
| `qasper-1910.04601-q1` | yes | no | yes | `oracle_span_mentioned_but_gold_evidence_missing_in_current_trace` |
| `qasper-2001.09899-q0` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |

## qasper-1701.00185-q1

- Paper: `1701.00185` - Self-Taught Convolutional Neural Networks for Short Text Clustering
- Question: Which popular clustering methods did they experiment with?
- Oracle spans: `K-means, Skip-thought Vectors, Recursive Neural Network and Paragraph Vector based clustering methods`
- Current trace: `data\traces\qasper-1701.00185-q1.jsonl`
- Current trace chunk count: 15
- Interpretation: `gold_evidence_recalled_in_current_trace`

### Targets

- `oracle_span`: K-means, Skip-thought Vectors, Recursive Neural Network and Paragraph Vector based clustering methods
- `highlighted_evidence`: In our experiment, some widely used text clustering methods are compared with our approach. Besides K-means, Skip-thought Vectors, Recursive Neural Network and Paragraph Vector based clustering methods, fo ...[truncated]
- `evidence_head`: In our experiment, some widely used text clustering methods are compared with our approach. Besides K-means, Skip-thought Vectors, Recursive Neural Network and Paragraph Vector based clustering methods, fo ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- `popular clustering methods compared with STC2`
- `baseline clustering methods K-means Skip-thought Recursive Neural Network Paragraph Vector`
- `experiment setup baseline clustering methods compared in this paper`

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `loose` |
| `highlighted_evidence` | yes | `loose` |
| `evidence_head` | yes | `loose` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `Which popular clustering methods did they experiment with?` | no | yes | 3 | 19.659 | . besides k - means, skip - thought vectors, recursive neural network and paragraph vector based clustering methods, four baseline clustering methods are directly based on the popular unsupervised dimensio ...[truncated] |
| `K-means, Skip-thought Vectors, Recursive Neural Network and Paragraph Vector based clustering methods` | no | yes | 3 | 21.886 | of computer terminology, and symbols and capital letters are meaningful, thus we do not do any pre - processed procedures. for biomedical, we remove the symbols and convert letters into lower case. # # pre ...[truncated] |
| `Which popular clustering methods did they experiment with? K-means, Skip-thought Vectors, Recursive Neural Network ...[truncated]` | no | yes | 2 | 22.403 | of computer terminology, and symbols and capital letters are meaningful, thus we do not do any pre - processed procedures. for biomedical, we remove the symbols and convert letters into lower case. # # pre ...[truncated] |
| `In our experiment, some widely used text clustering methods are compared with our approach. Besides K-means, Skip-t ...[truncated]` | no | yes | 2 | 22.703 | . besides k - means, skip - thought vectors, recursive neural network and paragraph vector based clustering methods, four baseline clustering methods are directly based on the popular unsupervised dimensio ...[truncated] |

## qasper-1808.05902-q0

- Paper: `1808.05902` - Learning Supervised Topic Models for Classification and Regression from Crowds
- Question: what are the advantages of the proposed model?
- Oracle spans: `he proposed model outperforms all the baselines, being the svi version the one that performs best.`, `the svi version converges much faster to higher values of the log marginal likelihood when compared to the batch version, which reflects the efficiency of the svi algorithm.`
- Current trace: `data\traces\qasper-1808.05902-q0.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: he proposed model outperforms all the baselines, being the svi version the one that performs best.
- `oracle_span`: the svi version converges much faster to higher values of the log marginal likelihood when compared to the batch version, which reflects the efficiency of the svi algorithm.
- `highlighted_evidence`: The results are shown in Fig. FIGREF87 for different numbers of topics, where we can see that the proposed model outperforms all the baselines, being the svi version the one that performs best.
- `highlighted_evidence`: In order to assess the computational advantages of the stochastic variational inference (svi) over the batch algorithm, the log marginal likelihood (or log evidence) was plotted against the number of itera ...[truncated]
- `evidence_head`: For all the experiments the hyper-parameters INLINEFORM0 , INLINEFORM1 and INLINEFORM2 were set using a simple grid search in the collection INLINEFORM3 . The same approach was used to optimize the hyper-p ...[truncated]
- `evidence_head`: In order to assess the computational advantages of the stochastic variational inference (svi) over the batch algorithm, the log marginal likelihood (or log evidence) was plotted against the number of itera ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `what are the advantages of the proposed model?` | yes | no | 1 | 22.625 | can see that the proposed model outperforms all the baselines, being the svi version the one that performs best. in order to assess the computational advantages of the stochastic variational inference ( sv ...[truncated] |
| `he proposed model outperforms all the baselines, being the svi version the one that performs best.` | yes | no | 1 | 24.687 | can see that the proposed model outperforms all the baselines, being the svi version the one that performs best. in order to assess the computational advantages of the stochastic variational inference ( sv ...[truncated] |
| `the svi version converges much faster to higher values of the log marginal likelihood when compared to the batch ve ...[truncated]` | yes | no | 1 | 24.378 | can see that the proposed model outperforms all the baselines, being the svi version the one that performs best. in order to assess the computational advantages of the stochastic variational inference ( sv ...[truncated] |
| `what are the advantages of the proposed model? he proposed model outperforms all the baselines, being the svi versi ...[truncated]` | yes | no | 1 | 24.29 | can see that the proposed model outperforms all the baselines, being the svi version the one that performs best. in order to assess the computational advantages of the stochastic variational inference ( sv ...[truncated] |
| `The results are shown in Fig. FIGREF87 for different numbers of topics, where we can see that the proposed model ou ...[truncated]` | yes | no | 2 | 23.332 | of bibref0 was used. it consists of applying lda to extract the documents ' topics distributions, which are then used to train a svm. similarly to the previous approach, the labels from the different annot ...[truncated] |

## qasper-1808.05902-q1

- Paper: `1808.05902` - Learning Supervised Topic Models for Classification and Regression from Crowds
- Question: what are the state of the art approaches?
- Oracle spans: `Bosch 2006 (mv)`, `LDA + LogReg (mv)`, `LDA + Raykar`, `LDA + Rodrigues`, `Blei 2003 (mv)`, `sLDA (mv)`
- Current trace: `data\traces\qasper-1808.05902-q1.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: Bosch 2006 (mv)
- `oracle_span`: LDA + LogReg (mv)
- `oracle_span`: LDA + Raykar
- `oracle_span`: LDA + Rodrigues
- `oracle_span`: Blei 2003 (mv)
- `oracle_span`: sLDA (mv)
- `highlighted_evidence`: With the purpose of comparing the proposed model with a popular state-of-the-art approach for image classification, for the LabelMe dataset, the following baseline was introduced: Bosch 2006 (mv): This bas ...[truncated]
- `highlighted_evidence`: Both the batch and the stochastic variational inference (svi) versions of the proposed model (MA-sLDAc) are compared with the following baselines: [itemsep=0.02cm] LDA + LogReg (mv): This baseline correspo ...[truncated]
- `evidence_head`: With the purpose of comparing the proposed model with a popular state-of-the-art approach for image classification, for the LabelMe dataset, the following baseline was introduced:
- `evidence_head`: Bosch 2006 (mv): This baseline is similar to one in BIBREF33 . The authors propose the use of pLSA to extract the latent topics, and the use of k-nearest neighbor (kNN) classifier using the documents' topi ...[truncated]
- `evidence_head`: The results obtained by the different approaches for the LabelMe data are shown in Fig. FIGREF94 , where the svi version is using mini-batches of 200 documents.
- `evidence_head`: Analyzing the results for the Reuters-21578 and LabelMe data, we can observe that MA-sLDAc outperforms all the baselines, with slightly better accuracies for the batch version, especially in the Reuters da ...[truncated]
- `evidence_head`: Both the batch and the stochastic variational inference (svi) versions of the proposed model (MA-sLDAc) are compared with the following baselines:
- `evidence_head`: [itemsep=0.02cm]
- `evidence_head`: LDA + LogReg (mv): This baseline corresponds to applying unsupervised LDA to the data, and learning a logistic regression classifier on the inferred topics distributions of the documents. The labels from t ...[truncated]
- `evidence_head`: LDA + Raykar: For this baseline, the model of BIBREF21 was applied using the documents' topic distributions inferred by LDA as features.
- `evidence_head`: LDA + Rodrigues: This baseline is similar to the previous one, but uses the model of BIBREF9 instead.
- `evidence_head`: Blei 2003 (mv): The idea of this baseline is to replicate a popular state-of-the-art approach for document classification. Hence, the approach of BIBREF0 was used. It consists of applying LDA to extract th ...[truncated]
- `evidence_head`: sLDA (mv): This corresponds to using the classification version of sLDA BIBREF2 with the labels obtained by performing majority voting (mv) on the annotators' answers.

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `what are the state of the art approaches?` | yes | no | 2 | 19.577 | , the worker accuracies are much higher and their distribution is much more concentrated than on the reuters - 21578 data ( see fig. figref90 ), which suggests that this is an easier task for the amt worke ...[truncated] |
| `Bosch 2006 (mv)` | yes | no | 1 | 20.349 | , the worker accuracies are much higher and their distribution is much more concentrated than on the reuters - 21578 data ( see fig. figref90 ), which suggests that this is an easier task for the amt worke ...[truncated] |
| `LDA + LogReg (mv)` | yes | no | 1 | 23.03 | [ itemsep = 0. 02cm ] lda + logreg ( mv ) : this baseline corresponds to applying unsupervised lda to the data, and learning a logistic regression classifier on the inferred topics distributions of the doc ...[truncated] |
| `LDA + Raykar` | yes | no | 1 | 19.221 | [ itemsep = 0. 02cm ] lda + logreg ( mv ) : this baseline corresponds to applying unsupervised lda to the data, and learning a logistic regression classifier on the inferred topics distributions of the doc ...[truncated] |
| `LDA + Rodrigues` | yes | no | 1 | 18.311 | [ itemsep = 0. 02cm ] lda + logreg ( mv ) : this baseline corresponds to applying unsupervised lda to the data, and learning a logistic regression classifier on the inferred topics distributions of the doc ...[truncated] |
| `Blei 2003 (mv)` | yes | no | 1 | 18.483 | [ itemsep = 0. 02cm ] lda + logreg ( mv ) : this baseline corresponds to applying unsupervised lda to the data, and learning a logistic regression classifier on the inferred topics distributions of the doc ...[truncated] |
| `sLDA (mv)` | yes | no | 1 | 20.151 | of bibref0 was used. it consists of applying lda to extract the documents ' topics distributions, which are then used to train a svm. similarly to the previous approach, the labels from the different annot ...[truncated] |
| `what are the state of the art approaches? Bosch 2006 (mv)` | yes | no | 1 | 21.653 | , the worker accuracies are much higher and their distribution is much more concentrated than on the reuters - 21578 data ( see fig. figref90 ), which suggests that this is an easier task for the amt worke ...[truncated] |
| `With the purpose of comparing the proposed model with a popular state-of-the-art approach for image classification, ...[truncated]` | yes | no | 1 | 23.868 | , the worker accuracies are much higher and their distribution is much more concentrated than on the reuters - 21578 data ( see fig. figref90 ), which suggests that this is an easier task for the amt worke ...[truncated] |

## qasper-1809.04960-q0

- Paper: `1809.04960` - Unsupervised Machine Commenting with Neural Variational Topic Model
- Question: Which paired corpora did they use in the other experiment?
- Oracle spans: `dataset that contains article-comment parallel contents INLINEFORM0 , and an unpaired dataset that contains the documents (articles or comments) INLINEFORM1`, `Chinese dataset BIBREF0`
- Current trace: `data\traces\qasper-1809.04960-q0.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: dataset that contains article-comment parallel contents INLINEFORM0 , and an unpaired dataset that contains the documents (articles or comments) INLINEFORM1
- `oracle_span`: Chinese dataset BIBREF0
- `highlighted_evidence`: In this scenario we have a paired dataset that contains article-comment parallel contents INLINEFORM0 , and an unpaired dataset that contains the documents (articles or comments) INLINEFORM1 . The supervis ...[truncated]
- `evidence_head`: In addition to the unsupervised training, we explore a semi-supervised training framework to combine the proposed unsupervised model and the supervised model. In this scenario we have a paired dataset that ...[truncated]
- `highlighted_evidence`: We select a large-scale Chinese dataset BIBREF0 with millions of real comments and a human-annotated test set to evaluate our model.
- `highlighted_evidence`: We further combine the supervised IR with our unsupervised model, which is trained with full unpaired data (4.8M) and different number of paired data (from 50K to 4.8M).
- `evidence_head`: We select a large-scale Chinese dataset BIBREF0 with millions of real comments and a human-annotated test set to evaluate our model. The dataset is collected from Tencent News, which is one of the most pop ...[truncated]
- `evidence_head`: We analyze the performance of the proposed method under the semi-supervised setting. We train the supervised IR model with different numbers of paired data. Figure FIGREF39 shows the curve (blue) of the re ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `Which paired corpora did they use in the other experiment?` | yes | no | 3 | 10.652 | the seq2seq model. with the help of our proposed model, both ir and s2s achieve an improvement under the semi - supervised scenarios. # # analysis and discussion we analyze the performance of the proposed ...[truncated] |
| `dataset that contains article-comment parallel contents INLINEFORM0 , and an unpaired dataset that contains the doc ...[truncated]` | yes | no | 1 | 23.885 | ##onstruct the input inlineform4. at the training stage, we train the neural variational topic model with the eq. eqref22. at the testing stage, we use inlineform0 to compute the topic representations of t ...[truncated] |
| `Chinese dataset BIBREF0` | yes | no | 1 | 24.26 | the loss function of the supervised learning ( e. g. the cross - entropy loss of seq2seq model ), and inlineform2 is a hyper - parameter to balance two parts of the loss function. hence, the model is train ...[truncated] |
| `Which paired corpora did they use in the other experiment? dataset that contains article-comment parallel contents ...[truncated]` | yes | no | 1 | 18.883 | ##onstruct the input inlineform4. at the training stage, we train the neural variational topic model with the eq. eqref22. at the testing stage, we use inlineform0 to compute the topic representations of t ...[truncated] |
| `In this scenario we have a paired dataset that contains article-comment parallel contents INLINEFORM0 , and an unpa ...[truncated]` | yes | no | 1 | 23.75 | ##onstruct the input inlineform4. at the training stage, we train the neural variational topic model with the eq. eqref22. at the testing stage, we use inlineform0 to compute the topic representations of t ...[truncated] |
| `We select a large-scale Chinese dataset BIBREF0 with millions of real comments and a human-annotated test set to ev ...[truncated]` | yes | no | 1 | 24.443 | the loss function of the supervised learning ( e. g. the cross - entropy loss of seq2seq model ), and inlineform2 is a hyper - parameter to balance two parts of the loss function. hence, the model is train ...[truncated] |

## qasper-1907.02030-q0

- Paper: `1907.02030` - Real-time Claim Detection from News Articles and Retrieval of Semantically-Similar Factchecks
- Question: How is the accuracy of the system measured?
- Oracle spans: `F1 score of 0.71 for this task without any specific training, simply by choosing a threshold below which all sentence pairs are considered duplicates`, `distances between duplicate and non-duplicate questions using different embedding systems`
- Current trace: `data\traces\qasper-1907.02030-q0.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: F1 score of 0.71 for this task without any specific training, simply by choosing a threshold below which all sentence pairs are considered duplicates
- `oracle_span`: distances between duplicate and non-duplicate questions using different embedding systems
- `highlighted_evidence`: The graphs in figure 1 show the distances between duplicate and non-duplicate questions using different embedding systems.
- `highlighted_evidence`: Large achieved a F1 score of 0.71 for this task without any specific training, simply by choosing a threshold below which all sentence pairs are considered duplicates.
- `highlighted_evidence`: In order to test whether these results generalised to our domain, we devised a test that would make use of what little data we had to evaluate.
- `evidence_head`: The graphs in figure 1 show the distances between duplicate and non-duplicate questions using different embedding systems. The X axis shows the euclidean distance between vectors and the Y axis frequency. ...[truncated]
- `evidence_head`: In order to test whether these results generalised to our domain, we devised a test that would make use of what little data we had to evaluate. We had no original data on whether sentences were semanticall ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `How is the accuracy of the system measured?` | yes | no | 2 | 11.358 | ##bref22 as the best match. to study the embeddings, we computed the euclidean distance between the two questions using various embeddings, to study the distance between semantically similar and dissimilar ...[truncated] |
| `F1 score of 0.71 for this task without any specific training, simply by choosing a threshold below which all senten ...[truncated]` | yes | no | 1 | 23.198 | ##bref22 as the best match. to study the embeddings, we computed the euclidean distance between the two questions using various embeddings, to study the distance between semantically similar and dissimilar ...[truncated] |
| `distances between duplicate and non-duplicate questions using different embedding systems` | yes | no | 1 | 25.896 | ##bref22 as the best match. to study the embeddings, we computed the euclidean distance between the two questions using various embeddings, to study the distance between semantically similar and dissimilar ...[truncated] |
| `How is the accuracy of the system measured? F1 score of 0.71 for this task without any specific training, simply by ...[truncated]` | yes | no | 1 | 19.13 | ##bref22 as the best match. to study the embeddings, we computed the euclidean distance between the two questions using various embeddings, to study the distance between semantically similar and dissimilar ...[truncated] |
| `The graphs in figure 1 show the distances between duplicate and non-duplicate questions using different embedding systems.` | yes | no | 1 | 26.882 | ##bref22 as the best match. to study the embeddings, we computed the euclidean distance between the two questions using various embeddings, to study the distance between semantically similar and dissimilar ...[truncated] |

## qasper-1910.03042-q2

- Paper: `1910.03042` - Gunrock: A Social Bot for Complex and Engaging Long Conversations
- Question: What the system designs introduced?
- Oracle spans: `Amazon Conversational Bot Toolkit`, `natural language understanding (NLU) (nlu) module`, `dialog manager`, `knowledge bases`, `natural language generation (NLG) (nlg) module`, `text to speech (TTS) (tts)`
- Current trace: `data\traces\qasper-1910.03042-q2.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: Amazon Conversational Bot Toolkit
- `oracle_span`: natural language understanding (NLU) (nlu) module
- `oracle_span`: dialog manager
- `oracle_span`: knowledge bases
- `oracle_span`: natural language generation (NLG) (nlg) module
- `oracle_span`: text to speech (TTS) (tts)
- `highlighted_evidence`: We extend the Amazon Conversational Bot Toolkit (CoBot) BIBREF6 which is a flexible event-driven framework. CoBot provides ASR results and natural language processing pipelines through the Alexa Skills Kit ...[truncated]
- `evidence_head`: Figure FIGREF3 provides an overview of Gunrock's architecture. We extend the Amazon Conversational Bot Toolkit (CoBot) BIBREF6 which is a flexible event-driven framework. CoBot provides ASR results and nat ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `What the system designs introduced?` | yes | no | 6 | 12.22 | to provide coherent profile information, a critical challenge in building social chatbots bibref3. compared to previous systems bibref4, gunrock generates more balanced conversations between human and mach ...[truncated] |
| `Amazon Conversational Bot Toolkit` | yes | no | 1 | 21.818 | to provide coherent profile information, a critical challenge in building social chatbots bibref3. compared to previous systems bibref4, gunrock generates more balanced conversations between human and mach ...[truncated] |
| `natural language understanding (NLU) (nlu) module` | yes | no | 1 | 21.485 | to provide coherent profile information, a critical challenge in building social chatbots bibref3. compared to previous systems bibref4, gunrock generates more balanced conversations between human and mach ...[truncated] |
| `dialog manager` | yes | no | 2 | 18.796 | to provide coherent profile information, a critical challenge in building social chatbots bibref3. compared to previous systems bibref4, gunrock generates more balanced conversations between human and mach ...[truncated] |
| `knowledge bases` | yes | no | 2 | 15.55 | to provide coherent profile information, a critical challenge in building social chatbots bibref3. compared to previous systems bibref4, gunrock generates more balanced conversations between human and mach ...[truncated] |
| `natural language generation (NLG) (nlg) module` | yes | no | 1 | 23.262 | to provide coherent profile information, a critical challenge in building social chatbots bibref3. compared to previous systems bibref4, gunrock generates more balanced conversations between human and mach ...[truncated] |
| `text to speech (TTS) (tts)` | yes | no | 3 | 17.53 | to provide coherent profile information, a critical challenge in building social chatbots bibref3. compared to previous systems bibref4, gunrock generates more balanced conversations between human and mach ...[truncated] |
| `What the system designs introduced? Amazon Conversational Bot Toolkit` | yes | no | 1 | 19.408 | to provide coherent profile information, a critical challenge in building social chatbots bibref3. compared to previous systems bibref4, gunrock generates more balanced conversations between human and mach ...[truncated] |
| `We extend the Amazon Conversational Bot Toolkit (CoBot) BIBREF6 which is a flexible event-driven framework. CoBot p ...[truncated]` | yes | no | 1 | 25.139 | to provide coherent profile information, a critical challenge in building social chatbots bibref3. compared to previous systems bibref4, gunrock generates more balanced conversations between human and mach ...[truncated] |

## qasper-1910.07181-q0

- Paper: `1910.07181` - BERTRAM: Improved Word Embeddings Have Big Impact on Contextualized Model Performance
- Question: How much is representaton improved for rare/medum frequency words compared to standalone BERT and previous work?
- Oracle spans: `improving the score for WNLaMPro-medium by 50% compared to BERT$_\text{base}$ and 31% compared to Attentive Mimicking`
- Current trace: `data\traces\qasper-1910.07181-q0.jsonl`
- Current trace chunk count: 28
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: improving the score for WNLaMPro-medium by 50% compared to BERT$_\text{base}$ and 31% compared to Attentive Mimicking
- `highlighted_evidence`: Moreover, the add and add-gated variants of Bertram perform surprisingly well for more frequent words, improving the score for WNLaMPro-medium by 50% compared to BERT$_\text{base}$ and 31% compared to Atte ...[truncated]
- `evidence_head`: Results on WNLaMPro rare and medium are shown in Table TABREF34, where the mean reciprocal rank (MRR) is reported for BERT, Attentive Mimicking and Bertram. As can be seen, supplementing BERT with any of t ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- `WNLaMPro rare medium frequency words MRR improvement over BERT and Attentive Mimicking`
- `Table 1 MRR WNLaMPro RARE MEDIUM BERT base BERTRAM-ADD 0.269 0.367 numerical results`
- `downstream task rarified datasets accuracy improvement BERTRAM over BERT and baselines MNLI AG News DBPedia`
- `58% improvement compared to BERTbase 37% compared to Attentive Mimicking WNLaMPro MEDIUM`
- `absolute improvement of up to 25% over BERT downstream tasks`
- `Table 3 accuracy BERT base 50.5 56.5 49.3 +AM 50.9 58.9 60.7 +BERTRAM 53.3 62.1 64.2 MNLI AG News DBPedia`

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `How much is representaton improved for rare/medum frequency words compared to standalone BERT and previous work?` | yes | no | 1 | 17.306 | of this dataset is to probe a language model ' s ability to understand rare words without any task - specific finetuning, bibref0 do not provide a training set. furthermore, the dataset is partitioned into ...[truncated] |
| `improving the score for WNLaMPro-medium by 50% compared to BERT$_\text{base}$ and 31% compared to Attentive Mimicking` | yes | no | 1 | 24.285 | of this dataset is to probe a language model ' s ability to understand rare words without any task - specific finetuning, bibref0 do not provide a training set. furthermore, the dataset is partitioned into ...[truncated] |
| `How much is representaton improved for rare/medum frequency words compared to standalone BERT and previous work? im ...[truncated]` | yes | no | 1 | 20.316 | of this dataset is to probe a language model ' s ability to understand rare words without any task - specific finetuning, bibref0 do not provide a training set. furthermore, the dataset is partitioned into ...[truncated] |
| `Moreover, the add and add-gated variants of Bertram perform surprisingly well for more frequent words, improving th ...[truncated]` | yes | no | 1 | 25.961 | of this dataset is to probe a language model ' s ability to understand rare words without any task - specific finetuning, bibref0 do not provide a training set. furthermore, the dataset is partitioned into ...[truncated] |

## qasper-1911.03385-q0

- Paper: `1911.03385` - Low-Level Linguistic Controls for Style Transfer and Content Preservation
- Question: How they perform manual evaluation, what is criteria?
- Oracle spans: `accuracy`
- Current trace: `data\traces\qasper-1911.03385-q0.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: accuracy
- `highlighted_evidence`: Each annotator annotated 90 reference sentences (i.e. from the training corpus) with which style they thought the sentence was from. The accuracy on this baseline task for annotators A1, A2, and A3 was 80% ...[truncated]
- `evidence_head`: Each annotator annotated 90 reference sentences (i.e. from the training corpus) with which style they thought the sentence was from. The accuracy on this baseline task for annotators A1, A2, and A3 was 80% ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `How they perform manual evaluation, what is criteria?` | yes | no | 4 | 11.624 | into a fluent sentence. the fluency of all outputs is lower than desired. we expect that incorporating pre - trained language models would increase the fluency of all outputs without requiring larger datas ...[truncated] |
| `accuracy` | yes | no | 5 | 14.586 | into a fluent sentence. the fluency of all outputs is lower than desired. we expect that incorporating pre - trained language models would increase the fluency of all outputs without requiring larger datas ...[truncated] |
| `How they perform manual evaluation, what is criteria? accuracy` | yes | no | 1 | 15.567 | into a fluent sentence. the fluency of all outputs is lower than desired. we expect that incorporating pre - trained language models would increase the fluency of all outputs without requiring larger datas ...[truncated] |
| `Each annotator annotated 90 reference sentences (i.e. from the training corpus) with which style they thought the s ...[truncated]` | yes | no | 1 | 25.749 | into a fluent sentence. the fluency of all outputs is lower than desired. we expect that incorporating pre - trained language models would increase the fluency of all outputs without requiring larger datas ...[truncated] |

## qasper-1911.03894-q1

- Paper: `1911.03894` - CamemBERT: a Tasty French Language Model
- Question: Which tasks does CamemBERT not improve on?
- Oracle spans: `its performance still lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa`
- Current trace: `data\traces\qasper-1911.03894-q1.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: its performance still lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa
- `highlighted_evidence`: Experiments ::: Results ::: Natural Language Inference: XNLI On the XNLI benchmark, CamemBERT obtains improved performance over multilingual language models on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 fo ...[truncated]
- `evidence_head`: Experiments ::: Results ::: Natural Language Inference: XNLI
- `evidence_head`: On the XNLI benchmark, CamemBERT obtains improved performance over multilingual language models on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 for XLM) while using less than half the parameters (110M vs. 25 ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `Which tasks does CamemBERT not improve on?` | no | no |  |  |  |
| `its performance still lags behind models trained on the original English training set in the TRANSLATE-TEST setting ...[truncated]` | yes | no | 1 | 22.67 | 3. 33 % for sequoia, 1. 70 % for spoken and 1. 65 % for partut. # # experiments : : : results : : : natural language inference : xnli on the xnli benchmark, camembert obtains improved performance over mult ...[truncated] |
| `Which tasks does CamemBERT not improve on? its performance still lags behind models trained on the original English ...[truncated]` | yes | no | 1 | 22.37 | 3. 33 % for sequoia, 1. 70 % for spoken and 1. 65 % for partut. # # experiments : : : results : : : natural language inference : xnli on the xnli benchmark, camembert obtains improved performance over mult ...[truncated] |
| `Experiments ::: Results ::: Natural Language Inference: XNLI On the XNLI benchmark, CamemBERT obtains improved perf ...[truncated]` | yes | no | 1 | 25.931 | 3. 33 % for sequoia, 1. 70 % for spoken and 1. 65 % for partut. # # experiments : : : results : : : natural language inference : xnli on the xnli benchmark, camembert obtains improved performance over mult ...[truncated] |

## qasper-1911.03894-q2

- Paper: `1911.03894` - CamemBERT: a Tasty French Language Model
- Question: How much better was results of CamemBERT than previous results on these tasks?
- Oracle spans: `2.36 point increase in the F1 score with respect to the best SEM architecture`, `on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 for XLM)`, `lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa`, `For POS tagging, we observe error reductions of respectively 0.71% for GSD, 0.81% for Sequoia, 0.7% for Spoken and 0.28% for ParTUT`, `For parsing, we observe error reductions in LAS of 2.96% for GSD, 3.33% for Sequoia, 1.70% for Spoken and 1.65% for ParTUT`
- Current trace: `data\traces\qasper-1911.03894-q2.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: 2.36 point increase in the F1 score with respect to the best SEM architecture
- `oracle_span`: on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 for XLM)
- `oracle_span`: lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa
- `oracle_span`: For POS tagging, we observe error reductions of respectively 0.71% for GSD, 0.81% for Sequoia, 0.7% for Spoken and 0.28% for ParTUT
- `oracle_span`: For parsing, we observe error reductions in LAS of 2.96% for GSD, 3.33% for Sequoia, 1.70% for Spoken and 1.65% for ParTUT
- `highlighted_evidence`: For POS tagging, we observe error reductions of respectively 0.71% for GSD, 0.81% for Sequoia, 0.7% for Spoken and 0.28% for ParTUT. For parsing, we observe error reductions in LAS of 2.96% for GSD, 3.33% ...[truncated]
- `highlighted_evidence`: On the XNLI benchmark, CamemBERT obtains improved performance over multilingual language models on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 for XLM) while using less than half the parameters (110M vs. 250M).
- `highlighted_evidence`: However, its performance still lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa.
- `highlighted_evidence`: Both improvements result in a 2.36 point increase in the F1 score with respect to the best SEM architecture (BiLSTM-CRF), giving CamemBERT the state of the art for NER on the FTB.
- `evidence_head`: CamemBERT also demonstrates higher performances than mBERT on those tasks. We observe a larger error reduction for parsing than for tagging. For POS tagging, we observe error reductions of respectively 0.7 ...[truncated]
- `evidence_head`: On the XNLI benchmark, CamemBERT obtains improved performance over multilingual language models on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 for XLM) while using less than half the parameters (110M vs. 25 ...[truncated]
- `evidence_head`: For named entity recognition, our experiments show that CamemBERT achieves a slightly better precision than the traditional CRF-based SEM architectures described above in Section SECREF25 (CRF and Bi-LSTM+ ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `How much better was results of CamemBERT than previous results on these tasks?` | yes | no | 1 | 22.774 | improvement in finding entity mentions, raising the recall score by 3. 5 points. both improvements result in a 2. 36 point increase in the f1 score with respect to the best sem architecture ( bilstm - crf ...[truncated] |
| `2.36 point increase in the F1 score with respect to the best SEM architecture` | yes | no | 1 | 24.224 | improvement in finding entity mentions, raising the recall score by 3. 5 points. both improvements result in a 2. 36 point increase in the f1 score with respect to the best sem architecture ( bilstm - crf ...[truncated] |
| `on the TRANSLATE-TRAIN setting (81.2 vs. 80.2 for XLM)` | yes | no | 1 | 20.423 | 3. 33 % for sequoia, 1. 70 % for spoken and 1. 65 % for partut. # # experiments : : : results : : : natural language inference : xnli on the xnli benchmark, camembert obtains improved performance over mult ...[truncated] |
| `lags behind models trained on the original English training set in the TRANSLATE-TEST setting, 81.2 vs. 82.91 for RoBERTa` | yes | no | 1 | 21.393 | 3. 33 % for sequoia, 1. 70 % for spoken and 1. 65 % for partut. # # experiments : : : results : : : natural language inference : xnli on the xnli benchmark, camembert obtains improved performance over mult ...[truncated] |
| `For POS tagging, we observe error reductions of respectively 0.71% for GSD, 0.81% for Sequoia, 0.7% for Spoken and ...[truncated]` | yes | no | 1 | 25.392 | extended to support camembert and dependency parsing bibref55. the nli experiments use the fairseq library following the roberta implementation. # # experiments : : : results : : : part - of - speech taggi ...[truncated] |
| `For parsing, we observe error reductions in LAS of 2.96% for GSD, 3.33% for Sequoia, 1.70% for Spoken and 1.65% for ParTUT` | yes | no | 1 | 24.591 | extended to support camembert and dependency parsing bibref55. the nli experiments use the fairseq library following the roberta implementation. # # experiments : : : results : : : part - of - speech taggi ...[truncated] |
| `How much better was results of CamemBERT than previous results on these tasks? 2.36 point increase in the F1 score ...[truncated]` | yes | no | 1 | 23.647 | improvement in finding entity mentions, raising the recall score by 3. 5 points. both improvements result in a 2. 36 point increase in the f1 score with respect to the best sem architecture ( bilstm - crf ...[truncated] |
| `For POS tagging, we observe error reductions of respectively 0.71% for GSD, 0.81% for Sequoia, 0.7% for Spoken and ...[truncated]` | yes | no | 1 | 25.392 | extended to support camembert and dependency parsing bibref55. the nli experiments use the fairseq library following the roberta implementation. # # experiments : : : results : : : part - of - speech taggi ...[truncated] |

## qasper-1804.10686-q1

- Paper: `1804.10686` - An Unsupervised Word Sense Disambiguation System for Under-Resourced Languages
- Question: Which corpus of synsets are used?
- Oracle spans: `Wiktionary`
- Current trace: `data\traces\qasper-1804.10686-q1.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: Wiktionary
- `highlighted_evidence`: The following different sense inventories have been used during the evaluation:
- `highlighted_evidence`: Watlink, a word sense network constructed automatically. It uses the synsets induced in an unsupervised way by the Watset[CWnolog, MCL] method BIBREF2 and the semantic relations from such dictionaries as W ...[truncated]
- `evidence_head`: The following different sense inventories have been used during the evaluation:
- `evidence_head`: Watlink, a word sense network constructed automatically. It uses the synsets induced in an unsupervised way by the Watset[CWnolog, MCL] method BIBREF2 and the semantic relations from such dictionaries as W ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `Which corpus of synsets are used?` | yes | no | 1 | 20.484 | and 5 words covered by 439 instances in the wiki - wiki subset. the following different sense inventories have been used during the evaluation : [ leftmargin = 4mm ] watlink, a word sense network construct ...[truncated] |
| `Wiktionary` | yes | no | 1 | 15.385 | and 5 words covered by 439 instances in the wiki - wiki subset. the following different sense inventories have been used during the evaluation : [ leftmargin = 4mm ] watlink, a word sense network construct ...[truncated] |
| `Which corpus of synsets are used? Wiktionary` | yes | no | 1 | 20.496 | and 5 words covered by 439 instances in the wiki - wiki subset. the following different sense inventories have been used during the evaluation : [ leftmargin = 4mm ] watlink, a word sense network construct ...[truncated] |
| `The following different sense inventories have been used during the evaluation:` | yes | no | 1 | 24.962 | and 5 words covered by 439 instances in the wiki - wiki subset. the following different sense inventories have been used during the evaluation : [ leftmargin = 4mm ] watlink, a word sense network construct ...[truncated] |

## qasper-1910.04601-q1

- Paper: `1910.04601` - RC-QED: Evaluating Natural Language Derivations in Multi-Hop Reading Comprehension
- Question: What dataset was used in the experiment?
- Oracle spans: `WikiHop`
- Current trace: `data\traces\qasper-1910.04601-q1.jsonl`
- Current trace chunk count: 15
- Interpretation: `oracle_span_mentioned_but_gold_evidence_missing_in_current_trace`

### Targets

- `oracle_span`: WikiHop
- `highlighted_evidence`: Our study uses WikiHop BIBREF0, as it is an entity-based multi-hop QA dataset and has been actively used.
- `evidence_head`: Our study uses WikiHop BIBREF0, as it is an entity-based multi-hop QA dataset and has been actively used. We randomly sampled 10,000 instances from 43,738 training instances and 2,000 instances from 5,129 ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- `What dataset was used in the experiment?`
- `dataset benchmark corpus used for evaluation`
- `experiment data collection source of questions`

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `loose` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `What dataset was used in the experiment?` | yes | no | 6 | 15.622 | the screen, and changes according to the length of nlds they write in real time. to discourage noisy annotations, we also warn crowdworkers that their work would be rejected for noisy submissions. we perio ...[truncated] |
| `WikiHop` | yes | no | 3 | 15.304 | the screen, and changes according to the length of nlds they write in real time. to discourage noisy annotations, we also warn crowdworkers that their work would be rejected for noisy submissions. we perio ...[truncated] |
| `What dataset was used in the experiment? WikiHop` | yes | no | 4 | 16.297 | the screen, and changes according to the length of nlds they write in real time. to discourage noisy annotations, we also warn crowdworkers that their work would be rejected for noisy submissions. we perio ...[truncated] |
| `Our study uses WikiHop BIBREF0, as it is an entity-based multi-hop QA dataset and has been actively used.` | yes | no | 1 | 24.583 | the screen, and changes according to the length of nlds they write in real time. to discourage noisy annotations, we also warn crowdworkers that their work would be rejected for noisy submissions. we perio ...[truncated] |

## qasper-2001.09899-q0

- Paper: `2001.09899` - Vocabulary-based Method for Quantifying Controversy in Social Media
- Question: What are the state of the art measures?
- Oracle spans: `Randomwalk`, `Walktrap`, `Louvain clustering`
- Current trace: `data\traces\qasper-2001.09899-q0.jsonl`
- Current trace chunk count: 0
- Interpretation: `current_query_missed_retrievable_gold_evidence`

### Targets

- `oracle_span`: Randomwalk
- `oracle_span`: Walktrap
- `oracle_span`: Louvain clustering
- `highlighted_evidence`: As Garimella et al. BIBREF23 have made their code public , we reproduced their best method Randomwalk on our datasets and measured the AUC ROC, obtaining a score of 0.935. An interesting finding was that t ...[truncated]
- `evidence_head`: As Garimella et al. BIBREF23 have made their code public , we reproduced their best method Randomwalk on our datasets and measured the AUC ROC, obtaining a score of 0.935. An interesting finding was that t ...[truncated]

### Full-Text Presence

| target | hit | match |
|---|---:|---|
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `oracle_span` | yes | `exact` |
| `highlighted_evidence` | yes | `exact` |
| `evidence_head` | yes | `exact` |

### Current Trace Recall

Queries:

- (none)

| target | hit | match |
|---|---:|---|
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `oracle_span` | no | `none` |
| `highlighted_evidence` | no | `none` |
| `evidence_head` | no | `none` |

### Gold-Oriented Probe

Probe source: QASPER enriched `full_text` diagnostic index.

| query | gold evidence hit | oracle mention only | best rank | best score | best chunk head |
|---|---:|---:|---:|---:|---|
| `What are the state of the art measures?` | yes | no | 7 | 9.085 | the black group shows the score distribution over controversial discussions and the white group over non - controversial ones. a larger separation of the two distributions indicates that the measure is bet ...[truncated] |
| `Randomwalk` | yes | no | 2 | 21.005 | the black group shows the score distribution over controversial discussions and the white group over non - controversial ones. a larger separation of the two distributions indicates that the measure is bet ...[truncated] |
| `Walktrap` | yes | no | 5 | 16.837 | the black group shows the score distribution over controversial discussions and the white group over non - controversial ones. a larger separation of the two distributions indicates that the measure is bet ...[truncated] |
| `Louvain clustering` | yes | no | 4 | 20.184 | the black group shows the score distribution over controversial discussions and the white group over non - controversial ones. a larger separation of the two distributions indicates that the measure is bet ...[truncated] |
| `What are the state of the art measures? Randomwalk` | yes | no | 1 | 16.904 | the black group shows the score distribution over controversial discussions and the white group over non - controversial ones. a larger separation of the two distributions indicates that the measure is bet ...[truncated] |
| `As Garimella et al. BIBREF23 have made their code public , we reproduced their best method Randomwalk on our datase ...[truncated]` | yes | no | 1 | 20.984 | the black group shows the score distribution over controversial discussions and the white group over non - controversial ones. a larger separation of the two distributions indicates that the measure is bet ...[truncated] |
