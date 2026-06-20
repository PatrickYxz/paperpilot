# Retrieval Recall Diagnosis

Date: 2026-06-16

Enriched source: `data\eval\qasper_subset_enriched.jsonl`

This report checks whether representative QASPER gold/oracle evidence is present in the eval full text, recalled by current PaperPilot traces, and retrievable with gold-oriented diagnostic queries.

Gold-oriented probe queries are diagnostic only. They must not be used by runtime PaperPilot.

The probe builds a separate diagnostic ColBERT index from QASPER enriched `full_text`, keyed by a content hash, so it does not reuse the runtime arXiv/PaperPilot index.

## Summary

| case_id | full text has gold | current trace recalls gold | probe retrieves gold | interpretation |
|---|---:|---:|---:|---|
| `qasper-1910.04601-q1` | yes | no | yes | `oracle_span_mentioned_but_gold_evidence_missing_in_current_trace` |
| `qasper-1701.00185-q1` | yes | yes | no | `gold_evidence_recalled_in_current_trace` |
| `qasper-1910.07181-q0` | yes | no | yes | `current_query_missed_retrievable_gold_evidence` |

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
