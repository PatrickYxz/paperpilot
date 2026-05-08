# Day 16 deep-read 召回精度 eval

**Date**: 2026-05-08
**Dataset**: AI2 QASPER NLP subset (extractive QA)
**Cases per baseline**: 150
**LLM**: DeepSeek (via anthropic SDK), shared across all baselines

## 三段对比

| Baseline | Pass | Fail | Error | Pass Rate | Avg latency |
|---|---|---|---|---|---|
| abstract_only | 3 | 147 | 0 | **2.0%** | 0.8s |
| full_text | 61 | 89 | 0 | **40.7%** | 3.5s |
| paperpilot | 68 | 82 | 0 | **45.3%** | 110.9s |

## PaperPilot 失败归因

| 桶 | 计数 | 占比 |
|---|---|---|
| synthesis_miss | 56 | 68% |
| colbert_searched_low | 25 | 30% |
| no_colbert_search | 1 | 1% |

## 解读

- abstract -> full_text 提升 +38.7pts:细节召回需要正文,abstract 远不够
- full_text -> paperpilot 提升 +4.7pts:colbert 选段 + 多次召回比全文一次性塞 LLM 更优,验证 RAG 路线价值
