# Day 18 deep-read 召回精度 eval

**Date**: 2026-05-10
**Dataset**: AI2 QASPER NLP subset (extractive QA)
**Cases per baseline**: 150
**LLM**: DeepSeek (via anthropic SDK), shared across all baselines
**Note**: abstract_only and full_text are the Day 16 baseline files; paperpilot
was rerun after the Day 18 deep-read prompt/evidence-span improvements.

## 三段对比

| Baseline | Pass | Fail | Error | Pass Rate | Avg latency |
|---|---|---|---|---|---|
| abstract_only | 3 | 147 | 0 | **2.0%** | 0.8s |
| full_text | 61 | 89 | 0 | **40.7%** | 3.5s |
| paperpilot | 99 | 51 | 0 | **66.0%** | 107.9s |

## PaperPilot 失败归因

| 桶 | 计数 | 占比 |
|---|---|---|
| synthesis_miss | 50 | 98% |
| colbert_searched_low | 1 | 2% |

## 解读

- abstract -> full_text 提升 +38.7pts:细节召回需要正文,abstract 远不够
- full_text -> paperpilot 提升 +25.3pts:colbert 选段 + 多次召回比全文一次性塞 LLM 更优,验证 RAG 路线价值
