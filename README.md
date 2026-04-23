# PaperPilot

> Academic research agent built on nanobot, with citation-graph 
> retrieval, ColBERT late interaction, and multimodal figure 
> understanding.

## Overview

PaperPilot is an LLM-powered agent that helps researchers track, 
read, and synthesize papers in their field of interest. Given a 
research query, it autonomously searches arXiv, traces citation 
lineages, performs concurrent paper reading via subagents, and 
generates structured research reports.

**Status**: 🚧 Under active development (Week 1 of 4)

## Architecture

┌─────────────────────────────────────────────┐
│  User Query                                  │
│    ↓                                         │
│  Query-aware Planner                         │
│    ↓                                         │
│  nanobot Agent Loop                          │
│    ↓                                         │
│  MCP Tools: arxiv_search, pdf_parse,         │
│             graph_retrieve, colbert_retrieve,│
│             vlm_figure                       │
│    ↓                                         │
│  SubagentManager (concurrent paper reading)  │
│    ↓                                         │
│  Report Generator → Markdown output          │
└─────────────────────────────────────────────┘

## Tech Stack

- **Agent Framework**: [nanobot](https://github.com/HKUDS/nanobot) 
  (~4k-line MCP-native framework)
- **Retrieval**: ColBERT v2 (via ragatouille) + NetworkX citation graph
- **Data Sources**: arXiv API, Semantic Scholar Graph API
- **Multimodal**: Qwen-VL for figure understanding
- **LLM**: DeepSeek-V3 / Qwen

## Roadmap

- [x] Project setup
- [ ] Week 1: Agent scaffold + arxiv_search + pdf_parse
- [ ] Week 2: ColBERT retrieval + NetworkX citation graph
- [ ] Week 3: VLM figure understanding + subagent concurrent reading
- [ ] Week 4: Evaluation harness + demo polish

## Acknowledgments

- [HKUDS/nanobot](https://github.com/HKUDS/nanobot) for the agent framework
- [Semantic Scholar](https://www.semanticscholar.org/product/api) for citation graph data
- [Allen AI](https://allenai.org/) for open research infrastructure

## License

MIT
