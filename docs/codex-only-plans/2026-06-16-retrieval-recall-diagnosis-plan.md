# Retrieval Recall Diagnosis Plan

Date: 2026-06-16

Codex only. This plan covers the next PaperPilot accuracy step after Evidence Selection V1.

## Background

Evidence Selection V1 showed that the wrapper can narrow answers when retrieved evidence is broad. It cannot fix cases where PaperPilot's retrieved chunks directly support an answer that disagrees with QASPER oracle spans.

The next question is whether those failures are caused by retrieval recall:

- Did the gold/oracle evidence exist in the eval full text?
- Did the current PaperPilot trace retrieve it?
- If not, can ColBERT retrieve it with a better query?
- Or is the QASPER oracle/gold evidence inconsistent with the paper text used by PaperPilot?

## Target Cases

Start with the three representative cases:

1. `qasper-1910.04601-q1`
2. `qasper-1701.00185-q1`
3. `qasper-1910.07181-q0`

## Diagnostic Layers

### Layer 1: Full-Text Presence

Check whether each oracle span and gold evidence snippet appears in `data/eval/qasper_subset_enriched.jsonl` full text.

This tells us whether the expected answer is even available in the eval text source.

### Layer 2: Current Trace Recall

Extract all current PaperPilot `colbert.search` chunks from `data/traces/<case_id>.jsonl`.

Check whether oracle spans or gold evidence snippets appear in retrieved chunks.

This tells us whether the running agent retrieved the gold-like evidence.

### Layer 3: Gold-Oriented Retrieval Probe

Run ColBERT search on the same paper with diagnostic queries:

- original question;
- oracle span;
- gold evidence head;
- question plus oracle span.

This is offline diagnosis only. These queries may use gold information and must not be used by runtime PaperPilot.

This tells us whether ColBERT can retrieve the gold evidence if asked more directly.

## Outputs

Add a reusable diagnostic script:

- `scripts/day24_retrieval_recall_diagnosis.py`

Write a markdown report:

- `docs/retrieval_recall_diagnosis_20260616.md`

The report should include:

- per-case summary table;
- full-text presence result;
- current trace recall result;
- gold-oriented probe result;
- interpretation of likely failure layer.

## Validation

Run unit-light script execution against the three target cases.

If the script imports project modules, run it with the project venv using escalation because `.venv\Scripts\python.exe` depends on a base interpreter outside the sandbox.

## Risks

- Gold evidence strings may be normalized differently from extracted PDF text.
- Exact substring matching may undercount recall.
- Gold-oriented queries are diagnostic only and can overestimate what runtime retrieval should achieve.
- ColBERT startup may require clearing proxy variables and increasing `MCP_INITIALIZE_TIMEOUT`.

