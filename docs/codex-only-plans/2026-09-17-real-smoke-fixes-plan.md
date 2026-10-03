# Real Smoke Test Blocking Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four production-boundary defects exposed by the isolated real DeepSeek/MCP smoke test, then rerun the same smoke scenario.

**Architecture:** Keep the existing graph and public APIs unchanged. Configure the existing DeepSeek chat model for non-thinking function calling, classify only deterministic non-retryable HTTP client errors at the executor boundary, count tool schemas locally in addition to provider-counted messages, and keep MCP stdout protocol-clean by importing PyMuPDF through its supported module name.

**Tech Stack:** Python 3.12, LangChain/LangGraph, `langchain-deepseek`, FastAPI task executors, MCP stdio, PyMuPDF, pytest.

**Spec:** `docs/codex-only-plans/2026-09-01-context-compression-design.md`

## Global Constraints

- Preserve the public Conversation, Message, Task, ResearchResult, and AnswerDraft contracts.
- Keep `PAPERPILOT_CONTEXT_MANAGEMENT_ENABLED` and `PAPERPILOT_FULL_COMPACTION_ENABLED` defaults unchanged.
- Do not introduce dependencies or expose prompts, paper bodies, credentials, or provider error bodies in durable events.
- Preserve retries for connection/timeouts, HTTP 408/409/425/429, HTTP 5xx, database errors, and unknown infrastructure failures.
- Treat other HTTP 4xx responses as deterministic non-retryable failures.
- Run every production change through a failing regression test first.

---

### Task 1: Disable DeepSeek thinking mode for structured tool calls

**Files:**
- Modify: `tests/deep_reading/test_runner.py`
- Modify: `paperpilot/deep_reading/runner.py`

**Interfaces:**
- Consumes: `build_deep_reading_model(model_name, research_max_output_tokens)`.
- Produces: the same `ChatDeepSeek` object, configured with `extra_body={"thinking": {"type": "disabled"}}`.

- [x] Update the model-construction test to require the explicit non-thinking request body.
- [x] Run that test and confirm it fails because `extra_body` is absent.
- [x] Add only the required `extra_body` argument to `build_deep_reading_model()`.
- [x] Run the focused test and related Runner tests.

### Task 2: Stop retrying deterministic HTTP 4xx workflow failures

**Files:**
- Modify: `tests/web/test_task_executor.py`
- Modify: `tests/web/test_worker_tasks.py`
- Modify: `paperpilot/web/task_executor.py`
- Modify: `paperpilot/web/worker_tasks.py`
- Modify: `paperpilot/deep_reading/runner.py`

**Interfaces:**
- Produces: `is_retryable_task_exception(exc: Exception) -> bool`.
- Produces: `DeepReadingRunner.fail_non_retryable(task_id, *, backend, attempts, exc) -> None`.
- Preserves: the existing `fail_retry_exhausted()` path for genuinely retryable failures.

- [x] Add a thread-executor test using an exception with `status_code=400`; assert one Runner call, no sleep, one non-retryable finalization, and no retry-exhaustion finalization.
- [x] Add classification tests for 400, 408, 409, 425, 429, 500, connection errors, and unknown exceptions.
- [x] Add the equivalent Celery test: 400 must not call `self.retry`, while retryable failures keep existing behavior.
- [x] Run the tests and confirm the new 400 cases fail because every exception is currently retried.
- [x] Implement the minimal classifier and terminal-finalization path for both executors.
- [x] Run thread, Celery, and Runner failure-boundary tests.

### Task 3: Count tool schemas even when the provider silently ignores them

**Files:**
- Modify: `tests/deep_reading/test_context_budget.py`
- Modify: `paperpilot/deep_reading/context_management/budget.py`

**Interfaces:**
- Preserves: `ModelAwareTokenCounter.count_messages(messages, *, tool_schemas=()) -> int`.
- Behavior: provider counter counts messages only; canonical local fallback always adds tool-schema tokens when schemas are present.

- [x] Add a model stub whose counter accepts `tools` but ignores them; assert the result still exceeds the message-only count and that the provider is not trusted to count schemas.
- [x] Run the test and confirm it fails at the message-only value.
- [x] Count messages with the provider and add canonical local tool-schema tokens unconditionally.
- [x] Run all context-budget and ContextView budget tests.

### Task 4: Keep MCP stdout free of PyMuPDF compatibility warnings

**Files:**
- Create: `tests/tools/test_mcp_protocol_stdout.py`
- Modify: `paperpilot/mcp_servers/arxiv.py`
- Modify: `paperpilot/mcp_servers/vlm/page_renderer.py`

**Interfaces:**
- Preserves: existing `fitz` call sites via `import pymupdf as fitz`.
- Behavior: importing MCP server modules writes no compatibility warning to stdout.

- [x] Add a subprocess import test for both modules and assert stdout is empty.
- [x] Run it and confirm the existing `import fitz` warning is captured on stdout.
- [x] Replace the deprecated import with `import pymupdf as fitz` in both modules.
- [x] Run MCP, arXiv, and VLM focused tests.

### Task 5: Verify locally and rerun the isolated real smoke test

**Files:**
- Modify only if a regression test exposes another root cause.

**Interfaces:**
- Uses the isolated database and checkpoint pattern from `/private/tmp/paperpilot-real-smoke.*`.

- [x] Run focused tests for all four fixes.
- [x] Run `.venv/bin/python -m pytest -q` and `git diff --check`.
- [x] Start a fresh isolated Web instance with context management enabled and full compaction disabled.
- [x] Submit one `quick` primary-paper question for arXiv `2001.09899v1` (the arXiv catalog lookup returned HTTP 500, so only conversation-creation metadata was injected; the Research/MCP/LLM path remained real).
- [x] Verify terminal Task status, final answer/citations, Artifact/Archive rows, redacted events, model usage, and absence of retry loops or JSON-RPC parse errors.
- [x] Close the temporary TestClient; no server process or cookie file was created. Retain isolated database evidence until the result is reported.

### Follow-up: Align the default graph-step budget with the 8-call structured attempt

The live smoke reached six Research model responses, then raised `GraphRecursionError` before the model-call or business-tool limits. In the current LangChain graph, a tool round traverses `before_model → model → after_model → tools`; a final structured response uses three nodes. Eight model calls therefore need 31 executed steps, and the invocation limit must be at least 32 to permit termination. The default 24 only accommodates the six-call path. Use 33 to leave one graph step of margin; keep explicit overrides and the 8/4 per-attempt model/tool budgets unchanged.

- [x] Add a real-Agent regression that follows seven tool rounds and a final structured response, and confirm it fails at the 24-step default.
- [x] Change the default to 33 consistently in Context, Runner, Web config, and `.env.example`; update default assertions without changing explicit 24-step tests.
- [x] Pass the focused Research, Runner, and config suites.
- [x] Run the full local suite and repeat the isolated real smoke scenario. The first repeat reached all 8 calls but ended without a decision, prompting the bounded-continuation follow-up below.

### Follow-up: Use the reserved attempt when the primary attempt exhausts its graph/model/tool budget

The first post-recursion smoke reached all 8 primary model calls and externalized two retrieval results, but still ended without a decision. The 4-call reserve was only reachable for malformed structured output, so budget exhaustion discarded it. Keep the total 12-call allocation and authoritative ledgers; invoke the second attempt once with a bounded continuation instruction. If that attempt also exhausts its budget, retain the existing safe terminal error. Do not retry unrelated infrastructure or contract failures.

- [x] Add failing tests for graph, model, and business-tool budget exhaustion followed by a successful reserved attempt.
- [x] Add a real-Agent test showing that a four-call primary attempt can complete through the reserve.
- [x] Preserve safe terminal behavior after the second attempt also exhausts its budget.
- [x] Run the full suite (`832 passed`) and one isolated real smoke; verify a completed task, assistant message with 10 citations, finalized head/checkpoint, two context artifacts, and one TurnArchive row.
