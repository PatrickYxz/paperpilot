# PaperPilot New-Architecture-Only Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every legacy PaperPilot product path and historical research artifact from the current checkout while preserving user/runtime data and leaving one FastAPI → Conversation → LangGraph → LangChain → MCP architecture.

**Architecture:** Refactor in place from the already verified Conversation vertical slice. First move the shared Tool contract and hidden retrieval LLM calls onto the new framework boundary, then add the conversation-scoped progress API, split FastAPI routers, make Thread/Celery invoke DeepReadingRunner directly, and consolidate the UI. Only after those dependencies are green may legacy code/tests/docs be deleted; database schema/history and all user/runtime data remain untouched.

**Tech Stack:** Python 3.12, FastAPI `APIRouter`, SQLAlchemy 2, Alembic, SQLite WAL, LangGraph 1.2, LangChain 1.3, `langgraph-checkpoint-sqlite` 3.1, `langchain-deepseek` 1.1, Pydantic 2, Celery/Redis, native HTML/CSS/JavaScript, pytest.

## Global Constraints

- The approved design is `docs/codex-only-plans/2026-08-11-new-architecture-cleanup-design.md`; implementation must not widen beyond it.
- Preserve `data/web/`, `data/langgraph/`, `data/papers/`, and `data/colbert_index/` byte-for-byte except for normal application writes explicitly initiated by the user; automated work must not write to them.
- Do not drop columns, rebuild tables, delete legacy rows, squash Alembic history, or run destructive migrations.
- `conversation.id` remains the LangGraph `thread_id`; rollback only moves the business head and performs zero model calls.
- FastAPI is the only product entrypoint. `/api/tasks/*`, `/api/eval/*`, legacy CLI, legacy Agent Loop, simulated tasks, and one-shot tasks must be absent at completion.
- Every new Task must be bound to both `conversation_id` and `user_message_id`; old rows may remain stored but are never listed, executed, or exposed.
- arXiv, ColBERT, Citation Graph, and VLM MCP servers all remain registered and startable.
- Retrieval planner/verifier must use `ChatDeepSeek` + Pydantic structured output. Planner performs at most one LangChain structured-output invoke per retrieval; verifier is disabled by default and capped at six requirement invokes when explicitly enabled. The shared `ChatDeepSeek` provider retry default remains `1`, so a transient transport failure may cause one additional provider attempt even though application-level invoke counts remain bounded.
- Keep current Agent defaults: model calls `8`, tool calls `12`, output tokens `4096`, provider retry `1`, recursion limit `24`.
- Keep current Task retry defaults: three retries after the first attempt, one-second initial exponential backoff capped at 30 seconds.
- Preserve atomic claim, completed-race behavior, terminal/transient exception identity, checkpoint validation, idempotent publication, owner isolation, and safe public error messages.
- Use FastAPI/Pydantic/LangChain/LangGraph native boundaries. Do not add Repository, Ports, Adapters, another workflow framework, or a generic service/container layer.
- All automated model tests use fake models; no real DeepSeek invocation, arXiv paper download, Semantic Scholar request, or other paid/network smoke without fresh user approval.
- Move the untracked user file to the exact external backup path before any deletion: `/Users/patrick/Documents/PaperPilot-archive/2026-08-05-architecture-audit-plan.md`.
- Delete tracked files with exact, reviewed targets. Before deleting untracked `data/eval/` or trace files, list the precise paths and obtain an additional user confirmation.
- Follow TDD for every behavior change and commit after each Task passes its focused gates.

---

## Pre-execution: Isolation, baseline, and user-file backup

This is execution setup, not an implementation commit.

- [ ] **Step 1: Confirm the source branch and dirty set**

Run from `/Users/patrick/Projects/paperpilot`:

```bash
git branch --show-current
git status --short
git rev-parse HEAD
```

Expected before backup: branch `codex/goal_test`; the only untracked project file is `docs/codex-only-plans/2026-08-05-architecture-audit-plan.md`. Stop if any other change exists.

- [ ] **Step 2: Back up the untracked user plan outside the project**

Resolve both paths literally; do not overwrite an existing backup:

```bash
test -f /Users/patrick/Projects/paperpilot/docs/codex-only-plans/2026-08-05-architecture-audit-plan.md
test ! -e /Users/patrick/Documents/PaperPilot-archive/2026-08-05-architecture-audit-plan.md
mkdir -p /Users/patrick/Documents/PaperPilot-archive
shasum -a 256 /Users/patrick/Projects/paperpilot/docs/codex-only-plans/2026-08-05-architecture-audit-plan.md
mv /Users/patrick/Projects/paperpilot/docs/codex-only-plans/2026-08-05-architecture-audit-plan.md /Users/patrick/Documents/PaperPilot-archive/2026-08-05-architecture-audit-plan.md
shasum -a 256 /Users/patrick/Documents/PaperPilot-archive/2026-08-05-architecture-audit-plan.md
```

Expected: the two SHA-256 values match; source is absent; target exists. Record the digest in the SDD progress ledger.

- [ ] **Step 3: Create an isolated worktree**

Use `superpowers:using-git-worktrees` and create:

```text
branch: codex/new-architecture-cleanup
worktree: /Users/patrick/Projects/paperpilot/.worktrees/new-architecture-cleanup
base: codex/goal_test
```

- [ ] **Step 4: Record protected-data metadata without opening writable connections**

Run from the new worktree, pointing explicitly at the main checkout data:

```bash
ls -l /Users/patrick/Projects/paperpilot/data/web/tasks.sqlite3
ls -l /Users/patrick/Projects/paperpilot/data/langgraph/checkpoints.sqlite3
sqlite3 'file:/Users/patrick/Projects/paperpilot/data/web/tasks.sqlite3?mode=ro' 'PRAGMA quick_check; SELECT version_num FROM alembic_version;'
sqlite3 'file:/Users/patrick/Projects/paperpilot/data/langgraph/checkpoints.sqlite3?mode=ro' "PRAGMA quick_check; SELECT name FROM sqlite_master WHERE type='table' AND name IN ('checkpoints','writes') ORDER BY name;"
```

Expected: both quick checks return `ok`; business Alembic version is `20260807_0002`; checkpoint tables are `checkpoints` and `writes`.

- [ ] **Step 5: Run the fresh code baseline**

```bash
LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests -q
uv pip check --python .venv/bin/python
git diff --check
```

Expected: current baseline `968 passed, 11 deselected`, only six known warnings; 143 packages compatible; diff check clean. Stop and diagnose if the fresh numbers differ.

---

### Task 1: Extract the new-stack Tool contract from legacy core

**Files:**
- Create: `paperpilot/tools/types.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/test_types.py`
- Modify: `paperpilot/deep_reading/nodes.py`
- Modify: `paperpilot/deep_reading/research_agent.py`
- Modify: `paperpilot/tools/mcp_client.py`
- Modify: `paperpilot/tools/mcp_runtime.py`
- Modify: `tests/deep_reading/test_nodes.py`
- Modify: `tests/deep_reading/test_research_agent.py`
- Modify: `tests/deep_reading/test_runner.py`
- Modify: `tests/test_mcp_client.py`
- Modify: `tests/test_mcp_runtime.py`
- Modify: `tests/web/test_conversation_api.py`

**Interfaces:**
- Consumes: existing `Tool(name: str, description: str, input_schema: dict, handler: Callable[[dict], Any])` semantics from `paperpilot.core.adapter`.
- Produces: `paperpilot.tools.types.Tool`; all retained Deep Reading and MCP code imports this type without importing `paperpilot.core`.

- [ ] **Step 1: Write the failing Tool contract test**

Create `tests/tools/test_types.py`:

```python
from paperpilot.tools.types import Tool


def test_tool_exposes_framework_neutral_mcp_contract() -> None:
    tool = Tool(
        name="mcp__test__echo",
        description="Echo input",
        input_schema={"type": "object"},
        handler=lambda args: {"echo": args["value"]},
    )

    assert tool.name == "mcp__test__echo"
    assert tool.handler({"value": "ok"}) == {"echo": "ok"}
```

- [ ] **Step 2: Verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/tools/test_types.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'paperpilot.tools.types'`.

- [ ] **Step 3: Implement only the shared Tool dataclass**

Create `paperpilot/tools/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], Any]
```

Do not migrate `LLMClient`, `ToolCall`, `ToolResult`, `ParsedResponse`, guardrails, or loop types.

- [ ] **Step 4: Move retained production imports**

Change only retained new-stack modules from:

```python
from paperpilot.core.adapter import Tool
```

to:

```python
from paperpilot.tools.types import Tool
```

Also update the MCP client module comment so it names `paperpilot.tools.types.Tool`.

- [ ] **Step 5: Move retained test imports and verify no retained direct core dependency**

Update the listed Deep Reading, MCP Runtime/Client, and Conversation API tests. Run:

```bash
rg -n "paperpilot\.core" paperpilot/deep_reading paperpilot/tools tests/deep_reading tests/test_mcp_client.py tests/test_mcp_runtime.py tests/web/test_conversation_api.py
```

Expected: no matches. Retrieval is handled in Task 2 and is not included in this scan yet.

- [ ] **Step 6: Run focused GREEN gates**

```bash
./.venv/bin/python -m pytest tests/tools tests/test_mcp_client.py tests/test_mcp_runtime.py tests/deep_reading tests/web/test_conversation_api.py -q
git diff --check
```

Expected: zero failures and clean diff.

- [ ] **Step 7: Commit**

```bash
git add paperpilot/tools/types.py paperpilot/deep_reading paperpilot/tools/mcp_client.py paperpilot/tools/mcp_runtime.py tests/tools tests/deep_reading tests/test_mcp_client.py tests/test_mcp_runtime.py tests/web/test_conversation_api.py
git commit -m "refactor(tools): extract new-stack tool contract"
```

---

### Task 2: Migrate retrieval planner and verifier to LangChain structured output

**Files:**
- Modify: `paperpilot/retrieval/query_plan_validator.py`
- Modify: `paperpilot/retrieval/llm_query_planner.py`
- Modify: `paperpilot/retrieval/evidence_verifier.py`
- Modify: `paperpilot/retrieval/planned_retrieval.py`
- Modify: `tests/retrieval/test_query_plan_validator.py`
- Modify: `tests/retrieval/test_llm_query_planner.py`
- Modify: `tests/retrieval/test_evidence_verifier.py`
- Modify: `tests/retrieval/test_planned_retrieval.py`
- Modify: `tests/mcp_servers/test_colbert_server_planned_retrieval.py`

**Interfaces:**
- Consumes: `ChatDeepSeek`, LangChain `with_structured_output(OutputModel, include_raw=True)`, existing `QueryPlan`, literal fallback, and current retrieval environment variables.
- Produces: `plan_with_llm(question: str, paper_title: str = "", abstract: str = "", model: BaseChatModel | None = None) -> tuple[QueryPlan, dict[str, Any]]`; `run_evidence_verification(plan: QueryPlan, pool: EvidencePool, model: BaseChatModel | None, summary_k: int, verifier_candidate_k: int = 6) -> EvidenceVerificationResult`; no import from `paperpilot.core`.

- [ ] **Step 1: Write RED tests for requirement cap and one-call structured planner**

Add to `tests/retrieval/test_query_plan_validator.py`:

```python
def valid_plan_dict(*, question: str = "question") -> dict:
    return {
        "version": "query_plan_v1",
        "question": question,
        "question_type": "other",
        "answer_shape": "freeform",
        "intent_summary": "Find direct evidence.",
        "focus_terms": [],
        "constraints": {
            "needs_numbers": False,
            "needs_comparison": False,
            "needs_table_or_figure": False,
            "polarity": "neutral",
        },
        "evidence_requirements": [
            {
                "id": "req_1",
                "description": "direct evidence",
                "required": True,
            }
        ],
        "queries": [
            {
                "id": "q_1",
                "role": "focused_rewrite",
                "query": "direct evidence",
                "targets": ["req_1"],
                "priority": 2,
            }
        ],
        "avoid": [],
        "expansion_hints": {
            "neighbor_window": 1,
            "prefer_tables": False,
            "prefer_captions": False,
        },
    }


def test_validate_query_plan_caps_evidence_requirements_at_six() -> None:
    data = valid_plan_dict()
    data["evidence_requirements"] = [
        {"id": f"req_{index}", "description": f"Requirement {index}"}
        for index in range(8)
    ]
    data["queries"] = [
        {"id": "q_1", "query": "evidence", "targets": ["req_1"]}
    ]

    plan = validate_query_plan(data, question="question")

    assert [item.id for item in plan.evidence_requirements] == [
        "req_0", "req_1", "req_2", "req_3", "req_4", "req_5"
    ]
```

Copy the exact `valid_plan_dict` helper above into `tests/retrieval/test_llm_query_planner.py` as a module-local test builder. Add a fake structured runnable that records `invoke()` calls and returns:

```python
{
    "raw": object(),
    "parsed": RetrievalQueryPlanOutput.model_validate(valid_plan_dict()),
    "parsing_error": None,
}
```

Assert `plan_with_llm(question="question", model=fake_model)` invokes exactly once and returns `fallback_used is False`.

- [ ] **Step 2: Verify RED**

```bash
./.venv/bin/python -m pytest tests/retrieval/test_query_plan_validator.py tests/retrieval/test_llm_query_planner.py -q
```

Expected: missing `MAX_EVIDENCE_REQUIREMENTS`/`RetrievalQueryPlanOutput` or wrong client interface.

- [ ] **Step 3: Add the Pydantic planner output contract**

In `llm_query_planner.py`, import `BaseModel`, `ConfigDict`, and `Field`, then define these strict Pydantic models for the existing JSON shape:

```python
class RetrievalConstraintsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_numbers: bool = False
    needs_comparison: bool = False
    needs_table_or_figure: bool = False
    polarity: Literal["neutral", "negative", "contrastive"] = "neutral"


class RetrievalRequirementOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    required: bool = True


class RetrievalQueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    query: str
    targets: list[str]
    priority: int = 1


class RetrievalExpansionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    neighbor_window: int = 1
    prefer_tables: bool = False
    prefer_captions: bool = False


class RetrievalQueryPlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["query_plan_v1"]
    question: str
    question_type: Literal[
        "dataset_used",
        "method_list",
        "metric_result",
        "comparison",
        "yes_no",
        "definition",
        "evidence_location",
        "other",
    ]
    answer_shape: Literal[
        "single_entity", "list", "number", "comparison", "yes_no", "freeform"
    ]
    intent_summary: str
    focus_terms: list[str] = Field(default_factory=list)
    constraints: RetrievalConstraintsOutput
    evidence_requirements: list[RetrievalRequirementOutput]
    queries: list[RetrievalQueryOutput]
    avoid: list[str] = Field(default_factory=list)
    expansion_hints: RetrievalExpansionOutput
```

- [ ] **Step 4: Implement one-call LangChain planning with safe fallback**

Replace the old `.call()` client boundary with:

```python
def plan_with_llm(
    *,
    question: str,
    paper_title: str = "",
    abstract: str = "",
    model: BaseChatModel | None = None,
) -> tuple[QueryPlan, dict[str, Any]]:
    planner_model = model or build_retrieval_model()
    runnable = planner_model.with_structured_output(
        RetrievalQueryPlanOutput,
        include_raw=True,
    )
    try:
        result = runnable.invoke(
            [HumanMessage(content=build_planner_prompt(
                question=question,
                paper_title=paper_title,
                abstract=abstract,
            ))]
        )
    except Exception as exc:
        return _fallback(question, f"planner_call_failed: {type(exc).__name__}")
    parsing_error = result.get("parsing_error")
    parsed = result.get("parsed")
    if parsing_error is not None or not isinstance(parsed, RetrievalQueryPlanOutput):
        error_name = type(parsing_error).__name__ if parsing_error else "missing_parsed"
        return _fallback(question, f"planner_parse_failed: {error_name}")
    plan = validate_query_plan(parsed.model_dump(mode="json"), question=question)
    return plan, {"fallback_used": False, "fallback_reason": None}


def _fallback(
    question: str,
    reason: str,
) -> tuple[QueryPlan, dict[str, Any]]:
    return minimal_fallback_plan(question), {
        "fallback_used": True,
        "fallback_reason": reason,
    }
```

Add exact environment parsing and model construction without a second retry layer:

```python
def _int_env(name: str, *, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        relation = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be {relation}")
    return value


def build_retrieval_model() -> BaseChatModel:
    return ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=_int_env(
            "PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS",
            default=4096,
            minimum=1,
        ),
        max_retries=_int_env(
            "PAPERPILOT_RESEARCH_MODEL_RETRIES",
            default=1,
            minimum=0,
        ),
    )
```

Add tests for invalid integer, zero output tokens, and negative retries. Do not catch these configuration errors as planner fallback; configuration is validated at construction/startup, while provider and structured-parse failures use literal fallback.

- [ ] **Step 5: Cap requirements in the validator**

Add:

```python
MAX_EVIDENCE_REQUIREMENTS = 6
```

Apply the cap immediately after `_requirements`: `requirements = _requirements(data.get("evidence_requirements"))[:MAX_EVIDENCE_REQUIREMENTS]`. Build `requirement_ids` from that capped list. `_queries` continues to discard unknown targets, assigns the first retained requirement when a query has no valid target, sorts by priority, removes model-produced literal/duplicate-question queries, prepends the deterministic `q_lit`, and caps the final query list at `MAX_QUERIES`.

- [ ] **Step 6: Write RED verifier structured-output tests**

Define and import the exact verifier contracts below. The fake model's `with_structured_output` must record the requested schema and return a runnable whose `invoke` yields `raw`, `parsed`, and `parsing_error` keys. Assert:

```python
class EvidenceVerificationDecisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    evidence_id: str
    support: Literal["direct", "partial", "no"]
    confidence: Literal["high", "medium", "low"]
    answer_atoms: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    reason: str = ""


class EvidenceVerificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[EvidenceVerificationDecisionOutput]
```

Then assert:

```python
assert fake_model.structured_invoke_count == len(required_requirements)
assert fake_model.structured_invoke_count <= 6
```

Rename `run_planned_retrieval`'s injection from `verifier_client` to `verifier_model` and forward it as `model=verifier_model`. Assert a complete call with `verify_evidence=False` performs zero verifier model calls by passing a fake model whose `with_structured_output` raises if accessed.

- [ ] **Step 7: Implement the verifier LangChain boundary**

Rename the `run_evidence_verification` injection parameter from `client` to `model`, set `verifier_model = model or build_retrieval_model()`, and keep selection/ranking functions unchanged. Replace only the old client call:

```python
runnable = verifier_model.with_structured_output(
    EvidenceVerificationOutput,
    include_raw=True,
)
result = runnable.invoke([HumanMessage(content=prompt)])
parsing_error = result.get("parsing_error")
parsed = result.get("parsed")
if parsing_error is not None or not isinstance(parsed, EvidenceVerificationOutput):
    parse_errors.append(f"{requirement.id}: structured output invalid")
    continue
decisions.extend(_materialize_decisions(parsed.decisions))


def _materialize_decisions(
    outputs: list[EvidenceVerificationDecisionOutput],
) -> list[EvidenceVerificationDecision]:
    decisions: list[EvidenceVerificationDecision] = []
    for output in outputs:
        decision = _normalize_decision(output.model_dump(mode="json"))
        if decision is not None:
            decisions.append(decision)
    return decisions
```

Construct `verifier_model = model or build_retrieval_model()` and its structured runnable once before the requirement loop, then reuse that runnable for each required requirement. On provider failure, return `EvidenceVerificationResult(enabled=True, method="llm_requirement_verifier_v1", verification_error=f"{type(exc).__name__}: {exc}", stats={"decision_count": 0, "direct_count": 0, "partial_count": 0, "no_count": 0})`. Do not add application-level retry loops; the shared provider retry setting remains separate and must be documented as such.

- [ ] **Step 8: Run focused retrieval/MCP GREEN gates and import scan**

```bash
./.venv/bin/python -m pytest tests/retrieval tests/mcp_servers/test_colbert_server.py tests/mcp_servers/test_colbert_server_planned_retrieval.py -q
rg -n "paperpilot\.core|LLMClient" paperpilot/retrieval paperpilot/mcp_servers/colbert
git diff --check
```

Expected: zero failures; no legacy imports; no whitespace errors.

- [ ] **Step 9: Commit**

```bash
git add paperpilot/retrieval tests/retrieval tests/mcp_servers/test_colbert_server_planned_retrieval.py
git commit -m "refactor(retrieval): use LangChain structured planning"
```

---

### Task 3: Add conversation-scoped task updates before removing legacy APIs

**Files:**
- Create: `paperpilot/web/routes/__init__.py`
- Create: `paperpilot/web/routes/task_updates.py`
- Create: `paperpilot/web/schemas.py`
- Modify: `paperpilot/web/task_store.py`
- Modify: `paperpilot/web/app.py`
- Modify: `paperpilot/web/static/conversations.js`
- Modify: `tests/web/test_conversation_store.py`
- Modify: `tests/web/test_conversation_api.py`
- Modify: `tests/web/test_conversation_ui.py`

**Interfaces:**
- Consumes: existing `TaskStore.get_task_updates`, Event/Artifact batches, authenticated `WebUser`, and current incremental cursor parameters.
- Produces: `TaskStore.get_conversation_task_updates(conversation_id: str, task_id: str, *, user_id: str, after_event_id: int = 0, after_artifact_id: int = 0, limit: int = 50) -> TaskUpdates | None`; `GET /api/conversations/{conversation_id}/tasks/{task_id}/updates`.

- [ ] **Step 1: Write Store RED tests for triple binding**

Create a Conversation task and assert:

```python
updates = store.get_conversation_task_updates(
    conversation.id,
    task.id,
    user_id=user.id,
)
assert updates is not None
assert updates.task.id == task.id

assert store.get_conversation_task_updates(
    other_conversation.id,
    task.id,
    user_id=user.id,
) is None
assert store.get_conversation_task_updates(
    conversation.id,
    task.id,
    user_id=other_user.id,
) is None
```

- [ ] **Step 2: Verify Store RED**

```bash
./.venv/bin/python -m pytest tests/web/test_conversation_store.py -k conversation_task_updates -q
```

Expected: `AttributeError` for the missing method.

- [ ] **Step 3: Implement one scoped Store query boundary**

Use one SQLAlchemy transaction/session: call `_select_owned_task_model(session, task_id, user_id)`, reject when the row is absent or `row.conversation_id != conversation_id`, then build `TaskUpdates` from `_task_from_model(row)`, `_read_event_batch(session, task_id, after_id=after_event_id, limit=limit)`, and `_read_artifact_batch(session, task_id, after_id=after_artifact_id, limit=limit)`. Do not implement the endpoint by trusting a Task fetched only by ID.

- [ ] **Step 4: Define update response schemas**

Move/copy the existing Task/Event/Artifact response shapes into `paperpilot/web/schemas.py` as strict Pydantic models:

```python
class TaskUpdatesResponse(StrictApiModel):
    task: TaskResponse
    events: TaskEventPageResponse
    artifacts: TaskArtifactPageResponse
```

Keep current field names and incremental pagination semantics.

- [ ] **Step 5: Write API RED tests**

Assert the new path returns 200 for the owner and 404 for wrong owner, wrong Conversation, wrong Task, or mismatched Conversation/Task. Also assert OpenAPI includes the exact new route.

- [ ] **Step 6: Implement and register `task_updates` router**

Expose:

```python
@router.get(
    "/api/conversations/{conversation_id}/tasks/{task_id}/updates",
    response_model=TaskUpdatesResponse,
)
def get_conversation_task_updates(
    conversation_id: str,
    task_id: str,
    after_event_id: int = Query(default=0, ge=0),
    after_artifact_id: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    user: WebUser = Depends(require_user),
) -> TaskUpdatesResponse:
    updates = store.get_conversation_task_updates(
        conversation_id,
        task_id,
        user_id=user.id,
        after_event_id=after_event_id,
        after_artifact_id=after_artifact_id,
        limit=limit,
    )
    if updates is None:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskUpdatesResponse.model_validate(task_updates_dict(updates))
```

Keep `task_updates_dict` in `routes/task_updates.py`; it returns `task.to_dict()` plus the existing Event/Artifact page fields `items`, `next_after_id`, and `has_more` without renaming them.

- [ ] **Step 7: Switch Conversation UI polling**

Replace the request with:

```javascript
`/api/conversations/${encodeURIComponent(conversationId)}` +
  `/tasks/${encodeURIComponent(taskId)}/updates?${params.toString()}`
```

Do not remove the old route yet; Task 4 removes it after this path is green.

- [ ] **Step 8: Run focused GREEN gates**

```bash
./.venv/bin/python -m pytest tests/web/test_conversation_store.py tests/web/test_conversation_api.py tests/web/test_conversation_ui.py -q
git diff --check
```

- [ ] **Step 9: Commit**

```bash
git add paperpilot/web/routes paperpilot/web/schemas.py paperpilot/web/task_store.py paperpilot/web/app.py paperpilot/web/static/conversations.js tests/web/test_conversation_store.py tests/web/test_conversation_api.py tests/web/test_conversation_ui.py
git commit -m "feat(api): scope task updates to conversations"
```

---

### Task 4: Make FastAPI composition Conversation-only

**Files:**
- Create: `paperpilot/web/routes/auth.py`
- Create: `paperpilot/web/routes/papers.py`
- Create: `paperpilot/web/routes/conversations.py`
- Modify: `paperpilot/web/routes/task_updates.py`
- Modify: `paperpilot/web/schemas.py`
- Modify: `paperpilot/web/app.py`
- Delete: `paperpilot/web/conversation_routes.py`
- Delete: `paperpilot/web/eval_summary.py`
- Delete: `paperpilot/web/pagination.py`
- Delete: `scripts/benchmark_web_admission.py`
- Modify: `tests/web/test_auth.py`
- Modify: `tests/web/test_conversation_api.py`
- Modify: `tests/web/test_conversation_ui.py`
- Modify: `tests/web/test_runtime_config.py`
- Modify: `tests/web/test_web_app.py`
- Delete: `tests/web/test_eval_summary.py`
- Delete: `tests/web/test_pagination.py`

**Interfaces:**
- Consumes: AuthService, TaskStore, TaskExecutorLike, DeepReadingRunner, paper search callable, and Task 3 updates router.
- Produces: `build_auth_router`, `build_paper_router`, `build_conversation_router`, `build_task_updates_router`; `create_app` with no legacy `simulation_delay_seconds` or `workflow_runner` arguments; OpenAPI contains only auth/paper/conversation/health APIs.

- [ ] **Step 1: Write the OpenAPI RED architecture test**

Add to `tests/web/test_web_app.py`:

```python
def test_openapi_exposes_only_new_product_routes(client) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert not any(path == "/api/tasks" or path.startswith("/api/tasks/") for path in paths)
    assert not any(path == "/api/eval" or path.startswith("/api/eval/") for path in paths)
    assert "/api/conversations/{conversation_id}/tasks/{task_id}/updates" in paths
```

- [ ] **Step 2: Verify RED**

```bash
./.venv/bin/python -m pytest tests/web/test_web_app.py -k new_product_routes -q
```

Expected: old `/api/tasks` and `/api/eval` paths violate assertions.

- [ ] **Step 3: Move Pydantic schemas without changing fields**

Move all auth, paper, Conversation, Message, rollback, alternatives, Task/Event/Artifact page schemas to `web/schemas.py`. Every model inherits:

```python
class StrictApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
```

Keep current response field names so the Conversation UI and existing clients do not change beyond the updates URL.

- [ ] **Step 4: Extract framework-native routers**

Each router file exposes one builder with explicit dependencies:

- `build_auth_router(*, auth: AuthService, require_user: RequireUser) -> APIRouter`, prefix `/api/auth`, containing the existing register, login, logout, and me handlers.
- `build_paper_router(*, require_user: RequireUser, paper_search: PaperSearch) -> APIRouter`, containing only `GET /api/papers/search`.
- `build_conversation_router(*, store: TaskStore, executor: TaskExecutorLike, require_user: RequireUser, deep_reading_runner: DeepReadingRunner, paper_search: PaperSearch) -> APIRouter`, containing create/list/detail/update, message creation/listing, alternatives, and rollback. The Conversation and paper routers receive the same injected `paper_search` callable so Conversation creation can resolve its paper reference without a hidden/default network dependency.
- `build_task_updates_router(*, store: TaskStore, require_user: RequireUser) -> APIRouter`, containing only the scoped task-updates endpoint.

Move the current handler bodies without changing validation or response fields. Builders close over only their explicit arguments and do not read module-global app state.

`papers.py` owns `default_web_paper_search` and `/api/papers/search`. `conversations.py` owns create/list/detail/update/messages/alternatives/rollback. `task_updates.py` owns only the scoped updates route.

- [ ] **Step 5: Reduce `app.py` to composition/lifecycle/health/static hosting**

`create_app` keeps exactly these injectable dependencies and delegates to a typed private composition function without accepting compatibility argument shapes:

```python
def create_app(
    task_store: TaskStore | None = None,
    *,
    task_executor: TaskExecutorLike | None = None,
    runtime_config: WebRuntimeConfig | None = None,
    checkpoint_runtime: SqliteCheckpointRuntime | None = None,
    mcp_runtime: MCPRuntime | None = None,
    deep_reading_runner: DeepReadingRunner | None = None,
    paper_search: PaperSearch | None = None,
) -> FastAPI:
    owned_resources = _OwnedAppResources()
    try:
        return _create_app(
            task_store=task_store,
            task_executor=task_executor,
            runtime_config=runtime_config,
            checkpoint_runtime=checkpoint_runtime,
            mcp_runtime=mcp_runtime,
            deep_reading_runner=deep_reading_runner,
            paper_search=paper_search,
            owned_resources=owned_resources,
        )
    except BaseException:
        _best_effort_construction_cleanup(owned_resources)
        raise
```

The private `_create_app` creates concrete defaults, the four routers, lifespan, health endpoints, middleware, and static mount; it registers no task/eval compatibility routes.

Remove `simulation_delay_seconds`, `workflow_runner`, legacy task/eval response helpers, cursor imports, and all `/api/tasks`/`/api/eval` decorators.

- [ ] **Step 6: Delete superseded route/eval/pagination modules, legacy admission benchmark, and tests**

Delete only the listed exact files after imports have moved. Delete `scripts/benchmark_web_admission.py` because it posts exclusively to the removed `/api/tasks` product path; remove its two legacy benchmark tests from `tests/web/test_runtime_config.py`. Rewrite `test_web_app.py` to cover app lifecycle, static root, health, middleware, and OpenAPI; legacy endpoint assertions are removed rather than inverted into compatibility tests. In `tests/web/test_conversation_ui.py`, remove only the deleted `workflow_runner` keyword from `create_app`; Task 6 owns the UI rewrite.

- [ ] **Step 7: Run Web GREEN gates**

```bash
./.venv/bin/python -m pytest tests/web/test_auth.py tests/web/test_conversation_api.py tests/web/test_conversation_ui.py tests/web/test_runtime_config.py tests/web/test_web_app.py tests/web/test_observability.py -q
./.venv/bin/python -c 'from paperpilot.web.app import app; paths=set(app.openapi()["paths"]); assert not any(p.startswith("/api/tasks") or p.startswith("/api/eval") for p in paths)'
git diff --check
```

- [ ] **Step 8: Commit**

```bash
git add paperpilot/web scripts/benchmark_web_admission.py tests/web
git commit -m "refactor(web): expose only Conversation APIs"
```

---

### Task 5: Make Thread and Celery execute DeepReadingRunner directly

**Files:**
- Modify: `paperpilot/deep_reading/runner.py`
- Modify: `paperpilot/web/task_executor.py`
- Modify: `paperpilot/web/worker_tasks.py`
- Modify: `paperpilot/web/celery_app.py`
- Modify: `paperpilot/web/app.py`
- Modify: `paperpilot/web/routes/conversations.py`
- Delete: `paperpilot/web/workflow.py`
- Modify: `tests/deep_reading/test_runner.py`
- Modify: `tests/web/test_auth.py`
- Modify: `tests/web/test_task_executor.py`
- Modify: `tests/web/test_conversation_worker.py`
- Modify: `tests/web/test_celery_worker.py`
- Modify: `tests/web/test_conversation_api.py`
- Modify: `tests/web/test_conversation_ui.py`
- Modify: `tests/web/test_web_app.py`
- Delete: `tests/web/test_workflow.py`

**Interfaces:**
- Consumes: TaskStore atomic `claim_task`, DeepReadingRunner terminal/transient semantics, Task retry configuration, TaskSubmissionReservation capacity ownership.
- Produces: `DeepReadingRunner.run(task_id: str, *, allow_running: bool = False) -> bool`; `TaskSubmissionReservation.submit(task_id: str)`; no `ExecutionMode`, `WorkflowRunnerLike`, simulated path, or legacy runner.

- [ ] **Step 1: Write Runner RED tests for atomic claim ownership**

Add tests asserting:

```python
assert runner.run(task.id) is True
assert runner.run(task.id) is False  # completed/claimed work is not duplicated
```

For a pending task, mock `TaskStore.claim_task` and assert claim occurs before `graph.invoke`. For a Celery redelivery fixture in `running`, assert `runner.run(task.id, allow_running=True)` may recover; `allow_running=False` returns `False`.

- [ ] **Step 2: Verify RED**

```bash
./.venv/bin/python -m pytest tests/deep_reading/test_runner.py -k atomic_claim -q
```

- [ ] **Step 3: Move claim into DeepReadingRunner**

Implement:

```python
def run(self, task_id: str, *, allow_running: bool = False) -> bool:
    claimed = self._task_store.claim_task(task_id, allow_running=allow_running)
    if claimed is None:
        existing = self._task_store.get_task(task_id)
        if existing is None:
            raise ValueError(f"task not found: {task_id}")
        return False
    if claimed.conversation_id is None:
        raise TaskBindingError("task is not bound to a Conversation")
    try:
        self._run(task_id)
    except DeepReadingTaskError as exc:
        self._fail_terminal(task_id, exc)
    return True
```

`ResearchTask` intentionally has no duplicated `user_message_id` column. Keep the existing `_load_business_binding` lookup as the authoritative validation that the claimed Task has a bound user Message in the same Conversation. Keep infrastructure exceptions escaping. Extract the current terminal logging/failure body to `_fail_terminal`; do not broad-catch unknown exceptions.

- [ ] **Step 4: Write Executor RED tests for mode-free submission**

Tests must call:

```python
reservation.submit("task_1")
executor.submit("task_2")
```

and assert the fake runner receives only task IDs. Assert passing a second positional mode raises `TypeError`.

- [ ] **Step 5: Simplify Executor protocols and retry loop**

Define:

```python
class ConversationTaskRunnerLike(Protocol):
    def run(self, task_id: str, *, allow_running: bool = False) -> bool:
        """Claim and execute one Conversation-bound task."""

    def fail_retry_exhausted(
        self,
        task_id: str,
        *,
        backend: str,
        attempts: int,
        max_retries: int,
        exc: Exception,
    ) -> None:
        """Persist retry exhaustion unless the task already completed."""
```

Remove `ExecutionMode`. `Submitter` becomes `Callable[[str], object]`. Thread attempt zero calls `runner.run(task_id)`; retry attempts call `runner.run(task_id, allow_running=True)` so a transient infrastructure failure does not strand the already claimed Task in `running`. Retry only exceptions that escape the Runner. Preserve Future capacity release and logging. Tests may record the `allow_running` keyword separately, but the public submitter receives only a Task ID.

- [ ] **Step 6: Move retry-exhaustion finalization to DeepReadingRunner**

Move the current idempotent completed-race logic from WorkflowRunner into `DeepReadingRunner.fail_retry_exhausted(task_id: str, *, backend: str, attempts: int, max_retries: int, exc: Exception) -> None`. It must call `TaskStore.fail_conversation_task` with stage `execution_retry_exhausted`, and treat an already completed Conversation Task as success.

- [ ] **Step 7: Simplify Celery payload and Worker**

Celery sends:

```python
sender.send_task(
    EXECUTE_RESEARCH_TASK_NAME,
    args=[task_id],
    task_id=task_id,
)
```

Worker signature becomes:

```python
def execute_research_task(self, task_id: str) -> None:
```

Build one TaskStore, checkpoint runtime, MCP runtime, and DeepReadingRunner; call `runner.run(task_id, allow_running=redelivered or retries > 0)`. Keep current typed retry signal, backoff, retry exhaustion, cleanup precedence, process-init reset, and process-shutdown close semantics.

- [ ] **Step 8: Update Conversation submission**

Change `reservation.submit(turn.task.id, "real")` to `reservation.submit(turn.task.id)`. Delete every simulated/real mode assertion from retained new API tests and remove new Conversation event writes containing `execution_mode`. The legacy `TaskStore.create_queued_task` method and its historical tests remain until Task 7; they are not a new product write path.

Delete `paperpilot/web/workflow.py` and `tests/web/test_workflow.py` in this Task after migrating every retained direct caller to `DeepReadingRunner` or a narrow fake implementing `ConversationTaskRunnerLike`. Update app/auth/UI/Web tests that constructed `WorkflowRunner` only as setup. This deletion is intentionally moved forward from Task 7 so the Task 5 runtime boundary and full suite can be green together.

- [ ] **Step 9: Run Thread/Celery/Runner GREEN gates**

```bash
./.venv/bin/python -m pytest tests/deep_reading/test_runner.py tests/web/test_task_executor.py tests/web/test_conversation_worker.py tests/web/test_celery_worker.py tests/web/test_conversation_api.py -q
rg -n "ExecutionMode|run_simulated|WorkflowRunner" paperpilot/deep_reading paperpilot/web
rg -n "execution_mode" paperpilot/deep_reading paperpilot/web --glob '*.py' --glob '!task_store.py'
git diff --check
```

Expected: no direct legacy runner/mode symbols in retained code and no Python runtime `execution_mode` write outside the explicitly deferred legacy `TaskStore` method. The old static workbench is removed in Task 6; the legacy TaskStore creation method and tests are removed in Task 7. Historical database fields remain readable, but no new Conversation product write includes mode.

- [ ] **Step 10: Commit**

```bash
git add paperpilot/deep_reading/runner.py paperpilot/web tests/deep_reading/test_runner.py tests/web
git commit -m "refactor(runtime): execute Conversation tasks directly"
```

---

### Task 6: Consolidate the frontend into one Conversation workspace

**Files:**
- Modify: `paperpilot/web/static/index.html`
- Modify: `paperpilot/web/static/app.js`
- Modify: `paperpilot/web/static/styles.css`
- Delete: `paperpilot/web/static/conversations.js`
- Delete: `paperpilot/web/static/conversations.css`
- Modify: `tests/web/test_conversation_ui.py`
- Modify: `tests/web/test_web_app.py`

**Interfaces:**
- Consumes: auth, paper search, Conversation CRUD/messages/alternatives/rollback, and conversation-scoped task updates APIs.
- Produces: exactly three static assets; no tabs or legacy API strings; current login/session and Conversation behaviors preserved.

- [ ] **Step 1: Write RED static allowlist tests**

Assert:

```python
assert sorted(path.name for path in STATIC_DIR.iterdir()) == [
    "app.js", "index.html", "styles.css"
]
assert "Legacy Workbench" not in html
assert "legacyTab" not in html + javascript
assert "/api/tasks" not in javascript
assert "/api/eval" not in javascript
assert "conversations.js" not in html
assert "conversations.css" not in html
```

Keep assertions for login/register/logout, paper search, create/list Conversation, messages, alternatives, rollback, full Assistant content, and scoped updates URL.

- [ ] **Step 2: Verify RED**

```bash
./.venv/bin/python -m pytest tests/web/test_conversation_ui.py -q
```

Expected: legacy tab/assets/API strings still exist.

- [ ] **Step 3: Rewrite `index.html` as one workspace**

Keep auth forms and Conversation DOM IDs consumed by the current new UI. Remove `legacyTab`, task form/list, eval snapshot, and both old split-asset references. Load only:

```html
<link rel="stylesheet" href="/static/styles.css">
<script src="/static/app.js" defer></script>
```

- [ ] **Step 4: Merge JavaScript around one state object**

Use one top-level state:

```javascript
const state = {
  currentUser: null,
  conversations: [],
  selectedConversationId: null,
  selectedConversation: null,
  activeTaskPoll: null,
};
```

Move retained `requestJson`, authentication message/rendering, `loadCurrentUser`, `submitAuth`, register/login/logout listeners, date formatting, and escaping behavior from old `app.js`. Move the complete Conversation function set from `conversations.js`: paper search/create, Conversation list/detail selection, message rendering/submission, progress polling, conflict refresh, alternatives, rollback, workspace reset, and mutation-control gating. Delete all generic task/eval rendering and polling functions. Replace custom auth window events with direct `setAuthenticatedUser(user)`/`clearAuthenticatedUser()` calls because only one script remains. Keep `authenticationVersion`, `conversationListVersion`, and per-selection `requestVersion` guards so stale Conversation/auth responses cannot repaint current state.

- [ ] **Step 5: Merge styles and preserve native hidden behavior**

Start `styles.css` with:

```css
[hidden] {
  display: none !important;
}
```

Merge auth/global and Conversation styles; delete every legacy tab/task/eval selector. Preserve responsive message layout and visible error/status styling.

- [ ] **Step 6: Delete split Conversation assets and run GREEN tests**

Update the existing static-serving assertion in `tests/web/test_web_app.py` so it requests only `app.js` and `styles.css`; the deleted split assets must not remain as expected-200 resources.

```bash
./.venv/bin/python -m pytest tests/web/test_conversation_ui.py tests/web/test_auth.py tests/web/test_web_app.py -q
rg -n "Legacy Workbench|legacyTab|/api/tasks|/api/eval|conversations\.(js|css)" paperpilot/web/static tests/web/test_conversation_ui.py
git diff --check
```

Expected: zero failures and no matches.

- [ ] **Step 7: Browser smoke without model calls**

Run the Web app with temporary business/checkpoint SQLite paths and a fake paper search/runner fixture. In the real browser verify: register, login, default Conversation workspace, search results, create Conversation, list/detail/messages rendering, logout, no console errors, and only three static assets requested.

- [ ] **Step 8: Commit**

```bash
git add paperpilot/web/static tests/web/test_conversation_ui.py tests/web/test_web_app.py
git commit -m "refactor(ui): keep only the Conversation workspace"
```

---

### Task 7: Delete legacy runtime and trim TaskStore to new call sites

**Files:**
- Delete: `paperpilot/agent/`
- Delete: `paperpilot/builtin_tools/`
- Delete: `paperpilot/core/`
- Delete: `paperpilot/main.py`
- Delete: `paperpilot/conversation.py`
- Delete: `paperpilot/bulk_input.py`
- Delete: `paperpilot/document_store.py`
- Delete: `paperpilot/message_codec.py`
- Delete: `paperpilot/session_store.py`
- Delete: `paperpilot/eval/`
- Delete: `paperpilot/web/event_mapper.py`
- Modify: `paperpilot/web/task_store.py`
- Modify: retained tests importing `paperpilot.tools.types.Tool`
- Delete: `tests/agent/`
- Delete: `tests/builtin_tools/`
- Delete: `tests/eval/`
- Delete: `tests/test_agent_loop.py`
- Delete: `tests/test_bulk_input.py`
- Delete: `tests/test_context_manager.py`
- Delete: `tests/test_conversation_session.py`
- Delete: `tests/test_deep_read_skill.py`
- Delete: `tests/test_document_store.py`
- Delete: `tests/test_main_integration.py`
- Delete: `tests/test_message_codec.py`
- Delete: `tests/test_session_store.py`
- Delete: `tests/web/test_event_mapper.py`

**Interfaces:**
- Consumes: Tasks 1–6 have removed every retained production dependency on these paths.
- Produces: no legacy runtime modules; TaskStore exposes only auth, Conversation/Message/Paper, Conversation Task lifecycle, event/artifact, and scoped update operations.

- [ ] **Step 1: Add RED architecture-boundary tests**

Create `tests/architecture/test_new_stack_only.py` with exact forbidden modules:

```python
FORBIDDEN_MODULES = {
    "paperpilot.agent",
    "paperpilot.builtin_tools",
    "paperpilot.core",
    "paperpilot.main",
    "paperpilot.conversation",
    "paperpilot.bulk_input",
    "paperpilot.document_store",
    "paperpilot.message_codec",
    "paperpilot.session_store",
    "paperpilot.web.workflow",
    "paperpilot.web.event_mapper",
    "paperpilot.web.eval_summary",
    "paperpilot.web.pagination",
}


def test_forbidden_legacy_modules_are_absent() -> None:
    for name in FORBIDDEN_MODULES:
        assert importlib.util.find_spec(name) is None
```

Add an AST/import-string scan over retained `paperpilot/**/*.py` asserting none of these module prefixes appears.

- [ ] **Step 2: Verify RED**

```bash
./.venv/bin/python -m pytest tests/architecture/test_new_stack_only.py -q
```

Expected: legacy modules still resolve.

- [ ] **Step 3: Generate and review the retained TaskStore call set**

Run:

```bash
rg -o "(?:store|task_store|self\._task_store)\.[A-Za-z_][A-Za-z0-9_]*" paperpilot/deep_reading paperpilot/web tests/deep_reading tests/web | sort -u
```

The retained public Store methods must cover auth/session; Conversation/Paper/Message; `claim_task`; pending compensation; Event/Artifact; scoped updates; publish/finalize/fail; rollback. Methods used only by deleted legacy tests/routes—`create_task`, `create_queued_task`, `list_tasks_page`, generic `update_status`, and unscoped `get_task_updates`—must be removed unless the scan proves a retained caller.

- [ ] **Step 4: Delete exact legacy runtime and test targets**

Use `apply_patch`/explicit tracked targets. Delete `paperpilot/eval/` and `tests/eval/` here instead of adapting their imports away from the removed runtime only to delete them in Task 8. Do not target `paperpilot/tools`, `mcp_servers`, `retrieval`, `deep_reading`, `web` data/config/migrations, or any `data/` path in this Task.

- [ ] **Step 5: Trim TaskStore and preserve database rows/schema**

Delete only methods with no retained caller. Keep `ResearchTask` mapping tolerant of nullable legacy columns. Do not modify `db_models.py` column nullability or migration files. Ensure new `create_conversation_turn` is the only Task creation method in production.

- [ ] **Step 6: Run architecture/import and retained shared tests**

```bash
./.venv/bin/python -m pytest tests/architecture tests/deep_reading tests/papers tests/mcp_servers tests/retrieval tests/test_mcp_client.py tests/test_mcp_runtime.py tests/web -q
rg -n "paperpilot\.(agent|builtin_tools|core|main|conversation|bulk_input|document_store|message_codec|session_store)|WorkflowRunner|/api/tasks|/api/eval" paperpilot tests
git diff --check
```

Expected: zero failures; no forbidden production/test references.

- [ ] **Step 7: Commit**

```bash
git add paperpilot tests
git commit -m "refactor(core): remove legacy PaperPilot runtime"
```

---

### Task 8: Remove historical eval/research assets and enforce repository allowlists

**Files:**
- Delete: `paperpilot/skills/`
- Delete: all tracked `scripts/day*.py`
- Delete: `data/eval/.gitkeep`
- Delete: `data/eval/summary.md`
- Delete: `data/traces/.gitkeep`
- Delete: all tracked files under `docs/superpowers/`
- Delete: historical tracked files under `docs/` except the explicit keep allowlist below
- Create: `tests/architecture/test_repository_allowlist.py`

**Interfaces:**
- Consumes: retained runtime/tests are green after Task 7.
- Produces: current checkout has no eval package, prompt-skill archive, Day scripts, eval/traces tracked artifacts, or historical docs; repository allowlist prevents them from returning.

**Document keep allowlist:**

```text
docs/codex-only-plans/README.md
docs/codex-only-plans/2026-08-07-paperpilot-conversation-langgraph-vertical-slice-design.md
docs/codex-only-plans/2026-08-07-paperpilot-conversation-langgraph-vertical-slice-plan.md
docs/codex-only-plans/2026-08-11-new-architecture-cleanup-design.md
docs/codex-only-plans/2026-08-11-new-architecture-cleanup-implementation-plan.md
```

- [ ] **Step 1: Inventory untracked eval/trace files and stop if any exist**

```bash
git status --short --untracked-files=all -- data/eval data/traces
find data/eval data/traces -type f -print 2>/dev/null | sort
```

If any untracked path exists, present the exact list to the user and obtain a fresh deletion confirmation. Do not continue on an assumption. Tracked `.gitkeep`/`summary.md` are already authorized by the approved design.

- [ ] **Step 2: Write RED repository allowlist tests**

Test exact absence/presence:

```python
def test_repository_contains_no_historical_runtime_or_eval_groups() -> None:
    root = Path(__file__).parents[2]
    for relative in (
        "paperpilot/eval",
        "paperpilot/skills",
        "tests/eval",
        "data/eval",
        "data/traces",
        "docs/superpowers",
    ):
        assert not (root / relative).exists()
    assert not list((root / "scripts").glob("day*.py"))
```

Add a docs assertion comparing tracked/current doc files to the five-entry keep allowlist.

- [ ] **Step 3: Verify RED**

```bash
./.venv/bin/python -m pytest tests/architecture/test_repository_allowlist.py -q
```

- [ ] **Step 4: Delete the exact tracked historical groups**

Remove the listed package/test/script/data/doc targets. Keep `scripts/benchmark_web_task_store.py` until Task 9 decides its new-stack value by import/run validation; the legacy admission benchmark was already removed with its `/api/tasks` dependency in Task 4. Do not remove migration history, fixtures, compose file, requirements, README, or protected runtime data.

- [ ] **Step 5: Run retained suite and allowlist GREEN gate**

```bash
./.venv/bin/python -m pytest tests -q
./.venv/bin/python -m pytest tests/architecture/test_repository_allowlist.py -q
git diff --check
```

Expected: zero failures; repository allowlist passes.

- [ ] **Step 6: Commit**

```bash
git add paperpilot tests scripts data/eval data/traces docs
git commit -m "chore(repo): remove historical eval and legacy assets"
```

---

### Task 9: Clean dependencies, operational scripts, and README

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-lock.txt`
- Modify: `.env.example`
- Modify: `README.md`
- Modify or Delete: `scripts/benchmark_web_task_store.py`
- Modify: `tests/deep_reading/test_dependency_contract.py`
- Modify: `tests/architecture/test_repository_allowlist.py`

**Interfaces:**
- Consumes: final retained imports and APIs from Tasks 1–8.
- Produces: a lock file with no legacy-only dependency, a new-architecture-only README/env example, and only runnable new-stack operational scripts.

- [ ] **Step 1: Write RED dependency/configuration tests**

Extend dependency tests:

```python
def test_legacy_anthropic_sdk_is_not_a_direct_requirement() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert not any(
        line.strip().startswith("anthropic")
        for line in requirements.splitlines()
    )
```

Extend architecture tests to parse the variable names from `.env.example`. Assert the active runtime keys listed in Step 5 are present and the removed keys `DEFAULT_MODEL`, `MAX_ITERATIONS`, and `BUDGET_TOKENS` are absent. Do not add pytest assertions over README prose; verify human-facing documentation with the explicit one-time checks in Steps 6–7 and Task 10 instead.

- [ ] **Step 2: Verify RED**

```bash
./.venv/bin/python -m pytest tests/deep_reading/test_dependency_contract.py tests/architecture -q
```

- [ ] **Step 3: Remove legacy-only direct dependencies and regenerate lock**

Remove `anthropic>=0.40.0`. Keep `python-dotenv`, MCP/arXiv, LangGraph/LangChain/DeepSeek, FastAPI/Uvicorn, Celery/Redis, SQLAlchemy/Alembic, PyMuPDF/PyLate, NetworkX/httpx, and DashScope because retained modules import them.

Regenerate with the project’s uv environment:

```bash
uv pip compile requirements.txt -o requirements-lock.txt
uv pip sync requirements-lock.txt --python .venv/bin/python
uv pip check --python .venv/bin/python
```

Expected: compatible environment; `anthropic` absent from direct and resolved requirements unless a retained transitive dependency demonstrably requires it.

- [ ] **Step 4: Validate or remove the retained TaskStore benchmark**

Run `scripts/benchmark_web_task_store.py --help` against current imports. If it imports deleted generic TaskStore methods, delete it; if it benchmarks retained Conversation/TaskStore paths, update names and fixtures to create Conversation-bound tasks. Do not keep the script merely because it predates cleanup.

- [ ] **Step 5: Rewrite `.env.example`**

Keep only active variables:

```text
DEEPSEEK_API_KEY
DASHSCOPE_API_KEY
S2_API_KEY
MCP_TOOL_TIMEOUT
PAPERPILOT_TASK_DB_PATH
PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH
LANGGRAPH_STRICT_MSGPACK
PAPERPILOT_TASK_EXECUTOR
PAPERPILOT_THREAD_WORKERS
PAPERPILOT_THREAD_QUEUE_CAPACITY
PAPERPILOT_OVERLOAD_RETRY_AFTER_SECONDS
PAPERPILOT_TASK_MAX_RETRIES
PAPERPILOT_TASK_RETRY_BACKOFF_SECONDS
PAPERPILOT_TASK_RETRY_BACKOFF_MAX_SECONDS
PAPERPILOT_CELERY_BROKER_URL
PAPERPILOT_TASK_SOFT_TIME_LIMIT_SECONDS
PAPERPILOT_TASK_TIME_LIMIT_SECONDS
PAPERPILOT_REDIS_VISIBILITY_TIMEOUT_SECONDS
PAPERPILOT_SUMMARY_TOKEN_THRESHOLD
PAPERPILOT_SUMMARY_RECENT_TURNS
PAPERPILOT_RESEARCH_RECURSION_LIMIT
PAPERPILOT_RESEARCH_MODEL_CALL_LIMIT
PAPERPILOT_RESEARCH_TOOL_CALL_LIMIT
PAPERPILOT_RESEARCH_MAX_OUTPUT_TOKENS
PAPERPILOT_RESEARCH_MODEL_RETRIES
```

Remove `DEFAULT_MODEL`, `MAX_ITERATIONS`, and `BUDGET_TOKENS` because they belonged to the old Agent Loop.

- [ ] **Step 6: Rewrite README around the single architecture**

README must include:

1. One architecture chain: FastAPI → Conversation → Executor → DeepReadingRunner → LangGraph → LangChain → MCP.
2. Business DB versus checkpoint DB ownership.
3. Thread and Celery startup order using the same explicit paths.
4. New API list, including conversation-scoped updates; no legacy routes.
5. Continuous follow-up, alternatives, and rollback semantics.
6. Four MCP extensions and ColBERT local model behavior.
7. Agent budget plus the explicit nested retrieval cost: at most one planner structured-output invoke per `retrieve_paper_evidence`; verifier default off and capped at six requirement invokes; provider `max_retries=1` can add one transport attempt after a transient failure.
8. Migration, backup, health, retry, and restore operations.
9. Automated fake-model scope and the separate approval requirement for real-model smoke.

- [ ] **Step 7: Run dependency/docs/full GREEN gates**

```bash
./.venv/bin/python -m pytest tests/deep_reading/test_dependency_contract.py tests/architecture -q
./.venv/bin/python -m pytest tests -q
uv pip check --python .venv/bin/python
rg -n "Legacy Workbench|/api/tasks|/api/eval|agent_loop|ConversationSession|execution_mode|DEFAULT_MODEL|MAX_ITERATIONS|BUDGET_TOKENS" README.md .env.example paperpilot tests scripts
git diff --check
```

Expected: zero test failures, compatible dependencies, and no forbidden matches.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt requirements-lock.txt .env.example README.md scripts tests/deep_reading/test_dependency_contract.py tests/architecture
git commit -m "docs(runtime): describe only the LangGraph service"
```

---

### Task 10: Final persistence, runtime, browser, and review gates

**Files:**
- Modify only if a gate reveals an in-scope defect; otherwise no production changes.
- Record: `.superpowers/sdd/2026-08-11-new-architecture-cleanup-implementation-plan/progress.md` (ignored execution ledger).

**Interfaces:**
- Consumes: completed Tasks 1–9.
- Produces: review-ready branch with protected data unchanged, clean worktree, no legacy architecture, and fresh evidence for every acceptance criterion.

- [ ] **Step 1: Run final architecture and complete test suites**

```bash
./.venv/bin/python -m pytest tests/architecture -q
LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m pytest tests -q
git diff --check codex/goal_test..HEAD
```

Expected: zero failures, no new warnings beyond known third-party warnings, diff check exit 0.

- [ ] **Step 2: Verify fresh Alembic database**

Use a newly created temporary directory, never `data/web/`:

```bash
PAPERPILOT_TASK_DB_PATH=/private/tmp/paperpilot-cleanup-business.sqlite3 ./.venv/bin/python -m alembic -c alembic.ini upgrade head
sqlite3 'file:/private/tmp/paperpilot-cleanup-business.sqlite3?mode=ro' 'SELECT version_num FROM alembic_version;'
```

Expected: `20260807_0002`. Remove the exact temporary file only after verifying its resolved path begins with `/private/tmp/paperpilot-cleanup-`.

- [ ] **Step 3: Verify fresh checkpoint database**

```bash
PAPERPILOT_LANGGRAPH_CHECKPOINT_DB_PATH=/private/tmp/paperpilot-cleanup-checkpoints.sqlite3 LANGGRAPH_STRICT_MSGPACK=true ./.venv/bin/python -m paperpilot.web.checkpoint --setup
sqlite3 'file:/private/tmp/paperpilot-cleanup-checkpoints.sqlite3?mode=ro' "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('checkpoints','writes') ORDER BY name;"
```

Expected: `checkpoints`, `writes`.

- [ ] **Step 4: Re-run protected-data read-only checks**

Repeat the exact Pre-execution Step 4 commands against `/Users/patrick/Projects/paperpilot/data/web/tasks.sqlite3` and `/Users/patrick/Projects/paperpilot/data/langgraph/checkpoints.sqlite3`. Compare sizes and quick-check results with the ledger. Database files may only differ if the user independently ran the app during implementation; if so, stop and ask before attributing the change.

- [ ] **Step 5: Run dependency and MCP startup gates**

```bash
uv pip check --python .venv/bin/python
./.venv/bin/python -c 'from dotenv import load_dotenv; load_dotenv(); from paperpilot.tools.mcp_runtime import MCPRuntime; runtime=MCPRuntime(); runtime.start(); names=sorted(tool.name for tool in runtime._client.list_tools()); print(names); runtime.close()'
```

Expected MCP tools include arXiv search/download, ColBERT build/search/planned retrieval, graph neighbors/common citations/shortest path, and VLM page understanding. This gate lists tools only and performs no external tool/model call.

- [ ] **Step 6: Run Web health and browser smoke on temporary databases**

Start one Uvicorn process with explicit temporary business/checkpoint paths and `PAPERPILOT_TASK_EXECUTOR=thread`. Verify `/health/live` and `/health/ready` return 200. In browser verify registration, login, paper-search UI rendering with a fake/injected search boundary, Conversation creation/list/detail, message page, and logout; do not submit real research work.

- [ ] **Step 7: Audit the final filesystem allowlist**

```bash
find paperpilot -maxdepth 2 -type f -not -path '*/__pycache__/*' | sort
find tests -maxdepth 2 -type f -not -path '*/__pycache__/*' | sort
find scripts -maxdepth 2 -type f -not -path '*/__pycache__/*' | sort
find docs -type f | sort
git status --short
```

Expected: only plan-approved new architecture groups; no dirty/untracked project files.

- [ ] **Step 8: Generate a whole-task review package and request independent review**

Use base commit recorded before Task 1 and current HEAD. Reviewer must check:

- protected data untouched;
- no legacy imports/routes/UI/files;
- scoped updates owner/conversation/task isolation;
- direct Thread/Celery claim/retry/idempotency behavior;
- retrieval LangChain structured call limits/fallback;
- fresh migration/checkpoint and four MCP servers;
- README/requirements match runtime;
- no Critical or Important regression.

If review finds an issue, fix only that issue with TDD, rerun its focused tests and the full suite, commit, regenerate the package, and re-review. Maximum five fix rounds.

- [ ] **Step 9: Final branch status**

```bash
git status --short
git log --oneline codex/goal_test..HEAD
git diff --stat codex/goal_test..HEAD
git diff --check codex/goal_test..HEAD
```

Expected: clean worktree, intentional commits only, clean diff. Do not merge, push, delete the worktree, or delete the feature branch until the user chooses an integration option through `superpowers:finishing-a-development-branch`.

---

## Spec-to-Task Coverage Map

| Approved requirement | Task |
| --- | --- |
| Preserve business/checkpoint/paper/index data | Pre-execution, 10 |
| External backup of untracked architecture plan | Pre-execution |
| Move Tool out of legacy core | 1 |
| LangChain/Pydantic retrieval planner/verifier with explicit limits | 2 |
| Conversation-scoped updates and triple ownership | 3 |
| Remove `/api/tasks`, `/api/eval`, split FastAPI routers/schemas | 4 |
| Remove WorkflowRunner/modes, direct Thread/Celery DeepReadingRunner | 5 |
| One Conversation UI and three static assets | 6 |
| Delete old CLI/Agent Loop/runtime while preserving DB rows/schema | 7 |
| Remove eval/Day/history assets, retain Git history | 8 |
| Keep four MCP extensions | 1, 2, 7, 10 |
| Remove legacy dependency/config/docs | 9 |
| Fresh tests/migration/checkpoint/MCP/health/browser/review | 10 |

## Stop Conditions

- Any protected data path resolves outside the exact approved files or changes during an automated step.
- External backup target already exists or backup digest cannot be verified.
- A retained new-stack module still imports a file scheduled for deletion.
- Scoped updates can expose a Task across owner or Conversation boundaries.
- Thread/Celery direct execution loses atomic claim, completed-race, bounded retry, terminal/transient identity, or idempotent publication behavior.
- Retrieval planner/verifier performs more calls than the documented limits or silently reintroduces the old LLM client.
- Fresh Alembic/checkpoint setup fails or attempts a destructive migration.
- Four-server MCP startup fails after legacy dependency removal.
- Full retained suite has a regression that cannot be explained and fixed within scope.
- Untracked eval/trace files are present without a fresh exact-path deletion confirmation.
