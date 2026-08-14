# PaperPilot Web Business Module Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `paperpilot/web/task_store.py` 与 Conversation Router 按业务职责拆分，同时保持所有公开 API、数据库 schema、事务、并发和 checkpoint 行为不变，并补充便于阅读的业务注释。

**Architecture:** 新增 `paperpilot.web.store` 包，将现有 SQLAlchemy 操作整理为接收共享 `SessionFactory` 的业务函数；`paperpilot.web.task_store.TaskStore` 保留为显式兼容门面。Conversation 路由改为同名包并由 `router.py` 组装 CRUD、消息和回滚子路由；测试按相同业务边界移动，不删除覆盖。

**Tech Stack:** Python 3.12、SQLAlchemy 2.x、Alembic、FastAPI、Pydantic 2、LangGraph、pytest、SQLite。

## Global Constraints

- 不修改 HTTP API 路径、请求/响应 schema、状态码或 `Retry-After` 行为。
- 不修改 SQLAlchemy 表、索引、约束、Alembic revision、现有业务数据库或 checkpoint 数据库。
- 不改变 Message 树、Conversation 双 head、Task 状态、连续追问、幂等发布、回滚或 Event/Artifact 水位语义。
- 保留 `paperpilot.web.task_store.TaskStore` 及现有 Record/异常导入路径。
- `TaskStore` 的公开方法必须是类上可见的显式方法；禁止 `__getattr__`、运行时注册和 Mixin 多继承。
- 每个关键数据库事务整体留在一个业务函数中，不能由门面串联多个写函数完成。
- Store 模块不得依赖 FastAPI、Router、`app.py` 或 `TaskStore`。
- 模块与公开业务函数必须解释职责、业务不变量、事务和幂等原因；不添加逐行翻译式注释。
- 自动测试不得调用真实 DeepSeek、MCP 网络服务或产生模型费用。
- 每个任务只提交该任务列出的文件，保留用户的其他工作树变更。

---

## Target File Map

### Production files

```text
paperpilot/web/store/__init__.py
    Internal business-persistence package marker; no facade behavior.

paperpilot/web/store/records.py
    Immutable business records and Store business exceptions.

paperpilot/web/store/helpers.py
    Shared row mapping, ownership queries, time/JSON helpers, and active-paper helpers.

paperpilot/web/store/users.py
    User and login-session persistence.

paperpilot/web/store/conversations.py
    Conversation CRUD, primary paper, and ConversationPaper reads.

paperpilot/web/store/messages.py
    User-turn creation, Message tree reads, alternatives, and unstable turn.

paperpilot/web/store/publications.py
    Assistant publication, Task terminalization, stable-head finalization, and rollback switch.

paperpilot/web/store/tasks.py
    Task lookup, claim, pending-submit failure, and business DB health.

paperpilot/web/store/updates.py
    Task Event/Artifact writes, incremental pages, and consistent updates snapshot.

paperpilot/web/task_store.py
    Engine/SessionFactory owner, explicit forwarding facade, and public re-exports.

paperpilot/web/routes/conversations/__init__.py
    Re-export build_conversation_router.

paperpilot/web/routes/conversations/router.py
    Compose the three Conversation subrouters under one prefix.

paperpilot/web/routes/conversations/crud.py
    Conversation create/list/detail/update HTTP handlers.

paperpilot/web/routes/conversations/messages.py
    Message reads, alternatives, admission, capacity reservation, and submission cleanup.

paperpilot/web/routes/conversations/rollback.py
    Rollback target/checkpoint validation and head switch HTTP handler.

paperpilot/web/routes/conversations/presenters.py
    Record-to-response-dictionary conversion only.
```

### Test files

```text
tests/architecture/test_web_module_boundaries.py
    Explicit facade, compatibility exports, package entrypoint, and forbidden reverse dependencies.

tests/web/store/conftest.py
    Shared Store fixtures and PaperCandidate builders.

tests/web/store/test_lifecycle.py
tests/web/store/test_users.py
tests/web/store/test_conversations.py
tests/web/store/test_messages.py
tests/web/store/test_publications.py
tests/web/store/test_tasks.py
tests/web/store/test_updates.py

tests/web/routes/conversations/conftest.py
tests/web/routes/conversations/test_crud.py
tests/web/routes/conversations/test_message_submission.py
tests/web/routes/conversations/test_message_reads.py
tests/web/routes/conversations/test_rollback.py
tests/web/routes/conversations/test_task_updates.py
```

---

### Task 1: Add structural and compatibility gates

**Files:**
- Create: `tests/architecture/test_web_module_boundaries.py`
- Modify: `tests/architecture/test_repository_allowlist.py`
- Test: `tests/architecture/test_web_module_boundaries.py`
- Test: `tests/architecture/test_repository_allowlist.py`

**Interfaces:**
- Consumes: current `TaskStore` public methods and `paperpilot.web.routes.conversations.build_conversation_router`.
- Produces: executable constraints for all later extraction tasks.

- [ ] **Step 0: Capture the protected-data and test baseline in the isolated execution worktree**

```bash
shasum -a 256 data/web/tasks.sqlite3 data/langgraph/checkpoints.sqlite3
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests -q
```

Record both hashes and the exact pytest summary. At this point the only accepted failure is `tests/architecture/test_repository_allowlist.py::test_repository_docs_match_keep_allowlist`, reporting the two approved August 13 design/plan documents as extra files. Any other failure must be investigated before Step 1.

- [ ] **Step 1: Update the Codex document allowlist**

Add the two approved documents to `DOCS_KEEP_ALLOWLIST`:

```python
"docs/codex-only-plans/2026-08-13-web-business-module-reorganization-design.md",
"docs/codex-only-plans/2026-08-13-web-business-module-reorganization-plan.md",
```

- [ ] **Step 2: Write facade and import compatibility tests**

Create `tests/architecture/test_web_module_boundaries.py` with the complete original method set:

```python
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

from paperpilot.web.task_store import (
    ConversationAlternative,
    ConversationBusyError,
    ConversationDetail,
    ConversationPaperRecord,
    ConversationRecord,
    ConversationTurn,
    DuplicateUsernameError,
    FinalizedConversationTask,
    MessageRecord,
    PaperRecord,
    PublishedConversationResult,
    ResearchTask,
    StaleConversationHeadError,
    TaskArtifact,
    TaskArtifactBatch,
    TaskEvent,
    TaskEventBatch,
    TaskStore,
    TaskUpdates,
    UsedPaperInput,
    VALID_DEPTHS,
    WebUser,
)


EXPECTED_TASK_STORE_METHODS = {
    "close",
    "create_user",
    "get_user_by_username",
    "get_user_by_id",
    "create_session",
    "get_user_for_session",
    "delete_session",
    "create_conversation",
    "list_conversations",
    "get_conversation_detail",
    "update_conversation",
    "create_conversation_turn",
    "publish_conversation_result",
    "finalize_conversation_task",
    "fail_conversation_task",
    "switch_conversation_head",
    "get_message",
    "get_task_message",
    "list_active_messages",
    "list_message_alternatives",
    "get_unstable_turn",
    "check_health",
    "get_task",
    "claim_task",
    "fail_pending_task",
    "add_event",
    "list_events_page",
    "add_artifact",
    "list_artifacts_page",
    "get_conversation_task_updates",
}


EXPECTED_TASK_STORE_SIGNATURES = {
    "close": "(self) -> 'None'",
    "create_user": (
        "(self, *, username: 'str', password_hash: 'str', "
        "password_salt: 'str') -> 'WebUser'"
    ),
    "get_user_by_username": "(self, username: 'str') -> 'WebUser | None'",
    "get_user_by_id": "(self, user_id: 'str') -> 'WebUser | None'",
    "create_session": "(self, user_id: 'str') -> 'str'",
    "get_user_for_session": "(self, token: 'str') -> 'WebUser | None'",
    "delete_session": "(self, token: 'str') -> 'None'",
    "create_conversation": (
        "(self, *, user_id: 'str', paper: 'PaperCandidate', "
        "title: 'str | None' = None) -> 'ConversationRecord'"
    ),
    "list_conversations": (
        "(self, *, user_id: 'str', include_archived: 'bool' = False, "
        "limit: 'int' = 100) -> 'list[ConversationRecord]'"
    ),
    "get_conversation_detail": (
        "(self, conversation_id: 'str', *, user_id: 'str') -> "
        "'ConversationDetail | None'"
    ),
    "update_conversation": (
        "(self, conversation_id: 'str', *, user_id: 'str', "
        "title: 'str | None' = None, archived: 'bool | None' = None) -> "
        "'ConversationRecord | None'"
    ),
    "create_conversation_turn": (
        "(self, *, user_id: 'str', conversation_id: 'str', content: 'str', "
        "depth: 'str', expected_head_message_id: 'str | None') -> "
        "'ConversationTurn'"
    ),
    "publish_conversation_result": (
        "(self, *, task_id: 'str', content: 'str', metadata: 'dict', "
        "used_papers: 'list[UsedPaperInput]') -> 'PublishedConversationResult'"
    ),
    "finalize_conversation_task": (
        "(self, *, task_id: 'str', assistant_message_id: 'str', "
        "final_checkpoint_id: 'str', result_quality: 'str', "
        "active_paper_ids: 'list[str]') -> 'FinalizedConversationTask'"
    ),
    "fail_conversation_task": (
        "(self, *, task_id: 'str', message: 'str', stage: 'str | None' = None, "
        "payload: 'dict | None' = None) -> 'ResearchTask | None'"
    ),
    "switch_conversation_head": (
        "(self, conversation_id: 'str', *, user_id: 'str', "
        "expected_head_message_id: 'str | None', target_message_id: 'str', "
        "target_checkpoint_id: 'str', active_paper_ids: 'list[str]') -> "
        "'ConversationRecord | None'"
    ),
    "get_message": (
        "(self, conversation_id: 'str', message_id: 'str', *, "
        "user_id: 'str') -> 'MessageRecord | None'"
    ),
    "get_task_message": (
        "(self, task_id: 'str', role: 'str') -> 'MessageRecord | None'"
    ),
    "list_active_messages": (
        "(self, conversation_id: 'str', *, user_id: 'str') -> "
        "'list[MessageRecord] | None'"
    ),
    "list_message_alternatives": (
        "(self, conversation_id: 'str', message_id: 'str', *, "
        "user_id: 'str') -> 'list[ConversationAlternative] | None'"
    ),
    "get_unstable_turn": (
        "(self, conversation_id: 'str', *, user_id: 'str') -> "
        "'ConversationTurn | None'"
    ),
    "check_health": "(self) -> 'None'",
    "get_task": (
        "(self, task_id: 'str', *, user_id: 'str | None' = None) -> "
        "'ResearchTask | None'"
    ),
    "claim_task": (
        "(self, task_id: 'str', *, allow_running: 'bool' = False) -> "
        "'ResearchTask | None'"
    ),
    "fail_pending_task": "(self, task_id: 'str') -> 'ResearchTask | None'",
    "add_event": (
        "(self, *, task_id: 'str', type: 'str', message: 'str', "
        "stage: 'str | None' = None, payload: 'dict | None' = None) -> "
        "'TaskEvent'"
    ),
    "list_events_page": (
        "(self, task_id: 'str', *, user_id: 'str | None', after_id: 'int', "
        "limit: 'int') -> 'TaskEventBatch | None'"
    ),
    "add_artifact": (
        "(self, *, task_id: 'str', kind: 'str', title: 'str', content: 'str', "
        "payload: 'dict | None' = None) -> 'TaskArtifact'"
    ),
    "list_artifacts_page": (
        "(self, task_id: 'str', *, user_id: 'str | None', after_id: 'int', "
        "limit: 'int') -> 'TaskArtifactBatch | None'"
    ),
    "get_conversation_task_updates": (
        "(self, conversation_id: 'str', task_id: 'str', *, user_id: 'str', "
        "after_event_id: 'int' = 0, after_artifact_id: 'int' = 0, "
        "limit: 'int' = 50) -> 'TaskUpdates | None'"
    ),
}


def test_task_store_keeps_explicit_public_methods() -> None:
    actual = {
        name
        for name, value in TaskStore.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert actual == EXPECTED_TASK_STORE_METHODS
    assert TaskStore.__bases__ == (object,)
    assert "__getattr__" not in TaskStore.__dict__
    assert {
        name: str(inspect.signature(TaskStore.__dict__[name]))
        for name in sorted(EXPECTED_TASK_STORE_METHODS)
    } == EXPECTED_TASK_STORE_SIGNATURES


def test_task_store_keeps_public_record_and_error_exports() -> None:
    assert all(
        item.__module__.startswith("paperpilot.web")
        for item in (
            DuplicateUsernameError,
            ConversationBusyError,
            StaleConversationHeadError,
            WebUser,
            ResearchTask,
            TaskEvent,
            TaskArtifact,
            TaskEventBatch,
            TaskArtifactBatch,
            TaskUpdates,
            PaperRecord,
            ConversationRecord,
            ConversationPaperRecord,
            ConversationDetail,
            MessageRecord,
            ConversationTurn,
            ConversationAlternative,
            UsedPaperInput,
            PublishedConversationResult,
            FinalizedConversationTask,
        )
    )
    assert VALID_DEPTHS == {"quick", "standard", "deep"}


def test_conversation_router_keeps_compatibility_entrypoint() -> None:
    module = importlib.import_module("paperpilot.web.routes.conversations")
    assert callable(module.build_conversation_router)
    assert str(inspect.signature(module.build_conversation_router)) == (
        "(*, store: 'TaskStore', executor: 'TaskExecutorLike', "
        "require_user: 'RequireUser', deep_reading_runner: "
        "'DeepReadingRunner', paper_search: 'PaperSearch', "
        "overload_retry_after_seconds: 'int') -> 'APIRouter'"
    )
```

- [ ] **Step 3: Add the Store dependency-direction scan**

In the same file, parse every future `paperpilot/web/store/*.py` file and reject imports from the Web edge or facade:

```python
FORBIDDEN_STORE_IMPORT_PREFIXES = (
    "fastapi",
    "paperpilot.web.app",
    "paperpilot.web.routes",
    "paperpilot.web.task_store",
)


def _resolved_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            raw_name = "." * node.level + (node.module or "")
            base = (
                importlib.util.resolve_name(raw_name, "paperpilot.web.store")
                if node.level
                else raw_name
            )
            imported.add(base)
            imported.update(
                f"{base}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
    return imported


def test_store_modules_do_not_depend_on_web_edges() -> None:
    root = Path(__file__).parents[2] / "paperpilot" / "web" / "store"
    assert root.is_dir()
    violations = {
        path.relative_to(root).as_posix(): sorted(
            name
            for name in _resolved_imports(path)
            if name.startswith(FORBIDDEN_STORE_IMPORT_PREFIXES)
        )
        for path in root.glob("*.py")
    }
    assert {path: names for path, names in violations.items() if names} == {}
```

- [ ] **Step 4: Run the new boundary test and confirm the intended RED state**

Run:

```bash
.venv/bin/python -m pytest tests/architecture/test_web_module_boundaries.py -q
```

Expected: the existing facade compatibility tests pass, while `test_store_modules_do_not_depend_on_web_edges` fails because `paperpilot/web/store/` does not exist yet.

- [ ] **Step 5: Create only the package marker and rerun the boundary test**

Create `paperpilot/web/store/__init__.py`:

```python
"""Business-grouped persistence operations behind the TaskStore facade."""
```

Run:

```bash
.venv/bin/python -m pytest tests/architecture/test_web_module_boundaries.py tests/architecture/test_repository_allowlist.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the gates**

```bash
git add \
  paperpilot/web/store/__init__.py \
  tests/architecture/test_web_module_boundaries.py \
  tests/architecture/test_repository_allowlist.py
git commit -m "test(web): lock module reorganization boundaries"
```

---

### Task 2: Extract business records and shared helpers

**Files:**
- Create: `paperpilot/web/store/records.py`
- Create: `paperpilot/web/store/helpers.py`
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/architecture/test_web_module_boundaries.py`
- Test: `tests/web/test_task_store.py`
- Test: `tests/web/test_conversation_store.py`

**Interfaces:**
- Produces: all Record and exception types currently exported by `task_store.py`.
- Produces: `SessionFactory = sessionmaker[Session]` and shared conversion/ownership/active-paper helpers.
- Preserves: public imports from `paperpilot.web.task_store` with identical object identity during one interpreter run.

- [ ] **Step 1: Strengthen the export test to require canonical Record ownership**

Replace the broad module assertion with exact identity checks:

```python
from paperpilot.web.store import records


def test_task_store_reexports_canonical_records_and_errors() -> None:
    assert VALID_DEPTHS is records.VALID_DEPTHS
    assert DuplicateUsernameError is records.DuplicateUsernameError
    assert ConversationBusyError is records.ConversationBusyError
    assert StaleConversationHeadError is records.StaleConversationHeadError
    assert WebUser is records.WebUser
    assert ResearchTask is records.ResearchTask
    assert TaskEvent is records.TaskEvent
    assert TaskArtifact is records.TaskArtifact
    assert TaskEventBatch is records.TaskEventBatch
    assert TaskArtifactBatch is records.TaskArtifactBatch
    assert TaskUpdates is records.TaskUpdates
    assert PaperRecord is records.PaperRecord
    assert ConversationRecord is records.ConversationRecord
    assert ConversationPaperRecord is records.ConversationPaperRecord
    assert ConversationDetail is records.ConversationDetail
    assert MessageRecord is records.MessageRecord
    assert ConversationTurn is records.ConversationTurn
    assert ConversationAlternative is records.ConversationAlternative
    assert UsedPaperInput is records.UsedPaperInput
    assert PublishedConversationResult is records.PublishedConversationResult
    assert FinalizedConversationTask is records.FinalizedConversationTask
```

- [ ] **Step 2: Run the identity test and verify it fails**

```bash
.venv/bin/python -m pytest \
  tests/architecture/test_web_module_boundaries.py::test_task_store_reexports_canonical_records_and_errors -q
```

Expected: FAIL because `paperpilot.web.store.records` does not exist.

- [ ] **Step 3: Move records and exceptions without changing fields**

Move `VALID_DEPTHS`, the three exception classes, and every dataclass from lines 45–257 of the current `task_store.py` into `store/records.py`. Preserve every field, default, `Literal`, `frozen=True`, `to_dict`, and `to_public_dict` exactly. Add a module docstring explaining that these are business records rather than SQLAlchemy rows.

`task_store.py` must import and re-export each symbol explicitly; do not use wildcard imports.

- [ ] **Step 4: Extract only genuinely shared helpers**

Create `store/helpers.py` with exact equivalents of:

```text
_task_from_model
_user_from_model
_paper_from_model
_conversation_from_model
_conversation_paper_from_model
_message_from_model
_event_from_model
_artifact_from_model
_select_owned_task_model
_select_owned_conversation_model
_validate_active_paper_ids
_read_active_paper_ids
_set_active_paper_ids
_decode_payload
_decode_json_list
_utc_now
_utc_in
```

Expose them internally without leading underscores so call sites state their intent:

```python
SessionFactory = sessionmaker[Session]

def task_from_row(row: ResearchTaskRow) -> ResearchTask:
    return ResearchTask(
        id=row.id,
        question=row.question,
        depth=row.depth,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
        user_id=row.user_id,
        conversation_id=row.conversation_id,
        base_checkpoint_id=row.base_checkpoint_id,
        final_checkpoint_id=row.final_checkpoint_id,
        result_quality=row.result_quality,
    )


def select_owned_task(
    session: Session,
    task_id: str,
    user_id: str | None,
) -> ResearchTaskRow | None:
    filters = [ResearchTaskRow.id == task_id]
    if user_id is not None:
        filters.append(ResearchTaskRow.user_id == user_id)
    return session.scalar(select(ResearchTaskRow).where(*filters))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
```

Use the same body-preserving rename rule for the remaining helpers: `_user_from_model → user_from_row`, `_paper_from_model → paper_from_row`, `_conversation_from_model → conversation_from_row`, `_conversation_paper_from_model → conversation_paper_from_row`, `_message_from_model → message_from_row`, `_event_from_model → event_from_row`, `_artifact_from_model → artifact_from_row`, `_select_owned_conversation_model → select_owned_conversation`, `_validate_active_paper_ids → validate_active_paper_ids`, `_read_active_paper_ids → read_active_paper_ids`, `_set_active_paper_ids → set_active_paper_ids`, `_decode_payload → decode_payload`, `_decode_json_list → decode_json_list`, and `_utc_in → utc_in`.

Keep these domain-owned helpers out of `helpers.py`:

```text
_new_task                         → messages.py
_select_task_result_chain         → publications.py
_raise_conversation_turn_conflict → messages.py
_is_active_task_unique_conflict   → messages.py
_raise_broken_message_reference   → messages.py
_read_event_batch                 → updates.py
_read_artifact_batch              → updates.py
_validate_incremental_page        → updates.py
_validate_conversation_title      → conversations.py
```

- [ ] **Step 5: Point the unchanged monolith at canonical records/helpers**

Before moving Store methods, update the existing method bodies in `task_store.py` to import the extracted functions under their old private names where that minimizes diff risk:

```python
from paperpilot.web.store.helpers import (
    artifact_from_row as _artifact_from_model,
    conversation_from_row as _conversation_from_model,
    conversation_paper_from_row as _conversation_paper_from_model,
    decode_json_list as _decode_json_list,
    decode_payload as _decode_payload,
    event_from_row as _event_from_model,
    message_from_row as _message_from_model,
    paper_from_row as _paper_from_model,
    read_active_paper_ids as _read_active_paper_ids,
    select_owned_conversation as _select_owned_conversation_model,
    select_owned_task as _select_owned_task_model,
    set_active_paper_ids as _set_active_paper_ids,
    task_from_row as _task_from_model,
    user_from_row as _user_from_model,
    utc_in as _utc_in,
    utc_now as _utc_now,
    validate_active_paper_ids as _validate_active_paper_ids,
)
```

Delete only the now-duplicated definitions from the bottom of `task_store.py`.

- [ ] **Step 6: Run focused regression tests**

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/architecture/test_web_module_boundaries.py \
  tests/web/test_task_store.py \
  tests/web/test_conversation_store.py -q
```

Expected: all tests pass with no real model calls.

- [ ] **Step 7: Commit the record/helper extraction**

```bash
git add paperpilot/web/store/records.py paperpilot/web/store/helpers.py \
  paperpilot/web/task_store.py tests/architecture/test_web_module_boundaries.py
git commit -m "refactor(web): extract store records and helpers"
```

---

### Task 3: Extract user and Conversation persistence

**Files:**
- Create: `paperpilot/web/store/users.py`
- Create: `paperpilot/web/store/conversations.py`
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/architecture/test_web_module_boundaries.py`
- Test: `tests/web/test_task_store.py`
- Test: `tests/web/test_conversation_store.py`
- Test: `tests/web/test_auth.py`

**Interfaces:**
- `users.py` produces six functions with the same keyword semantics as the facade methods.
- `conversations.py` produces four functions with the same return values and exceptions as the facade methods.
- `TaskStore` forwards its owned `SessionFactory` explicitly.

- [ ] **Step 1: Write delegation tests before moving code**

Use a constructed Store and monkeypatch the future domain functions. Verify exact argument forwarding for representative methods:

```python
def test_task_store_explicitly_delegates_user_and_conversation_operations(
    monkeypatch,
) -> None:
    from paperpilot.web.store import conversations, users

    store = object.__new__(TaskStore)
    session_factory = object()
    store._session_factory = session_factory
    sentinel_user = object()
    sentinel_conversation = object()
    calls: list[tuple[str, object, object]] = []

    def fake_get_user_by_id(factory, user_id):
        calls.append(("get_user_by_id", factory, user_id))
        return sentinel_user

    def fake_list_conversations(factory, **kwargs):
        calls.append(("list_conversations", factory, kwargs))
        return [sentinel_conversation]

    monkeypatch.setattr(users, "get_user_by_id", fake_get_user_by_id)
    monkeypatch.setattr(
        conversations,
        "list_conversations",
        fake_list_conversations,
    )
    assert store.get_user_by_id("user_1") is sentinel_user
    assert store.list_conversations(user_id="user_1") == [sentinel_conversation]
    assert calls == [
        ("get_user_by_id", session_factory, "user_1"),
        ("list_conversations", session_factory, {"user_id": "user_1"}),
    ]
```

- [ ] **Step 2: Run the delegation test and verify it fails**

Expected: FAIL because `users.py` and `conversations.py` are absent.

- [ ] **Step 3: Extract user/session functions**

Move the six methods into module functions whose first argument is `session_factory: SessionFactory`. Preserve username validation, token format, seven-day expiry, duplicate error mapping, and session ownership behavior.

Exact contract: `create_user(session_factory: SessionFactory, *, username: str, password_hash: str, password_salt: str) -> WebUser`.

`create_session` must check user existence inside `users.py`; it must not call the `TaskStore` facade.

- [ ] **Step 4: Extract Conversation CRUD as whole transactions**

Move these methods as complete functions. Their exact contracts are `create_conversation(session_factory: SessionFactory, *, user_id: str, paper: PaperCandidate, title: str | None = None) -> ConversationRecord` and `list_conversations(session_factory: SessionFactory, *, user_id: str, include_archived: bool = False, limit: int = 100) -> list[ConversationRecord]`.

Also move `get_conversation_detail`, `update_conversation`, and `_validate_conversation_title`. Preserve the primary `ConversationPaperRow(role="primary", added_by="user")` write in the same transaction as Paper/Conversation creation.

- [ ] **Step 5: Add explicit facade forwarding methods**

Every moved method remains defined on `TaskStore`, with its original signature and return annotation, for example:

```python
def create_conversation(
    self,
    *,
    user_id: str,
    paper: PaperCandidate,
    title: str | None = None,
) -> ConversationRecord:
    return conversations.create_conversation(
        self._session_factory,
        user_id=user_id,
        paper=paper,
        title=title,
    )
```

- [ ] **Step 6: Add reader-oriented docstrings**

Each module docstring states responsibilities and exclusions. `create_conversation` documents the Paper upsert + Conversation + primary association atomic write; `update_conversation` documents why an active Task blocks archival.

- [ ] **Step 7: Run focused tests**

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/architecture/test_web_module_boundaries.py \
  tests/web/test_auth.py \
  tests/web/test_task_store.py::test_user_and_session_persist_across_store_instances \
  tests/web/test_conversation_store.py -k 'conversation and not turn and not publish and not finalize and not switch' -q
```

- [ ] **Step 8: Commit**

```bash
git add paperpilot/web/store/users.py paperpilot/web/store/conversations.py \
  paperpilot/web/task_store.py tests/architecture/test_web_module_boundaries.py
git commit -m "refactor(web): group users and conversations"
```

---

### Task 4: Extract Message tree and user-turn persistence

**Files:**
- Create: `paperpilot/web/store/messages.py`
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/architecture/test_web_module_boundaries.py`
- Modify: `tests/web/test_conversation_store.py`
- Test: `tests/web/test_conversation_store.py`

**Interfaces:**
- Produces: `create_conversation_turn`, `get_message`, `get_task_message`, `list_active_messages`, `list_message_alternatives`, and `get_unstable_turn` module functions.
- Preserves: atomic User Message + pending Task + queued Event creation and Message tree diagnostics.

- [ ] **Step 1: Add a facade delegation test for `create_conversation_turn`**

In `tests/architecture/test_web_module_boundaries.py`, patch `paperpilot.web.store.messages.create_conversation_turn`, assert that `TaskStore.create_conversation_turn` forwards the owned SessionFactory and every named argument, and confirm the facade method remains subclass-overridable.

- [ ] **Step 2: Run the new test and verify it fails**

Expected: FAIL because `store/messages.py` does not exist.

- [ ] **Step 3: Move the entire turn transaction and Message queries**

Move the six public methods together with these private helpers:

```text
_new_task
_raise_conversation_turn_conflict
_is_active_task_unique_conflict
_raise_broken_message_reference
```

Rename only for local clarity; do not change SQL statement order, exception precedence, unique-conflict mapping, active-path traversal, cycle detection, or unstable-turn queued-event ordering.

- [ ] **Step 4: Update deterministic private monkeypatch targets**

In `tests/web/test_conversation_store.py`, change only the helper that patches ID/time generation:

```python
from paperpilot.web.store import messages as messages_module

values = iter(hex_values)
monkeypatch.setattr(
    messages_module.uuid,
    "uuid4",
    lambda: SimpleNamespace(hex=next(values)),
)
monkeypatch.setattr(
    messages_module,
    "utc_now",
    lambda: "2026-08-07T09:00:00+00:00",
)
```

Do not preserve private `task_store.uuid` or `_utc_now` compatibility.

- [ ] **Step 5: Add business comments at the critical points**

Document:

- why capacity reservation happens outside this Store function;
- why the User Message points at the current stable `head_message_id`;
- why the stable head remains unchanged until finalization;
- why busy conflict is reported before stale-head conflict;
- why active-path reads traverse parents from head rather than sort all messages by time.

- [ ] **Step 6: Run Message and Deep Reading integration tests**

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/web/test_conversation_store.py -k \
  'turn or message or active_path or alternatives or unstable' -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/deep_reading/test_nodes.py \
  tests/deep_reading/test_runner.py -q
```

- [ ] **Step 7: Commit**

```bash
git add paperpilot/web/store/messages.py paperpilot/web/task_store.py \
  tests/web/test_conversation_store.py tests/architecture/test_web_module_boundaries.py
git commit -m "refactor(web): isolate conversation message storage"
```

---

### Task 5: Extract publication, finalization, failure, and head switching

**Files:**
- Create: `paperpilot/web/store/publications.py`
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/architecture/test_web_module_boundaries.py`
- Test: `tests/web/test_conversation_store.py`
- Test: `tests/deep_reading/test_graph.py`
- Test: `tests/deep_reading/test_nodes.py`
- Test: `tests/deep_reading/test_runner.py`

**Interfaces:**
- Produces: four transaction-owning functions with the original TaskStore method contracts.
- Preserves: exactly-once semantic result rows, terminal winner semantics, double-head consistency, and rollback active-paper restoration.

- [ ] **Step 1: Add explicit facade delegation tests for publication and rollback**

In `tests/architecture/test_web_module_boundaries.py`, patch `publications.publish_conversation_result` and `publications.switch_conversation_head`; assert the facade passes one SessionFactory plus unchanged named arguments and returns the exact sentinel objects.

- [ ] **Step 2: Run the tests and verify the RED state**

Expected: FAIL because `store/publications.py` does not exist.

- [ ] **Step 3: Move publication as one transaction**

Move `publish_conversation_result` intact with its result-chain and used-paper logic. Move `_select_task_result_chain` into this module. Retain the no-op Task UPDATE that acquires SQLite write ownership before idempotency reads.

Exact contract: `publish_conversation_result(session_factory: SessionFactory, *, task_id: str, content: str, metadata: dict, used_papers: list[UsedPaperInput]) -> PublishedConversationResult`.

- [ ] **Step 4: Move finalization and failure as complete transactions**

Move `finalize_conversation_task` and `fail_conversation_task` without extracting internal writes into other modules. Preserve:

- idempotent replay validation;
- stale message/checkpoint head checks;
- one semantic terminal winner under concurrent finalize/fail;
- one `completed` or `failed` Event;
- refusal to overwrite completed work.

- [ ] **Step 5: Move rollback head switching as one transaction**

Move `switch_conversation_head` intact. It must continue to reject archived/busy/stale Conversations before partially modifying active papers or either head.

- [ ] **Step 6: Add focused explanations**

Keep or improve comments explaining:

```text
SQLite no-op UPDATE → serialize idempotency reads
publish result       → create immutable answer but do not move stable head
finalize             → advance message and checkpoint heads together
rollback             → switch heads and active papers without deleting history
```

- [ ] **Step 7: Run all publication concurrency and graph tests**

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/web/test_conversation_store.py -k \
  'publish or finalize or fail_conversation or head_switch or switch_conversation' -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/deep_reading/test_graph.py \
  tests/deep_reading/test_nodes.py \
  tests/deep_reading/test_runner.py -q
```

- [ ] **Step 8: Commit**

```bash
git add paperpilot/web/store/publications.py paperpilot/web/task_store.py \
  tests/architecture/test_web_module_boundaries.py
git commit -m "refactor(web): isolate publication transactions"
```

---

### Task 6: Extract Task lifecycle and Event/Artifact updates

**Files:**
- Create: `paperpilot/web/store/tasks.py`
- Create: `paperpilot/web/store/updates.py`
- Modify: `paperpilot/web/task_store.py`
- Modify: `tests/architecture/test_web_module_boundaries.py`
- Test: `tests/web/test_task_store.py`
- Test: `tests/web/test_conversation_store.py`
- Test: `tests/web/test_conversation_worker.py`
- Test: `tests/web/test_celery_worker.py`

**Interfaces:**
- `tasks.py`: `check_health(engine)`, `get_task`, `claim_task`, `fail_pending_task`.
- `updates.py`: Event/Artifact add/list and conversation-scoped consistent snapshot.
- Final `task_store.py`: resource owner plus explicit forwarding only; target 250–400 lines.

- [ ] **Step 1: Add delegation tests for Task and updates groups**

In `tests/architecture/test_web_module_boundaries.py`, patch `tasks.claim_task` and `updates.get_conversation_task_updates`; assert facade delegation. Assert `check_health` receives the owned Engine rather than SessionFactory.

- [ ] **Step 2: Run tests and verify the RED state**

Expected: FAIL because the domain modules do not exist.

- [ ] **Step 3: Extract Task lifecycle functions**

Move `check_health`, `get_task`, `claim_task`, and `fail_pending_task`. Preserve atomic compare-and-update statements and `allow_running` recovery behavior.

- [ ] **Step 4: Extract Event/Artifact update functions and local helpers**

Move:

```text
add_event
list_events_page
add_artifact
list_artifacts_page
get_conversation_task_updates
_read_event_batch
_read_artifact_batch
_validate_incremental_page
```

`get_conversation_task_updates` must keep one `session_factory.begin()` scope around Task, Event, and Artifact reads. Do not implement it by calling the two public page functions, because that would produce three separate snapshots.

- [ ] **Step 5: Finish and inspect the explicit facade**

Delete the final SQLAlchemy query implementations from `task_store.py`. Verify that the file contains only:

```text
imports and public re-exports
TaskStore.__init__
TaskStore.close
29 explicit public forwarding methods
```

Run:

```bash
wc -l paperpilot/web/task_store.py
```

Expected: 250–400 lines. Do not reduce line count using dynamic forwarding.

- [ ] **Step 6: Run Store, Worker, and compatibility tests**

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/architecture/test_web_module_boundaries.py \
  tests/web/test_task_store.py \
  tests/web/test_conversation_store.py \
  tests/web/test_conversation_worker.py \
  tests/web/test_celery_worker.py -q
```

- [ ] **Step 7: Commit**

```bash
git add paperpilot/web/store/tasks.py paperpilot/web/store/updates.py \
  paperpilot/web/task_store.py tests/architecture/test_web_module_boundaries.py
git commit -m "refactor(web): isolate task updates and lifecycle"
```

---

### Task 7: Split Store tests by business responsibility

**Files:**
- Create: `tests/web/store/conftest.py`
- Create: `tests/web/store/test_lifecycle.py`
- Create: `tests/web/store/test_users.py`
- Create: `tests/web/store/test_conversations.py`
- Create: `tests/web/store/test_messages.py`
- Create: `tests/web/store/test_publications.py`
- Create: `tests/web/store/test_tasks.py`
- Create: `tests/web/store/test_updates.py`
- Delete: `tests/web/test_task_store.py`
- Delete: `tests/web/test_conversation_store.py`

**Interfaces:**
- Consumes: the unchanged public `TaskStore` API.
- Produces: the same Store test coverage organized by business group.
- Baseline: `test_task_store.py` has 26 test functions / 29 collected cases; `test_conversation_store.py` has 39 test functions / 41 collected cases. The reorganized Store suite must therefore have 65 test functions / 70 collected cases.

- [ ] **Step 1: Record the pre-move Store test inventory**

```bash
rg '^def test_' tests/web/test_task_store.py tests/web/test_conversation_store.py
.venv/bin/python -m pytest --collect-only -q \
  tests/web/test_task_store.py tests/web/test_conversation_store.py
```

Save the function-name list in the implementation notes for comparison. Do not infer completeness from green tests alone.

- [ ] **Step 2: Extract only shared fixtures/helpers into `conftest.py`**

Move reusable PaperCandidate and user/conversation builders as pytest fixtures or fixtures that return callables. Plain helper functions are not injected automatically by pytest and must not be imported directly from `conftest.py`. Keep concurrency barriers, fault injection, direct SQL helpers, and specialized result builders in the test module that owns the scenario.

For example, replace direct calls to the shared `_paper` helper with a fixture factory:

```python
@pytest.fixture
def paper_factory():
    def build(**overrides: object) -> PaperCandidate:
        values: dict[str, object] = {
            "external_id": "2401.12345v2",
            "title": "A Test Paper",
            "authors": ["Ada Lovelace"],
            "abstract": "abstract",
            "source_url": "https://arxiv.org/abs/2401.12345v2",
        }
        values.update(overrides)
        return PaperCandidate(**values)

    return build
```

Test-specific helpers such as `_insert_completed_turn`, `_publish_turn`, SQLite barriers, and fault injectors stay beside the tests that use them.

- [ ] **Step 3: Move lifecycle and user tests unchanged**

`test_lifecycle.py` receives constructor migration, close, WAL/PRAGMA, indexes, environment path, health, and legacy migration tests. `test_users.py` receives user/login-session persistence.

- [ ] **Step 4: Move Conversation and Message tests unchanged**

`test_conversations.py` receives the first ten Conversation CRUD/primary-paper tests. `test_messages.py` receives turn creation, busy/stale precedence, concurrent turns, active path, alternatives, ownership, unstable turn, broken parent, cross-conversation parent, and cycle tests.

- [ ] **Step 5: Move publication and rollback-storage tests unchanged**

`test_publications.py` receives all tests from `test_publish_conversation_result_is_task_idempotent_and_keeps_head_stable` through `test_switch_conversation_head_rejects_archived_conversation`, including concurrent publication, finalize/fail winner, retry after rollback, and head-switch atomicity.

- [ ] **Step 6: Move Task and updates tests unchanged**

`test_tasks.py` receives task mapping, claim, running recovery, terminal refusal, and pending failure. `test_updates.py` receives Event/Artifact add/list, watermarks, index-plan assertions, ownership-scoped updates, and single-snapshot tests, including the corresponding ownership test currently in `test_conversation_store.py`.

- [ ] **Step 7: Delete original files only after inventory matches**

Run this exact function-inventory comparison before deleting the originals:

```python
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import subprocess


def test_names(source: str) -> Counter[str]:
    tree = ast.parse(source)
    return Counter(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


before: Counter[str] = Counter()
for path in (
    "tests/web/test_task_store.py",
    "tests/web/test_conversation_store.py",
):
    source = subprocess.run(
        ["git", "show", f"17cc2e6:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    before.update(test_names(source))

after: Counter[str] = Counter()
for path in sorted(Path("tests/web/store").glob("test_*.py")):
    after.update(test_names(path.read_text(encoding="utf-8")))

assert sum(before.values()) == 65
assert before == after, {
    "missing": before - after,
    "duplicated": after - before,
}
```

Then delete the two original files.

- [ ] **Step 8: Run collection and Store tests**

```bash
.venv/bin/python -m pytest --collect-only -q tests/web/store
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/web/store -q
```

Expected: exactly 65 test functions and 70 collected cases; every original function name appears once; no duplicate original files remain.

- [ ] **Step 9: Commit**

```bash
git add -A tests/web/store tests/web/test_task_store.py tests/web/test_conversation_store.py
git commit -m "test(web): group store tests by business domain"
```

---

### Task 8: Split the Conversation Router without changing HTTP behavior

**Files:**
- Create: `paperpilot/web/routes/conversations/__init__.py`
- Create: `paperpilot/web/routes/conversations/router.py`
- Create: `paperpilot/web/routes/conversations/crud.py`
- Create: `paperpilot/web/routes/conversations/messages.py`
- Create: `paperpilot/web/routes/conversations/rollback.py`
- Create: `paperpilot/web/routes/conversations/presenters.py`
- Delete: `paperpilot/web/routes/conversations.py`
- Modify: `tests/architecture/test_web_module_boundaries.py`
- Test: `tests/web/test_conversation_api.py`
- Test: `tests/web/test_web_app.py`

**Interfaces:**
- `build_conversation_router` keeps the exact six-keyword-only-argument signature locked in Task 1 and still returns `APIRouter`.
- `paperpilot.web.routes.conversations` keeps the same import path despite changing from module to package.
- Child routers use no public API prefix; the root router owns `/api/conversations`.

- [ ] **Step 1: Add package-shape and route-set tests**

Extend the architecture test to assert these modules import:

```python
for name in (
    "paperpilot.web.routes.conversations.router",
    "paperpilot.web.routes.conversations.crud",
    "paperpilot.web.routes.conversations.messages",
    "paperpilot.web.routes.conversations.rollback",
    "paperpilot.web.routes.conversations.presenters",
):
    importlib.import_module(name)
```

In `tests/web/test_conversation_api.py`, add an explicit OpenAPI route-method contract for the router being split. The task-updates endpoint is excluded here because it is owned by `routes/task_updates.py` and already covered by the app route allowlist:

```python
def test_conversation_router_keeps_route_method_set(tmp_path) -> None:
    harness = _harness(tmp_path)
    schema = harness.client.get("/openapi.json").json()
    actual = {
        (path, method)
        for path, operations in schema["paths"].items()
        if path.startswith("/api/conversations") and "/tasks/" not in path
        for method in operations
    }
    assert actual == {
        ("/api/conversations", "get"),
        ("/api/conversations", "post"),
        ("/api/conversations/{conversation_id}", "get"),
        ("/api/conversations/{conversation_id}", "patch"),
        ("/api/conversations/{conversation_id}/messages", "get"),
        ("/api/conversations/{conversation_id}/messages", "post"),
        (
            "/api/conversations/{conversation_id}/messages/"
            "{message_id}/alternatives",
            "get",
        ),
        ("/api/conversations/{conversation_id}/rollback", "post"),
    }
```

- [ ] **Step 2: Run the package-shape test and verify RED**

Expected: FAIL because `conversations` is still a single module.

- [ ] **Step 3: Extract pure presenters first**

Move `_paper_dict`, `_conversation_dict`, `_task_dict`, `_message_dict`, and `_detail_dict` into `presenters.py` with public-within-package names:

```python
paper_dict
conversation_dict
task_dict
message_dict
detail_dict
```

Presenters must not import `TaskStore`, SQLAlchemy, FastAPI Request, or `HTTPException`.

- [ ] **Step 4: Extract CRUD routes**

Implement the exact internal contract `build_crud_router(*, store: TaskStore, require_user: RequireUser, paper_search: PaperSearch) -> APIRouter`.

Move create/list/detail/update handlers and paper-reference resolution without changing path suffixes, response models, status codes, owner-scoped 404 behavior, or title validation mapping. Move `_resolve_paper_reference` into `crud.py`. Keep a small module-local ownership lookup in each route module that needs it; do not make pure presenters query the Store merely to deduplicate three lines.

- [ ] **Step 5: Extract Message routes**

Implement the exact internal contract `build_message_router(*, store: TaskStore, executor: TaskExecutorLike, require_user: RequireUser, overload_retry_after_seconds: int) -> APIRouter`.

Move list messages, alternatives, submit, `_validate_turn_admission`, reservation, and ambiguous submit cleanup intact. Keep the comment explaining why a failed broker submission may not mark a Task failed after a Worker already claimed it.

- [ ] **Step 6: Extract rollback routes**

Implement the exact internal contract `build_rollback_router(*, store: TaskStore, require_user: RequireUser, deep_reading_runner: DeepReadingRunner) -> APIRouter`.

Move rollback admission and checkpoint validation intact, including schema version, graph version, completeness, Task/message binding, active paper uniqueness, and primary-paper membership checks. Keep the rollback admission check local instead of making rollback depend on the Message route module.

- [ ] **Step 7: Compose the public router**

In `router.py`:

```python
def build_conversation_router(
    *,
    store: TaskStore,
    executor: TaskExecutorLike,
    require_user: RequireUser,
    deep_reading_runner: DeepReadingRunner,
    paper_search: PaperSearch,
    overload_retry_after_seconds: int,
) -> APIRouter:
    router = APIRouter(prefix="/api/conversations")
    router.include_router(
        build_crud_router(
            store=store,
            require_user=require_user,
            paper_search=paper_search,
        )
    )
    router.include_router(
        build_message_router(
            store=store,
            executor=executor,
            require_user=require_user,
            overload_retry_after_seconds=overload_retry_after_seconds,
        )
    )
    router.include_router(
        build_rollback_router(
            store=store,
            require_user=require_user,
            deep_reading_runner=deep_reading_runner,
        )
    )
    return router
```

`__init__.py` explicitly re-exports this function. Delete the old module only after imports and OpenAPI route-set tests pass.

- [ ] **Step 8: Run API and app tests**

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/architecture/test_web_module_boundaries.py \
  tests/web/test_conversation_api.py \
  tests/web/test_web_app.py \
  tests/web/test_conversation_ui.py -q
```

- [ ] **Step 9: Commit**

```bash
git add -A paperpilot/web/routes/conversations.py \
  paperpilot/web/routes/conversations \
  tests/architecture/test_web_module_boundaries.py \
  tests/web/test_conversation_api.py
git commit -m "refactor(web): group conversation routes"
```

---

### Task 9: Split Conversation API tests by route responsibility

**Files:**
- Create: `tests/web/routes/conversations/conftest.py`
- Create: `tests/web/routes/conversations/test_crud.py`
- Create: `tests/web/routes/conversations/test_message_submission.py`
- Create: `tests/web/routes/conversations/test_message_reads.py`
- Create: `tests/web/routes/conversations/test_rollback.py`
- Create: `tests/web/routes/conversations/test_task_updates.py`
- Delete: `tests/web/test_conversation_api.py`

**Interfaces:**
- Consumes: unchanged HTTP surface and existing test harness behavior.
- Produces: all original Conversation API cases collected exactly once under route-focused files.
- Baseline: `test_conversation_api.py` has 22 test functions / 31 collected cases.

- [ ] **Step 1: Record the original API test inventory**

```bash
rg '^def test_' tests/web/test_conversation_api.py
.venv/bin/python -m pytest --collect-only -q tests/web/test_conversation_api.py
```

- [ ] **Step 2: Move shared harness code into local `conftest.py`**

Move only reusable app/client/auth/Conversation constructors and fake checkpoint/runner/executor types. Expose callable helpers through named fixtures such as `harness_factory`, `register_user`, `create_conversation`, and `complete_turn`; do not import plain functions directly from `conftest.py`. Keep test-specific fault injection beside its test.

- [ ] **Step 3: Move CRUD and read tests**

`test_crud.py` receives authentication requirements, create/list/detail/title/archive, paper resolution, owner isolation, expected-head schema presence, and the existing paper-search normalization case because it is part of Conversation creation setup.

`test_message_reads.py` receives active-path/unstable-turn and alternatives responses.

- [ ] **Step 4: Move submission and updates tests**

`test_message_submission.py` receives reservation ordering, busy/stale mapping, capacity rejection, configured `Retry-After`, submit failure cleanup, and ambiguous ownership transfer cases.

`test_task_updates.py` receives conversation/user/task scoped update authorization.

- [ ] **Step 5: Move rollback tests**

`test_rollback.py` receives OpenAPI rollback contract, cross-owner checkpoint protection, restart/rollback/branch lifecycle, valid rollback without model call, invalid checkpoint cases, and active-task/stale-head prevalidation.

- [ ] **Step 6: Delete the original file only after inventory equality**

Run this exact function-inventory comparison before deleting the original:

```python
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import subprocess


def test_names(source: str) -> Counter[str]:
    tree = ast.parse(source)
    return Counter(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


source = subprocess.run(
    ["git", "show", "17cc2e6:tests/web/test_conversation_api.py"],
    check=True,
    capture_output=True,
    text=True,
).stdout
before = test_names(source)

after: Counter[str] = Counter()
for path in sorted(Path("tests/web/routes/conversations").glob("test_*.py")):
    after.update(test_names(path.read_text(encoding="utf-8")))

assert sum(before.values()) == 22
assert before == after, {
    "missing": before - after,
    "duplicated": after - before,
}
```

Then delete `tests/web/test_conversation_api.py`.

- [ ] **Step 7: Run route tests and full Web tests**

```bash
.venv/bin/python -m pytest --collect-only -q tests/web/routes/conversations
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/web/routes/conversations \
  tests/web/test_web_app.py \
  tests/web/test_conversation_ui.py -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/web -q
```

Expected: exactly 22 test functions and 31 collected cases under the new directory, followed by a green full Web suite.

- [ ] **Step 8: Commit**

```bash
git add -A tests/web/routes/conversations tests/web/test_conversation_api.py
git commit -m "test(web): group conversation API coverage"
```

---

### Task 10: Final readability, dependency, schema, and regression verification

**Files:**
- Modify only if verification finds a scoped defect in files created by Tasks 1–9.
- Test: all project tests and architecture gates.

**Interfaces:**
- Produces: verified behavior-preserving reorganization ready for independent review.

- [ ] **Step 1: Audit module documentation and comments**

For each new production module, verify the top-level docstring answers:

```text
What does this module own?
What does it explicitly not own?
Which records/tables does it use?
Which transaction or concurrency boundary matters?
```

For `create_conversation_turn`, `publish_conversation_result`, `finalize_conversation_task`, `fail_conversation_task`, and `switch_conversation_head`, verify the docstring explains head effects and atomicity/idempotency. Remove comments that merely restate Python or SQLAlchemy syntax.

- [ ] **Step 2: Verify the public facade and package entrypoint**

```bash
.venv/bin/python - <<'PY'
from paperpilot.web.task_store import TaskStore
from paperpilot.web.routes.conversations import build_conversation_router

expected = {
    "create_user", "create_session", "create_conversation",
    "create_conversation_turn", "publish_conversation_result",
    "finalize_conversation_task", "switch_conversation_head",
    "get_conversation_task_updates",
}
assert expected <= TaskStore.__dict__.keys()
assert callable(build_conversation_router)
print("compatibility imports: ok")
PY
```

- [ ] **Step 3: Verify file sizes and dependency direction**

```bash
wc -l paperpilot/web/task_store.py paperpilot/web/store/*.py \
  paperpilot/web/routes/conversations/*.py
.venv/bin/python -m pytest tests/architecture/test_web_module_boundaries.py -q
```

Expected: no monolithic Store/query implementation remains in `task_store.py`; no Store module imports FastAPI, routes, app, or the facade.

- [ ] **Step 4: Verify database schema and migrations remain unchanged**

```bash
.venv/bin/python -m pytest \
  tests/web/test_db_models.py \
  tests/web/test_db_migrations.py \
  tests/web/test_database.py -q
git diff 17cc2e6 -- paperpilot/web/db_models.py migrations alembic.ini
```

Expected: tests PASS and the schema/migration diff is empty.

- [ ] **Step 5: Run focused reorganized tests**

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest \
  tests/web/store \
  tests/web/routes/conversations \
  tests/deep_reading -q
```

- [ ] **Step 6: Run the complete Web and project suites**

```bash
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests/web -q
LANGGRAPH_STRICT_MSGPACK=true .venv/bin/python -m pytest tests -q
```

Record exact pass/deselect/warning counts. Do not report success from focused tests alone.

- [ ] **Step 7: Check environment consistency and diff hygiene**

```bash
uv pip check
git diff --check
git status --short
```

If `uv pip check` fails only because the sandbox cannot read the uv cache, rerun the identical command with user approval rather than changing cache configuration.

- [ ] **Step 8: Verify protected local databases were not modified**

Compare SHA-256 hashes of `data/web/tasks.sqlite3` and `data/langgraph/checkpoints.sqlite3` with hashes captured before implementation. Tests must use `tmp_path`; no implementation step should open the production files for writes.

- [ ] **Step 9: Request independent code review**

Invoke `superpowers:requesting-code-review` against the full branch diff. Require review of:

```text
public compatibility
transaction boundaries
concurrency/idempotency
Message/checkpoint double-head consistency
dependency direction
test inventory preservation
comment usefulness
```

Address Critical and Important findings before completion.

- [ ] **Step 10: Commit any final scoped corrections**

If no corrections are needed, do not create an empty commit. Otherwise:

```bash
git add \
  paperpilot/web/task_store.py \
  paperpilot/web/store \
  paperpilot/web/routes/conversations \
  tests/architecture/test_web_module_boundaries.py \
  tests/architecture/test_repository_allowlist.py \
  tests/web/store \
  tests/web/routes/conversations
git commit -m "fix(web): close module reorganization review findings"
```

---

## Execution Notes

- Start implementation in an isolated worktree via `superpowers:using-git-worktrees`; do not reorganize directly in a dirty user checkout.
- Capture protected database hashes and the baseline full-suite result before Task 1 implementation. Because the already-approved design document and this plan are intentionally not yet in `DOCS_KEEP_ALLOWLIST`, the only accepted baseline failure is `test_repository_docs_match_keep_allowlist`; Task 1 must make it green. Any other failure blocks the refactor until understood.
- Complete Tasks 2–6 before mechanically moving tests, so failures indicate behavioral regressions rather than import mistakes from simultaneous test relocation.
- Keep each high-risk transaction extraction in its own commit and rerun its concurrency tests immediately.
- Never use a script that rewrites function bodies or SQL automatically. Mechanical test-file moves may use a deterministic split helper only after comparing every test function name before and after.
- The design source of truth is `docs/codex-only-plans/2026-08-13-web-business-module-reorganization-design.md`.
