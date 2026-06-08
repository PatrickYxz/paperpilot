# PaperPilot Current Work Audit And Web MVP Plan

## Background And Goal

The next PaperPilot phase should first close out the existing uncommitted work,
then define a small local Web research workstation MVP. The goal of this plan is
to separate completed work, half-finished work, and directly reusable pieces
before designing the Web MVP.

This pass is intentionally limited to audit and design. It does not modify
runtime behavior, public interfaces, configuration, dependencies, or data
structures.

## Constraints

- Preserve the current local-first direction: simple HTML and JavaScript,
  FastAPI, SQLite, local background execution, explicit research workflow state.
- Do not introduce auth, cloud deployment, Redis/Celery, PostgreSQL, React/Vue,
  or new MCP servers in the first MVP.
- Do not delete or revert any existing work without explicit confirmation.
- Treat current uncommitted changes as user work unless proven otherwise.
- If later implementation requires interface, data model, config, or behavior
  changes, ask for confirmation before editing.

## Step-By-Step Plan

1. Inspect current git status and identify modified, untracked, and generated
   files.
2. Group current work into:
   - completed and likely test-backed changes,
   - Web-MVP-relevant pieces,
   - half-finished or risky pieces that should be parked,
   - unrelated personal or documentation artifacts.
3. Read the core changed modules and their tests at a high level:
   - `paperpilot/main.py`
   - `paperpilot/core/loop.py`
   - `paperpilot/builtin_tools/compact.py`
   - conversation, session, document, bulk-input modules
   - directly related tests
4. Check the current README and existing roadmap to keep product direction
   aligned with the already agreed local research workstation scope.
5. Propose 2-3 MVP directions with trade-offs.
6. Recommend one direction and list the first implementation slice, but wait for
   user confirmation before changing runtime code.

## Verification Method

- Use read-only git and file inspection commands for the audit.
- Optionally run targeted tests only if needed to classify current work.
- Report exactly which checks were run and which areas remain unverified.

## Current Audit Result

The current uncommitted work is mostly coherent and test-backed. It is best
classified as a CLI/conversation enhancement layer rather than the Web MVP
itself.

Reusable pieces for the Web MVP:

- `ConversationSession` gives a long-lived in-process conversation boundary.
- `SessionStore` persists named message histories under `data/sessions`.
- `DocumentStore` persists user-pasted long documents and supports lightweight
  lexical evidence search.
- `BulkPaperInputDetector` prevents very long user-pasted papers from being
  kept directly in the model message history.
- `ContextManager` and `compact_messages` support context preflight and automatic
  compaction before LLM calls.
- `ask_user` is useful for CLI interaction, but a Web MVP should translate this
  into an explicit workflow pause instead of blocking on terminal input.

Areas that should not be treated as final Web-MVP implementation yet:

- There is no FastAPI/SQLite task API layer.
- There is no task table, workflow state machine, task event log, or report
  persistence model.
- Some Chinese prompt/test strings appear mojibake in PowerShell output and
  should be checked before user-facing or model-facing release.
- `_personal/` and unrelated plan documents are outside the Web MVP scope.

Targeted verification run:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_agent_loop.py tests\test_conversation_session.py tests\test_session_store.py tests\test_document_store.py tests\test_bulk_input.py tests\test_message_codec.py tests\test_context_manager.py tests\builtin_tools\test_ask_user.py tests\test_main_integration.py -q
```

Result after running outside the sandbox with user-approved access:

```text
47 passed, 1 deselected in 7.95s
```

## Risks And Confirmation Points

- The working tree contains many uncommitted files. Some may represent completed
  work, and some may be drafts. Do not assume they are safe to modify.
- A Web MVP will likely require API and persistence design. That must be
  confirmed before implementation.
- Current plan documents and some terminal output may show encoding issues in
  PowerShell, so conclusions should be based on readable source files and tests
  rather than mojibake output.
