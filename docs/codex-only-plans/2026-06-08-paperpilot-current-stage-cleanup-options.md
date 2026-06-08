# PaperPilot Current Stage Cleanup And Commit Options

Date: 2026-06-08

## Goal

This document organizes the current PaperPilot worktree after the Web workbench
A1-A3.3 work. It is meant to support a discussion about how to stage, commit, or
continue the work. It does not change runtime behavior.

## Current Situation

The working tree is intentionally broad. It contains at least three different
tracks:

1. Web workbench MVP work.
2. CLI/conversation/session/context enhancements that predate the Web work.
3. Personal/interview/learning documents.

These should not be blindly committed as one lump unless the goal is only to
checkpoint a private worktree.

## Track 1: Web Workbench MVP

This is the cleanest current product track. It includes:

```text
paperpilot/web/
tests/web/
requirements.txt
docs/codex-only-plans/2026-06-05-paperpilot-audit-and-web-mvp-plan.md
docs/codex-only-plans/2026-06-05-paperpilot-web-task-mvp-a1-plan.md
docs/codex-only-plans/2026-06-05-paperpilot-web-workflow-a2-plan.md
docs/codex-only-plans/2026-06-05-paperpilot-web-event-structure-a25-plan.md
docs/codex-only-plans/2026-06-05-paperpilot-web-runner-artifacts-a30-plan.md
docs/codex-only-plans/2026-06-05-paperpilot-web-real-runner-a31-plan.md
docs/codex-only-plans/2026-06-08-paperpilot-web-mvp-a1-a31-closure-audit.md
docs/codex-only-plans/2026-06-08-paperpilot-web-real-runner-event-mapping-a32-plan.md
docs/codex-only-plans/2026-06-08-paperpilot-web-event-ui-a33-plan.md
docs/learning/2026-06-05-paperpilot-web-workbench-zero-basics.md
```

Current capability:

- FastAPI local Web app.
- SQLite task storage.
- Task events and structured payloads.
- Task artifacts.
- Simulated workflow as default.
- Explicit real PaperPilot mode.
- Real runner event mapping into `task_events`.
- Improved event UI with categories and folded payloads.

Recommended commit label if committed alone:

```text
Web MVP: add local research task workbench
```

## Track 2: CLI Conversation And Context Enhancements

This track includes modified and untracked files such as:

```text
paperpilot/main.py
paperpilot/conversation.py
paperpilot/session_store.py
paperpilot/message_codec.py
paperpilot/document_store.py
paperpilot/bulk_input.py
paperpilot/core/context_manager.py
paperpilot/core/loop.py
paperpilot/builtin_tools/ask_user.py
paperpilot/builtin_tools/user_document.py
paperpilot/builtin_tools/compact.py
tests/test_agent_loop.py
tests/test_conversation_session.py
tests/test_session_store.py
tests/test_message_codec.py
tests/test_document_store.py
tests/test_bulk_input.py
tests/test_context_manager.py
tests/builtin_tools/test_ask_user.py
tests/test_main_integration.py
```

Current capability:

- CLI chat mode and named sessions.
- Session persistence.
- Message encoding/decoding.
- User-pasted long document storage/search.
- Bulk input detection.
- Context preflight and auto-compaction.
- `ask_user` tool.

This track is test-backed, but it is broader than the Web MVP. It should be
reviewed as its own feature set before committing.

Recommended commit label if committed separately:

```text
Conversation: add recoverable sessions and context safeguards
```

## Track 3: Learning, Interview, And Personal Documents

This includes:

```text
_personal/
docs/learning/
docs/codex-only-plans/2026-05-*.md
docs/codex-only-plans/2026-06-01-paperpilot-ai-application-learning-roadmap-plan.md
docs/superpowers/plans/2026-06-02-exam-mate-plan.md
```

Some documents are useful local context, but they should not automatically go
into a product commit.

Recommended handling:

- Keep `_personal/` out of product commits.
- Commit learning docs only if the repo is intentionally keeping study material.
- Move unrelated `exam-mate` docs out of PaperPilot if this repository should
  stay clean.

## Validation Snapshot

The current Web and nearby agent/session scope passed:

```text
76 passed, 1 deselected, 1 warning
```

The broader suite passed when excluding the known SOCKS proxy environment issue:

```text
199 passed, 10 deselected, 1 warning
```

Known environmental risk:

```text
tests/mcp_servers/test_ss_client.py
```

can fail when a SOCKS proxy is configured but `socksio` is not installed.

## Cleanup Options

### Option A: Commit Web MVP Only

Stage and commit only Track 1.

Pros:

- Produces the cleanest product milestone.
- Keeps Web work independent from CLI/session changes.
- Easier to explain and review.

Cons:

- Some Web real-mode behavior imports `paperpilot.conversation`, which is
  currently part of the broader untracked conversation track. If committed alone,
  exact dependencies must be checked carefully.

Best when:

- The goal is a clean, reviewable Web MVP commit.

### Option B: Commit Web MVP Plus Required Conversation Foundation

Stage Track 1 plus the minimum conversation/session files required for A3.1 real
mode and existing tests.

Pros:

- Captures the Web MVP as it actually runs today.
- Avoids committing a Web layer that references uncommitted conversation code.
- Still avoids personal documents.

Cons:

- Larger commit.
- Mixes Web product shell with CLI/session enhancements.
- Needs careful commit message and review notes.

Best when:

- The goal is a working local milestone rather than a tiny commit.

### Option C: No Commit Yet, Continue A3.x

Keep the worktree dirty and continue with more Web features.

Pros:

- Fastest forward momentum.
- Avoids spending time on staging decisions now.

Cons:

- Risk of losing clear boundaries.
- Harder to review, roll back, or explain.
- Future commits become more painful.

Best when:

- This is a purely local exploratory branch and checkpoint cleanliness is not
  important yet.

## Recommendation

Choose Option B if the next step is to make a durable project milestone.

Reason:

- Web A3.1/A3.2 real mode relies on the conversation runner path.
- The current repo state already treats the conversation/session work as a
  tested foundation.
- Keeping `_personal/` and unrelated documents out still preserves a meaningful
  boundary.

If strict reviewability matters more, choose Option A, but first verify whether
the committed baseline already contains the required conversation APIs. If it
does not, Option A will be incomplete.

## Proposed Next Discussion

Before staging anything, decide:

1. Should this repository keep the learning docs as committed project material?
2. Should `_personal/` stay completely untracked?
3. Is the next milestone a clean Web MVP commit, or a broader local-app
   milestone including sessions/context/document support?

