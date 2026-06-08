# PaperPilot Option B Staging Plan

Date: 2026-06-08

## Goal

Prepare a staged milestone that includes the Web workbench MVP plus the
conversation/session/context foundation it depends on.

This plan deliberately excludes personal files and unrelated project documents.

## Include

### Web Workbench MVP

- `paperpilot/web/`
- `tests/web/`
- `requirements.txt`
- Web MVP planning documents from 2026-06-05 and 2026-06-08.
- `docs/learning/2026-06-05-paperpilot-web-workbench-zero-basics.md`

### Conversation And Runtime Foundation

- `paperpilot/main.py`
- `paperpilot/conversation.py`
- `paperpilot/session_store.py`
- `paperpilot/message_codec.py`
- `paperpilot/document_store.py`
- `paperpilot/bulk_input.py`
- `paperpilot/core/context_manager.py`
- `paperpilot/core/loop.py`
- `paperpilot/builtin_tools/compact.py`
- `paperpilot/builtin_tools/ask_user.py`
- `paperpilot/builtin_tools/user_document.py`

### Tests For Included Runtime Work

- `tests/test_agent_loop.py`
- `tests/test_main_integration.py`
- `tests/test_conversation_session.py`
- `tests/test_session_store.py`
- `tests/test_message_codec.py`
- `tests/test_document_store.py`
- `tests/test_bulk_input.py`
- `tests/test_context_manager.py`
- `tests/builtin_tools/test_ask_user.py`

## Exclude

- `_personal/`
- `docs/superpowers/plans/2026-06-02-exam-mate-plan.md`
- unrelated resume/interview/browser workflow plans from May unless explicitly
  requested later
- local runtime database files under `data/`

## Verification

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\web tests\test_agent_loop.py tests\test_conversation_session.py tests\test_session_store.py tests\test_document_store.py tests\test_bulk_input.py tests\test_message_codec.py tests\test_context_manager.py tests\builtin_tools\test_ask_user.py tests\test_main_integration.py -q
```

Then run broader regression excluding the known local SOCKS proxy issue:

```powershell
.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/mcp_servers/test_ss_client.py
```

## Commit Boundary

Suggested commit message:

```text
Add local Web research workbench
```

Do not commit until staged content has been reviewed.

