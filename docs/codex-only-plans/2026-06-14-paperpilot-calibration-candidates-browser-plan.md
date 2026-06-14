# PaperPilot Calibration Candidates Browser Plan

Date: 2026-06-14

## Background And Goal

The current static Web workbench shows aggregate eval scores through
`GET /api/eval/summary`. The next useful step is to make the 31 manually
reviewed calibration candidates inspectable from the browser, so the aggregate
calibrated score can be traced back to individual cases.

## Scope

This slice adds a read-only calibration candidates browser:

- backend loading of `data/eval/semantic_calibration_candidates_20260614.jsonl`;
- `GET /api/eval/calibration-candidates`;
- optional filters for `category` and `review_decision`;
- frontend filter controls and a compact expandable list;
- tests for loading, filtering, and static UI assets.

This slice does not add:

- editing `review_decision` in the browser;
- write-back API;
- charts;
- React/Vite migration;
- changes to task execution or PaperPilot research workflow.

## API Design

Endpoint:

```text
GET /api/eval/calibration-candidates
```

Optional query parameters:

```text
category=partial_high|partial_medium|strict_pass_semantic_bad
review_decision=accept_as_correct|keep_partial|downgrade_to_incorrect|judge_error|TODO
```

Missing file behavior:

- return `available: false`;
- include a readable message;
- return an empty `candidates` list.

## Frontend Design

Add a section below the Evaluation Snapshot metrics:

- category filter;
- review decision filter;
- candidate count;
- compact candidate rows.

Each candidate row should show:

- `case_id`;
- `category`;
- `review_decision`;
- strict pass;
- semantic label and confidence;
- question.

Each row should expand to show:

- oracle spans;
- predicted excerpt;
- judge reason;
- manual review notes.

## Validation

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\web -q
.venv\Scripts\python.exe -m pytest tests\web tests\eval -q
```

Also check:

```powershell
GET http://127.0.0.1:8000/api/eval/calibration-candidates
```

## Risks

- Long predictions can make the panel noisy. The backend should provide a
  bounded excerpt for the frontend.
- The static page is getting denser. Keep this browser compact and read-only.
