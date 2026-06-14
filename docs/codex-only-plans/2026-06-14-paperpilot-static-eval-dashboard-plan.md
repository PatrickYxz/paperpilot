# PaperPilot Static Eval Dashboard Plan

Date: 2026-06-14

## Background And Goal

PaperPilot currently has a local FastAPI Web workbench implemented with static
HTML/CSS/JS. We decided not to migrate to React yet. The next frontend step is
to add a small read-only evaluation snapshot to the existing workbench so the
recent QASPER semantic/calibration work is visible in the UI.

## Scope

This slice adds:

- a backend snapshot helper that reads existing eval JSONL files;
- a `GET /api/eval/summary` endpoint;
- a read-only frontend panel showing strict, semantic, and calibrated scores;
- tests for the summary helper and API endpoint.

This slice does not add:

- React/Vite or any JavaScript build pipeline;
- editable calibration review UI;
- charts;
- task deletion, cancellation, or rerun controls;
- changes to PaperPilot's research workflow.

## Files

Expected backend files:

- `paperpilot/web/eval_summary.py`
- `paperpilot/web/app.py`

Expected frontend files:

- `paperpilot/web/static/index.html`
- `paperpilot/web/static/app.js`
- `paperpilot/web/static/styles.css`

Expected tests:

- `tests/web/test_eval_summary.py`
- `tests/web/test_web_app.py`

## Data Inputs

Default source files:

- `data/eval/semantic_audit_paperpilot_full_20260611.jsonl`
- `data/eval/semantic_calibration_candidates_20260614.jsonl`

If files are missing, the endpoint should return `available: false` with a
clear message instead of raising a 500 error.

## Displayed Metrics

The UI should show:

- strict pass score;
- semantic correct-only score;
- semantic weighted score with `correct=1`, `partial=0.5`;
- calibrated correct-only score;
- calibrated weighted score;
- semantic label counts;
- calibration decision counts.

## Validation

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests\web -q
```

If time allows, also run:

```powershell
.venv\Scripts\python.exe -m pytest tests\web tests\eval -q
```

## Risks

- The current eval input filenames are date-specific. This slice intentionally
  hardcodes the latest known filenames as defaults, because the goal is a small
  read-only snapshot rather than a general eval browser.
- The existing static UI can become crowded. Keep this panel compact and
  summary-oriented.
