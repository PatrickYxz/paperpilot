# Answer Synthesis Guard Repair V1 Design

Date: 2026-06-14

## Goal

Build a conservative Answer Guard + Repair Loop for the QASPER `deep-read-paper`
evaluation path.

The goal is not to replace semantic evaluation. The goal is to make PaperPilot's
final answer cleaner and more controllable before it reaches the scorer:

- remove internal process traces such as `Step 4` and `Answer span candidates`;
- enforce a direct `Short answer` followed by `Evidence`;
- reduce overbroad answer dumps;
- require numeric claims in the short answer to be supported by the evidence section;
- retry final-answer synthesis once when the first answer has severe quality issues.

This first version is intentionally narrow. It targets the single-paper QASPER
deep-read path first, not every PaperPilot conversation.

## Current Context

The current QASPER PaperPilot baseline runs through:

- `paperpilot/eval/baselines.py::run_paperpilot`
- `paperpilot.main.run`
- `paperpilot/conversation.py::_build_system_prompt`
- `paperpilot/skills/deep-read-paper.md`

The failure-mode analysis found that many remaining errors are not pure retrieval
failures. They include:

- answer scope is too broad;
- multi-part answers miss required items;
- numeric or entity answers are not verified tightly enough;
- final answers contain internal process text or noisy candidate lists.

Because of that, the first implementation should improve final answer synthesis
before changing retrieval.

## Non-Goals

This version will not:

- change ColBERT retrieval;
- change MCP tools;
- change the LLM judge;
- change the strict QASPER scorer;
- change the web dashboard;
- add a full answer-synthesis pipeline;
- globally intercept ordinary chat, multi-paper comparison, or subagent answers.

## Components

### 1. Prompt Tightening

Update `paperpilot/skills/deep-read-paper.md` so final answers are constrained:

- `Short answer:` must directly answer the question first.
- `Evidence:` must contain only retrieved evidence that directly supports the short answer.
- `Notes:` is optional and only for necessary caveats.
- Do not include `Step 4`, tool progress, chain-of-thought style narration, or `Answer span candidates` in the final answer.
- Do not include related but unasked methods, datasets, metrics, or baselines in the direct answer.
- For list questions, include only items directly asked by the question and directly supported by evidence.
- For numeric/entity questions, repeat the supporting number/entity in `Evidence`.

### 2. Answer Quality Checker

Add an eval-path checker, likely `paperpilot/eval/answer_quality.py`.

It returns a structured result:

```json
{
  "passed": false,
  "severity": "severe",
  "issues": [
    {
      "code": "internal_trace_marker",
      "severity": "severe",
      "message": "Final answer contains internal process marker: Step 4"
    }
  ]
}
```

Initial issue codes:

- `internal_trace_marker`
- `missing_short_answer_section`
- `missing_evidence_section`
- `short_answer_too_long`
- `numeric_not_supported_in_evidence`
- `list_dumping_risk`
- `evidence_too_thin`

Only severe issues should trigger repair in V1:

- `internal_trace_marker`
- `missing_short_answer_section`
- `missing_evidence_section`
- `short_answer_too_long`
- `numeric_not_supported_in_evidence`

Risk-only issues should be recorded but should not trigger repair:

- `list_dumping_risk`
- `evidence_too_thin`

### 3. Repair Loop

Add a repair step in `paperpilot/eval/baselines.py::run_paperpilot`.

Flow:

1. Run PaperPilot normally.
2. Extract the first final answer as `predicted_raw`.
3. Run answer quality checker.
4. If no severe issue exists, return the original final answer.
5. If severe issues exist, call the LLM once to rewrite the final answer.
6. Run the checker again on the repaired answer.
7. Return the repaired answer as `predicted`, while preserving the original answer and both quality reports.

The repair prompt must be narrow:

- do not call tools;
- do not add new facts;
- use only evidence and claims already present in the conversation/final answer;
- rewrite into `Short answer`, `Evidence`, optional `Notes`;
- if evidence is insufficient, say so explicitly;
- do not include process traces or answer-span candidate sections.

V1 allows at most one repair attempt.

### 4. Eval Result Shape

`run_paperpilot()` should return extra fields:

```json
{
  "predicted_raw": "...",
  "predicted": "...",
  "answer_quality": {...},
  "answer_repaired": true,
  "repair_answer_quality": {...},
  "repair_error": null
}
```

Compatibility rule:

- existing consumers should still read `predicted`;
- if no repair happens, `predicted_raw == predicted`;
- if repair fails, keep the best available answer and record `repair_error`.

## Data Flow

```text
QASPER case
  -> run_paperpilot prompt
  -> PaperPilot deep-read workflow
  -> raw final answer
  -> answer quality checker
  -> if severe issue: one repair LLM call
  -> repaired or original final answer
  -> strict scorer / semantic audit / calibration
```

## Error Handling

- If the initial agent run fails, preserve the current `error` behavior.
- If quality checking fails unexpectedly, record a quality-check error and keep the raw answer.
- If repair LLM call fails, keep the raw answer and set `repair_error`.
- If repair output still fails quality checks, return the repaired answer only if it is not empty; otherwise return the raw answer.

## Testing

Add focused tests for:

- answer quality checker catches internal process markers;
- answer quality checker catches missing `Short answer` and `Evidence`;
- answer quality checker catches overlong short answers;
- answer quality checker catches unsupported numeric claims;
- list dumping is recorded as risk-only;
- repair trigger selects only severe issues;
- `run_paperpilot()` result includes `predicted_raw`, `answer_quality`, `answer_repaired`, and `repair_answer_quality`;
- prompt text in `deep-read-paper.md` contains the new scope, checklist, numeric/entity, and no-process-trace constraints.

## Rollout

Implement behind the QASPER evaluation path first.

After implementation:

1. Run unit tests.
2. Run a small smoke eval on the known failure cases.
3. Compare raw vs repaired answers.
4. Only then rerun larger QASPER evaluation.

## Open Decisions

All decisions are fixed for V1:

- one repair attempt only;
- no tool calls during repair;
- repair enabled only in `run_paperpilot()` evaluation path;
- ordinary interactive PaperPilot answers are unchanged in this slice.
