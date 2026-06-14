const taskForm = document.querySelector("#taskForm");
const questionInput = document.querySelector("#question");
const depthInput = document.querySelector("#depth");
const executionModeInput = document.querySelector("#executionMode");
const formMessage = document.querySelector("#formMessage");
const refreshButton = document.querySelector("#refreshTasks");
const refreshEvalButton = document.querySelector("#refreshEval");
const statusFilter = document.querySelector("#statusFilter");
const taskList = document.querySelector("#taskList");
const taskDetail = document.querySelector("#taskDetail");
const taskArtifacts = document.querySelector("#taskArtifacts");
const taskEvents = document.querySelector("#taskEvents");
const evalSnapshot = document.querySelector("#evalSnapshot");
const candidateCategory = document.querySelector("#candidateCategory");
const candidateDecision = document.querySelector("#candidateDecision");
const candidateCount = document.querySelector("#candidateCount");
const candidateList = document.querySelector("#candidateList");

let selectedTaskId = null;
let pollTimer = null;

function setMessage(text, kind = "") {
  formMessage.textContent = text;
  formMessage.className = kind ? `message ${kind}` : "message";
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function formatPercent(value) {
  const number = Number(value);
  if (Number.isNaN(number)) {
    return "0.0%";
  }
  return `${(number * 100).toFixed(1)}%`;
}

function renderTasks(tasks) {
  taskList.innerHTML = "";
  if (!tasks.length) {
    taskList.innerHTML = '<p class="task-detail empty">No tasks yet.</p>';
    return;
  }

  for (const task of tasks) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = task.id === selectedTaskId ? "task-row active" : "task-row";
    button.innerHTML = `
      <span class="task-question">${escapeHtml(task.question)}</span>
      <span class="status">${escapeHtml(task.status)}</span>
      <span class="task-meta">${escapeHtml(task.depth)} - ${escapeHtml(formatDate(task.created_at))}</span>
    `;
    button.addEventListener("click", () => loadTask(task.id));
    taskList.appendChild(button);
  }
}

function renderEvalSnapshot(snapshot) {
  if (!snapshot.available) {
    evalSnapshot.className = "eval-snapshot empty";
    evalSnapshot.textContent = snapshot.message || "Evaluation snapshot is not available.";
    return;
  }

  evalSnapshot.className = "eval-snapshot";
  evalSnapshot.innerHTML = `
    <div class="metric-grid">
      ${renderMetricCard("Strict pass", snapshot.strict.pass_count, snapshot.total_cases, snapshot.strict.rate)}
      ${renderMetricCard("Semantic correct", snapshot.semantic.correct_count, snapshot.total_cases, snapshot.semantic.correct_rate)}
      ${renderMetricCard("Semantic weighted", snapshot.semantic.weighted_count, snapshot.total_cases, snapshot.semantic.weighted_rate)}
      ${renderMetricCard("Calibrated correct", snapshot.calibrated.correct_count, snapshot.total_cases, snapshot.calibrated.correct_rate)}
      ${renderMetricCard("Calibrated weighted", snapshot.calibrated.weighted_count, snapshot.total_cases, snapshot.calibrated.weighted_rate)}
      ${renderMetricCard("Review candidates", snapshot.calibrated.candidate_count, snapshot.total_cases, null)}
    </div>
    <div class="eval-tables">
      ${renderCountTable("Semantic labels", snapshot.semantic.label_counts)}
      ${renderCountTable("Review decisions", snapshot.calibrated.decision_counts)}
    </div>
    <p class="eval-source">Source: ${escapeHtml(snapshot.audit_path)} · ${escapeHtml(snapshot.calibration_path)}</p>
  `;
}

function renderCalibrationCandidates(payload) {
  if (!payload.available) {
    candidateCount.textContent = "Unavailable";
    candidateList.className = "candidate-list empty";
    candidateList.textContent = payload.message || "Calibration candidates are not available.";
    return;
  }

  const candidates = payload.candidates || [];
  candidateCount.textContent = `${candidates.length} of ${payload.total_candidates}`;
  candidateList.className = candidates.length ? "candidate-list" : "candidate-list empty";
  if (!candidates.length) {
    candidateList.textContent = "No candidates match the current filters.";
    return;
  }

  candidateList.innerHTML = candidates.map(renderCandidate).join("");
}

function renderCandidate(candidate) {
  const strictText = candidate.strict_pass ? "strict pass" : "strict fail";
  return `
    <details class="candidate-card">
      <summary>
        <span class="candidate-main">
          <span class="candidate-id">${escapeHtml(candidate.case_id)}</span>
          <span class="candidate-question">${escapeHtml(candidate.question)}</span>
        </span>
        <span class="candidate-tags">
          <span class="status">${escapeHtml(candidate.review_decision)}</span>
          <span class="mini-tag">${escapeHtml(candidate.category)}</span>
          <span class="mini-tag">${escapeHtml(strictText)}</span>
          <span class="mini-tag">${escapeHtml(candidate.semantic_label)} / ${escapeHtml(candidate.confidence)}</span>
        </span>
      </summary>
      <div class="candidate-body">
        <h4>Oracle spans</h4>
        ${renderInlineList(candidate.oracle_spans || [])}
        <h4>PaperPilot predicted excerpt</h4>
        <pre>${escapeHtml(candidate.predicted_excerpt || "(empty prediction)")}</pre>
        <h4>Judge reason</h4>
        <p>${escapeHtml(candidate.judge_reason || "")}</p>
        <h4>Manual notes</h4>
        <p>${escapeHtml(candidate.review_notes || "")}</p>
        ${candidate.trace_path ? `<p class="eval-source">Trace: ${escapeHtml(candidate.trace_path)}</p>` : ""}
      </div>
    </details>
  `;
}

function renderInlineList(items) {
  if (!items.length) {
    return '<p class="task-detail empty">No oracle spans.</p>';
  }
  return `
    <ul class="oracle-list">
      ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
    </ul>
  `;
}

function renderMetricCard(label, count, total, rate) {
  const rateText = rate === null ? `${escapeHtml(total)} total` : formatPercent(rate);
  return `
    <div class="metric-card">
      <span class="metric-label">${escapeHtml(label)}</span>
      <strong>${escapeHtml(count)}</strong>
      <span class="metric-rate">${escapeHtml(rateText)}</span>
    </div>
  `;
}

function renderCountTable(title, counts) {
  const rows = Object.entries(counts || {});
  if (!rows.length) {
    return `
      <div class="count-table">
        <h3>${escapeHtml(title)}</h3>
        <p class="task-detail empty">No data.</p>
      </div>
    `;
  }
  return `
    <div class="count-table">
      <h3>${escapeHtml(title)}</h3>
      <table>
        <tbody>
          ${rows
            .map(([name, count]) => `
              <tr>
                <th>${escapeHtml(name)}</th>
                <td>${escapeHtml(count)}</td>
              </tr>
            `)
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderTaskDetail(task, events, artifacts) {
  taskDetail.className = "task-detail";
  taskDetail.innerHTML = `
    <p class="task-question">${escapeHtml(task.question)}</p>
    <span class="status">${escapeHtml(task.status)}</span>
    <dl>
      <dt>ID</dt><dd>${escapeHtml(task.id)}</dd>
      <dt>Depth</dt><dd>${escapeHtml(task.depth)}</dd>
      <dt>Created</dt><dd>${escapeHtml(formatDate(task.created_at))}</dd>
      <dt>Updated</dt><dd>${escapeHtml(formatDate(task.updated_at))}</dd>
    </dl>
  `;
  renderTaskArtifacts(artifacts);
  renderTaskEvents(events);
}

function renderTaskArtifacts(artifacts) {
  taskArtifacts.innerHTML = "";
  if (!artifacts.length) {
    taskArtifacts.innerHTML = '<p class="task-detail empty">No artifacts yet.</p>';
    return;
  }
  for (const artifact of artifacts) {
    const card = document.createElement("div");
    card.className = "artifact-card";
    card.innerHTML = `
      <div class="artifact-title">
        <span>${escapeHtml(artifact.title)}</span>
        <span class="artifact-kind">${escapeHtml(artifact.kind)}</span>
      </div>
      <div class="artifact-content">${escapeHtml(artifact.content)}</div>
      <span class="event-time">${escapeHtml(formatDate(artifact.created_at))}</span>
    `;
    taskArtifacts.appendChild(card);
  }
}

function renderTaskEvents(events) {
  taskEvents.innerHTML = "";
  if (!events.length) {
    taskEvents.innerHTML = '<p class="task-detail empty">No events yet.</p>';
    return;
  }
  for (const event of events) {
    const row = document.createElement("div");
    const category = eventCategory(event);
    row.className = `event-row event-${category}`;
    row.innerHTML = renderEventRow(event, category);
    taskEvents.appendChild(row);
  }
}

function renderEventRow(event, category) {
  const payload = event.payload && Object.keys(event.payload).length
    ? JSON.stringify(event.payload, null, 2)
    : "";
  const details = payload
    ? `
      <details class="event-payload">
        <summary>Details</summary>
        <pre>${escapeHtml(payload)}</pre>
      </details>
    `
    : "";
  return `
    <div class="event-header">
      <span class="event-category">${escapeHtml(category)}</span>
      <span class="event-stage">${escapeHtml(event.stage || event.type)}</span>
    </div>
    <div class="event-message">${escapeHtml(event.message)}</div>
    <span class="event-time">${escapeHtml(formatDate(event.created_at))}</span>
    ${details}
  `;
}

function eventCategory(event) {
  const stage = event.stage || "";
  if (stage === "agent_turn") {
    return "agent";
  }
  if (stage === "tool_call" || stage === "tool_result") {
    return "tool";
  }
  if (stage === "context_preflight" || stage === "auto_compact") {
    return "context";
  }
  if (stage === "failure" || stage === "guardrail" || event.type === "failed") {
    return "failure";
  }
  if (
    [
      "queue",
      "start",
      "prepare",
      "deep_read_placeholder",
      "complete",
      "real_start",
      "real_complete",
    ].includes(stage)
  ) {
    return "workflow";
  }
  return "other";
}

async function loadTasks() {
  const status = statusFilter.value;
  const url = status ? `/api/tasks?status=${encodeURIComponent(status)}` : "/api/tasks";
  const tasks = await requestJson(url);
  renderTasks(tasks);
}

async function loadEvalSnapshot() {
  const snapshot = await requestJson("/api/eval/summary");
  renderEvalSnapshot(snapshot);
}

async function loadCalibrationCandidates() {
  const params = new URLSearchParams();
  if (candidateCategory.value) {
    params.set("category", candidateCategory.value);
  }
  if (candidateDecision.value) {
    params.set("review_decision", candidateDecision.value);
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const payload = await requestJson(`/api/eval/calibration-candidates${suffix}`);
  renderCalibrationCandidates(payload);
}

async function loadTask(taskId) {
  selectedTaskId = taskId;
  const encodedTaskId = encodeURIComponent(taskId);
  const [task, events, artifacts] = await Promise.all([
    requestJson(`/api/tasks/${encodedTaskId}`),
    requestJson(`/api/tasks/${encodedTaskId}/events`),
    requestJson(`/api/tasks/${encodedTaskId}/artifacts`),
  ]);
  renderTaskDetail(task, events, artifacts);
  await loadTasks();
  schedulePolling(task);
}

taskForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setMessage("");
  const question = questionInput.value.trim();
  if (!question) {
    setMessage("Question is required.", "error");
    return;
  }

  try {
    const task = await requestJson("/api/tasks", {
      method: "POST",
      body: JSON.stringify({
        question,
        depth: depthInput.value,
        execution_mode: executionModeInput.value,
      }),
    });
    questionInput.value = "";
    setMessage("Task created.", "success");
    await loadTask(task.id);
  } catch (error) {
    setMessage(error.message, "error");
  }
});

refreshButton.addEventListener("click", () => {
  loadTasks().catch((error) => setMessage(error.message, "error"));
});

refreshEvalButton.addEventListener("click", () => {
  Promise.all([loadEvalSnapshot(), loadCalibrationCandidates()]).catch((error) => {
    evalSnapshot.className = "eval-snapshot empty";
    evalSnapshot.textContent = error.message;
  });
});

candidateCategory.addEventListener("change", () => {
  loadCalibrationCandidates().catch((error) => {
    candidateList.className = "candidate-list empty";
    candidateList.textContent = error.message;
  });
});

candidateDecision.addEventListener("change", () => {
  loadCalibrationCandidates().catch((error) => {
    candidateList.className = "candidate-list empty";
    candidateList.textContent = error.message;
  });
});

statusFilter.addEventListener("change", () => {
  loadTasks().catch((error) => setMessage(error.message, "error"));
});

function schedulePolling(task) {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  if (task.status !== "pending" && task.status !== "running") {
    return;
  }
  pollTimer = setTimeout(() => {
    if (selectedTaskId) {
      loadTask(selectedTaskId).catch((error) => setMessage(error.message, "error"));
    }
  }, 1000);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

loadTasks().catch((error) => setMessage(error.message, "error"));
loadEvalSnapshot().catch((error) => {
  evalSnapshot.className = "eval-snapshot empty";
  evalSnapshot.textContent = error.message;
});
loadCalibrationCandidates().catch((error) => {
  candidateList.className = "candidate-list empty";
  candidateList.textContent = error.message;
});
