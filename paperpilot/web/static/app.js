const taskForm = document.querySelector("#taskForm");
const questionInput = document.querySelector("#question");
const depthInput = document.querySelector("#depth");
const executionModeInput = document.querySelector("#executionMode");
const formMessage = document.querySelector("#formMessage");
const refreshButton = document.querySelector("#refreshTasks");
const statusFilter = document.querySelector("#statusFilter");
const taskList = document.querySelector("#taskList");
const taskDetail = document.querySelector("#taskDetail");
const taskArtifacts = document.querySelector("#taskArtifacts");
const taskEvents = document.querySelector("#taskEvents");

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
