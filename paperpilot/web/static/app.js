const authForm = document.querySelector("#authForm");
const usernameInput = document.querySelector("#username");
const passwordInput = document.querySelector("#password");
const registerButton = document.querySelector("#registerButton");
const logoutButton = document.querySelector("#logoutButton");
const currentUser = document.querySelector("#currentUser");
const currentUsername = document.querySelector("#currentUsername");
const authMessage = document.querySelector("#authMessage");
const workspace = document.querySelector("#workspace");
const conversationList = document.querySelector("#conversationList");
const paperSearch = document.querySelector("#paperSearch");
const paperQuery = document.querySelector("#paperQuery");
const paperSearchMessage = document.querySelector("#paperSearchMessage");
const paperCandidates = document.querySelector("#paperCandidates");
const conversationTitle = document.querySelector("#conversationTitle");
const conversationPaper = document.querySelector("#conversationPaper");
const conversationMessage = document.querySelector("#conversationMessage");
const messageList = document.querySelector("#messageList");
const messageComposer = document.querySelector("#messageComposer");
const conversationQuestion = document.querySelector("#conversationQuestion");
const conversationDepth = document.querySelector("#conversationDepth");
const sendConversationMessage = document.querySelector("#sendConversationMessage");
const refreshConversations = document.querySelector("#refreshConversations");
const conversationEvents = document.querySelector("#conversationEvents");
const branchSelector = document.querySelector("#branchSelector");
const branchOptions = document.querySelector("#branchOptions");
const closeBranchSelector = document.querySelector("#closeBranchSelector");

const state = {
  currentUser: null,
  conversations: [],
  selectedConversationId: null,
  selectedConversation: null,
  activeTaskPoll: null,
};

let authenticationVersion = 0;
let conversationListVersion = 0;
let requestVersion = 0;

function makeElement(tagName, className = "", text = "") {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  element.textContent = text;
  return element;
}

function setAuthMessage(text, kind = "") {
  authMessage.textContent = text;
  authMessage.className = kind ? `message ${kind}` : "message";
}

function setConversationMessage(text, kind = "") {
  conversationMessage.textContent = text;
  conversationMessage.className = kind ? `message ${kind}` : "message";
}

function setPaperMessage(text, kind = "") {
  paperSearchMessage.textContent = text;
  paperSearchMessage.className = kind ? `message ${kind}` : "message";
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...options,
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : response.statusText;
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    error.status = response.status;
    throw error;
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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function isCurrentAuthentication(version) {
  return state.currentUser !== null && authenticationVersion === version;
}

function setAuthenticatedUser(user) {
  authenticationVersion += 1;
  state.currentUser = user;
  authForm.hidden = true;
  currentUser.hidden = false;
  workspace.hidden = false;
  currentUsername.textContent = user.username;
  setAuthMessage("");
  resetConversationWorkspace();
  loadConversations().catch((error) => {
    setConversationMessage(error.message, "error");
  });
}

function clearAuthenticatedUser(message = "") {
  authenticationVersion += 1;
  state.currentUser = null;
  authForm.hidden = false;
  currentUser.hidden = true;
  workspace.hidden = true;
  currentUsername.textContent = "";
  resetConversationWorkspace();
  setAuthMessage(message);
}

async function loadCurrentUser() {
  const authRequestVersion = authenticationVersion;
  try {
    const user = await requestJson("/api/auth/me");
    if (authRequestVersion !== authenticationVersion) {
      return;
    }
    setAuthenticatedUser(user);
  } catch (_error) {
    if (authRequestVersion !== authenticationVersion) {
      return;
    }
    clearAuthenticatedUser("Log in or register to use the workbench.");
  }
}

async function submitAuth(mode) {
  setAuthMessage("");
  const username = usernameInput.value.trim();
  const password = passwordInput.value;
  if (!username || !password) {
    setAuthMessage("Username and password are required.", "error");
    return;
  }
  authenticationVersion += 1;
  const authRequestVersion = authenticationVersion;
  try {
    const user = await requestJson(`/api/auth/${mode}`, {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    if (authRequestVersion !== authenticationVersion) {
      return;
    }
    passwordInput.value = "";
    setAuthenticatedUser(user);
  } catch (error) {
    if (authRequestVersion === authenticationVersion) {
      setAuthMessage(error.message, "error");
    }
  }
}

function clearPollTimer() {
  if (state.activeTaskPoll && state.activeTaskPoll.timer !== null) {
    clearTimeout(state.activeTaskPoll.timer);
    state.activeTaskPoll.timer = null;
  }
}

function resetConversationSelection(conversationId) {
  clearPollTimer();
  requestVersion += 1;
  state.selectedConversationId = conversationId;
  state.selectedConversation = null;
  state.activeTaskPoll = null;
  conversationEvents.replaceChildren();
  branchOptions.replaceChildren();
  branchSelector.hidden = true;
  return requestVersion;
}

function isCurrentConversation(conversationId, version) {
  return (
    state.currentUser !== null
    && state.selectedConversationId === conversationId
    && requestVersion === version
  );
}

function hasActiveTask() {
  return state.activeTaskPoll !== null;
}

function setMutationControlsDisabled() {
  const disabled = hasActiveTask();
  conversationQuestion.disabled = !state.selectedConversation || disabled;
  conversationDepth.disabled = !state.selectedConversation || disabled;
  sendConversationMessage.disabled = !state.selectedConversation || disabled;
  for (const button of document.querySelectorAll(".conversation-mutation")) {
    button.disabled = disabled;
  }
}

function renderPaperCandidates(items) {
  paperCandidates.replaceChildren();
  if (!items.length) {
    paperCandidates.appendChild(
      makeElement("p", "conversation-empty", "No matching papers found."),
    );
    return;
  }
  for (const paper of items) {
    const card = makeElement("article", "paper-candidate");
    card.appendChild(makeElement("h3", "paper-candidate-title", paper.title));
    card.appendChild(
      makeElement(
        "p",
        "paper-candidate-meta",
        `${paper.authors.join(", ")} · arXiv ${paper.external_id}`,
      ),
    );
    if (paper.abstract) {
      card.appendChild(makeElement("p", "paper-candidate-abstract", paper.abstract));
    }
    const button = makeElement("button", "secondary-button", "Start conversation");
    button.type = "button";
    button.addEventListener("click", () => {
      createConversation(paper);
    });
    card.appendChild(button);
    paperCandidates.appendChild(card);
  }
}

async function searchPapers() {
  const query = paperQuery.value.trim();
  if (!query) {
    setPaperMessage("Enter a paper title, topic, arXiv ID, or URL.", "error");
    return;
  }
  const authRequestVersion = authenticationVersion;
  setPaperMessage("Searching…");
  paperCandidates.replaceChildren();
  try {
    const params = new URLSearchParams({ q: query, limit: "10" });
    const payload = await requestJson(`/api/papers/search?${params.toString()}`);
    if (
      !isCurrentAuthentication(authRequestVersion)
      || paperQuery.value.trim() !== query
    ) {
      return;
    }
    renderPaperCandidates(payload.items);
    setPaperMessage(payload.items.length ? "Select a paper to begin." : "");
  } catch (error) {
    if (
      isCurrentAuthentication(authRequestVersion)
      && paperQuery.value.trim() === query
    ) {
      setPaperMessage(error.message, "error");
    }
  }
}

async function createConversation(paper) {
  const authRequestVersion = authenticationVersion;
  setPaperMessage("Creating conversation…");
  try {
    const created = await requestJson("/api/conversations", {
      method: "POST",
      body: JSON.stringify({
        paper: {
          source: paper.source,
          external_id: paper.external_id,
        },
      }),
    });
    if (!isCurrentAuthentication(authRequestVersion)) {
      return;
    }
    setPaperMessage("Conversation created.", "success");
    await loadConversation(created.id);
    await loadConversations(false);
  } catch (error) {
    if (isCurrentAuthentication(authRequestVersion)) {
      setPaperMessage(error.message, "error");
    }
  }
}

function renderConversationList(items) {
  conversationList.replaceChildren();
  if (!items.length) {
    conversationList.appendChild(
      makeElement("p", "conversation-empty", "Search for a paper to start a conversation."),
    );
    return;
  }
  for (const conversation of items) {
    const button = makeElement("button", "conversation-row");
    button.type = "button";
    button.dataset.conversationId = conversation.id;
    if (conversation.id === state.selectedConversationId) {
      button.classList.add("active");
    }
    button.appendChild(makeElement("span", "conversation-row-title", conversation.title));
    button.appendChild(
      makeElement(
        "span",
        "conversation-row-meta",
        conversation.head_message_id ? "Reading in progress" : "New conversation",
      ),
    );
    button.addEventListener("click", () => {
      loadConversation(conversation.id);
    });
    conversationList.appendChild(button);
  }
}

async function loadConversations(selectFirst = true) {
  if (state.currentUser === null) {
    return;
  }
  const authRequestVersion = authenticationVersion;
  const listVersion = ++conversationListVersion;
  try {
    const payload = await requestJson("/api/conversations?limit=100");
    if (
      !isCurrentAuthentication(authRequestVersion)
      || listVersion !== conversationListVersion
    ) {
      return;
    }
    state.conversations = payload.items;
    renderConversationList(state.conversations);
    if (
      selectFirst
      && state.selectedConversationId === null
      && state.conversations.length
    ) {
      await loadConversation(state.conversations[0].id);
    }
  } catch (error) {
    if (
      isCurrentAuthentication(authRequestVersion)
      && listVersion === conversationListVersion
    ) {
      setConversationMessage(error.message, "error");
    }
  }
}

function renderConversationHeader(detail) {
  conversationTitle.textContent = detail.conversation.title;
  const activePaperTitles = detail.active_papers.map((paper) => paper.title);
  conversationPaper.textContent = activePaperTitles.join(" · ");
}

function addMessageControls(card, message) {
  if (message.role !== "assistant" || message.status !== "complete") {
    return;
  }
  const controls = makeElement("div", "message-controls");
  const rollbackButton = makeElement(
    "button",
    "secondary-button conversation-mutation",
    "Rollback here",
  );
  rollbackButton.type = "button";
  rollbackButton.addEventListener("click", () => {
    rollbackToMessage(message.id);
  });
  const alternativesButton = makeElement(
    "button",
    "secondary-button conversation-mutation",
    "其他版本",
  );
  alternativesButton.type = "button";
  alternativesButton.addEventListener("click", () => {
    loadAlternatives(message.id);
  });
  controls.append(rollbackButton, alternativesButton);
  card.appendChild(controls);
}

function renderMessage(message, extraClass = "") {
  const className = ["conversation-message", `message-${message.role}`, extraClass]
    .filter(Boolean)
    .join(" ");
  const card = makeElement("article", className);
  const role = message.role === "assistant" ? "PaperPilot" : "You";
  card.appendChild(makeElement("span", "message-role", role));
  card.appendChild(makeElement("div", "message-body", message.content));
  addMessageControls(card, message);
  return card;
}

function renderMessages() {
  messageList.replaceChildren();
  const messages = state.selectedConversation ? state.selectedConversation.messages : [];
  const unstableTurn = state.selectedConversation
    ? state.selectedConversation.unstableTurn
    : null;
  if (!messages.length && unstableTurn === null) {
    messageList.appendChild(
      makeElement("p", "conversation-empty", "Ask your first question about this paper."),
    );
  }
  for (const message of messages) {
    messageList.appendChild(renderMessage(message));
  }
  if (unstableTurn !== null) {
    messageList.appendChild(renderMessage(unstableTurn.user_message, "message-unstable"));
    const taskStatus = makeElement(
      "p",
      `unstable-status status-${unstableTurn.task.status}`,
      unstableTurn.task.status === "failed"
        ? "This question failed. You can ask a new question."
        : `Question ${unstableTurn.task.status}.`,
    );
    messageList.appendChild(taskStatus);
  }
  setMutationControlsDisabled();
}

async function refreshSelectedConversation(conversationId, version) {
  const encodedConversationId = encodeURIComponent(conversationId);
  const [detail, messagesPayload] = await Promise.all([
    requestJson(`/api/conversations/${encodedConversationId}`),
    requestJson(`/api/conversations/${encodedConversationId}/messages`),
  ]);
  if (!isCurrentConversation(conversationId, version)) {
    return false;
  }
  state.selectedConversation = {
    detail,
    messages: messagesPayload.items,
    unstableTurn: messagesPayload.unstable_turn,
  };
  const task = detail.active_task
    || (state.selectedConversation.unstableTurn && state.selectedConversation.unstableTurn.task);
  const activeTaskId = (
    task && (task.status === "pending" || task.status === "running")
      ? task.id
      : null
  );
  if (activeTaskId === null) {
    clearPollTimer();
    state.activeTaskPoll = null;
  } else if (!state.activeTaskPoll || state.activeTaskPoll.taskId !== activeTaskId) {
    clearPollTimer();
    state.activeTaskPoll = {
      taskId: activeTaskId,
      eventAfterId: 0,
      artifactAfterId: 0,
      timer: null,
    };
  }
  renderConversationHeader(detail);
  renderMessages();
  return true;
}

async function loadConversation(conversationId) {
  const version = resetConversationSelection(conversationId);
  conversationTitle.textContent = "Loading conversation…";
  conversationPaper.textContent = "";
  messageList.replaceChildren();
  setConversationMessage("");
  setMutationControlsDisabled();
  try {
    const loaded = await refreshSelectedConversation(conversationId, version);
    if (!loaded) {
      return;
    }
    renderConversationListSelection();
    if (state.activeTaskPoll !== null) {
      setConversationMessage("PaperPilot is working. Progress appears below.");
      pollConversationTask(state.activeTaskPoll.taskId, version);
    }
  } catch (error) {
    if (isCurrentConversation(conversationId, version)) {
      setConversationMessage(error.message, "error");
    }
  }
}

function renderConversationListSelection() {
  for (const row of conversationList.querySelectorAll(".conversation-row")) {
    row.classList.toggle(
      "active",
      row.dataset.conversationId === state.selectedConversationId,
    );
  }
}

function appendProgressEvents(events) {
  for (const event of events) {
    const row = makeElement("div", "conversation-event");
    row.appendChild(makeElement("span", "conversation-event-stage", event.stage || event.type));
    row.appendChild(makeElement("span", "conversation-event-message", event.message));
    conversationEvents.appendChild(row);
  }
}

function scheduleConversationPoll(taskId, version) {
  clearPollTimer();
  const poll = state.activeTaskPoll;
  if (
    !poll
    || poll.taskId !== taskId
    || !isCurrentConversation(state.selectedConversationId, version)
  ) {
    return;
  }
  poll.timer = setTimeout(() => {
    if (state.activeTaskPoll === poll) {
      poll.timer = null;
      pollConversationTask(taskId, version);
    }
  }, 1000);
}

async function pollConversationTask(taskId, version) {
  const conversationId = state.selectedConversationId;
  if (!conversationId || !isCurrentConversation(conversationId, version)) {
    return;
  }
  try {
    let payload;
    do {
      const poll = state.activeTaskPoll;
      if (!poll || poll.taskId !== taskId) {
        return;
      }
      const params = new URLSearchParams({
        after_event_id: String(poll.eventAfterId),
        after_artifact_id: String(poll.artifactAfterId),
        limit: "100",
      });
      payload = await requestJson(
        `/api/conversations/${encodeURIComponent(conversationId)}`
          + `/tasks/${encodeURIComponent(taskId)}/updates?${params.toString()}`,
      );
      if (
        !isCurrentConversation(conversationId, version)
        || !state.activeTaskPoll
        || taskId !== state.activeTaskPoll.taskId
      ) {
        return;
      }
      appendProgressEvents(payload.events.items);
      state.activeTaskPoll.eventAfterId = payload.events.next_after_id;
      state.activeTaskPoll.artifactAfterId = payload.artifacts.next_after_id;
    } while (payload.events.has_more || payload.artifacts.has_more);

    if (payload.task.status === "completed" || payload.task.status === "failed") {
      state.activeTaskPoll = null;
      const refreshed = await refreshSelectedConversation(conversationId, version);
      if (!refreshed) {
        return;
      }
      setConversationMessage(
        payload.task.status === "completed"
          ? "Answer complete."
          : "The question failed. You can ask a new question.",
        payload.task.status === "completed" ? "success" : "error",
      );
      loadConversations(false);
      return;
    }
    scheduleConversationPoll(taskId, version);
  } catch (error) {
    if (!isCurrentConversation(conversationId, version)) {
      return;
    }
    setConversationMessage(`Progress update failed: ${error.message}`, "error");
    scheduleConversationPoll(taskId, version);
  }
}

async function reloadAfterConflict(conversationId, version) {
  if (!isCurrentConversation(conversationId, version)) {
    return;
  }
  try {
    await refreshSelectedConversation(conversationId, version);
    if (state.activeTaskPoll !== null) {
      pollConversationTask(state.activeTaskPoll.taskId, version);
    }
  } catch (_refreshError) {
    // The conflict notice remains actionable even if the refresh also fails.
  } finally {
    if (isCurrentConversation(conversationId, version)) {
      setConversationMessage("会话已更新，请重试", "error");
    }
  }
}

async function submitMessage() {
  if (!state.selectedConversation || hasActiveTask()) {
    return;
  }
  const content = conversationQuestion.value.trim();
  if (!content) {
    setConversationMessage("Enter a question.", "error");
    return;
  }
  const conversationId = state.selectedConversationId;
  const version = requestVersion;
  const expectedHead = state.selectedConversation.detail.conversation.head_message_id;
  try {
    const payload = await requestJson(
      `/api/conversations/${encodeURIComponent(conversationId)}/messages`,
      {
        method: "POST",
        body: JSON.stringify({
          content,
          depth: conversationDepth.value,
          expected_head_message_id: expectedHead,
        }),
      },
    );
    if (!isCurrentConversation(conversationId, version)) {
      return;
    }
    conversationQuestion.value = "";
    state.selectedConversation.unstableTurn = {
      user_message: payload.user_message,
      task: payload.task,
    };
    state.activeTaskPoll = {
      taskId: payload.task.id,
      eventAfterId: 0,
      artifactAfterId: 0,
      timer: null,
    };
    conversationEvents.replaceChildren();
    branchOptions.replaceChildren();
    branchSelector.hidden = true;
    renderMessages();
    setConversationMessage("Question queued. Progress appears below.");
    pollConversationTask(payload.task.id, version);
  } catch (error) {
    if (!isCurrentConversation(conversationId, version)) {
      return;
    }
    if (error.status === 409) {
      await reloadAfterConflict(conversationId, version);
      return;
    }
    try {
      await refreshSelectedConversation(conversationId, version);
    } catch (_refreshError) {
      // The original request failure remains the useful user-facing error.
    }
    if (isCurrentConversation(conversationId, version)) {
      setConversationMessage(error.message, "error");
    }
  }
}

async function loadAlternatives(messageId) {
  if (hasActiveTask() || !state.selectedConversation) {
    return;
  }
  const conversationId = state.selectedConversationId;
  const version = requestVersion;
  try {
    const payload = await requestJson(
      `/api/conversations/${encodeURIComponent(conversationId)}/messages/${encodeURIComponent(messageId)}/alternatives`,
    );
    if (!isCurrentConversation(conversationId, version)) {
      return;
    }
    branchOptions.replaceChildren();
    branchSelector.hidden = false;
    if (!payload.items.length) {
      branchOptions.appendChild(
        makeElement("p", "conversation-empty", "No other versions are available here."),
      );
      return;
    }
    for (const alternative of payload.items) {
      const card = makeElement("article", "branch-option");
      card.appendChild(
        makeElement("p", "branch-user", `You: ${alternative.user_message.content}`),
      );
      card.appendChild(
        makeElement("p", "branch-assistant", alternative.assistant_message.content),
      );
      const chooseButton = makeElement(
        "button",
        "secondary-button conversation-mutation",
        "Use this version",
      );
      chooseButton.type = "button";
      chooseButton.addEventListener("click", () => {
        rollbackToMessage(alternative.assistant_message.id);
      });
      card.appendChild(chooseButton);
      branchOptions.appendChild(card);
    }
    setMutationControlsDisabled();
  } catch (error) {
    if (!isCurrentConversation(conversationId, version)) {
      return;
    }
    setConversationMessage(error.message, "error");
  }
}

async function rollbackToMessage(messageId) {
  if (hasActiveTask() || !state.selectedConversation) {
    return;
  }
  const conversationId = state.selectedConversationId;
  const version = requestVersion;
  const expectedHead = state.selectedConversation.detail.conversation.head_message_id;
  try {
    await requestJson(`/api/conversations/${encodeURIComponent(conversationId)}/rollback`, {
      method: "POST",
      body: JSON.stringify({
        message_id: messageId,
        expected_head_message_id: expectedHead,
      }),
    });
    if (!isCurrentConversation(conversationId, version)) {
      return;
    }
    await loadConversation(conversationId);
    const reloadVersion = requestVersion;
    await loadConversations(false);
    if (
      !isCurrentConversation(conversationId, reloadVersion)
      || !state.selectedConversation
    ) {
      return;
    }
    setConversationMessage("Conversation rolled back.", "success");
  } catch (error) {
    if (!isCurrentConversation(conversationId, version)) {
      return;
    }
    if (error.status === 409) {
      await reloadAfterConflict(conversationId, version);
      return;
    }
    setConversationMessage(error.message, "error");
  }
}

function resetConversationWorkspace() {
  conversationListVersion += 1;
  resetConversationSelection(null);
  state.conversations = [];
  conversationList.replaceChildren();
  paperCandidates.replaceChildren();
  messageList.replaceChildren();
  conversationTitle.textContent = "Select a conversation";
  conversationPaper.textContent = "";
  setConversationMessage("");
  setPaperMessage("");
  setMutationControlsDisabled();
}

authForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitAuth("login");
});

registerButton.addEventListener("click", () => {
  submitAuth("register");
});

logoutButton.addEventListener("click", async () => {
  authenticationVersion += 1;
  const authRequestVersion = authenticationVersion;
  try {
    await requestJson("/api/auth/logout", { method: "POST" });
  } finally {
    if (authRequestVersion === authenticationVersion) {
      clearAuthenticatedUser("Logged out.");
    }
  }
});

paperSearch.addEventListener("submit", (event) => {
  event.preventDefault();
  searchPapers();
});

messageComposer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitMessage();
});

refreshConversations.addEventListener("click", () => {
  loadConversations(false);
});

closeBranchSelector.addEventListener("click", () => {
  branchSelector.hidden = true;
  branchOptions.replaceChildren();
});

setMutationControlsDisabled();
loadCurrentUser();
