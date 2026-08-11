(function () {
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
  const conversationsTab = document.querySelector("#conversationsTab");

  const conversationState = {
    selectedConversationId: null,
    selectedConversation: null,
    messages: [],
    activeTaskId: null,
    eventAfterId: 0,
    artifactAfterId: 0,
    pollTimer: null,
    requestVersion: 0,
  };

  let unstableTurn = null;
  let authenticated = false;
  let authenticationVersion = 0;
  let conversationListVersion = 0;

  function makeElement(tagName, className = "", text = "") {
    const element = document.createElement(tagName);
    if (className) {
      element.className = className;
    }
    element.textContent = text;
    return element;
  }

  function setConversationMessage(text, kind = "") {
    conversationMessage.textContent = text;
    conversationMessage.className = kind ? `message ${kind}` : "message";
  }

  function setPaperMessage(text, kind = "") {
    paperSearchMessage.textContent = text;
    paperSearchMessage.className = kind ? `message ${kind}` : "message";
  }

  function clearPollTimer() {
    if (conversationState.pollTimer !== null) {
      clearTimeout(conversationState.pollTimer);
      conversationState.pollTimer = null;
    }
  }

  function resetConversationSelection(conversationId) {
    clearPollTimer();
    conversationState.requestVersion += 1;
    conversationState.selectedConversationId = conversationId;
    conversationState.selectedConversation = null;
    conversationState.messages = [];
    conversationState.activeTaskId = null;
    conversationState.eventAfterId = 0;
    conversationState.artifactAfterId = 0;
    unstableTurn = null;
    conversationEvents.replaceChildren();
    branchOptions.replaceChildren();
    branchSelector.hidden = true;
    return conversationState.requestVersion;
  }

  function isCurrentConversation(conversationId, version) {
    return (
      authenticated
      && conversationState.selectedConversationId === conversationId
      && conversationState.requestVersion === version
    );
  }

  function isCurrentAuthentication(version) {
    return authenticated && authenticationVersion === version;
  }

  function hasActiveTask() {
    return conversationState.activeTaskId !== null;
  }

  function setMutationControlsDisabled() {
    const disabled = hasActiveTask();
    conversationQuestion.disabled = !conversationState.selectedConversation || disabled;
    conversationDepth.disabled = !conversationState.selectedConversation || disabled;
    sendConversationMessage.disabled = !conversationState.selectedConversation || disabled;
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
      if (conversation.id === conversationState.selectedConversationId) {
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
    if (!authenticated) {
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
      renderConversationList(payload.items);
      if (
        selectFirst
        && conversationState.selectedConversationId === null
        && payload.items.length
      ) {
        await loadConversation(payload.items[0].id);
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
    if (!conversationState.messages.length && unstableTurn === null) {
      messageList.appendChild(
        makeElement("p", "conversation-empty", "Ask your first question about this paper."),
      );
    }
    for (const message of conversationState.messages) {
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
    conversationState.selectedConversation = detail;
    conversationState.messages = messagesPayload.items;
    unstableTurn = messagesPayload.unstable_turn;
    const task = detail.active_task || (unstableTurn && unstableTurn.task);
    conversationState.activeTaskId = (
      task && (task.status === "pending" || task.status === "running")
        ? task.id
        : null
    );
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
      if (conversationState.activeTaskId !== null) {
        setConversationMessage("PaperPilot is working. Progress appears below.");
        pollConversationTask(conversationState.activeTaskId, version);
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
        row.dataset.conversationId === conversationState.selectedConversationId,
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
    if (!isCurrentConversation(conversationState.selectedConversationId, version)) {
      return;
    }
    conversationState.pollTimer = setTimeout(() => {
      conversationState.pollTimer = null;
      pollConversationTask(taskId, version);
    }, 1000);
  }

  async function pollConversationTask(taskId, version) {
    const conversationId = conversationState.selectedConversationId;
    if (!conversationId || !isCurrentConversation(conversationId, version)) {
      return;
    }
    try {
      let payload;
      do {
        const params = new URLSearchParams({
          after_event_id: String(conversationState.eventAfterId),
          after_artifact_id: String(conversationState.artifactAfterId),
          limit: "100",
        });
        payload = await requestJson(
          `/api/conversations/${encodeURIComponent(conversationId)}` +
            `/tasks/${encodeURIComponent(taskId)}/updates?${params.toString()}`,
        );
        if (!isCurrentConversation(conversationId, version) || taskId !== conversationState.activeTaskId) {
          return;
        }
        appendProgressEvents(payload.events.items);
        conversationState.eventAfterId = payload.events.next_after_id;
        conversationState.artifactAfterId = payload.artifacts.next_after_id;
      } while (payload.events.has_more || payload.artifacts.has_more);

      if (payload.task.status === "completed" || payload.task.status === "failed") {
        conversationState.activeTaskId = null;
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
      if (conversationState.activeTaskId !== null) {
        pollConversationTask(conversationState.activeTaskId, version);
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
    if (!conversationState.selectedConversation || hasActiveTask()) {
      return;
    }
    const content = conversationQuestion.value.trim();
    if (!content) {
      setConversationMessage("Enter a question.", "error");
      return;
    }
    const conversationId = conversationState.selectedConversationId;
    const version = conversationState.requestVersion;
    const expectedHead = conversationState.selectedConversation.conversation.head_message_id;
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
      unstableTurn = {
        user_message: payload.user_message,
        task: payload.task,
      };
      conversationState.activeTaskId = payload.task.id;
      conversationState.eventAfterId = 0;
      conversationState.artifactAfterId = 0;
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
    if (hasActiveTask() || !conversationState.selectedConversation) {
      return;
    }
    const conversationId = conversationState.selectedConversationId;
    const version = conversationState.requestVersion;
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
    if (hasActiveTask() || !conversationState.selectedConversation) {
      return;
    }
    const conversationId = conversationState.selectedConversationId;
    const version = conversationState.requestVersion;
    const expectedHead = conversationState.selectedConversation.conversation.head_message_id;
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
      const reloadVersion = conversationState.requestVersion;
      await loadConversations(false);
      if (
        !isCurrentConversation(conversationId, reloadVersion)
        || !conversationState.selectedConversation
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
    conversationList.replaceChildren();
    paperCandidates.replaceChildren();
    messageList.replaceChildren();
    conversationTitle.textContent = "Select a conversation";
    conversationPaper.textContent = "";
    setConversationMessage("");
    setPaperMessage("");
    setMutationControlsDisabled();
  }

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

  conversationsTab.addEventListener("click", () => {
    if (authenticated) {
      loadConversations(false);
    }
  });

  document.addEventListener("paperpilot:authenticated", () => {
    authenticated = true;
    authenticationVersion += 1;
    resetConversationWorkspace();
    loadConversations();
  });

  document.addEventListener("paperpilot:unauthenticated", () => {
    authenticated = false;
    authenticationVersion += 1;
    resetConversationWorkspace();
  });

  setMutationControlsDisabled();
})();
