const isLocalFrontendOnly =
  (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") &&
  window.location.port === "5500";
const API_BASE_URL = window.CHAT_API_BASE_URL || (isLocalFrontendOnly ? "http://localhost:8000" : window.location.origin);
const CHAT_ENDPOINT = `${API_BASE_URL}/chat`;
const HEALTH_ENDPOINT = `${API_BASE_URL}/health`;

const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const statusText = document.getElementById("statusText");

let sessionId = localStorage.getItem("chat_session_id");
let isSending = false;

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.classList.toggle("error", isError);
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function createMessageElement(role, text, isError = false) {
  const messageEl = document.createElement("article");
  messageEl.className = `message ${role}`;

  const avatarEl = document.createElement("div");
  avatarEl.className = "avatar";
  avatarEl.textContent = role === "assistant" ? "AI" : "You";

  const bubbleEl = document.createElement("div");
  bubbleEl.className = "bubble";
  if (isError) {
    bubbleEl.classList.add("error");
  }
  bubbleEl.textContent = text;

  messageEl.appendChild(avatarEl);
  messageEl.appendChild(bubbleEl);
  return messageEl;
}

function appendMessage(role, text, isError = false) {
  const messageEl = createMessageElement(role, text, isError);
  chatWindow.appendChild(messageEl);
  scrollToBottom();
  return messageEl;
}

function createLoadingMessage() {
  const loadingEl = document.createElement("article");
  loadingEl.className = "message assistant";
  loadingEl.id = "loadingMessage";

  const avatarEl = document.createElement("div");
  avatarEl.className = "avatar";
  avatarEl.textContent = "AI";

  const bubbleEl = document.createElement("div");
  bubbleEl.className = "bubble loading";
  bubbleEl.innerHTML = "<span class='dot'></span><span class='dot'></span><span class='dot'></span>";

  loadingEl.appendChild(avatarEl);
  loadingEl.appendChild(bubbleEl);
  chatWindow.appendChild(loadingEl);
  scrollToBottom();
  return loadingEl;
}

function updateComposerState(disabled) {
  messageInput.disabled = disabled;
  sendButton.disabled = disabled;
}

function autoResizeTextarea() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 180)}px`;
}

function classifyStatus(httpStatus, detailText) {
  const detail = (detailText || "").toLowerCase();
  if (httpStatus === 401 || detail.includes("authentication failed")) {
    return "API key error";
  }
  if (httpStatus === 429 || detail.includes("rate limit")) {
    return "Rate limited";
  }
  if (httpStatus === 503) {
    return "Service unavailable";
  }
  if (httpStatus >= 500) {
    return "Server error";
  }
  return "Request failed";
}

async function sendMessage(message) {
  if (isSending) {
    return;
  }

  isSending = true;
  updateComposerState(true);
  setStatus("Thinking...");

  appendMessage("user", message);
  const loadingEl = createLoadingMessage();

  try {
    const response = await fetch(CHAT_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        session_id: sessionId || undefined,
      }),
    });

    let data = {};
    try {
      data = await response.json();
    } catch {
      data = {};
    }

    if (!response.ok) {
      const errorMessage = data.detail || `Request failed with status ${response.status}`;
      const error = new Error(errorMessage);
      error.httpStatus = response.status;
      throw error;
    }

    if (data.session_id) {
      sessionId = data.session_id;
      localStorage.setItem("chat_session_id", sessionId);
    }

    appendMessage("assistant", data.response || "No response was returned by the server.");
    setStatus("Ready");
  } catch (error) {
    const text = error instanceof Error ? error.message : "Unknown error while contacting the server.";
    const httpStatus = error && typeof error === "object" && "httpStatus" in error
      ? error.httpStatus
      : 0;
    const statusLabel = classifyStatus(typeof httpStatus === "number" ? httpStatus : 0, text);
    appendMessage("assistant", `Error: ${text}`, true);
    setStatus(statusLabel, true);
  } finally {
    loadingEl.remove();
    isSending = false;
    updateComposerState(false);
    messageInput.focus();
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  messageInput.value = "";
  autoResizeTextarea();
  await sendMessage(message);
});

messageInput.addEventListener("input", autoResizeTextarea);

messageInput.addEventListener("keydown", async (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    const message = messageInput.value.trim();
    if (!message) {
      return;
    }

    messageInput.value = "";
    autoResizeTextarea();
    await sendMessage(message);
  }
});

async function checkBackendHealth() {
  try {
    const response = await fetch(HEALTH_ENDPOINT, { method: "GET" });
    if (!response.ok) {
      setStatus("Backend offline", true);
      return;
    }

    const data = await response.json();
    const providerConfigured = Object.prototype.hasOwnProperty.call(data, "provider_configured")
      ? data.provider_configured
      : data.openai_configured;
    if (!providerConfigured) {
      setStatus("Provider not configured", true);
      return;
    }

    setStatus("Ready");
  } catch {
    setStatus("Backend offline", true);
  }
}

window.addEventListener("load", () => {
  checkBackendHealth();
  messageInput.focus();
});
