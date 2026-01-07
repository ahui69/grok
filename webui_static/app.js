const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const healthInfo = document.getElementById("healthInfo");
const sessionsEl = document.getElementById("sessions");
const attachmentsEl = document.getElementById("attachments");
const chatEl = document.getElementById("chat");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const newSessionBtn = document.getElementById("newSession");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const messageInput = document.getElementById("messageInput");
const sessionTitle = document.getElementById("sessionTitle");
const userName = document.getElementById("userName");
const projectName = document.getElementById("projectName");

const state = {
  sessionId: null,
  pending: false,
  sessions: [],
};

function setStatus(ok, text) {
  statusText.textContent = text;
  statusDot.classList.toggle("ok", ok);
  statusDot.classList.toggle("err", !ok);
}

function setHealthInfo(data) {
  if (!data) {
    healthInfo.textContent = "";
    return;
  }
  const parts = [
    data.grok_model ? `model: ${data.grok_model}` : null,
    data.promot_loaded ? "promot: ok" : "promot: brak",
    data.memory_started ? "memory: on" : "memory: off",
  ].filter(Boolean);
  healthInfo.textContent = parts.join(" · ");
}

function escapeText(text) {
  return (text || "").toString();
}

function addMessage(role, text, context) {
  const card = document.createElement("div");
  card.className = `message ${role}`;
  card.textContent = escapeText(text);

  if (context && context.length) {
    const details = document.createElement("details");
    details.className = "context";
    const summary = document.createElement("summary");
    summary.textContent = "Pamięć (kontekst)";
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(context.slice(0, 12), null, 2);
    details.appendChild(summary);
    details.appendChild(pre);
    card.appendChild(details);
  }

  chatEl.appendChild(card);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function clearChat() {
  chatEl.innerHTML = "";
}

function setSessionTitle(text) {
  sessionTitle.textContent = text || "Aktywna sesja";
}

function setActiveSession(id) {
  state.sessionId = id;
  for (const item of sessionsEl.querySelectorAll(".list-item")) {
    item.classList.toggle("active", item.dataset.sessionId === id);
  }
}

async function loadHealth() {
  try {
    const res = await fetch("/health");
    if (!res.ok) {
      throw new Error(`health ${res.status}`);
    }
    const data = await res.json();
    setStatus(true, "Online");
    setHealthInfo(data);
  } catch (err) {
    setStatus(false, "Brak połączenia");
    setHealthInfo(null);
  }
}

function renderSessions(list) {
  sessionsEl.innerHTML = "";
  if (!list.length) {
    const empty = document.createElement("div");
    empty.className = "list-item";
    empty.textContent = "Brak sesji";
    sessionsEl.appendChild(empty);
    return;
  }
  for (const s of list) {
    const item = document.createElement("div");
    item.className = "list-item";
    item.dataset.sessionId = s.session_id;
    item.textContent = s.title || s.session_id;
    item.addEventListener("click", () => {
      setActiveSession(s.session_id);
      setSessionTitle(`Sesja: ${s.title || s.session_id}`);
      clearChat();
      addMessage("system", "Załadowano sesję. Historia czatu jest lokalna w tej karcie.");
      loadAttachments();
    });
    sessionsEl.appendChild(item);
  }
  if (state.sessionId) {
    setActiveSession(state.sessionId);
  }
}

async function loadSessions() {
  try {
    const res = await fetch("/api/session/list");
    if (!res.ok) {
      throw new Error(`sessions ${res.status}`);
    }
    const data = await res.json();
    state.sessions = data.sessions || [];
    renderSessions(state.sessions);
  } catch (err) {
    renderSessions([]);
  }
}

function renderAttachments(items) {
  attachmentsEl.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "list-item";
    empty.textContent = "Brak plików";
    attachmentsEl.appendChild(empty);
    return;
  }
  for (const a of items) {
    const item = document.createElement("div");
    item.className = "list-item";
    const link = document.createElement("a");
    link.textContent = a.filename;
    link.href = a.download_url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.style.color = "inherit";
    link.style.textDecoration = "none";
    item.appendChild(link);
    attachmentsEl.appendChild(item);
  }
}

async function loadAttachments() {
  if (!state.sessionId) {
    renderAttachments([]);
    return;
  }
  try {
    const res = await fetch(`/api/attachments?session_id=${encodeURIComponent(state.sessionId)}`);
    if (!res.ok) {
      throw new Error(`attachments ${res.status}`);
    }
    const data = await res.json();
    renderAttachments(data.items || []);
  } catch (err) {
    renderAttachments([]);
  }
}

async function newSession() {
  try {
    const res = await fetch("/api/session/new", { method: "POST" });
    if (!res.ok) {
      throw new Error(`new session ${res.status}`);
    }
    const data = await res.json();
    setActiveSession(data.session_id);
    setSessionTitle(`Sesja: ${data.session_id}`);
    clearChat();
    addMessage("system", "Nowa sesja gotowa.");
    await loadSessions();
    await loadAttachments();
  } catch (err) {
    addMessage("system", "Nie udało się utworzyć sesji.");
  }
}

async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || state.pending) {
    return;
  }
  addMessage("user", text);
  messageInput.value = "";
  state.pending = true;
  sendBtn.disabled = true;

  const payload = {
    session_id: state.sessionId,
    message: text,
    user: userName.value.trim() || null,
    project: projectName.value.trim() || null,
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      throw new Error(`chat ${res.status}`);
    }
    const data = await res.json();
    setActiveSession(data.session_id);
    setSessionTitle(`Sesja: ${data.session_id}`);
    addMessage("assistant", data.answer, data.context || []);
    await loadSessions();
    await loadAttachments();
  } catch (err) {
    addMessage("system", "Błąd połączenia lub odpowiedzi.");
  } finally {
    state.pending = false;
    sendBtn.disabled = false;
  }
}

async function uploadFile(file) {
  if (!file) {
    return;
  }
  if (!state.sessionId) {
    await newSession();
  }
  const fd = new FormData();
  fd.append("session_id", state.sessionId);
  fd.append("user", userName.value.trim() || "");
  fd.append("project", projectName.value.trim() || "");
  fd.append("file", file);

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: fd,
    });
    if (!res.ok) {
      throw new Error(`upload ${res.status}`);
    }
    const data = await res.json();
    setActiveSession(data.session_id);
    addMessage("system", `Dodano plik: ${data.attachment.filename}`);
    await loadAttachments();
  } catch (err) {
    addMessage("system", "Upload nie powiódł się.");
  }
}

function initInputs() {
  userName.value = localStorage.getItem("nm_user") || "";
  projectName.value = localStorage.getItem("nm_project") || "";
  userName.addEventListener("input", () => {
    localStorage.setItem("nm_user", userName.value);
  });
  projectName.addEventListener("input", () => {
    localStorage.setItem("nm_project", projectName.value);
  });
}

sendBtn.addEventListener("click", sendMessage);
clearBtn.addEventListener("click", clearChat);
newSessionBtn.addEventListener("click", newSession);
uploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (evt) => uploadFile(evt.target.files[0]));
messageInput.addEventListener("keydown", (evt) => {
  if (evt.key === "Enter" && !evt.shiftKey) {
    evt.preventDefault();
    sendMessage();
  }
});

initInputs();
loadHealth();
loadSessions();
setInterval(loadHealth, 15000);
