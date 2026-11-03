import {
  $,
  diagnosisCard,
  el,
  historyRow,
  meterRow,
  message,
  paragraphs,
  quickReplies,
  replaceChildren,
  thinking,
  typeInto,
} from './components.js';
import { API, CONSENTS, HINTS, HOW_IT_WORKS, INSIGHT, INTRO, LIMITS, PLACEHOLDERS, QUICK_REPLIES, STATUS, STORAGE, TYPING } from './constants.js';

const ui = {
  chat: document.querySelector('.chat'),
  sidebar: $('sidebar'),
  backdrop: $('backdrop'),
  menuToggle: $('menu-toggle'),
  insightToggle: $('insight-toggle'),
  scroll: $('chat-scroll'),
  messages: $('messages'),
  heroHint: $('hero-hint'),
  form: $('chat-form'),
  input: $('message-input'),
  send: $('send-button'),
  status: $('status'),
  consents: $('consents'),
  history: $('history'),
  newChat: $('new-chat'),
  insight: $('insight'),
  insightStatus: $('insight-status'),
  insightAgreement: $('insight-agreement'),
  insightModels: $('insight-models'),
  insightCandidates: $('insight-candidates'),
  modal: $('modal'),
  modalTitle: $('modal-title'),
  modalBody: $('modal-body'),
  modalAgree: $('modal-agree'),
  modalClose: $('modal-close'),
};

const state = {
  online: false,
  busy: false,
  agreed: loadJson(STORAGE.consents, []),
  history: loadJson(STORAGE.history, []),
  active: null,
  modalConsent: null,
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function saveJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {}
}

const newId = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`);

async function readJson(res) {
  const type = res.headers.get('content-type') ?? '';
  return type.includes('application/json') ? res.json() : null;
}

function drawer(panel, toggle, closeButton) {
  const set = (open) => {
    panel.classList.toggle('drawer-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    ui.backdrop.hidden = !document.querySelector('.drawer-open');
    if (open) closeButton.focus();
    else if (document.activeElement === closeButton) toggle.focus();
  };
  toggle.addEventListener('click', () => set(!panel.classList.contains('drawer-open')));
  closeButton.addEventListener('click', () => set(false));
  return set;
}

const setSidebar = drawer(ui.sidebar, ui.menuToggle, $('sidebar-close'));
const setInsight = drawer(ui.insight, ui.insightToggle, $('insight-close'));
const closeDrawers = () => {
  setSidebar(false);
  setInsight(false);
};

ui.backdrop.addEventListener('click', closeDrawers);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !ui.modal.open) closeDrawers();
});

const allAgreed = () => Object.keys(CONSENTS).every((id) => state.agreed.includes(id));

function setStatus(text, kind) {
  ui.status.dataset.state = kind;
  ui.status.textContent = kind === 'online' ? '' : text;
}

function updateGate() {
  const locked = state.busy || !state.online || !allAgreed();
  ui.input.disabled = locked;
  ui.send.disabled = locked;
  const mode = !state.online ? 'offline' : !allAgreed() ? 'consent' : 'ready';
  ui.heroHint.textContent = HINTS[mode];
  ui.input.placeholder = PLACEHOLDERS[mode];
  if (!locked && document.activeElement === document.body) ui.input.focus();
}

function openModal({ title, body }, consentId = null) {
  state.modalConsent = consentId;
  ui.modalTitle.textContent = title;
  replaceChildren(ui.modalBody, paragraphs(body));
  ui.modalAgree.hidden = !consentId || state.agreed.includes(consentId);
  ui.modal.showModal();
}

ui.modalAgree.addEventListener('click', () => {
  if (state.modalConsent && !state.agreed.includes(state.modalConsent)) {
    state.agreed = [...state.agreed, state.modalConsent];
    saveJson(STORAGE.consents, state.agreed);
    renderConsents();
    updateGate();
  }
  ui.modal.close();
});
ui.modalClose.addEventListener('click', () => ui.modal.close());
ui.modal.addEventListener('click', (event) => event.target === ui.modal && ui.modal.close());

function renderConsents() {
  ui.consents.querySelectorAll('.consent').forEach((node) => {
    const done = state.agreed.includes(node.dataset.consent);
    node.setAttribute('aria-pressed', String(done));
    node.classList.toggle('consent-done', done);
    node.querySelector('input').checked = done;
  });
}

ui.consents.addEventListener('click', (event) => {
  const node = event.target.closest('.consent');
  if (node) openModal(CONSENTS[node.dataset.consent], node.dataset.consent);
});

$('how-it-works').addEventListener('click', () => {
  setSidebar(false);
  openModal(HOW_IT_WORKS);
});

$('symptom-index').addEventListener('click', async () => {
  setSidebar(false);
  try {
    const list = (await readJson(await fetch(API.symptoms))) ?? [];
    openModal({
      title: `Recognised symptoms (${list.length})`,
      body: ['HIRO understands the symptom names below plus everyday phrasings of them. You can also answer its yes / no questions.', list.join(', ')],
    });
  } catch {
    openModal({ title: 'Recognised symptoms', body: ['The symptom list is unavailable while the server is offline.'] });
  }
});

function titleFor(text) {
  const clean = text.replace(/^i have\s+/i, '').trim();
  return clean.length > LIMITS.titleLength ? `${clean.slice(0, LIMITS.titleLength)}...` : clean;
}

function persistHistory() {
  state.history = state.history.slice(0, LIMITS.history);
  saveJson(STORAGE.history, state.history);
  renderHistory();
}

function renderHistory() {
  replaceChildren(
    ui.history,
    state.history.map((conv) => historyRow(conv, { active: conv.id === state.active?.id, fallbackMeta: 'In progress' })),
  );
}

ui.history.addEventListener('click', (event) => {
  const row = event.target.closest('.row');
  const conv = row && state.history.find((c) => c.id === row.dataset.id);
  if (!conv) return;
  openConversation(conv);
  setSidebar(false);
});

ui.newChat.addEventListener('click', () => {
  startNew();
  setSidebar(false);
});

function scrollToBottom(smooth = true) {
  requestAnimationFrame(() => ui.scroll.scrollTo({ top: ui.scroll.scrollHeight, behavior: smooth ? 'smooth' : 'auto' }));
}

function renderMessage(msg, last) {
  if (msg.role === 'user') return message('user', msg.text);
  if (msg.kind === 'diagnosis') {
    return message(
      'hiro',
      diagnosisCard(msg.data, {
        emptyPrecautions: 'No precautions are available for this condition.',
        notice: 'This is a model prediction, not a medical diagnosis. Please consult a doctor to confirm.',
      }),
    );
  }
  if (msg.kind === 'question' && last) {
    return message('hiro', el('div', {}, [el('p', { class: 'whitespace-pre-wrap', text: msg.text }), quickReplies(QUICK_REPLIES)]));
  }
  return message('hiro', msg.text, { error: msg.kind === 'error' });
}

function renderConversation() {
  ui.chat.classList.toggle('has-conversation', Boolean(state.active));
  const msgs = state.active?.messages ?? [];
  replaceChildren(
    ui.messages,
    msgs.map((msg, i) => renderMessage(msg, i === msgs.length - 1)),
  );
  scrollToBottom(false);
}

async function appendMessage(msg, { animate = false } = {}) {
  state.active.messages.push(msg);
  ui.messages.querySelectorAll('.quick-replies').forEach((node) => node.remove());
  const node = renderMessage(msg, true);
  ui.messages.appendChild(node);
  scrollToBottom();
  if (!animate || msg.role !== 'hiro' || !msg.text) return;
  const paragraph = node.querySelector('p');
  const replies = node.querySelector('.quick-replies');
  replies?.classList.add('hidden');
  await typeInto(paragraph, msg.text, TYPING, () => scrollToBottom(false));
  replies?.classList.remove('hidden');
  scrollToBottom();
}

function openConversation(conv) {
  state.active = conv;
  renderConversation();
  renderHistory();
  renderInsight();
  updateGate();
}

function startNew() {
  openConversation(null);
}

function ensureActive(firstMessage) {
  if (state.active) return;
  state.active = {
    id: newId(),
    title: titleFor(firstMessage),
    createdAt: Date.now(),
    result: null,
    state: null,
    insight: null,
    messages: [{ role: 'hiro', kind: 'text', text: INTRO }],
  };
  state.history = [state.active, ...state.history];
  ui.chat.classList.add('has-conversation');
  replaceChildren(ui.messages, renderMessage(state.active.messages[0], false));
}

async function postChat(message, convState) {
  let lastError = null;
  for (let attempt = 1; attempt <= LIMITS.sendRetries; attempt++) {
    let waitMs = 700 * attempt;
    let label = STATUS.reconnecting(attempt);
    try {
      const res = await fetch(API.chat, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, state: convState }),
      });
      const data = await readJson(res);
      if (res.ok && data) return data;
      if (res.status === 400 || res.status === 422) throw new Error(data?.detail ?? 'The server rejected the message.');
      if (res.status === 503) {
        waitMs = Math.max(waitMs, Number(res.headers.get('Retry-After') ?? 0) * 1000);
        label = STATUS.busy(attempt);
      }
      lastError = new Error(data?.detail ?? `HTTP ${res.status}`);
    } catch (err) {
      if (err.message.startsWith('The server rejected')) throw err;
      lastError = err;
    }
    setStatus(label, 'connecting');
    await sleep(waitMs);
  }
  throw lastError ?? new Error('Unknown error');
}

async function sendMessage(text) {
  if (state.busy || !state.online || !allAgreed()) return;
  ensureActive(text);
  appendMessage({ role: 'user', text });
  persistHistory();

  state.busy = true;
  updateGate();
  const pending = thinking();
  ui.messages.appendChild(pending);
  scrollToBottom();

  try {
    const data = await postChat(text, state.active.state);
    pending.remove();
    state.active.state = data.state;
    if (data.insight) state.active.insight = { ...data.insight, final: data.type === 'diagnosis' };
    renderInsight();
    setStatus('', 'online');
    if (data.type === 'diagnosis') {
      state.active.result = data.final_prediction;
      appendMessage({ role: 'hiro', kind: 'diagnosis', data });
    } else {
      await appendMessage({ role: 'hiro', kind: data.type === 'question' ? 'question' : 'text', text: data.reply }, { animate: true });
    }
  } catch (err) {
    pending.remove();
    appendMessage({ role: 'hiro', kind: 'error', text: `Something went wrong: ${err.message}. Please try again.` });
    checkHealth();
  } finally {
    state.busy = false;
    persistHistory();
    renderInsight();
    updateGate();
  }
}

ui.form.addEventListener('submit', (event) => {
  event.preventDefault();
  const text = ui.input.value.trim();
  if (!text) return;
  ui.input.value = '';
  sendMessage(text);
});

ui.messages.addEventListener('click', (event) => {
  const chip = event.target.closest('.quick-replies button');
  if (chip) sendMessage(chip.textContent.trim());
});

document.addEventListener('keydown', (event) => {
  if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'o') {
    event.preventDefault();
    startNew();
    ui.input.focus();
  }
});

async function healthy() {
  try {
    return (await fetch(API.health, { cache: 'no-store' })).ok;
  } catch {
    return false;
  }
}

async function checkHealth() {
  state.online = await healthy();
  if (state.online) setStatus('', 'online');
  else connect();
  updateGate();
}

async function connect() {
  setStatus(STATUS.connecting, 'connecting');
  updateGate();
  for (let attempt = 1; attempt <= LIMITS.connectAttempts; attempt++) {
    if (await healthy()) {
      state.online = true;
      setStatus('', 'online');
      updateGate();
      return;
    }
    setStatus(STATUS.waiting(attempt), 'connecting');
    await sleep(Math.min(1000 * attempt, 5000));
  }
  setStatus(STATUS.unreachable, 'error');
}

function renderInsight() {
  const insight = state.active?.insight;
  ui.insight.dataset.state = insight ? 'ready' : 'empty';
  if (!insight) return;
  ui.insightStatus.textContent = INSIGHT.symptoms(insight.symptom_count) + (insight.final ? INSIGHT.final : INSIGHT.live);
  ui.insightAgreement.textContent = INSIGHT.agreement(insight.agreement, insight.model_count);
  ui.insightAgreement.dataset.level = INSIGHT.level(insight.agreement, insight.model_count);
  replaceChildren(
    ui.insightModels,
    insight.models.map((m, i) =>
      meterRow({ label: m.name, meta: m.prediction, value: m.probability, color: i + 1, muted: !m.agrees, metaNote: m.agrees ? null : INSIGHT.disagrees }),
    ),
  );
  replaceChildren(
    ui.insightCandidates,
    insight.candidates.map((c, i) => meterRow({ label: c.disease, value: c.probability, color: i + 1 })),
  );
}

renderConsents();
renderHistory();
renderConversation();
renderInsight();
updateGate();
connect();
