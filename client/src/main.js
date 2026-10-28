const $ = (id) => document.getElementById(id);

const chatEl = $('chat');
const messagesEl = $('messages');
const formEl = $('chat-form');
const inputEl = $('message-input');
const sendEl = $('send-button');
const statusEl = $('status');
const statusBarEl = $('status-bar');
const heroHintEl = $('hero-hint');
const historyEl = $('history');
const historyEmptyEl = $('history-empty');
const consentsEl = $('consents');
const chipsEl = $('chips');
const sidebarEl = $('sidebar');
const backdropEl = $('backdrop');
const menuToggleEl = $('menu-toggle');
const modalEl = $('modal');
const modalTitleEl = $('modal-title');
const modalBodyEl = $('modal-body');
const modalAgreeEl = $('modal-agree');
const diagnosisTpl = $('tpl-diagnosis');
const quickRepliesTpl = $('tpl-quick-replies');

const CONSENT_KEY = 'hiro.consents';
const HISTORY_KEY = 'hiro.history';
const MAX_HISTORY = 30;
const MAX_CONNECT_ATTEMPTS = 120;
const SEND_RETRIES = 3;

const CONSENTS = {
  advice: {
    title: 'Not medical advice',
    body: [
      'HIRO is an educational symptom checker. Its results are produced by machine-learning models trained on a small public dataset and may be wrong or incomplete.',
      'It cannot diagnose, treat or rule out any condition, and it does not know your medical history, medications or allergies.',
      'Always consult a qualified healthcare professional before acting on anything HIRO suggests.',
    ],
  },
  emergency: {
    title: 'Emergencies',
    body: [
      'HIRO is not an emergency service and cannot assess the severity of your condition.',
      'If you have chest pain, difficulty breathing, severe bleeding, sudden weakness or numbness, confusion, a seizure, a severe allergic reaction, or any symptom that feels life-threatening, stop using HIRO and contact your local emergency number immediately.',
    ],
  },
  privacy: {
    title: 'Privacy',
    body: [
      'Your messages are sent only to the local HIRO server to generate a reply. The server keeps nothing between requests; the state of this assessment travels with each message.',
      'Your assessment history is saved in this browser only (local storage) so you can revisit it. Clear it at any time with your browser settings. Nothing is sent to third parties.',
    ],
  },
  terms: {
    title: 'Terms of use',
    body: [
      'By using HIRO you accept that it is provided as-is for educational purposes without any warranty of accuracy, availability or fitness for a particular purpose.',
      'You are responsible for decisions about your health. The authors are not liable for any loss or harm arising from use of this tool.',
      'Do not enter information about other people without their consent.',
    ],
  },
};

const HOW_IT_WORKS = {
  title: 'How it works',
  body: [
    '1. Describe what you are feeling in plain language, for example "high fever and headache".',
    '2. HIRO picks out the symptoms it recognises and asks a few yes / no follow-up questions, choosing the ones that best narrow down the possibilities. Answer with the buttons or type freely; say "that\'s all" at any point to skip ahead.',
    '3. Three classifiers (Random Forest, SVC, Naive Bayes) vote on the recorded symptoms. You get a possible condition, a confidence level based on how many symptoms were gathered and whether the models agree, a short description and suggested precautions.',
    'Use the result as a starting point for a conversation with a doctor, not as a diagnosis.',
  ],
};

const INTRO =
  'Hello, I am HIRO. Describe what you are feeling and I will ask a few follow-up questions before suggesting a possible condition. This is not a medical diagnosis. Say "that\'s all" at any time to see the result.';

let online = false;
let busy = false;
let agreed = loadJson(CONSENT_KEY, []);
let history = loadJson(HISTORY_KEY, []);
let active = null; // current conversation object or null when on the empty screen

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
  } catch {
    /* storage full or disabled */
  }
}

function newId() {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

async function readJson(res) {
  const type = res.headers.get('content-type') ?? '';
  return type.includes('application/json') ? res.json() : null;
}

function setStatus(text, state) {
  statusEl.textContent = text;
  statusBarEl.dataset.state = state;
}

function allAgreed() {
  return Object.keys(CONSENTS).every((id) => agreed.includes(id));
}

function updateGate() {
  const locked = busy || !online || !allAgreed();
  inputEl.disabled = locked;
  sendEl.disabled = locked;
  chipsEl.classList.toggle('chips-row--disabled', locked);

  if (!online) {
    heroHintEl.textContent = 'Connecting to the HIRO server.';
    inputEl.placeholder = 'Waiting for the server';
  } else if (!allAgreed()) {
    const remaining = Object.keys(CONSENTS).length - agreed.length;
    heroHintEl.textContent = `Open and agree to the ${remaining === 4 ? 'four' : remaining} remaining notice${remaining === 1 ? '' : 's'} in the sidebar to begin.`;
    inputEl.placeholder = 'Agree to the notices in the sidebar to start';
  } else {
    heroHintEl.textContent = 'Describe your symptoms in plain language. HIRO will ask a few questions before suggesting a possible condition.';
    inputEl.placeholder = 'Describe your symptoms, e.g. headache and mild fever';
  }
  if (!locked && document.activeElement !== inputEl && !modalEl.open) inputEl.focus();
}

/* Drawer */

function setDrawer(open) {
  sidebarEl.classList.toggle('sidebar--open', open);
  backdropEl.hidden = !open;
  menuToggleEl.setAttribute('aria-expanded', String(open));
}

menuToggleEl.addEventListener('click', () => setDrawer(!sidebarEl.classList.contains('sidebar--open')));
backdropEl.addEventListener('click', () => setDrawer(false));
$('sidebar-close').addEventListener('click', () => setDrawer(false));

/* Modal */

let modalConsentId = null;

function openModal({ title, body }, consentId = null) {
  modalConsentId = consentId;
  modalTitleEl.textContent = title;
  modalBodyEl.replaceChildren(
    ...body.map((text) => {
      const p = document.createElement('p');
      p.textContent = text;
      return p;
    }),
  );
  const alreadyAgreed = consentId && agreed.includes(consentId);
  modalAgreeEl.hidden = !consentId || alreadyAgreed;
  modalEl.showModal();
}

modalAgreeEl.addEventListener('click', () => {
  if (modalConsentId && !agreed.includes(modalConsentId)) {
    agreed = [...agreed, modalConsentId];
    saveJson(CONSENT_KEY, agreed);
    renderConsents();
    updateGate();
  }
  modalEl.close();
});

$('modal-close').addEventListener('click', () => modalEl.close());
modalEl.addEventListener('click', (event) => {
  if (event.target === modalEl) modalEl.close();
});

/* Consents */

function renderConsents() {
  consentsEl.querySelectorAll('.consent').forEach((button) => {
    const done = agreed.includes(button.dataset.consent);
    button.setAttribute('aria-pressed', String(done));
    button.classList.toggle('consent--done', done);
  });
}

consentsEl.addEventListener('click', (event) => {
  const button = event.target.closest('.consent');
  if (!button) return;
  const id = button.dataset.consent;
  openModal(CONSENTS[id], id);
});

$('how-it-works').addEventListener('click', () => {
  setDrawer(false);
  openModal(HOW_IT_WORKS);
});

$('symptom-index').addEventListener('click', async () => {
  setDrawer(false);
  try {
    const res = await fetch('/api/symptoms');
    const list = (await readJson(res)) ?? [];
    openModal({
      title: `Recognised symptoms (${list.length})`,
      body: [
        'HIRO only understands the symptom names below. Use these words (or answer its yes / no questions) for the best results.',
        list.join(', '),
      ],
    });
  } catch {
    openModal({ title: 'Recognised symptoms', body: ['The symptom list is unavailable while the server is offline.'] });
  }
});

/* History */

function titleFor(text) {
  const clean = text.replace(/^i have\s+/i, '').trim();
  return clean.length > 42 ? `${clean.slice(0, 42)}...` : clean;
}

function persistHistory() {
  history = history.slice(0, MAX_HISTORY);
  saveJson(HISTORY_KEY, history);
  renderHistory();
}

function renderHistory() {
  historyEmptyEl.hidden = history.length > 0;
  historyEl.replaceChildren(
    ...history.map((conv) => {
      const li = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.id = conv.id;
      button.className = conv.id === active?.id ? 'history__item history__item--active' : 'history__item';
      button.textContent = conv.title;
      if (conv.result) {
        const meta = document.createElement('span');
        meta.className = 'history__meta';
        meta.textContent = conv.result;
        button.appendChild(meta);
      }
      li.appendChild(button);
      return li;
    }),
  );
}

historyEl.addEventListener('click', (event) => {
  const button = event.target.closest('.history__item');
  if (!button) return;
  const conv = history.find((c) => c.id === button.dataset.id);
  if (!conv) return;
  setDrawer(false);
  openConversation(conv);
});

/* Rendering */

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function bubble(role, content) {
  const wrapper = document.createElement('div');
  wrapper.className = `bubble bubble--${role}`;
  const label = document.createElement('span');
  label.className = 'bubble__label';
  label.textContent = role === 'user' ? 'You' : 'HIRO';
  wrapper.appendChild(label);
  if (typeof content === 'string') {
    const p = document.createElement('p');
    p.textContent = content;
    wrapper.appendChild(p);
  } else {
    wrapper.appendChild(content);
  }
  return wrapper;
}

function renderMessage(msg, withQuickReplies) {
  if (msg.role === 'user') return bubble('user', msg.text);
  if (msg.kind === 'diagnosis') return bubble('hiro', renderDiagnosis(msg.data));
  if (msg.kind === 'question' && withQuickReplies) {
    const body = document.createElement('div');
    const p = document.createElement('p');
    p.textContent = msg.text;
    body.append(p, quickRepliesTpl.content.firstElementChild.cloneNode(true));
    return bubble('hiro', body);
  }
  const node = bubble('hiro', msg.text);
  if (msg.kind === 'error') node.classList.add('bubble--error');
  return node;
}

function renderDiagnosis(data) {
  const node = diagnosisTpl.content.firstElementChild.cloneNode(true);

  const confidence = node.querySelector('.result__confidence');
  confidence.textContent = `${data.confidence} confidence`;
  confidence.dataset.level = data.confidence;

  const symptoms = node.querySelector('.result__symptoms');
  data.symptoms.forEach((s) => {
    const li = document.createElement('li');
    li.textContent = s;
    symptoms.appendChild(li);
  });

  node.querySelector('.result__disease').textContent = data.final_prediction;
  node.querySelector('.result__description').textContent = data.description;

  const tests = node.querySelector('.result__tests');
  Object.entries(data.tests).forEach(([model, disease]) => {
    const li = document.createElement('li');
    li.dataset.agrees = String(disease === data.final_prediction);
    const name = document.createElement('span');
    name.textContent = model;
    const result = document.createElement('strong');
    result.textContent = disease;
    li.append(name, result);
    tests.appendChild(li);
  });

  const precautions = node.querySelector('.result__precautions');
  if (data.precautions.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'No precautions are available for this condition.';
    precautions.appendChild(li);
  } else {
    data.precautions.forEach((p) => {
      const li = document.createElement('li');
      li.textContent = p;
      precautions.appendChild(li);
    });
  }
  return node;
}

function renderConversation() {
  messagesEl.replaceChildren();
  if (!active) {
    chatEl.classList.add('chat--empty');
    return;
  }
  chatEl.classList.remove('chat--empty');
  const last = active.messages.length - 1;
  active.messages.forEach((msg, i) => {
    messagesEl.appendChild(renderMessage(msg, i === last));
  });
  scrollToBottom();
}

function appendMessage(msg) {
  active.messages.push(msg);
  messagesEl.querySelectorAll('.quick-replies').forEach((el) => el.remove());
  messagesEl.appendChild(renderMessage(msg, true));
  scrollToBottom();
}

/* Conversations */

function openConversation(conv) {
  active = conv;
  renderConversation();
  renderHistory();
  updateGate();
}

function startNew() {
  active = null;
  renderConversation();
  renderHistory();
  updateGate();
}

function ensureActive(firstMessage) {
  if (active) return;
  active = {
    id: newId(),
    title: titleFor(firstMessage),
    createdAt: Date.now(),
    result: null,
    state: null,
    messages: [{ role: 'hiro', kind: 'text', text: INTRO }],
  };
  history = [active, ...history];
  chatEl.classList.remove('chat--empty');
  messagesEl.replaceChildren(renderMessage(active.messages[0], false));
}

async function postChat(message, state) {
  let lastError = null;
  for (let attempt = 1; attempt <= SEND_RETRIES; attempt++) {
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, state }),
      });
      const data = await readJson(res);
      if (res.ok && data) return data;
      if (res.status === 400 || res.status === 422) {
        throw new Error(data?.detail ?? 'The server rejected the message.');
      }
      lastError = new Error(data?.detail ?? `HTTP ${res.status}`);
    } catch (err) {
      if (err.message.startsWith('The server rejected')) throw err;
      lastError = err;
    }
    setStatus(`Reconnecting (${attempt})`, 'connecting');
    await sleep(700 * attempt);
  }
  throw lastError ?? new Error('Unknown error');
}

async function sendMessage(message) {
  if (busy || !online || !allAgreed()) return;
  ensureActive(message);
  appendMessage({ role: 'user', text: message });
  persistHistory();

  busy = true;
  updateGate();
  const typing = bubble('hiro', Object.assign(document.createElement('p'), { className: 'typing', textContent: 'Thinking' }));
  messagesEl.appendChild(typing);
  scrollToBottom();

  try {
    const data = await postChat(message, active.state);
    typing.remove();
    active.state = data.state;
    setStatus('Online', 'online');

    if (data.type === 'diagnosis') {
      active.result = data.final_prediction;
      appendMessage({ role: 'hiro', kind: 'diagnosis', data });
    } else if (data.type === 'question') {
      appendMessage({ role: 'hiro', kind: 'question', text: data.reply });
    } else {
      appendMessage({ role: 'hiro', kind: 'text', text: data.reply });
    }
  } catch (err) {
    typing.remove();
    appendMessage({ role: 'hiro', kind: 'error', text: `Something went wrong: ${err.message}. Please try again.` });
    checkHealth();
  } finally {
    busy = false;
    persistHistory();
    updateGate();
  }
}

/* Events */

formEl.addEventListener('submit', (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = '';
  sendMessage(message);
});

messagesEl.addEventListener('click', (event) => {
  const button = event.target.closest('.quick-replies button');
  if (button) sendMessage(button.textContent.trim());
});

chipsEl.addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (button && !inputEl.disabled) sendMessage(`I have ${button.textContent.trim().toLowerCase()}`);
});

$('new-chat').addEventListener('click', () => {
  setDrawer(false);
  startNew();
});
$('composer-new').addEventListener('click', startNew);

document.addEventListener('keydown', (event) => {
  if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'o') {
    event.preventDefault();
    startNew();
  }
});

/* Connectivity */

async function healthy() {
  try {
    const res = await fetch('/api/health', { cache: 'no-store' });
    return res.ok;
  } catch {
    return false;
  }
}

async function checkHealth() {
  if (await healthy()) {
    online = true;
    setStatus('Online', 'online');
  } else {
    online = false;
    connect();
  }
  updateGate();
}

// The server binds its port only after the ML models finish loading, so keep retrying.
async function connect() {
  setStatus('Connecting', 'connecting');
  updateGate();
  for (let attempt = 1; attempt <= MAX_CONNECT_ATTEMPTS; attempt++) {
    if (await healthy()) {
      online = true;
      setStatus('Online', 'online');
      updateGate();
      return;
    }
    setStatus(`Waiting for server (${attempt})`, 'connecting');
    await sleep(Math.min(1000 * attempt, 5000));
  }
  setStatus('Server unavailable', 'error');
  heroHintEl.textContent = 'The HIRO server on port 8000 is not reachable. Start it and reload this page.';
}

renderConsents();
renderHistory();
renderConversation();
updateGate();
connect();
