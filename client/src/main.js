import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const $ = (id) => document.getElementById(id);

/* Elements */

const chatEl = document.querySelector('.chat');
const sidebarEl = $('sidebar');
const backdropEl = $('backdrop');
const menuToggleEl = $('menu-toggle');
const hospitalsToggleEl = $('hospitals-toggle');

const scrollEl = $('chat-scroll');
const messagesEl = $('messages');
const heroHintEl = $('hero-hint');
const formEl = $('chat-form');
const inputEl = $('message-input');
const sendEl = $('send-button');
const statusEl = $('status');
const consentsEl = $('consents');

const historyEl = $('history');
const newChatEl = $('new-chat');

const hospitalsEl = $('hospitals');
const hospitalsStatusEl = $('hospitals-status');
const hospitalsMessageEl = $('hospitals-message');
const mapEl = $('map');
const mapPlaceholderEl = $('map-placeholder');
const locateEl = $('locate');
const radiusEl = $('radius');

const modalEl = $('modal');
const modalTitleEl = $('modal-title');
const modalBodyEl = $('modal-body');
const modalAgreeEl = $('modal-agree');
const modalCloseEl = $('modal-close');

const diagnosisTpl = $('tpl-diagnosis');
const quickRepliesTpl = $('tpl-quick-replies');
const hospitalTpl = $('tpl-hospital');

/* Constants */

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
      'Your assessment history is saved in this browser only (local storage) so you can revisit it. Clear it at any time with your browser settings.',
      'On the Hospitals screen your approximate coordinates are sent through the local server to OpenStreetMap to look up hospitals, and map tiles are loaded from OpenStreetMap. They are not stored. The Roboto font is loaded from Google Fonts, which receives a standard web request when the page loads.',
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

const HINTS = {
  offline: 'Connecting to the HIRO server.',
  consent: 'Read and agree to the four notices below the box to begin.',
  ready: 'Describe your symptoms in plain language. HIRO asks a few follow-up questions before suggesting a possible condition.',
};

/* State */

let online = false;
let busy = false;
let agreed = loadJson(CONSENT_KEY, []);
let history = loadJson(HISTORY_KEY, []);
let active = null; // current conversation, or null for the empty state

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

/* Drawers (sidebar below 768 px, hospitals below 1200 px) */

function syncBackdrop() {
  backdropEl.hidden = !sidebarEl.classList.contains('sidebar--open') && !hospitalsEl.classList.contains('hospitals--open');
}

function setSidebar(open) {
  sidebarEl.classList.toggle('sidebar--open', open);
  menuToggleEl.setAttribute('aria-expanded', String(open));
  syncBackdrop();
  if (open) $('sidebar-close').focus();
}

function setHospitals(open) {
  hospitalsEl.classList.toggle('hospitals--open', open);
  hospitalsToggleEl.setAttribute('aria-expanded', String(open));
  syncBackdrop();
  if (open) $('hospitals-close').focus();
}

menuToggleEl.addEventListener('click', () => setSidebar(!sidebarEl.classList.contains('sidebar--open')));
$('sidebar-close').addEventListener('click', () => {
  setSidebar(false);
  menuToggleEl.focus();
});
hospitalsToggleEl.addEventListener('click', () => setHospitals(!hospitalsEl.classList.contains('hospitals--open')));
$('hospitals-close').addEventListener('click', () => {
  setHospitals(false);
  hospitalsToggleEl.focus();
});
backdropEl.addEventListener('click', () => {
  setSidebar(false);
  setHospitals(false);
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !modalEl.open) {
    setSidebar(false);
    setHospitals(false);
  }
});

/* Gate: input is usable only when online and every notice is agreed */

function allAgreed() {
  return Object.keys(CONSENTS).every((id) => agreed.includes(id));
}

function setStatus(text, state) {
  statusEl.dataset.state = state;
  statusEl.textContent = state === 'online' ? '' : text;
}

function updateGate() {
  const locked = busy || !online || !allAgreed();
  inputEl.disabled = locked;
  sendEl.disabled = locked;

  if (!online) {
    heroHintEl.textContent = HINTS.offline;
    inputEl.placeholder = 'Waiting for the server';
  } else if (!allAgreed()) {
    heroHintEl.textContent = HINTS.consent;
    inputEl.placeholder = 'Agree to the notices below to start';
  } else {
    heroHintEl.textContent = HINTS.ready;
    inputEl.placeholder = 'e.g. headache and mild fever';
  }

  if (!locked && document.activeElement === document.body) inputEl.focus();
}

/* Dialog */

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
  modalAgreeEl.hidden = !consentId || agreed.includes(consentId);
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

modalCloseEl.addEventListener('click', () => modalEl.close());
modalEl.addEventListener('click', (event) => {
  if (event.target === modalEl) modalEl.close();
});

/* Consents */

function renderConsents() {
  consentsEl.querySelectorAll('.consent').forEach((button) => {
    const done = agreed.includes(button.dataset.consent);
    button.setAttribute('aria-pressed', String(done));
    button.classList.toggle('consent--done', done);
    button.querySelector('input').checked = done;
  });
}

consentsEl.addEventListener('click', (event) => {
  const button = event.target.closest('.consent');
  if (button) openModal(CONSENTS[button.dataset.consent], button.dataset.consent);
});

$('how-it-works').addEventListener('click', () => {
  setSidebar(false);
  openModal(HOW_IT_WORKS);
});

$('symptom-index').addEventListener('click', async () => {
  setSidebar(false);
  try {
    const res = await fetch('/api/symptoms');
    const list = (await readJson(res)) ?? [];
    openModal({
      title: `Recognised symptoms (${list.length})`,
      body: [
        'HIRO only understands the symptom names below. Use these words, or answer its yes / no questions, for the best results.',
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
  return clean.length > 48 ? `${clean.slice(0, 48)}...` : clean;
}

function persistHistory() {
  history = history.slice(0, MAX_HISTORY);
  saveJson(HISTORY_KEY, history);
  renderHistory();
}

function renderHistory() {
  historyEl.replaceChildren(
    ...history.map((conv) => {
      const li = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.id = conv.id;
      button.className = conv.id === active?.id ? 'row row--active' : 'row';

      const title = document.createElement('span');
      title.className = 'row__title';
      title.textContent = conv.title;
      const meta = document.createElement('span');
      meta.className = 'row__meta';
      meta.textContent = conv.result ?? 'In progress';
      const chevron = document.createElement('span');
      chevron.className = 'row__chevron';
      chevron.setAttribute('aria-hidden', 'true');
      chevron.textContent = '\u203a';

      button.append(title, meta, chevron);
      li.appendChild(button);
      return li;
    }),
  );
}

historyEl.addEventListener('click', (event) => {
  const button = event.target.closest('.row');
  const conv = button && history.find((c) => c.id === button.dataset.id);
  if (!conv) return;
  openConversation(conv);
  setSidebar(false);
});

newChatEl.addEventListener('click', () => {
  startNew();
  setSidebar(false);
});

/* Rendering */

function scrollToBottom(smooth = true) {
  requestAnimationFrame(() => {
    scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
  });
}

function messageNode(role, content) {
  const wrapper = document.createElement('div');
  wrapper.className = `msg msg--${role}`;
  if (role === 'hiro') {
    const avatar = document.createElement('span');
    avatar.className = 'msg__avatar';
    avatar.setAttribute('aria-hidden', 'true');
    const icon = document.createElement('span');
    icon.className = 'logo-icon';
    avatar.appendChild(icon);
    wrapper.appendChild(avatar);
  }
  const body = document.createElement('div');
  body.className = 'msg__body';
  if (typeof content === 'string') {
    const p = document.createElement('p');
    p.textContent = content;
    body.appendChild(p);
  } else {
    body.appendChild(content);
  }
  wrapper.appendChild(body);
  return wrapper;
}

function renderMessage(msg, withQuickReplies) {
  if (msg.role === 'user') return messageNode('user', msg.text);
  if (msg.kind === 'diagnosis') return messageNode('hiro', renderDiagnosis(msg.data));
  if (msg.kind === 'question' && withQuickReplies) {
    const body = document.createElement('div');
    const p = document.createElement('p');
    p.textContent = msg.text;
    body.append(p, quickRepliesTpl.content.firstElementChild.cloneNode(true));
    return messageNode('hiro', body);
  }
  const node = messageNode('hiro', msg.text);
  if (msg.kind === 'error') node.classList.add('msg--error');
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
  chatEl.classList.toggle('has-conversation', Boolean(active));
  if (!active) return;
  const last = active.messages.length - 1;
  active.messages.forEach((msg, i) => messagesEl.appendChild(renderMessage(msg, i === last)));
  scrollToBottom(false);
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
  chatEl.classList.add('has-conversation');
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
      if (res.status === 400 || res.status === 422) throw new Error(data?.detail ?? 'The server rejected the message.');
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
  const dots = document.createElement('span');
  dots.className = 'thinking';
  dots.setAttribute('role', 'status');
  dots.setAttribute('aria-label', 'HIRO is thinking');
  dots.append(document.createElement('i'), document.createElement('i'), document.createElement('i'));
  const thinking = messageNode('hiro', dots);
  thinking.classList.add('msg--thinking');
  messagesEl.appendChild(thinking);
  scrollToBottom();

  try {
    const data = await postChat(message, active.state);
    thinking.remove();
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
    thinking.remove();
    appendMessage({ role: 'hiro', kind: 'error', text: `Something went wrong: ${err.message}. Please try again.` });
    checkHealth();
  } finally {
    busy = false;
    persistHistory();
    updateGate();
  }
}

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

document.addEventListener('keydown', (event) => {
  if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === 'o') {
    event.preventDefault();
    startNew();
    inputEl.focus();
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
  setStatus('Connecting to server', 'connecting');
  updateGate();
  for (let attempt = 1; attempt <= MAX_CONNECT_ATTEMPTS; attempt++) {
    if (await healthy()) {
      online = true;
      setStatus('Online', 'online');
      updateGate();
      if (lastPosition && !map) loadHospitals();
      return;
    }
    setStatus(`Waiting for server (${attempt})`, 'connecting');
    await sleep(Math.min(1000 * attempt, 5000));
  }
  setStatus('The HIRO server on port 8000 is not reachable. Start it and reload this page.', 'error');
}

/* Hospitals */

let lastPosition = null;
let map = null;
let markersLayer = null;
let userMarker = null;
let radiusCircle = null;
let lastBounds = null;

function setHospitalsState(state, message = '') {
  hospitalsEl.dataset.state = state;
  hospitalsMessageEl.textContent = message;
  if (mapPlaceholderEl.isConnected) mapPlaceholderEl.textContent = state === 'locating' ? 'Finding your location' : '';
}

function markerIcon(kind) {
  return L.divIcon({ className: `map-pin map-pin--${kind}`, iconSize: [14, 14], iconAnchor: [7, 7], popupAnchor: [0, -10] });
}

function ensureMap() {
  if (map) return map;
  mapPlaceholderEl.remove();
  map = L.map(mapEl, { zoomControl: true, attributionControl: true });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
  }).addTo(map);
  markersLayer = L.layerGroup().addTo(map);
  // The column resizes with the viewport and the drawer; keep tiles and bounds in step.
  new ResizeObserver(() => fitMap()).observe(mapEl);
  return map;
}

function popupFor(h) {
  const node = hospitalTpl.content.firstElementChild.cloneNode(true);
  node.querySelector('.hospital__name').textContent = h.name;
  node.querySelector('.hospital__meta').textContent = [
    h.kind === 'clinic' ? 'Clinic' : 'Hospital',
    `${h.distance_km} km`,
    h.emergency ? 'Emergency' : null,
    h.address,
  ]
    .filter(Boolean)
    .join(' \u00b7 ');
  node.querySelector('.hospital__directions').href = h.directions_url;
  const call = node.querySelector('.hospital__call');
  if (h.phone) {
    call.hidden = false;
    call.href = `tel:${h.phone.replace(/[^+\d]/g, '')}`;
  }
  return node;
}

function fitMap() {
  if (!map || !lastBounds) return;
  map.invalidateSize({ animate: false });
  map.fitBounds(lastBounds, { padding: [16, 16], animate: false });
}

function renderHospitals(list) {
  const { lat, lon } = lastPosition;
  ensureMap();
  markersLayer.clearLayers();
  // Circle bounds can only be projected once the map has a view.
  if (!map._loaded) map.setView([lat, lon], 13, { animate: false });

  const accent = getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim();
  if (radiusCircle) radiusCircle.remove();
  radiusCircle = L.circle([lat, lon], {
    radius: Number(radiusEl.value),
    color: accent,
    weight: 1,
    fillColor: accent,
    fillOpacity: 0.06,
    interactive: false,
  }).addTo(map);

  if (userMarker) userMarker.remove();
  userMarker = L.marker([lat, lon], { icon: markerIcon('you'), zIndexOffset: 1000, title: 'You' })
    .bindPopup('You are here')
    .addTo(map);

  list.forEach((h) => {
    L.marker([h.lat, h.lon], { icon: markerIcon(h.kind), title: h.name })
      .bindPopup(popupFor(h), { maxWidth: 260, minWidth: 220 })
      .addTo(markersLayer);
  });
  lastBounds = radiusCircle.getBounds();
  setTimeout(fitMap, 0);
}

async function loadHospitals() {
  if (!lastPosition) return;
  if (!online) {
    hospitalsStatusEl.textContent = 'Waiting for the server';
    return;
  }
  const { lat, lon } = lastPosition;
  hospitalsStatusEl.textContent = 'Searching OpenStreetMap';
  try {
    const params = new URLSearchParams({ lat, lon, radius: radiusEl.value });
    const res = await fetch(`/api/hospitals?${params}`);
    const data = await readJson(res);
    if (!res.ok || !Array.isArray(data)) throw new Error(data?.detail ?? `HTTP ${res.status}`);
    renderHospitals(data);
    hospitalsStatusEl.textContent = data.length
      ? `${data.length} place${data.length === 1 ? '' : 's'} within ${radiusEl.value / 1000} km`
      : `Nothing found within ${radiusEl.value / 1000} km. Try a larger radius.`;
  } catch (err) {
    renderHospitals([]);
    hospitalsStatusEl.textContent = `Could not load hospitals: ${err.message}`;
  }
}

function requestLocation() {
  if (!navigator.geolocation) {
    setHospitalsState('denied', 'Location is not supported by this browser.');
    return;
  }
  setHospitalsState('locating');
  locateEl.disabled = true;
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      locateEl.disabled = false;
      lastPosition = { lat: pos.coords.latitude.toFixed(5), lon: pos.coords.longitude.toFixed(5) };
      setHospitalsState('ready');
      loadHospitals();
    },
    (err) => {
      locateEl.disabled = false;
      setHospitalsState(
        'denied',
        err.code === err.PERMISSION_DENIED
          ? 'Location permission was denied. Allow it for this site in your browser settings, then try again.'
          : 'Could not determine your location. Check that location services are on and try again.',
      );
    },
    { enableHighAccuracy: false, timeout: 15000, maximumAge: 300000 },
  );
}

// Ask on startup; the manual button only appears when the browser will not give a position.
async function initLocation() {
  if (!navigator.geolocation) {
    setHospitalsState('denied', 'Location is not supported by this browser.');
    return;
  }
  try {
    const status = await navigator.permissions.query({ name: 'geolocation' });
    if (status.state === 'denied') {
      setHospitalsState('denied', 'Location permission was denied. Allow it for this site in your browser settings, then try again.');
      status.onchange = () => status.state !== 'denied' && requestLocation();
      return;
    }
  } catch {
    /* Permissions API unavailable: just ask. */
  }
  requestLocation();
}

locateEl.addEventListener('click', requestLocation);
radiusEl.addEventListener('change', loadHospitals);

/* Boot */

renderConsents();
renderHistory();
renderConversation();
updateGate();
connect();
initLocation();
