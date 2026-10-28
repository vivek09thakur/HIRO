const messagesEl = document.getElementById('messages');
const formEl = document.getElementById('chat-form');
const inputEl = document.getElementById('message-input');
const sendEl = document.getElementById('send-button');
const statusEl = document.getElementById('status');
const diagnosisTpl = document.getElementById('tpl-diagnosis');

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addBubble(role, content) {
  const wrapper = document.createElement('div');
  wrapper.className = `bubble bubble--${role}`;

  const label = document.createElement('span');
  label.className = 'bubble__label';
  label.textContent = role === 'user' ? '<user>' : '<hiro>';
  wrapper.appendChild(label);

  if (typeof content === 'string') {
    const text = document.createElement('p');
    text.textContent = content;
    wrapper.appendChild(text);
  } else {
    wrapper.appendChild(content);
  }

  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function addTyping() {
  const dots = document.createElement('p');
  dots.className = 'typing';
  dots.textContent = '…';
  return addBubble('hiro', dots);
}

function renderDiagnosis(data) {
  const node = diagnosisTpl.content.firstElementChild.cloneNode(true);

  const tests = node.querySelector('.diagnosis__tests');
  Object.entries(data.tests).forEach(([model, disease], i) => {
    const li = document.createElement('li');
    const name = document.createElement('span');
    name.textContent = `TEST ${i + 1} (${model})`;
    const result = document.createElement('strong');
    result.textContent = disease;
    li.append(name, result);
    tests.appendChild(li);
  });

  node.querySelector('.diagnosis__disease').textContent = data.final_prediction;
  node.querySelector('.diagnosis__description').textContent = data.description;

  const precautions = node.querySelector('.diagnosis__precautions');
  if (data.precautions.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'Sorry I could not find the precautions of the disease';
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

function setBusy(busy) {
  inputEl.disabled = busy;
  sendEl.disabled = busy;
  if (!busy) inputEl.focus();
}

async function sendMessage(message) {
  addBubble('user', message);
  setBusy(true);
  const typing = addTyping();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    typing.remove();

    if (!res.ok) {
      addBubble('hiro', data.detail ?? 'ERROR OCCURED');
      return;
    }

    if (data.type === 'diagnosis') {
      addBubble('hiro', `Okay I have founded these following symptoms : ${data.symptoms.join(', ')}`);
      addBubble('hiro', renderDiagnosis(data));
    } else {
      addBubble('hiro', data.reply);
    }
  } catch (err) {
    typing.remove();
    addBubble('hiro', `ERROR OCCURED => ${err.message}`);
  } finally {
    setBusy(false);
  }
}

formEl.addEventListener('submit', (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;
  inputEl.value = '';
  sendMessage(message);
});

async function introduce() {
  try {
    const res = await fetch('/api/intro');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    addBubble('hiro', data.message);
    statusEl.textContent = 'Online';
    statusEl.classList.add('status--online');
    setBusy(false);
  } catch (err) {
    statusEl.textContent = 'Backend unavailable';
    statusEl.classList.add('status--error');
    addBubble('hiro', `ERROR OCCURED => ${err.message}. Is the FastAPI server running on port 8000?`);
  }
}

introduce();
