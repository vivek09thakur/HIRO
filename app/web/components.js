export const $ = (id) => document.getElementById(id);

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key in node && typeof value === 'boolean') node[key] = value;
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

export const text = (tag, content, cls) => el(tag, { class: cls, text: content });
export const muted = (content, tag = 'p') => text(tag, content, 'muted');
export const paragraphs = (lines) => lines.map((line) => text('p', line));

export function button(label, { variant = 'secondary', size = '', type = 'button', ...attrs } = {}) {
  return el('button', { type, class: `btn btn-${variant} ${size}`.trim(), text: label, ...attrs });
}

export const chip = (label, attrs = {}) => el('button', { type: 'button', class: 'chip', text: label, ...attrs });

export const link = (label, href, { variant = 'secondary', size = 'btn-sm', ...attrs } = {}) =>
  el('a', { class: `btn btn-${variant} ${size}`.trim(), href, target: '_blank', rel: 'noopener noreferrer', text: label, ...attrs });

export function badge(label, level) {
  return el('span', { class: 'badge', dataset: { level }, text: label });
}

export function logoIcon(cls = '') {
  return el('span', { class: `logo-icon ${cls}`.trim(), 'aria-hidden': 'true' });
}

export function message(role, content, { error = false } = {}) {
  const body = el('div', { class: role === 'user' ? 'msg-user' : error ? 'msg-error' : 'msg-body' });
  body.append(typeof content === 'string' ? el('p', { class: 'whitespace-pre-wrap', text: content }) : content);
  return el('div', { class: `msg ${role === 'user' ? 'justify-end' : ''}` }, [
    role === 'hiro' ? el('span', { class: 'msg-avatar', 'aria-hidden': 'true' }, logoIcon('h-4 w-4')) : null,
    body,
  ]);
}

export function thinking() {
  const dots = el('span', { class: 'thinking', role: 'status', 'aria-label': 'HIRO is thinking' }, [el('i'), el('i'), el('i')]);
  const node = message('hiro', dots);
  node.classList.add('msg-thinking');
  return node;
}

export function quickReplies(labels) {
  return el('div', { class: 'quick-replies flex flex-wrap gap-2 mt-3' }, labels.map((label) => chip(label)));
}

export function section(title, child) {
  return el('section', {}, [text('h4', title, 'card-heading'), child]);
}

export function chips(items) {
  return el('ul', { class: 'flex flex-wrap gap-2 list-none p-0' }, items.map((item) => el('li', { class: 'chip-static', text: item })));
}

export function keyValueList(entries, highlight) {
  return el(
    'ul',
    { class: 'kv-list' },
    entries.map(([key, value]) =>
      el('li', { dataset: { agrees: String(value === highlight) } }, [text('span', key, 'text-muted'), text('strong', value)]),
    ),
  );
}

export function orderedList(items, emptyText) {
  const list = items.length ? items : [emptyText];
  return el('ol', { class: 'grid gap-1 pl-6 text-sm list-decimal' }, list.map((item) => el('li', { class: 'first-letter:uppercase', text: item })));
}

export function diagnosisCard(data, { emptyPrecautions, notice }) {
  return el('article', { class: 'card' }, [
    el('header', { class: 'flex items-center justify-between gap-3' }, [
      text('h3', 'Assessment result', 'text-base font-medium'),
      badge(`${data.confidence} confidence`, data.confidence),
    ]),
    section('Symptoms reviewed', chips(data.symptoms)),
    section(
      'Possible condition',
      el('div', {}, [text('p', data.final_prediction, 'text-lg font-medium text-accent'), text('p', data.description, 'mt-1 text-sm')]),
    ),
    section('Model agreement', keyValueList(Object.entries(data.tests), data.final_prediction)),
    section('Suggested precautions', orderedList(data.precautions, emptyPrecautions)),
    text('p', notice, 'notice'),
  ]);
}

export function historyRow(conv, { active, fallbackMeta }) {
  return el('li', {}, [
    el('button', { type: 'button', class: `row ${active ? 'row-active' : ''}`.trim(), dataset: { id: conv.id } }, [
      text('span', conv.title, 'row-title'),
      text('span', conv.result ?? fallbackMeta, 'row-meta'),
      el('span', { class: 'text-muted', 'aria-hidden': 'true', text: '\u203a' }),
    ]),
  ]);
}

export function meterRow({ label, meta, value, color, muted: isMuted = false, metaNote }) {
  const pct = Math.round(value * 100);
  return el('li', { class: 'reading' }, [
    el('div', { class: 'reading-head' }, [
      el('span', { class: 'min-w-0' }, [
        color ? el('span', { class: 'swatch', dataset: { color }, 'aria-hidden': 'true' }) : null,
        text('span', label, 'font-medium'),
        meta ? muted(` · ${meta}${metaNote ? ` (${metaNote})` : ''}`, 'span') : null,
      ]),
      text('span', `${pct}%`, 'reading-pct'),
    ]),
    el('div', { class: `meter ${isMuted ? 'meter-muted' : ''}`.trim(), dataset: color ? { color } : {}, role: 'meter', 'aria-valuenow': pct, 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-label': `${label} ${pct}%` }, [
      el('span', { style: `width:${pct}%` }),
    ]),
  ]);
}

export function replaceChildren(parent, children) {
  parent.replaceChildren(...[].concat(children).filter(Boolean));
  return parent;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export async function typeInto(node, fullText, { msPerChar, pauseAfter, pauseMs, tickMs }, onTick) {
  if (reducedMotion()) {
    node.textContent = fullText;
    return;
  }
  node.classList.add('typing');
  node.textContent = '';
  const started = performance.now();
  let shown = 0;
  let paused = 0;
  while (shown < fullText.length) {
    await sleep(tickMs);
    const target = Math.min(fullText.length, Math.floor((performance.now() - started - paused) / msPerChar));
    if (target <= shown) continue;
    const slice = fullText.slice(shown, target);
    const stop = slice.search(pauseAfter);
    shown = stop >= 0 ? shown + stop + 1 : target;
    if (stop >= 0) paused += pauseMs;
    node.textContent = fullText.slice(0, shown);
    onTick?.();
  }
  node.classList.remove('typing');
}
