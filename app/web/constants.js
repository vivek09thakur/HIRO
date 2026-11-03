export const STORAGE = { consents: 'hiro.consents', history: 'hiro.history' };
export const LIMITS = { history: 30, connectAttempts: 120, sendRetries: 3, titleLength: 48 };
export const TYPING = { msPerChar: 9, tickMs: 16, pauseAfter: /[.!?](?=\s|$)/, pauseMs: 180 };
export const API = { health: '/api/health', chat: '/api/chat', symptoms: '/api/symptoms' };

export const CONSENTS = {
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
      'Your messages are sent only to the HIRO server to generate a reply. The server keeps nothing between requests; the state of this assessment travels with each message.',
      'Your assessment history is saved in this browser only (local storage) so you can revisit it. Clear it at any time with your browser settings.',
      'Fonts and styles are loaded from Google Fonts and the Tailwind CDN, which receive a standard web request when the page loads. Nothing else leaves your browser.',
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

export const HOW_IT_WORKS = {
  title: 'How it works',
  body: [
    '1. Describe what you are feeling in plain language, for example "high fever and headache". Typos, everyday phrasing ("my tummy hurts") and negations ("no cough") are understood.',
    '2. HIRO picks out the symptoms it recognises and asks a few yes / no follow-up questions, choosing the ones that best narrow down the possibilities. Answer with the buttons or type freely; say "that\'s all" at any point to skip ahead.',
    '3. Three classifiers (Random Forest, SVC, Naive Bayes) vote on the recorded symptoms. You get a possible condition, a confidence level based on how many symptoms were gathered and whether the models agree, a short description and suggested precautions. The Model insight panel shows each model\'s prediction and how sure it is after every message.',
    'Use the result as a starting point for a conversation with a doctor, not as a diagnosis.',
  ],
};

export const INTRO =
  'Hello, I am HIRO. Describe what you are feeling and I will ask a few follow-up questions before suggesting a possible condition. This is not a medical diagnosis. Say "that\'s all" at any time to see the result.';

export const HINTS = {
  offline: 'Connecting to the HIRO server.',
  consent: 'Read and agree to the four notices below the box to begin.',
  ready: 'Describe your symptoms in plain language. HIRO asks a few follow-up questions before suggesting a possible condition.',
};

export const PLACEHOLDERS = {
  offline: 'Waiting for the server',
  consent: 'Agree to the notices below to start',
  ready: 'e.g. headache and mild fever',
};

export const STATUS = {
  connecting: 'Connecting to server',
  waiting: (n) => `Waiting for server (${n})`,
  reconnecting: (n) => `Reconnecting (${n})`,
  unreachable: 'The HIRO server is not reachable. Start it and reload this page.',
};

export const INSIGHT = {
  symptoms: (n) => `Based on ${n} symptom${n === 1 ? '' : 's'}`,
  live: ' · updates after every message',
  final: ' · final reading',
  agreement: (agree, total) => `${agree} of ${total} agree`,
  level: (agree, total) => (agree === total ? 'high' : agree > 1 ? 'medium' : 'low'),
  disagrees: 'differs',
};

export const QUICK_REPLIES = ['Yes', 'No', "That's all"];
