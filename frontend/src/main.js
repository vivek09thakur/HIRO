import axios from 'axios';

const authSection = document.getElementById('auth');
const chatSection = document.getElementById('chat');
const loginButton = document.getElementById('login');
const signupLink = document.getElementById('signup-link');
const chatHistory = document.getElementById('chat-history');
const messageInput = document.getElementById('message-input');
const sendMessageButton = document.getElementById('send-message');
const messagesDiv = document.getElementById('messages');

let token = '';

loginButton.addEventListener('click', async () => {
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;

  try {
    const response = await axios.post('http://localhost:8000/api/token/', { username, password });
    token = response.data.access;
    authSection.classList.add('hidden');
    chatSection.classList.remove('hidden');
    loadChatHistory();
  } catch (error) {
    alert('Login failed');
  }
});

sendMessageButton.addEventListener('click', async () => {
  const message = messageInput.value;

  try {
    const response = await axios.post(
      'http://localhost:8000/api/chat/',
      { message },
      { headers: { Authorization: `Bearer ${token}` } }
    );

    addMessage('You', message);
    addMessage('Bot', response.data.response);
    messageInput.value = '';
    loadChatHistory();
  } catch (error) {
    alert('Failed to send message');
  }
});

async function loadChatHistory() {
  try {
    const response = await axios.get('http://localhost:8000/api/chat/', {
      headers: { Authorization: `Bearer ${token}` },
    });

    chatHistory.innerHTML = '';
    response.data.forEach((chat) => {
      const li = document.createElement('li');
      li.textContent = chat.message;
      chatHistory.appendChild(li);
    });
  } catch (error) {
    alert('Failed to load chat history');
  }
}

function addMessage(sender, text) {
  const div = document.createElement('div');
  div.textContent = `${sender}: ${text}`;
  messagesDiv.appendChild(div);
}
