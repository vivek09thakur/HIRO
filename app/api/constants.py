from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "app" / "web"

APP_TITLE = "HIRO Healthcare Companion"

INFERENCE_WORKERS = 4
MAX_QUEUE = 200
RETRY_AFTER_SECONDS = 2

MSG_EMPTY = "Message cannot be empty"
MSG_CHAT_FAILED = "Could not process the message ({reason})."
MSG_BUSY = "HIRO is busy right now. Please try again in a moment."
