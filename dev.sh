#!/usr/bin/env bash
# Starts the FastAPI server (port 8000) and the Vite client (port 5173) together.
# Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"

if [ -x venv/bin/python ]; then
  PYTHON=venv/bin/python
elif [ -x venv/Scripts/python.exe ]; then
  PYTHON=venv/Scripts/python.exe
else
  echo "No venv found. Run: python -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [ ! -d client/node_modules ]; then
  echo "Installing client dependencies..."
  npm --prefix client install
fi

export TF_CPP_MIN_LOG_LEVEL=2

cleanup() {
  trap - INT TERM EXIT
  kill "$SERVER_PID" "$CLIENT_PID" 2>/dev/null || true
  wait "$SERVER_PID" "$CLIENT_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

"$PYTHON" -m uvicorn server.app.main:app --reload --port 8000 &
SERVER_PID=$!

npm --prefix client run dev &
CLIENT_PID=$!

echo
echo "HIRO server  -> http://127.0.0.1:8000"
echo "HIRO client  -> http://localhost:5173"
echo "Press Ctrl+C to stop both."
echo

# Exit as soon as either process ends (wait -n needs bash >= 4.3; fall back to waiting for all).
wait -n "$SERVER_PID" "$CLIENT_PID" 2>/dev/null || wait
