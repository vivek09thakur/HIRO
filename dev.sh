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
HEALTH_URL=http://127.0.0.1:8000/api/health
SERVER_PID=""
CLIENT_PID=""

port_in_use() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -q ":$1 "
  else
    return 1
  fi
}

for port in 8000 5173; do
  if port_in_use "$port"; then
    echo "Port $port is already in use. Stop the other HIRO instance (or process) first." >&2
    exit 1
  fi
done

cleanup() {
  trap - INT TERM EXIT
  [ -n "$CLIENT_PID" ] && kill "$CLIENT_PID" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

server_ready() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1
  else
    "$PYTHON" -c "import sys,urllib.request; urllib.request.urlopen('$HEALTH_URL', timeout=2)" >/dev/null 2>&1
  fi
}

"$PYTHON" -m uvicorn server.app.main:app --reload --reload-dir server --port 8000 &
SERVER_PID=$!

echo
echo "Starting HIRO server on http://127.0.0.1:8000 ..."
until server_ready; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited before becoming healthy." >&2
    exit 1
  fi
  sleep 0.5
done
echo "Server is online. Starting client..."

npm --prefix client run dev &
CLIENT_PID=$!

echo
echo "HIRO server  -> http://127.0.0.1:8000"
echo "HIRO client  -> http://localhost:5173"
echo "Press Ctrl+C to stop both."
echo

# Exit as soon as either process ends (wait -n needs bash >= 4.3; fall back to waiting for all).
wait -n "$SERVER_PID" "$CLIENT_PID" 2>/dev/null || wait
