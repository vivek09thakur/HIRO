#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x venv/bin/python ]; then
  PYTHON=venv/bin/python
elif [ -x venv/Scripts/python.exe ]; then
  PYTHON=venv/Scripts/python.exe
else
  echo "No venv found. Run: python -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

PORT="${PORT:-8000}"
echo "HIRO -> http://127.0.0.1:$PORT"
exec "$PYTHON" -m uvicorn app.api.main:app --reload --reload-dir app --reload-dir model --port "$PORT"
