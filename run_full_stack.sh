#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
VENV_PYTHON="$ROOT_DIR/venv/bin/python"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
API_BASE_URL="${VITE_API_BASE_URL:-http://$BACKEND_HOST:$BACKEND_PORT}"
REPLACE_EXISTING="${REPLACE_EXISTING:-0}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing Python virtualenv at $VENV_PYTHON"
  exit 1
fi

if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
  echo "Missing frontend/package.json"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is not installed or not on PATH"
  exit 1
fi

BACKEND_PID=""
FRONTEND_PID=""

find_existing_processes() {
  ps -eo pid=,args= | grep -E "uvicorn app\.main:app|node .*vite|vite --host|node scripts/dev-with-chrome" | grep -v grep || true
}

stop_existing_processes() {
  pkill -f "uvicorn app.main:app" >/dev/null 2>&1 || true
  pkill -f "node .*vite" >/dev/null 2>&1 || true
  pkill -f "vite --host" >/dev/null 2>&1 || true
  pkill -f "node scripts/dev-with-chrome" >/dev/null 2>&1 || true
}

cleanup() {
  local exit_code=$?
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

EXISTING_PROCESSES="$(find_existing_processes)"
if [[ -n "$EXISTING_PROCESSES" ]]; then
  echo "Existing backend/frontend dev processes were found:"
  echo "$EXISTING_PROCESSES"
  if [[ "$REPLACE_EXISTING" == "1" ]]; then
    echo "Stopping existing dev processes because REPLACE_EXISTING=1"
    stop_existing_processes
    sleep 1
  else
    echo
    echo "Refusing to start another copy because stale processes cause old-tab/new-tab confusion."
    echo "Run one of these:"
    echo "  REPLACE_EXISTING=1 ./run_full_stack.sh"
    echo "or stop them manually:"
    echo "  pkill -f 'uvicorn app.main:app'"
    echo "  pkill -f 'node .*vite'"
    exit 1
  fi
fi

echo "Starting backend on http://$BACKEND_HOST:$BACKEND_PORT"
(
  cd "$BACKEND_DIR"
  exec "$VENV_PYTHON" -m uvicorn app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload
) &
BACKEND_PID=$!

echo "Starting frontend on http://$FRONTEND_HOST:$FRONTEND_PORT"
(
  cd "$FRONTEND_DIR"
  export VITE_API_BASE_URL="$API_BASE_URL"
  exec npm exec vite -- --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Frontend API base URL: $API_BASE_URL"
echo "Open http://$FRONTEND_HOST:$FRONTEND_PORT in your browser."
echo "Press Ctrl+C to stop both services."

wait -n "$BACKEND_PID" "$FRONTEND_PID"
