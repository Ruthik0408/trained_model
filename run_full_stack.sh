#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
PYTHON_BIN="$ROOT_DIR/venv/bin/python"

BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8000"
FRONTEND_HOST="0.0.0.0"
FRONTEND_PORT="5173"
LOG_DIR="$ROOT_DIR/.run-logs"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

# Override with TULIP_LOCAL_IP when you need a specific LAN address.
LOCAL_IP="${TULIP_LOCAL_IP:-$(hostname -I | awk '{print $1}')}"

if [[ -z "$LOCAL_IP" ]]; then
  echo "Could not detect local IP. Run: hostname -I"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Backend venv not found at $PYTHON_BIN"
  echo "Create it first, then install:"
  echo "python3 -m venv venv"
  echo "./venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Frontend dependencies missing. Run:"
  echo "cd frontend && npm install"
  exit 1
fi

port_is_busy() {
  local port="$1"
  (echo >"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1
}

wait_for_port() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local log_file="$4"
  local attempts=60

  for ((attempt = 1; attempt <= attempts; attempt += 1)); do
    if port_is_busy "$port"; then
      return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name failed to start. Last log lines:"
      tail -n 20 "$log_file" 2>/dev/null || true
      return 1
    fi

    sleep 1
  done

  echo "$name did not become ready on port $port. Last log lines:"
  tail -n 20 "$log_file" 2>/dev/null || true
  return 1
}

if port_is_busy "$BACKEND_PORT"; then
  echo "Port $BACKEND_PORT is already in use. Stop the existing backend or change the backend port."
  exit 1
fi

if port_is_busy "$FRONTEND_PORT"; then
  echo "Port $FRONTEND_PORT is already in use. Stop the existing frontend or change the frontend port."
  exit 1
fi

cleanup() {
  echo
  echo "Stopping backend and frontend..."

  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi

  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi

  wait 2>/dev/null || true
}

trap cleanup EXIT INT TERM

mkdir -p "$LOG_DIR"
: >"$BACKEND_LOG"
: >"$FRONTEND_LOG"

# Frontend should call backend using your actual IP
export VITE_API_BASE_URL="http://$LOCAL_IP:$BACKEND_PORT"
export TULIP_CORS_ALLOWED_ORIGINS="${TULIP_CORS_ALLOWED_ORIGINS:-http://localhost:$FRONTEND_PORT,http://127.0.0.1:$FRONTEND_PORT,http://$LOCAL_IP:$FRONTEND_PORT}"

echo "Starting Tulip anomaly stack..."
echo "Local IP: $LOCAL_IP"
echo "Detailed logs:"
echo "  Backend:  $BACKEND_LOG"
echo "  Frontend: $FRONTEND_LOG"
echo

echo "Starting backend..."
(
  cd "$BACKEND_DIR"
  "$PYTHON_BIN" -m uvicorn app.main:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT"
) >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

echo "Starting frontend..."
(
  cd "$FRONTEND_DIR"
  npx vite --host "$FRONTEND_HOST" --port "$FRONTEND_PORT"
) >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

wait_for_port "Backend" "$BACKEND_PORT" "$BACKEND_PID" "$BACKEND_LOG"
wait_for_port "Frontend" "$FRONTEND_PORT" "$FRONTEND_PID" "$FRONTEND_LOG"

echo
echo "Backend:  http://$LOCAL_IP:$BACKEND_PORT"
echo "Frontend: http://$LOCAL_IP:$FRONTEND_PORT"
echo "Open on same Wi-Fi: http://$LOCAL_IP:$FRONTEND_PORT"
echo
echo "Stack is running. Press Ctrl+C to stop."

if wait -n "$BACKEND_PID" "$FRONTEND_PID"; then
  echo "A service stopped."
else
  echo "A service stopped with an error. Check logs:"
  echo "  Backend:  $BACKEND_LOG"
  echo "  Frontend: $FRONTEND_LOG"
  exit 1
fi
