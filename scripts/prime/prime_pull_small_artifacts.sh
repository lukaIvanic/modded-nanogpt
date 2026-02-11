#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id> [--watch]"
  exit 2
fi

RUN_ID="$1"
WATCH_MODE="${2:-}"

SSH_USER="${SSH_USER:-ubuntu}"
SSH_HOST="${SSH_HOST:-216.81.248.26}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/prime_intellect_codex_ed25519}"

if [[ -z "${REMOTE_BASE:-}" ]]; then
  if [[ "$SSH_USER" == "root" ]]; then
    REMOTE_BASE="/root/projects/nanogpt-speedrun"
  else
    REMOTE_BASE="/home/$SSH_USER/projects/nanogpt-speedrun"
  fi
fi
REMOTE_RUNS="${REMOTE_RUNS:-$REMOTE_BASE/runs}"
REMOTE_RUN_DIR="$REMOTE_RUNS/$RUN_ID"

LOCAL_REPO="${LOCAL_REPO:-/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt}"
LOCAL_ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-$LOCAL_REPO/logs/prime_a100}"
LOCAL_RUN_DIR="$LOCAL_ARTIFACT_ROOT/$RUN_ID"

mkdir -p "$LOCAL_RUN_DIR"

pull_once() {
  rsync -az \
    -e "ssh -i '$SSH_KEY' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p '$SSH_PORT'" \
    --include '*/' \
    --include '*.log' \
    --include '*.csv' \
    --include '*.json' \
    --include '*.txt' \
    --include 'command.sh' \
    --include 'env.txt' \
    --exclude '*' \
    "$SSH_USER@$SSH_HOST:$REMOTE_RUN_DIR/" \
    "$LOCAL_RUN_DIR/"
}

echo "[INFO] Pulling small artifacts for run $RUN_ID..."
pull_once

if [[ "$WATCH_MODE" == "--watch" ]]; then
  echo "[INFO] Watch mode enabled. Polling every 20s..."
  while true; do
    sleep 20
    pull_once
    if [[ -f "$LOCAL_RUN_DIR/status.json" ]]; then
      status="$(python3 - <<PY
import json
with open("$LOCAL_RUN_DIR/status.json","r",encoding="utf-8") as f:
    print(json.load(f).get("status","running"))
PY
)"
      if [[ "$status" != "running" ]]; then
        echo "[INFO] Remote run status=$status; final pull complete."
        break
      fi
    fi
  done
fi

echo "[INFO] Local run artifacts at: $LOCAL_RUN_DIR"
