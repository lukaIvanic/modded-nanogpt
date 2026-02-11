#!/usr/bin/env bash
set -euo pipefail

SSH_USER="${SSH_USER:-ubuntu}"
SSH_HOST="${SSH_HOST:-216.81.248.26}"
SSH_PORT="${SSH_PORT:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/prime_intellect_codex_ed25519}"

LOCAL_REPO="${LOCAL_REPO:-/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt}"
LOCAL_TRAIN_SHARD="${LOCAL_TRAIN_SHARD:-$LOCAL_REPO/data/fineweb10B/fineweb_train_000001.bin}"
LOCAL_VAL_SHARD="${LOCAL_VAL_SHARD:-$LOCAL_REPO/data/fineweb10B/fineweb_val_000000.bin}"

if [[ -z "${REMOTE_BASE:-}" ]]; then
  if [[ "$SSH_USER" == "root" ]]; then
    REMOTE_BASE="/root/projects/nanogpt-speedrun"
  else
    REMOTE_BASE="/home/$SSH_USER/projects/nanogpt-speedrun"
  fi
fi
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_BASE/modded-nanogpt}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-$REMOTE_REPO/data/fineweb10B}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_BASE/.venv}"
ORIGIN_URL="${ORIGIN_URL:-https://github.com/lukaIvanic/modded-nanogpt.git}"

SSH_OPTS=(
  -i "$SSH_KEY"
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -p "$SSH_PORT"
)

if [[ ! -f "$LOCAL_TRAIN_SHARD" || ! -f "$LOCAL_VAL_SHARD" ]]; then
  echo "[ERROR] Missing local shard(s):"
  echo "  train: $LOCAL_TRAIN_SHARD"
  echo "  val:   $LOCAL_VAL_SHARD"
  exit 2
fi

echo "[INFO] Preparing remote repository path..."
ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" "mkdir -p '$REMOTE_BASE'"
ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" \
  "if [[ ! -d '$REMOTE_REPO/.git' ]]; then rm -rf '$REMOTE_REPO'; git clone '$ORIGIN_URL' '$REMOTE_REPO'; fi"

echo "[INFO] Syncing local repo to remote..."
rsync -az --delete \
  -e "ssh -i '$SSH_KEY' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p '$SSH_PORT'" \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='logs/' \
  --exclude='__pycache__/' \
  --exclude='data/fineweb10B/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  "$LOCAL_REPO/" \
  "$SSH_USER@$SSH_HOST:$REMOTE_REPO/"

echo "[INFO] Uploading first-smoke FineWeb shards..."
ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" "mkdir -p '$REMOTE_DATA_DIR'"
rsync -az \
  -e "ssh -i '$SSH_KEY' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p '$SSH_PORT'" \
  "$LOCAL_TRAIN_SHARD" \
  "$LOCAL_VAL_SHARD" \
  "$SSH_USER@$SSH_HOST:$REMOTE_DATA_DIR/"

echo "[INFO] Validating remote shard headers (magic/version/token_count)..."
ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" "REMOTE_DATA_DIR='$REMOTE_DATA_DIR' '$REMOTE_VENV/bin/python' - <<'PY'
import os
import struct

paths = [
    os.path.join(os.environ['REMOTE_DATA_DIR'], 'fineweb_train_000001.bin'),
    os.path.join(os.environ['REMOTE_DATA_DIR'], 'fineweb_val_000000.bin'),
]
for path in paths:
    with open(path, 'rb') as f:
        header = f.read(256 * 4)
    ints = struct.unpack('<256i', header)
    magic, version, num_tokens = ints[0], ints[1], ints[2]
    size = os.path.getsize(path)
    expected = 256 * 4 + num_tokens * 2
    print(path)
    print(f'  magic={magic} version={version} tokens={num_tokens} size={size}')
    if magic != 20240520:
        raise SystemExit(f'magic mismatch for {path}')
    if version != 1:
        raise SystemExit(f'version mismatch for {path}')
    if expected != size:
        raise SystemExit(f'size mismatch for {path}: expected {expected}, got {size}')
print('HEADER_VALIDATION_OK')
PY"

echo "[INFO] Sync complete."
echo "[INFO] Optional later command for full dataset flow:"
echo "  ssh -i '$SSH_KEY' -p '$SSH_PORT' '$SSH_USER@$SSH_HOST' \"cd '$REMOTE_REPO' && '$REMOTE_VENV/bin/python' data/cached_fineweb10B.py 9\""
