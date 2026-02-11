#!/usr/bin/env bash
set -euo pipefail

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
REMOTE_REPO="${REMOTE_REPO:-$REMOTE_BASE/modded-nanogpt}"
REMOTE_RUNS="${REMOTE_RUNS:-$REMOTE_BASE/runs}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_BASE/.venv}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu126}"
AUTO_FIX_FABRIC_MANAGER="${AUTO_FIX_FABRIC_MANAGER:-0}"

ssh_cmd() {
  ssh -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p "$SSH_PORT" \
    "$SSH_USER@$SSH_HOST" "$@"
}

echo "[INFO] Checking remote connectivity..."
ssh_cmd "echo CONNECT_OK && hostname && whoami"

echo "[INFO] Creating remote workspace directories..."
ssh_cmd "mkdir -p '$REMOTE_BASE' '$REMOTE_REPO' '$REMOTE_RUNS'"

echo "[INFO] Installing python3-pip and python3-venv (idempotent)..."
ssh_cmd "set -e; if [[ \$(id -u) -eq 0 ]]; then SUDO=''; else SUDO='sudo'; fi; export DEBIAN_FRONTEND=noninteractive; \$SUDO apt-get update -y && \$SUDO apt-get install -y python3-pip python3-venv"

if [[ "$AUTO_FIX_FABRIC_MANAGER" == "1" ]]; then
  echo "[INFO] Ensuring fabric-manager matches NVIDIA 580 driver (NVSwitch-safe path)..."
  ssh_cmd "set -e; if [[ \$(id -u) -eq 0 ]]; then SUDO=''; else SUDO='sudo'; fi; if nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | grep -q '^580\\.'; then \
    export DEBIAN_FRONTEND=noninteractive; \
    \$SUDO apt-get remove -y nvidia-fabricmanager nvidia-fabricmanager-590 >/dev/null 2>&1 || true; \
    \$SUDO apt-get -o Dpkg::Options::='--force-overwrite' install -y -f >/dev/null 2>&1 || true; \
    \$SUDO apt-get -o Dpkg::Options::='--force-overwrite' install -y nvidia-fabricmanager-580 nvidia-kernel-common-580-server; \
    \$SUDO systemctl daemon-reload; \
    \$SUDO systemctl enable nvidia-fabricmanager >/dev/null 2>&1 || true; \
    \$SUDO systemctl restart nvidia-fabricmanager || true; \
  fi"
fi

echo "[INFO] Creating/refreshing virtual environment at $REMOTE_VENV..."
ssh_cmd "python3 -m venv '$REMOTE_VENV'"

echo "[INFO] Installing Python dependencies..."
ssh_cmd "'$REMOTE_VENV/bin/python' -m pip install --upgrade pip setuptools wheel"
ssh_cmd "'$REMOTE_VENV/bin/pip' install --pre --upgrade torch --index-url '$TORCH_INDEX_URL'"
ssh_cmd "'$REMOTE_VENV/bin/pip' install --upgrade numpy tqdm huggingface-hub datasets tiktoken"

echo "[INFO] Verifying CUDA visibility from venv..."
ssh_cmd "'$REMOTE_VENV/bin/python' - <<'PY'
import torch
print('torch_version=', torch.__version__)
count = torch.cuda.device_count()
avail = torch.cuda.is_available()
print('cuda_available=', avail)
print('cuda_device_count=', count)
if not avail:
    raise SystemExit('CUDA unavailable after bootstrap')
for i in range(count):
    print(f'gpu[{i}]={torch.cuda.get_device_name(i)}')
PY"

echo "[INFO] Remote bootstrap complete."
