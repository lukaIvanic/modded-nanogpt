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

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"
DISABLE_COMPILE="${DISABLE_COMPILE:-1}"
TRAIN_BS_SCHEDULE="${TRAIN_BS_SCHEDULE:-}"
TRAIN_BS_EXTENSION="${TRAIN_BS_EXTENSION:-}"
TRAIN_MAX_SEQ_LEN="${TRAIN_MAX_SEQ_LEN:-}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
VAL_TOKENS="${VAL_TOKENS:-}"

RUN_ID="${RUN_ID:-prime_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_RUN_DIR="$REMOTE_RUNS/$RUN_ID"

ssh_cmd() {
  ssh -i "$SSH_KEY" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p "$SSH_PORT" \
    "$SSH_USER@$SSH_HOST" "$@"
}

echo "[INFO] Preparing remote run dir: $REMOTE_RUN_DIR"
ssh_cmd "mkdir -p '$REMOTE_RUN_DIR'"

echo "[INFO] Writing remote command.sh..."
ssh -i "$SSH_KEY" \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -p "$SSH_PORT" \
  "$SSH_USER@$SSH_HOST" "cat > '$REMOTE_RUN_DIR/command.sh'" <<EOF
#!/usr/bin/env bash
set -euo pipefail

RUN_ID='${RUN_ID}'
REMOTE_BASE='${REMOTE_BASE}'
REPO='${REMOTE_REPO}'
RUN_DIR='${REMOTE_RUN_DIR}'
VENV='${REMOTE_VENV}'

export RUN_ID
export RUN_DIR
export NUM_ITERATIONS='${NUM_ITERATIONS}'
export VAL_LOSS_EVERY='${VAL_LOSS_EVERY}'
export SAVE_CHECKPOINT='${SAVE_CHECKPOINT}'
export DISABLE_COMPILE='${DISABLE_COMPILE}'
export CKPT_DIR="\$RUN_DIR/checkpoints"
if [[ -n '${TRAIN_BS_SCHEDULE}' ]]; then export TRAIN_BS_SCHEDULE='${TRAIN_BS_SCHEDULE}'; fi
if [[ -n '${TRAIN_BS_EXTENSION}' ]]; then export TRAIN_BS_EXTENSION='${TRAIN_BS_EXTENSION}'; fi
if [[ -n '${TRAIN_MAX_SEQ_LEN}' ]]; then export TRAIN_MAX_SEQ_LEN='${TRAIN_MAX_SEQ_LEN}'; fi
if [[ -n '${VAL_BATCH_SIZE}' ]]; then export VAL_BATCH_SIZE='${VAL_BATCH_SIZE}'; fi
if [[ -n '${VAL_TOKENS}' ]]; then export VAL_TOKENS='${VAL_TOKENS}'; fi

cd "\$REPO"
mkdir -p "\$RUN_DIR"

cat > "\$RUN_DIR/status.json" <<JSON
{"run_id":"\$RUN_ID","status":"running","start_utc":"\$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON

{
  echo "run_id=\$RUN_ID"
  echo "start_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=\$(hostname)"
  echo "whoami=\$(whoami)"
  echo "git_sha=\$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=\$("\$VENV/bin/python" --version 2>&1)"
  "\$VENV/bin/python" - <<'PY'
import torch
print('torch_version=' + torch.__version__)
print('cuda_available=' + str(torch.cuda.is_available()))
print('cuda_device_count=' + str(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print(f'gpu[{i}]=' + torch.cuda.get_device_name(i))
PY
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
} > "\$RUN_DIR/env.txt"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free \
  --format=csv -l 2 > "\$RUN_DIR/gpu.csv" &
SMI_PID=\$!
cleanup() {
  kill "\$SMI_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

set +e
"\$VENV/bin/torchrun" --standalone --nproc_per_node='${NPROC_PER_NODE}' train_gpt.py 2>&1 | tee "\$RUN_DIR/train.log"
EXIT_CODE=\${PIPESTATUS[0]}
set -e

"\$VENV/bin/python" - <<'PY'
import csv
import os
import re

run_dir = os.environ['RUN_DIR']
train_log = os.path.join(run_dir, 'train.log')
out_csv = os.path.join(run_dir, 'val_metrics.csv')
pattern = re.compile(r'step:(\d+)/(\d+)\s+val_loss:([0-9.]+)')

rows = []
if os.path.exists(train_log):
    with open(train_log, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))

with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['step', 'step_total', 'val_loss'])
    for row in rows:
        w.writerow(row)
PY

STATUS='finished'
if [[ "\$EXIT_CODE" -ne 0 ]]; then
  STATUS='failed'
fi

cat > "\$RUN_DIR/status.json" <<JSON
{"run_id":"\$RUN_ID","status":"\$STATUS","end_utc":"\$(date -u +%Y-%m-%dT%H:%M:%SZ)","exit_code":\$EXIT_CODE}
JSON

exit "\$EXIT_CODE"
EOF
ssh_cmd "chmod +x '$REMOTE_RUN_DIR/command.sh'"

echo "[INFO] Launching training run on remote..."
ssh_cmd "RUN_DIR='$REMOTE_RUN_DIR' bash '$REMOTE_RUN_DIR/command.sh'"

echo "[INFO] Run complete."
echo "[INFO] RUN_ID=$RUN_ID"
echo "[INFO] REMOTE_RUN_DIR=$REMOTE_RUN_DIR"
