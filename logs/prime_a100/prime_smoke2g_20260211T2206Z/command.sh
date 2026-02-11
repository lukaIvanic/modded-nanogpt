#!/usr/bin/env bash
set -euo pipefail

RUN_ID='prime_smoke2g_20260211T2206Z'
REMOTE_BASE='/root/projects/nanogpt-speedrun'
REPO='/root/projects/nanogpt-speedrun/modded-nanogpt'
RUN_DIR='/root/projects/nanogpt-speedrun/runs/prime_smoke2g_20260211T2206Z'
VENV='/root/projects/nanogpt-speedrun/.venv'

export RUN_ID
export RUN_DIR
export NUM_ITERATIONS='5'
export VAL_LOSS_EVERY='1'
export SAVE_CHECKPOINT='0'
export DISABLE_COMPILE='1'
export CKPT_DIR="$RUN_DIR/checkpoints"

cd "$REPO"
mkdir -p "$RUN_DIR"

cat > "$RUN_DIR/status.json" <<JSON
{"run_id":"$RUN_ID","status":"running","start_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON

{
  echo "run_id=$RUN_ID"
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "whoami=$(whoami)"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=$("$VENV/bin/python" --version 2>&1)"
  "$VENV/bin/python" - <<'PY'
import torch
print('torch_version=' + torch.__version__)
print('cuda_available=' + str(torch.cuda.is_available()))
print('cuda_device_count=' + str(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print(f'gpu[{i}]=' + torch.cuda.get_device_name(i))
PY
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
} > "$RUN_DIR/env.txt"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free   --format=csv -l 2 > "$RUN_DIR/gpu.csv" &
SMI_PID=$!
cleanup() {
  kill "$SMI_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

set +e
"$VENV/bin/torchrun" --standalone --nproc_per_node='2' train_gpt.py 2>&1 | tee "$RUN_DIR/train.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e

"$VENV/bin/python" - <<'PY'
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
if [[ "$EXIT_CODE" -ne 0 ]]; then
  STATUS='failed'
fi

cat > "$RUN_DIR/status.json" <<JSON
{"run_id":"$RUN_ID","status":"$STATUS","end_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","exit_code":$EXIT_CODE}
JSON

exit "$EXIT_CODE"
