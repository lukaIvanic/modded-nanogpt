#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
campaign_root="${CAMPAIGN_ROOT:-$repo_root/logs/kd_softloss_debug/$ts}"
mkdir -p "$campaign_root/cases"

torchrun_bin="${TORCHRUN_BIN:-}"
if [[ -z "$torchrun_bin" ]]; then
  if [[ -x "$repo_root/../.venv/bin/torchrun" ]]; then
    torchrun_bin="$repo_root/../.venv/bin/torchrun"
  elif command -v torchrun >/dev/null 2>&1; then
    torchrun_bin="$(command -v torchrun)"
  fi
fi
if [[ -z "$torchrun_bin" || ! -x "$torchrun_bin" ]]; then
  echo "Unable to find executable torchrun. Set TORCHRUN_BIN explicitly." >&2
  exit 2
fi

num_steps="${NUM_SCHEDULED_ITERATIONS:-80}"
nproc_per_node="${NPROC_PER_NODE:-1}"
base_seed="${BASE_SEED:-42}"
teacher_layers="${KD_TEACHER_NUM_LAYERS:-3}"
teacher_heads="${KD_TEACHER_NUM_HEADS:-6}"
teacher_head_dim="${KD_TEACHER_HEAD_DIM:-128}"
teacher_model_dim="${KD_TEACHER_MODEL_DIM:-768}"
ga_candidates="${GA_CANDIDATES:-8 10 12 16}"
drop_first_deltas="${DROP_FIRST_DELTAS:-20}"

{
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=$repo_root"
  echo "campaign_root=$campaign_root"
  echo "num_steps=$num_steps"
  echo "nproc_per_node=$nproc_per_node"
  echo "base_seed=$base_seed"
  echo "ga_candidates=$ga_candidates"
  echo "teacher_layers=$teacher_layers"
  echo "teacher_heads=$teacher_heads"
  echo "teacher_head_dim=$teacher_head_dim"
  echo "teacher_model_dim=$teacher_model_dim"
  echo "drop_first_deltas=$drop_first_deltas"
  echo "torchrun_bin=$torchrun_bin"
  echo "git_sha=$(git rev-parse HEAD 2>/dev/null || true)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi
} > "$campaign_root/preflight.txt"

common_env=(
  "BASE_SEED=$base_seed"
  "NUM_SCHEDULED_ITERATIONS=$num_steps"
  "NUM_EXTENSION_ITERATIONS=0"
  "VAL_LOSS_EVERY=0"
  "SAVE_CHECKPOINT=0"
  "DISABLE_COMPILE=0"
  "SKIP_WARMUP=0"
  "KD_PROFILE_TIMING=1"
)

run_case() {
  local case_name="$1"
  shift
  local case_dir="$campaign_root/cases/$case_name"
  local run_id="kd_softdbg_${ts}_${case_name}"
  local trainer_log="$repo_root/logs/${run_id}.txt"
  local driver_log="$case_dir/driver.log"
  local wall_start wall_end wall_sec

  mkdir -p "$case_dir"
  wall_start="$(date +%s)"

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '$repo_root'"
    echo -n "env RUN_ID='$run_id' "
    for kv in "${common_env[@]}"; do
      printf "%q " "$kv"
    done
    for kv in "$@"; do
      printf "%q " "$kv"
    done
    printf "%q " "$torchrun_bin"
    echo "--standalone --nproc_per_node='$nproc_per_node' train_gpt.py"
  } > "$case_dir/command.sh"
  chmod +x "$case_dir/command.sh"

  cat > "$case_dir/status.json" <<JSON
{"case_name":"$case_name","run_id":"$run_id","status":"running","start_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
JSON

  local exit_code=0
  if "$case_dir/command.sh" > "$driver_log" 2>&1; then
    exit_code=0
  else
    exit_code=$?
  fi

  if [[ -f "$trainer_log" ]]; then
    cp "$trainer_log" "$case_dir/train.log"
  fi

  wall_end="$(date +%s)"
  wall_sec=$((wall_end - wall_start))
  echo "$wall_sec" > "$case_dir/wall_seconds.txt"
  tail -n 200 "$driver_log" > "$case_dir/driver_tail.txt" || true

  local status="finished"
  if [[ "$exit_code" -ne 0 ]]; then
    status="failed"
  fi
  cat > "$case_dir/status.json" <<JSON
{"case_name":"$case_name","run_id":"$run_id","status":"$status","end_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","exit_code":$exit_code}
JSON

  return "$exit_code"
}

reached_target_steps() {
  local case_name="$1"
  local log="$campaign_root/cases/$case_name/train.log"
  [[ -f "$log" ]] || return 1
  grep -Eq "step:${num_steps}/" "$log"
}

is_oom_case() {
  local case_name="$1"
  local log="$campaign_root/cases/$case_name/driver.log"
  [[ -f "$log" ]] || return 1
  grep -Eiq "out of memory|cuda error: out of memory|cublas_status_alloc_failed|cudnn_status_alloc_failed" "$log"
}

first_fit_ga=""

echo "[phase1] Unchunked GA fit search: $ga_candidates"
for ga in $ga_candidates; do
  case_name="ga${ga}_full_kd_unchunked"
  echo "[run] $case_name"
  if run_case \
    "$case_name" \
    "GRAD_ACCUM_STEPS=$ga" \
    "KD_ENABLED=1" \
    "KD_TEACHER_INFER_ONLY=0" \
    "KD_RANDOM_TEACHER=0" \
    "KD_RANDOM_MODEL_TEACHER=1" \
    "KD_TEACHER_NUM_LAYERS=$teacher_layers" \
    "KD_TEACHER_NUM_HEADS=$teacher_heads" \
    "KD_TEACHER_HEAD_DIM=$teacher_head_dim" \
    "KD_TEACHER_MODEL_DIM=$teacher_model_dim" \
    "KD_TEACHER_FORCE_SDPA=0" \
    "KD_COMPILE_TEACHER=1" \
    "KD_SOFT_LOSS_TOKEN_CHUNK=0" \
    "KD_SOFT_LOSS_TOPK=0" \
    "KD_SOFT_LOSS_TOKEN_STRIDE=1" \
    "KD_DYNAMIC_NORM=1" \
    "KD_APPLY_EVERY_STEPS=1" \
    "KD_MICROBATCHES_PER_STEP=0"; then
    exit_code=0
  else
    exit_code=$?
  fi

  if [[ "$exit_code" -eq 0 ]] && reached_target_steps "$case_name"; then
    first_fit_ga="$ga"
    echo "[phase1] first fit at GA=$first_fit_ga"
    break
  fi
  if is_oom_case "$case_name"; then
    echo "[phase1] $case_name -> OOM (continuing)"
  else
    echo "[phase1] $case_name -> failed (non-OOM or partial)"
  fi
done

if [[ -z "$first_fit_ga" ]]; then
  echo "[phase1] no unchunked fit up to GA candidates: $ga_candidates"
  echo "NONE" > "$campaign_root/first_fit_ga.txt"
else
  echo "$first_fit_ga" > "$campaign_root/first_fit_ga.txt"

  echo "[phase2] decomposition at GA=$first_fit_ga"
  run_case \
    "decomp_ce_only_ga${first_fit_ga}" \
    "GRAD_ACCUM_STEPS=$first_fit_ga" \
    "KD_ENABLED=0"

  run_case \
    "decomp_tandem_infer_only_ga${first_fit_ga}" \
    "GRAD_ACCUM_STEPS=$first_fit_ga" \
    "KD_ENABLED=1" \
    "KD_TEACHER_INFER_ONLY=1" \
    "KD_RANDOM_TEACHER=0" \
    "KD_RANDOM_MODEL_TEACHER=1" \
    "KD_TEACHER_NUM_LAYERS=$teacher_layers" \
    "KD_TEACHER_NUM_HEADS=$teacher_heads" \
    "KD_TEACHER_HEAD_DIM=$teacher_head_dim" \
    "KD_TEACHER_MODEL_DIM=$teacher_model_dim" \
    "KD_TEACHER_FORCE_SDPA=0" \
    "KD_COMPILE_TEACHER=1" \
    "KD_SOFT_LOSS_TOKEN_CHUNK=0" \
    "KD_SOFT_LOSS_TOPK=0" \
    "KD_SOFT_LOSS_TOKEN_STRIDE=1" \
    "KD_DYNAMIC_NORM=1" \
    "KD_APPLY_EVERY_STEPS=1" \
    "KD_MICROBATCHES_PER_STEP=0"

  run_case \
    "decomp_full_kd_unchunked_ga${first_fit_ga}" \
    "GRAD_ACCUM_STEPS=$first_fit_ga" \
    "KD_ENABLED=1" \
    "KD_TEACHER_INFER_ONLY=0" \
    "KD_RANDOM_TEACHER=0" \
    "KD_RANDOM_MODEL_TEACHER=1" \
    "KD_TEACHER_NUM_LAYERS=$teacher_layers" \
    "KD_TEACHER_NUM_HEADS=$teacher_heads" \
    "KD_TEACHER_HEAD_DIM=$teacher_head_dim" \
    "KD_TEACHER_MODEL_DIM=$teacher_model_dim" \
    "KD_TEACHER_FORCE_SDPA=0" \
    "KD_COMPILE_TEACHER=1" \
    "KD_SOFT_LOSS_TOKEN_CHUNK=0" \
    "KD_SOFT_LOSS_TOPK=0" \
    "KD_SOFT_LOSS_TOKEN_STRIDE=1" \
    "KD_DYNAMIC_NORM=1" \
    "KD_APPLY_EVERY_STEPS=1" \
    "KD_MICROBATCHES_PER_STEP=0"

  run_case \
    "decomp_full_kd_chunk1024_ga${first_fit_ga}" \
    "GRAD_ACCUM_STEPS=$first_fit_ga" \
    "KD_ENABLED=1" \
    "KD_TEACHER_INFER_ONLY=0" \
    "KD_RANDOM_TEACHER=0" \
    "KD_RANDOM_MODEL_TEACHER=1" \
    "KD_TEACHER_NUM_LAYERS=$teacher_layers" \
    "KD_TEACHER_NUM_HEADS=$teacher_heads" \
    "KD_TEACHER_HEAD_DIM=$teacher_head_dim" \
    "KD_TEACHER_MODEL_DIM=$teacher_model_dim" \
    "KD_TEACHER_FORCE_SDPA=0" \
    "KD_COMPILE_TEACHER=1" \
    "KD_SOFT_LOSS_TOKEN_CHUNK=1024" \
    "KD_SOFT_LOSS_TOPK=0" \
    "KD_SOFT_LOSS_TOKEN_STRIDE=1" \
    "KD_DYNAMIC_NORM=1" \
    "KD_APPLY_EVERY_STEPS=1" \
    "KD_MICROBATCHES_PER_STEP=0"
fi

python3 "$repo_root/scripts/ablation/analyze_kd_unchunked_ga_sweep.py" \
  --campaign-root "$campaign_root" \
  --drop-first-deltas "$drop_first_deltas" \
  --steps "$num_steps"

echo "campaign_root=$campaign_root"
