#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_id>"
  exit 2
fi

RUN_ID="$1"

LOCAL_REPO="${LOCAL_REPO:-/Users/lukaivanic/projects/nanogpt-speedrun/modded-nanogpt}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$LOCAL_REPO/logs/prime_a100/$RUN_ID}"
TARGET_BRANCH="${TARGET_BRANCH:-codex/prime-a100-spot}"
POLL_SECONDS="${POLL_SECONDS:-20}"

LAST_STEP_FILE="$LOCAL_RUN_DIR/.last_pushed_step"
LAST_STEP="-1"
if [[ -f "$LAST_STEP_FILE" ]]; then
  LAST_STEP="$(cat "$LAST_STEP_FILE" 2>/dev/null || echo -1)"
fi

echo "[INFO] Monitoring $LOCAL_RUN_DIR for new eval rows..."
echo "[INFO] Branch target: $TARGET_BRANCH"

while true; do
  VAL_CSV="$LOCAL_RUN_DIR/val_metrics.csv"
  STATUS_JSON="$LOCAL_RUN_DIR/status.json"

  NEW_STEP="$LAST_STEP"
  NEW_VAL=""

  if [[ -f "$VAL_CSV" ]]; then
    LAST_ROW="$(tail -n +2 "$VAL_CSV" | tail -n 1 || true)"
    if [[ -n "$LAST_ROW" ]]; then
      STEP="$(echo "$LAST_ROW" | awk -F, '{print $1}')"
      VAL="$(echo "$LAST_ROW" | awk -F, '{print $3}')"
      if [[ "$STEP" =~ ^[0-9]+$ ]]; then
        NEW_STEP="$STEP"
        NEW_VAL="$VAL"
      fi
    fi
  fi

  SHOULD_PUSH=0
  if [[ "$NEW_STEP" =~ ^[0-9]+$ ]] && [[ "$NEW_STEP" -gt "$LAST_STEP" ]]; then
    SHOULD_PUSH=1
  fi

  STATUS="running"
  if [[ -f "$STATUS_JSON" ]]; then
    STATUS="$(python3 - <<PY
import json
with open("$STATUS_JSON","r",encoding="utf-8") as f:
    print(json.load(f).get("status","running"))
PY
)"
  fi

  if [[ "$STATUS" != "running" ]]; then
    SHOULD_PUSH=1
  fi

  if [[ "$SHOULD_PUSH" -eq 1 ]]; then
    cd "$LOCAL_REPO"
    git switch "$TARGET_BRANCH" >/dev/null 2>&1 || true
    git add "$LOCAL_RUN_DIR" >/dev/null 2>&1 || true

    if ! git diff --cached --quiet; then
      MSG="prime-a100 ${RUN_ID} step ${NEW_STEP}"
      if [[ -n "$NEW_VAL" ]]; then
        MSG="$MSG val ${NEW_VAL}"
      fi
      git commit -m "$MSG"
      git push origin "$TARGET_BRANCH"
      LAST_STEP="$NEW_STEP"
      echo "$LAST_STEP" > "$LAST_STEP_FILE"
      echo "[INFO] pushed commit for step=$LAST_STEP"
    fi
  fi

  if [[ "$STATUS" != "running" ]]; then
    echo "[INFO] status=$STATUS; exiting monitor loop."
    break
  fi

  sleep "$POLL_SECONDS"
done
