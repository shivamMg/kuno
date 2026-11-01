#!/usr/bin/env bash
set -Eeuo pipefail

FATAL_EXIT=78
TRAINING_ENABLED="${KUNO_TRAINING_ENABLED:-false}"

if [[ "$TRAINING_ENABLED" != "true" && "$TRAINING_ENABLED" != "false" ]]; then
  printf 'kuno-entrypoint: KUNO_TRAINING_ENABLED must be true or false\n' >&2
  exit "$FATAL_EXIT"
fi

if [[ "${KUNO_HEARTBEAT_TRACKED:-false}" != "true" ]]; then
  exec /opt/venv/bin/python -m kuno.train.heartbeat "$0" "$@"
fi

write_dry_run_status() {
  local status="$1" step="$2"
  local run_dir="$KUNO_DURABLE_PATH/runs/$KUNO_RUN_ID"
  local pending="$run_dir/.status.$$.json"
  install -d -m 0755 "$run_dir"
  /opt/venv/bin/python - "$pending" "$KUNO_RUN_ID" "$status" "$step" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, run_id, status, step = sys.argv[1:]
payload = {
    "dry_run": True,
    "reason": "storage dry run",
    "run_id": run_id,
    "status": status,
    "step": int(step),
    "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
  mv -f "$pending" "$run_dir/status.json"
}

run_storage_dry_run() {
  : "${KUNO_RUN_ID:?KUNO_RUN_ID is required}"
  : "${KUNO_DURABLE_PATH:?KUNO_DURABLE_PATH is required}"
  local raw_steps="${KUNO_DRY_RUN_STEPS:-3}"
  local raw_interval="${KUNO_DRY_RUN_INTERVAL_SECONDS:-1}"
  local steps interval step
  [[ -d "$KUNO_DURABLE_PATH" && -w "$KUNO_DURABLE_PATH" ]] || {
    printf 'kuno-entrypoint: durable storage is not writable\n' >&2
    exit 75
  }
  [[ "$raw_steps" =~ ^[0-9]+$ ]] || {
    printf 'kuno-entrypoint: KUNO_DRY_RUN_STEPS must be a positive integer\n' >&2
    exit "$FATAL_EXIT"
  }
  [[ "$raw_interval" =~ ^[0-9]+$ ]] || {
    printf 'kuno-entrypoint: KUNO_DRY_RUN_INTERVAL_SECONDS must be nonnegative\n' >&2
    exit "$FATAL_EXIT"
  }
  steps=$((10#$raw_steps))
  interval=$((10#$raw_interval))
  [[ "$steps" -gt 0 ]] || {
    printf 'kuno-entrypoint: KUNO_DRY_RUN_STEPS must be a positive integer\n' >&2
    exit "$FATAL_EXIT"
  }
  printf 'kuno-entrypoint: starting storage dry run with %s steps\n' "$steps" >&2
  for ((step = 1; step <= steps; step++)); do
    write_dry_run_status running "$step"
    printf 'kuno-entrypoint: storage dry-run step %s/%s\n' "$step" "$steps" >&2
    if [[ "$step" -lt "$steps" && "$interval" -gt 0 ]]; then
      sleep "$interval"
    fi
  done
  write_dry_run_status completed "$steps"
  printf 'kuno-entrypoint: storage dry run completed\n' >&2
}

if [[ "$TRAINING_ENABLED" != "true" ]]; then
  run_storage_dry_run
  exit 0
fi

: "${KUNO_CONFIG_PATH:?KUNO_CONFIG_PATH is required}"
if [[ $# -gt 0 ]]; then
  exec "$@"
fi
/opt/venv/bin/kuno-train "$KUNO_CONFIG_PATH"
exec accelerate launch --num_processes "${KUNO_NUM_PROCESSES:-8}" \
  /opt/venv/bin/kuno-train "$KUNO_CONFIG_PATH" --execute
