#!/usr/bin/env bash
set -Eeuo pipefail

RETRYABLE_EXIT=75
FATAL_EXIT=78
TRAINING_ENABLED="${KUNO_TRAINING_ENABLED:-false}"

if [[ "$TRAINING_ENABLED" != "true" && "$TRAINING_ENABLED" != "false" ]]; then
  printf 'kuno-training: KUNO_TRAINING_ENABLED must be true or false\n' >&2
  exit "$FATAL_EXIT"
fi
[[ "$TRAINING_ENABLED" == "true" ]] && DRY_RUN=false || DRY_RUN=true

: "${KUNO_RUN_ID:?KUNO_RUN_ID is required}"
: "${KUNO_DURABLE_PATH:?KUNO_DURABLE_PATH is required}"
: "${KUNO_IMAGE:?KUNO_IMAGE is required}"
if [[ "$DRY_RUN" != "true" ]]; then
  : "${KUNO_LOCAL_PATH:?KUNO_LOCAL_PATH is required}"
  : "${KUNO_CONFIG_PATH:?KUNO_CONFIG_PATH is required}"
fi

NUM_PROCESSES="${KUNO_NUM_PROCESSES:-8}"
MIN_SCRATCH_GB="${KUNO_MIN_SCRATCH_GB:-100}"
# Fixed name so the unit's ExecStopPost can reap a container that outlives the
# docker client, and so a restart cannot run two containers for the same run.
CONTAINER_NAME="${KUNO_CONTAINER_NAME:-kuno-training}"
CHILD_PID=""

log() {
  printf 'kuno-training: %s\n' "$*" >&2
}

halt_report() {
  local reason="$1"
  local run_dir="$KUNO_DURABLE_PATH/runs/$KUNO_RUN_ID"
  local pending="$run_dir/.HALT.$$.json"
  install -d -m 0755 "$run_dir" || return 1
  python3 - "$pending" "$KUNO_RUN_ID" "$reason" <<'PY' || return 1
import json
import sys
from datetime import datetime, timezone

path, run_id, reason = sys.argv[1:]
payload = {
    "run_id": run_id,
    "status": "fatal",
    "reason": reason,
    "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
  mv -f "$pending" "$run_dir/HALT.json" || return 1
}

write_status() {
  local status="$1" reason="${2:-}" step="${3:-}" dry_run="${4:-false}"
  local run_dir="$KUNO_DURABLE_PATH/runs/$KUNO_RUN_ID"
  local pending="$run_dir/.status.$$.json"
  install -d -m 0755 "$run_dir" || return 1
  python3 - "$pending" "$KUNO_RUN_ID" "$status" "$reason" "$step" "$dry_run" <<'PY' || return 1
import json
import sys
from datetime import datetime, timezone

path, run_id, status, reason, step, dry_run = sys.argv[1:]
payload = {
    "run_id": run_id,
    "status": status,
    "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
}
if reason:
    payload["reason"] = reason
if step:
    payload["step"] = int(step)
if dry_run == "true":
    payload["dry_run"] = True
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
PY
  mv -f "$pending" "$run_dir/status.json" || return 1
}

forward_signal() {
  docker stop --time 60 "$CONTAINER_NAME" >/dev/null 2>&1 || true
  if [[ -n "$CHILD_PID" ]]; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
  fi
}
trap forward_signal TERM INT

retryable_failure() {
  log "$* (retryable)"
  exit "$RETRYABLE_EXIT"
}

fatal_failure() {
  log "$* (fatal)"
  halt_report "$*" || log "unable to record halt report"
  exit "$FATAL_EXIT"
}

code_in_list() {
  local wanted="$1" list="$2" value
  local -a values
  IFS=',' read -r -a values <<<"$list"
  for value in "${values[@]}"; do
    [[ "$wanted" == "$value" ]] && return 0
  done
  return 1
}

check_image() {
  [[ "$KUNO_IMAGE" =~ @sha256:[[:xdigit:]]{64}$ ]] \
    || fatal_failure "KUNO_IMAGE is not an exact image digest"
  command -v docker >/dev/null 2>&1 \
    || retryable_failure "docker is unavailable"
  local digests
  digests="$(docker image inspect --format '{{index .RepoDigests 0}}' "$KUNO_IMAGE" 2>/dev/null)" \
    || retryable_failure "immutable image is not available locally"
  printf '%s\n' "$digests" | grep -Fx "$KUNO_IMAGE" >/dev/null \
    || fatal_failure "local image digest does not match desired image"
}

check_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || retryable_failure "nvidia-smi is unavailable"
  local names count
  names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)" \
    || retryable_failure "unable to query GPUs"
  count="$(printf '%s\n' "$names" | sed '/^[[:space:]]*$/d' | wc -l)"
  [[ "$count" -eq 8 ]] || retryable_failure "expected 8 GPUs, found $count"
  if printf '%s\n' "$names" | grep -qiv 'H100'; then
    retryable_failure "all GPUs must be NVIDIA H100"
  fi
  local topology
  topology="$(nvidia-smi topo -m 2>/dev/null)" || retryable_failure "NVLink topology query failed"
  printf '%s\n' "$topology" | grep -Eiq 'NV[0-9]*|NVLINK' \
    || retryable_failure "NVLink topology was not detected"
}

check_storage() {
  command -v mountpoint >/dev/null 2>&1 || retryable_failure "mountpoint is unavailable"
  mountpoint -q "$KUNO_DURABLE_PATH" \
    || retryable_failure "durable BlobFuse path is not mounted"
  [[ -r "$KUNO_DURABLE_PATH" && -d "$KUNO_DURABLE_PATH" ]] \
    || retryable_failure "durable BlobFuse path is not readable"
}

check_scratch() {
  install -d -m 0755 "$KUNO_LOCAL_PATH" \
    || retryable_failure "unable to create local scratch path"
  command -v findmnt >/dev/null 2>&1 \
    || retryable_failure "findmnt is unavailable"
  mountpoint -q "$KUNO_LOCAL_PATH" \
    && retryable_failure "local scratch path must be node-local"
  local fstype available required
  fstype="$(findmnt -T "$KUNO_LOCAL_PATH" -n -o FSTYPE 2>/dev/null || true)"
  case "$fstype" in
    fuse*|nfs*|cifs|blobfuse*) retryable_failure "local scratch is on $fstype" ;;
  esac
  available="$(df -Pk "$KUNO_LOCAL_PATH" | awk 'NR==2 {print $4}')"
  required="$((MIN_SCRATCH_GB * 1024 * 1024))"
  [[ "$available" =~ ^[0-9]+$ && "$available" -ge "$required" ]] \
    || retryable_failure "insufficient local scratch capacity"
}

verify_hash() {
  local path="$1" expected="$2" actual
  [[ -n "$expected" ]] || return 0
  [[ -n "$path" ]] || fatal_failure "hash path is not configured"
  [[ -f "$path" ]] || fatal_failure "hash input does not exist: $path"
  actual="$(sha256sum "$path" | awk '{print $1}')" \
    || fatal_failure "unable to hash $path"
  [[ "$actual" == "${expected#sha256:}" ]] \
    || fatal_failure "checksum mismatch for $path"
}

launch_container() {
  local -a command
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  command=(
    docker run --rm
    --name "$CONTAINER_NAME"
    --user "$(id -u):$(id -g)"
    -e "KUNO_RUN_ID=$KUNO_RUN_ID"
    -e "KUNO_COMPUTE_ID=${KUNO_COMPUTE_ID:-}"
    -e "KUNO_NODE_ID=${KUNO_NODE_ID:-}"
    -e "KUNO_IMAGE=$KUNO_IMAGE"
    -e "KUNO_TRAINING_ENABLED=$TRAINING_ENABLED"
    -e "KUNO_DURABLE_PATH=/mnt/kuno-training"
    -e "KUNO_CONFIG_PATH=${KUNO_CONFIG_PATH:-/mnt/kuno-training}"
    -v "$KUNO_DURABLE_PATH:/mnt/kuno-training"
  )
  if [[ "$DRY_RUN" == "true" ]]; then
    command+=(
      --network none
      -e "KUNO_DRY_RUN_STEPS=${KUNO_DRY_RUN_STEPS:-3}"
      -e "KUNO_DRY_RUN_INTERVAL_SECONDS=${KUNO_DRY_RUN_INTERVAL_SECONDS:-1}"
    )
  else
    command+=(
      --gpus all --network host --ipc host --shm-size=16g
      -e "KUNO_NUM_PROCESSES=$NUM_PROCESSES"
      -e "KUNO_CONFIG_HASH=${KUNO_CONFIG_HASH:-}"
      -v "$KUNO_LOCAL_PATH:/mnt/kuno-local"
    )
  fi
  command+=("$KUNO_IMAGE")
  log "starting immutable image $KUNO_IMAGE"
  "${command[@]}" &
  CHILD_PID=$!
  wait "$CHILD_PID"
}

main() {
  check_image
  check_storage
  if [[ "$DRY_RUN" != "true" ]]; then
    check_gpu
    check_scratch
    verify_hash "${KUNO_CONFIG_PATH:-}" "${KUNO_CONFIG_HASH:-}"
  fi
  local code
  if [[ "$DRY_RUN" != "true" ]]; then
    write_status "running" || retryable_failure "unable to record running status"
  fi
  set +e
  launch_container
  code=$?
  set -e
  CHILD_PID=""
  case "$code" in
    0)
      if [[ "$DRY_RUN" != "true" ]]; then
        write_status "completed" || log "unable to record completed status"
      fi
      return 0
      ;;
    *)
      if code_in_list "$code" "${KUNO_RETRYABLE_EXIT_CODES:-75,137,143}"; then
        write_status "retryable" "container exited with status $code" \
          || log "unable to record retryable status"
        return "$RETRYABLE_EXIT"
      fi
      local reason
      if code_in_list "$code" "${KUNO_FATAL_EXIT_CODES:-78}"; then
        reason="container exited with fatal status $code"
      else
        reason="container exited with unexpected status $code"
      fi
      write_status "fatal" "$reason" || log "unable to record fatal status"
      halt_report "$reason" || log "unable to record halt report"
      return "$FATAL_EXIT"
      ;;
  esac
}

main "$@"
