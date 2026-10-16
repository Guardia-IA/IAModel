#!/usr/bin/env bash
# Campaign2 — Sprint 1: probe, preflight (caché 12Hz), train paralelo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TRAINING_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"

PY="${PYTHON:-python3}"
CURRENT_RUN_FILE="${SCRIPT_DIR}/artifacts/runs/.current_run"
RUN_ID="${RUN_ID:-}"

CMD="${1:-help}"
shift || true
EXTRA_ARGS=("$@")

parse_run_id_from_args() {
  local args=("$@")
  local i=0
  while [[ $i -lt ${#args[@]} ]]; do
    if [[ "${args[$i]}" == "--run-id" && $((i + 1)) -lt ${#args[@]} ]]; then
      RUN_ID="${args[$((i + 1))]}"
      return 0
    fi
    i=$((i + 1))
  done
  return 1
}

ensure_run_id() {
  if [[ -n "$RUN_ID" ]]; then
    return 0
  fi
  if [[ -f "$CURRENT_RUN_FILE" ]]; then
    RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
  fi
  if [[ -z "$RUN_ID" ]]; then
    echo "ERROR: sin RUN_ID. Usa: $0 new-run" >&2
    exit 2
  fi
}

case "$CMD" in
  help)
    cat <<EOF
Uso: $0 <comando> [args...]

  new-run              Crea RUN_ID campaign2_YYYYMMDD_HHMMSS
  probe [--run-id ID]  gpu_probe_campaign2.py
  preflight            preflight_campaign2.py (+ --data-root PATH repetible)
  train                train_campaign2_parallel.py
  pipeline             new-run + probe + preflight + train (smoke)

Variables:
  RUN_ID, CAMPAIGN2_MAX_PARALLEL, PYTHON

Ejemplo:
  $0 new-run
  $0 preflight --data-root /media/8TB/.../data_result
  $0 probe && $0 train
EOF
    ;;
  new-run)
    mkdir -p "$(dirname "$CURRENT_RUN_FILE")"
    RUN_ID="$("$PY" -c "from campaign2_paths import new_run_id; print(new_run_id())")"
    echo "$RUN_ID" > "$CURRENT_RUN_FILE"
    mkdir -p "${SCRIPT_DIR}/artifacts/runs/${RUN_ID}"
    echo "RUN_ID=${RUN_ID}"
    ;;
  probe)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" gpu_probe_campaign2.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  preflight)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" preflight_campaign2.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  train)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    if [[ -n "${CAMPAIGN2_MAX_PARALLEL:-}" ]]; then
      "$PY" train_campaign2_parallel.py --run-id "$RUN_ID" --max-parallel "$CAMPAIGN2_MAX_PARALLEL" "${EXTRA_ARGS[@]}"
    else
      "$PY" train_campaign2_parallel.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    fi
    ;;
  pipeline)
    "$0" new-run
    RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
    export RUN_ID
    "$0" probe --run-id "$RUN_ID"
    "$0" preflight --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    "$0" train --run-id "$RUN_ID"
    ;;
  *)
    echo "Comando desconocido: $CMD" >&2
    exit 1
    ;;
esac
