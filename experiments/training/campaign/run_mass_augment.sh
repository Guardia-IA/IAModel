#!/usr/bin/env bash
# Orquestador augmentación masiva (~100k filas train).
# Preflight mass → train (mismo train_campaign.py) → eval real + sintético.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TRAINING_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"

PY="${PYTHON:-python3}"
CURRENT_RUN_FILE="${SCRIPT_DIR}/artifacts/runs/.current_mass_run"
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

strip_run_id_from_extra() {
  local out=()
  local skip=0
  local a
  for a in "${EXTRA_ARGS[@]}"; do
    if [[ $skip -eq 1 ]]; then
      skip=0
      continue
    fi
    if [[ "$a" == "--run-id" ]]; then
      skip=1
      continue
    fi
    out+=("$a")
  done
  EXTRA_ARGS=("${out[@]}")
}

ensure_run_id() {
  if [[ -n "$RUN_ID" ]]; then
    return 0
  fi
  if [[ -f "$CURRENT_RUN_FILE" ]]; then
    RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
  fi
  if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$("$PY" -c "from campaign_paths import new_run_id; print(new_run_id('mass'))")"
    echo "$RUN_ID" > "$CURRENT_RUN_FILE"
    echo "[!] RUN_ID auto-creado: ${RUN_ID}" >&2
  fi
}

run_root() {
  echo "${SCRIPT_DIR}/artifacts/runs/${RUN_ID}"
}

logs_dir() {
  mkdir -p "$(run_root)/logs"
  echo "$(run_root)/logs"
}

write_run_meta() {
  local phase="${1:-init}"
  "$PY" - <<PY
import json
from datetime import datetime, timezone
from campaign_paths import run_meta_path, load_campaign_config, resolve_experiment_ids

run_id = ${RUN_ID@Q}
cfg = load_campaign_config()
meta = {
    "run_id": run_id,
    "pipeline": "mass_augment",
    "phase": ${phase@Q},
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "experiment_ids": resolve_experiment_ids(cfg.get("experiment_ids")),
}
path = run_meta_path(run_id)
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(path)
PY
}

run_args() {
  echo --run-id "$RUN_ID"
}

extra_py_args() {
  if ((${#EXTRA_ARGS[@]})); then
    printf '%s' "${EXTRA_ARGS[*]}"
  fi
}

log_banner() {
  local logfile="$1"
  local title="$2"
  {
    echo ""
    echo "================================================================"
    echo "${title} — $(date -Iseconds)"
    echo "RUN_ID=${RUN_ID}"
    echo "================================================================"
  } >> "$logfile"
}

run_preflight() {
  ensure_run_id
  mkdir -p "$(run_root)/logs"
  echo "$RUN_ID" > "$CURRENT_RUN_FILE"
  write_run_meta "preflight_start" >/dev/null

  local logfile
  logfile="$(logs_dir)/preflight.log"
  log_banner "$logfile" "PREFLIGHT MASS AUG"

  echo "Run: ${RUN_ID}"
  echo "Artefactos: $(run_root)"
  echo "Log: ${logfile}"

  {
    echo "--- validate ---"
    "$PY" validate_campaign.py $(run_args) "${EXTRA_ARGS[@]}"
    echo "--- preflight_mass_augment --write-all ---"
    "$PY" preflight_mass_augment.py --write-all $(run_args) "${EXTRA_ARGS[@]}"
    echo "--- validate --require-plans ---"
    "$PY" validate_campaign.py --require-plans $(run_args) "${EXTRA_ARGS[@]}"
  } 2>&1 | tee -a "$logfile"

  write_run_meta "preflight_done" >/dev/null
  echo ""
  echo "[OK] Preflight mass aug completado. Siguiente: ./run_mass_augment.sh train-bg"
}

run_train_bg() {
  ensure_run_id
  local logfile pidfile
  logfile="$(logs_dir)/train.log"
  pidfile="$(logs_dir)/train.pid"
  log_banner "$logfile" "TRAIN MASS AUG (background)"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Train ya en curso (PID $(cat "$pidfile")). Log: ${logfile}" >&2
    exit 1
  fi

  echo "Run: ${RUN_ID}"
  echo "Log: ${logfile}"
  write_run_meta "train_start" >/dev/null

  local extra
  extra="$(extra_py_args)"

  nohup bash >> "${logfile}" 2>&1 <<SCRIPT &
set -euo pipefail
cd "${SCRIPT_DIR}"
${PY} validate_campaign.py --require-plans --run-id "${RUN_ID}" ${extra}
${PY} train_campaign.py --all --resume --run-id "${RUN_ID}" ${extra}
SCRIPT

  echo $! > "$pidfile"
  echo "[OK] Train en background PID $(cat "$pidfile")"
  echo "  tail -f ${logfile}"
}

run_eval_bg() {
  ensure_run_id
  local logfile pidfile train_pidfile
  logfile="$(logs_dir)/eval.log"
  pidfile="$(logs_dir)/eval.pid"
  train_pidfile="$(logs_dir)/train.pid"

  if [[ -f "$train_pidfile" ]] && kill -0 "$(cat "$train_pidfile")" 2>/dev/null; then
    echo "Esperando train (PID $(cat "$train_pidfile"))..." >&2
    while kill -0 "$(cat "$train_pidfile")" 2>/dev/null; do sleep 30; done
  fi

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Eval ya en curso (PID $(cat "$pidfile")). Log: ${logfile}" >&2
    exit 1
  fi

  log_banner "$logfile" "EVAL MASS AUG (real + sintético)"
  write_run_meta "eval_start" >/dev/null

  local extra
  extra="$(extra_py_args)"

  nohup bash >> "${logfile}" 2>&1 <<SCRIPT &
set -euo pipefail
cd "${SCRIPT_DIR}"
${PY} evaluate_mass_augment.py --all --run-id "${RUN_ID}" ${extra}
${PY} summarize_campaign.py --run-id "${RUN_ID}" ${extra}
SCRIPT

  echo $! > "$pidfile"
  echo "[OK] Eval en background PID $(cat "$pidfile")"
  echo "  tail -f ${logfile}"
}

run_all_bg() {
  RUN_ID="$("$PY" -c "from campaign_paths import new_run_id; print(new_run_id('mass'))")"
  echo "$RUN_ID" > "$CURRENT_RUN_FILE"
  mkdir -p "$(run_root)/logs"
  write_run_meta "pipeline_start" >/dev/null

  local pipeline_log extra run_root_path
  pipeline_log="$(logs_dir)/pipeline.log"
  extra="$(extra_py_args)"
  run_root_path="$(run_root)"
  echo "Nuevo pipeline MASS RUN_ID=${RUN_ID}"

  nohup bash >> "${pipeline_log}" 2>&1 <<SCRIPT &
set -euo pipefail
cd "${SCRIPT_DIR}"
export RUN_ID="${RUN_ID}"
./run_mass_augment.sh preflight --run-id "${RUN_ID}" ${extra}
./run_mass_augment.sh train-bg --run-id "${RUN_ID}" ${extra}
train_pid=\$(cat "${run_root_path}/logs/train.pid")
while kill -0 "\$train_pid" 2>/dev/null; do sleep 60; done
./run_mass_augment.sh eval-bg --run-id "${RUN_ID}" ${extra}
eval_pid=\$(cat "${run_root_path}/logs/eval.pid")
while kill -0 "\$eval_pid" 2>/dev/null; do sleep 30; done
echo "[FIN] Pipeline mass aug RUN_ID=${RUN_ID}"
SCRIPT

  echo $! > "$(logs_dir)/pipeline.pid"
  echo "[OK] Pipeline PID $(cat "$(logs_dir)/pipeline.pid")"
  echo "  tail -f ${pipeline_log}"
}

show_status() {
  ensure_run_id
  echo "RUN_ID mass aug: ${RUN_ID}"
  echo "Carpeta: $(run_root)"
}

parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
strip_run_id_from_extra

case "$CMD" in
  new-run)
    RUN_ID="$("$PY" -c "from campaign_paths import new_run_id; print(new_run_id('mass'))")"
    mkdir -p "${SCRIPT_DIR}/artifacts/runs"
    echo "$RUN_ID" > "$CURRENT_RUN_FILE"
    mkdir -p "$(run_root)/logs"
    write_run_meta "created" >/dev/null
    echo "Nuevo RUN_ID: ${RUN_ID}"
    echo "Siguiente: ./run_mass_augment.sh preflight --cells mc_full mc_filtered bin_full bin_filtered"
    ;;
  status)
    show_status
    ;;
  check-ready)
    ensure_run_id
    exec "$PY" validate_campaign.py --require-plans $(run_args) "${EXTRA_ARGS[@]}"
    ;;
  preflight)
    run_preflight
    ;;
  train|train-bg)
    run_train_bg
    ;;
  eval|eval-bg)
    run_eval_bg
    ;;
  all-bg|pipeline-bg)
    run_all_bg
    ;;
  help|*)
    cat <<'EOF'
Uso: ./run_mass_augment.sh <comando> [args Python...]

Pipeline augmentación masiva (~100k filas train, val/test reales):
  artifacts/runs/<RUN_ID>/plans/<cell>/config_mass_augmentation.json

Comandos:
  new-run       RUN_ID con prefijo mass_
  preflight     preflight_mass_augment.py --write-all (+ estimación tiempo)
  train-bg      train_campaign.py --all (usa mass_augmentation del plan)
  eval-bg       evaluate_mass_augment.py --all (real + sintético)
  all-bg        preflight → train → eval en background
  check-ready   validate --require-plans

Flujo recomendado (4 celdas: mc_full, mc_filtered, bin_full, bin_filtered):
  ./run_mass_augment.sh new-run
  ./run_mass_augment.sh preflight --cells mc_full mc_filtered bin_full bin_filtered
  ./run_mass_augment.sh train-bg --cells mc_full mc_filtered bin_full bin_filtered
  tail -f artifacts/runs/<RUN_ID>/logs/train.log
  ./run_mass_augment.sh eval-bg --cells mc_full mc_filtered bin_full bin_filtered

Config recetas: experiments/training/config_mass_augmentation.json
Bloque campaña: campaign_config.json → mass_augment
EOF
    ;;
esac
