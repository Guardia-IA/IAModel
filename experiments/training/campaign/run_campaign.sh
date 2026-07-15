#!/usr/bin/env bash
# Orquestador campaña: cada run aislado en artifacts/runs/<RUN_ID>/
# Preflight (foreground + log) → train/eval (background + log por fase)
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

# Extra args para Python (puede incluir --run-id explícito)
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

strip_data_root_from_extra() {
  local out=()
  local skip=0
  local a
  for a in "${EXTRA_ARGS[@]}"; do
    if [[ $skip -eq 1 ]]; then
      skip=0
      continue
    fi
    if [[ "$a" == "--data-root" ]]; then
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
    RUN_ID="$("$PY" -c "from campaign_paths import new_run_id; print(new_run_id())")"
    echo "$RUN_ID" > "$CURRENT_RUN_FILE"
    echo "[!] RUN_ID auto-creado: ${RUN_ID} (usa 'new-run' antes del pipeline para uno nuevo explícito)" >&2
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
    "phase": ${phase@Q},
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "experiment_ids": resolve_experiment_ids(cfg.get("experiment_ids")),
    "experiment_ids_spec": cfg.get("experiment_ids"),
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
  log_banner "$logfile" "PREFLIGHT"

  echo "Run: ${RUN_ID}"
  echo "Artefactos: $(run_root)"
  echo "Log preflight: ${logfile}"

  {
    echo "--- validate ---"
    "$PY" validate_campaign.py $(run_args) "${EXTRA_ARGS[@]}"
    echo "--- preflight --write-all ---"
    "$PY" preflight_campaign.py --write-all $(run_args) "${EXTRA_ARGS[@]}"
    echo "--- validate --require-plans ---"
    "$PY" validate_campaign.py --require-plans $(run_args) "${EXTRA_ARGS[@]}"
  } 2>&1 | tee -a "$logfile"

  write_run_meta "preflight_done" >/dev/null
  echo ""
  echo "[OK] Preflight completado. Siguiente: ./run_campaign.sh train"
}

run_train_fg() {
  ensure_run_id
  strip_data_root_from_extra
  local logfile
  logfile="$(logs_dir)/train.log"
  log_banner "$logfile" "TRAIN (foreground)"
  echo "Log train: ${logfile}"
  {
    "$PY" validate_campaign.py --require-plans $(run_args) "${EXTRA_ARGS[@]}"
    "$PY" train_campaign.py --all --resume $(run_args) "${EXTRA_ARGS[@]}"
  } 2>&1 | tee -a "$logfile"
  write_run_meta "train_done" >/dev/null
}

run_train_bg() {
  ensure_run_id
  strip_data_root_from_extra
  local logfile pidfile
  logfile="$(logs_dir)/train.log"
  pidfile="$(logs_dir)/train.pid"
  log_banner "$logfile" "TRAIN (background)"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Train ya en curso (PID $(cat "$pidfile")). Log: ${logfile}" >&2
    exit 1
  fi

  echo "Run: ${RUN_ID}"
  echo "Log train: ${logfile}"
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
  echo "  Cuando termine: ./run_campaign.sh eval --run-id ${RUN_ID}"
}

run_eval_fg() {
  ensure_run_id
  strip_data_root_from_extra
  local logfile
  logfile="$(logs_dir)/eval.log"
  log_banner "$logfile" "EVAL (foreground)"
  {
    "$PY" evaluate_campaign.py --all --export-fp-videos $(run_args) "${EXTRA_ARGS[@]}"
    "$PY" summarize_campaign.py $(run_args) "${EXTRA_ARGS[@]}"
  } 2>&1 | tee -a "$logfile"
  write_run_meta "eval_done" >/dev/null
}

run_eval_bg() {
  ensure_run_id
  strip_data_root_from_extra
  local logfile pidfile train_pidfile
  logfile="$(logs_dir)/eval.log"
  pidfile="$(logs_dir)/eval.pid"
  train_pidfile="$(logs_dir)/train.pid"

  if [[ -f "$train_pidfile" ]] && kill -0 "$(cat "$train_pidfile")" 2>/dev/null; then
    echo "Esperando a que termine train (PID $(cat "$train_pidfile"))..." >&2
    while kill -0 "$(cat "$train_pidfile")" 2>/dev/null; do
      sleep 30
    done
    echo "Train finalizado." >&2
  fi

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Eval ya en curso (PID $(cat "$pidfile")). Log: ${logfile}" >&2
    exit 1
  fi

  log_banner "$logfile" "EVAL (background)"
  echo "Run: ${RUN_ID}"
  echo "Log eval: ${logfile}"
  write_run_meta "eval_start" >/dev/null

  local extra
  extra="$(extra_py_args)"

  nohup bash >> "${logfile}" 2>&1 <<SCRIPT &
set -euo pipefail
cd "${SCRIPT_DIR}"
${PY} evaluate_campaign.py --all --export-fp-videos --run-id "${RUN_ID}" ${extra}
${PY} summarize_campaign.py --run-id "${RUN_ID}" ${extra}
SCRIPT

  echo $! > "$pidfile"
  echo "[OK] Eval en background PID $(cat "$pidfile")"
  echo "  tail -f ${logfile}"
}

run_all_bg() {
  RUN_ID="$("$PY" -c "from campaign_paths import new_run_id; print(new_run_id())")"
  echo "$RUN_ID" > "$CURRENT_RUN_FILE"
  mkdir -p "$(run_root)/logs"
  write_run_meta "pipeline_start" >/dev/null

  local pipeline_log extra run_root_path
  pipeline_log="$(logs_dir)/pipeline.log"
  extra="$(extra_py_args)"
  run_root_path="$(run_root)"
  echo "Nuevo pipeline RUN_ID=${RUN_ID}"
  echo "Log unificado: ${pipeline_log}"

  nohup bash >> "${pipeline_log}" 2>&1 <<SCRIPT &
set -euo pipefail
cd "${SCRIPT_DIR}"
export RUN_ID="${RUN_ID}"
./run_campaign.sh preflight --run-id "${RUN_ID}" ${extra}
./run_campaign.sh train-bg --run-id "${RUN_ID}" ${extra}
train_pid=\$(cat "${run_root_path}/logs/train.pid")
while kill -0 "\$train_pid" 2>/dev/null; do sleep 60; done
./run_campaign.sh eval-bg --run-id "${RUN_ID}" ${extra}
eval_pid=\$(cat "${run_root_path}/logs/eval.pid")
while kill -0 "\$eval_pid" 2>/dev/null; do sleep 30; done
echo "[FIN] Pipeline completado RUN_ID=${RUN_ID}"
SCRIPT

  echo $! > "$(logs_dir)/pipeline.pid"
  echo "[OK] Pipeline en background PID $(cat "$(logs_dir)/pipeline.pid")"
  echo "  tail -f ${pipeline_log}"
}

run_sliding_window_fg() {
  ensure_run_id
  local mode="${SLIDING_MODE:-full}"
  local logfile
  logfile="$(logs_dir)/sliding_window.log"
  log_banner "$logfile" "SLIDING WINDOW (${mode})"

  local sw_cell="${SW_CELL:-bin_full}"
  local sw_model="${SW_MODEL:-modelo_12}"
  local sw_split="${SW_SPLIT:-val}"
  local sw_max_fp="${SW_MAX_FP:-1}"

  if [[ -z "${GUADIA_DATA_RESULT_ROOT:-}" ]]; then
    echo "[!] GUADIA_DATA_RESULT_ROOT no está definido — el eval puede fallar si los UIDs no resuelven poses." >&2
  fi

  echo "Run: ${RUN_ID}"
  echo "Modo: ${mode} | celda=${sw_cell} modelo=${sw_model} split=${sw_split} max_fp=${sw_max_fp}"
  echo "Log: ${logfile}"

  {
    echo "--- sliding_window mode=${mode} ---"
    RUN_ID="${RUN_ID}" \
      CELL="${sw_cell}" \
      MODEL="${sw_model}" \
      SPLIT="${sw_split}" \
      MAX_FP="${sw_max_fp}" \
      "${SCRIPT_DIR}/run_sliding_window_eval.sh" "${mode}" "${EXTRA_ARGS[@]}"
    write_run_meta "sliding_window_done" >/dev/null
  } 2>&1 | tee -a "$logfile"
}

run_sliding_window_bg() {
  ensure_run_id
  local logfile pidfile
  logfile="$(logs_dir)/sliding_window.log"
  pidfile="$(logs_dir)/sliding_window.pid"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Sliding-window ya en curso (PID $(cat "$pidfile")). Log: ${logfile}" >&2
    exit 1
  fi

  log_banner "$logfile" "SLIDING WINDOW (background)"
  write_run_meta "sliding_window_start" >/dev/null

  local mode="${SLIDING_MODE:-full}"
  local sw_cell="${SW_CELL:-bin_full}"
  local sw_model="${SW_MODEL:-modelo_12}"
  local sw_split="${SW_SPLIT:-val}"
  local sw_max_fp="${SW_MAX_FP:-1}"
  local extra
  extra="$(extra_py_args)"

  nohup bash >> "${logfile}" 2>&1 <<SCRIPT &
set -euo pipefail
cd "${SCRIPT_DIR}"
export RUN_ID="${RUN_ID}"
export GUADIA_DATA_RESULT_ROOT="${GUADIA_DATA_RESULT_ROOT:-}"
export SLIDING_MODE="${mode}"
export SW_CELL="${sw_cell}"
export SW_MODEL="${sw_model}"
export SW_SPLIT="${sw_split}"
export SW_MAX_FP="${sw_max_fp}"
./run_sliding_window_eval.sh "${mode}" ${extra}
SCRIPT

  echo $! > "$pidfile"
  echo "[OK] Sliding-window en background PID $(cat "$pidfile")"
  echo "  tail -f ${logfile}"
}

run_sliding_window_battery() {
  ensure_run_id
  local logfile
  logfile="$(logs_dir)/sliding_window_battery.log"
  log_banner "$logfile" "SLIDING WINDOW BATTERY (modelo_12 + ensemble F1 + fp-only + mc)"

  if [[ -z "${GUADIA_DATA_RESULT_ROOT:-}" ]]; then
    echo "[!] Define GUADIA_DATA_RESULT_ROOT antes de lanzar la batería." >&2
    exit 1
  fi

  echo "Run: ${RUN_ID} | Log: ${logfile}"

  {
    echo "=== 1/3 bin_full — modelo_12 BINARIO argmax + ensemble F1 11|12 (both) + sweep ==="
    SLIDING_MODE=full-both SW_CELL=bin_full "${SCRIPT_DIR}/run_sliding_window_eval.sh" full-both "${EXTRA_ARGS[@]}"
    echo ""
    echo "=== 2/3 bin_full — fp-only modelo_12 + ensemble (both) + sweep ==="
    SLIDING_MODE=fp-only SW_CELL=bin_full "${SCRIPT_DIR}/run_sliding_window_eval.sh" fp-only "${EXTRA_ARGS[@]}"
    echo ""
    echo "=== 3/3 bin_full — ensemble conservador val_best_ensemble.json + sweep ==="
    SLIDING_MODE=ensemble-low-fp SW_CELL=bin_full "${SCRIPT_DIR}/run_sliding_window_eval.sh" ensemble-low-fp "${EXTRA_ARGS[@]}"
    if [[ "${SW_INCLUDE_MC:-0}" == "1" ]]; then
      echo ""
      echo "=== EXTRA mc_full — modelo_12 MULTICLASE (distinto checkpoint, opcional) ==="
      SLIDING_MODE=multiclass SW_CELL=mc_full "${SCRIPT_DIR}/run_sliding_window_eval.sh" multiclass "${EXTRA_ARGS[@]}"
    fi
    write_run_meta "sliding_window_battery_done" >/dev/null
    echo ""
    echo "[FIN] Batería anti-FP completada RUN_ID=${RUN_ID}"
  } 2>&1 | tee -a "$logfile"
}

run_sliding_window_battery_bg() {
  ensure_run_id
  local logfile pidfile
  logfile="$(logs_dir)/sliding_window_battery.log"
  pidfile="$(logs_dir)/sliding_window_battery.pid"
  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Batería sliding-window ya en curso (PID $(cat "$pidfile"))." >&2
    exit 1
  fi
  write_run_meta "sliding_window_battery_start" >/dev/null
  nohup bash >> "${logfile}" 2>&1 <<SCRIPT &
set -euo pipefail
cd "${SCRIPT_DIR}"
export RUN_ID="${RUN_ID}"
export GUADIA_DATA_RESULT_ROOT="${GUADIA_DATA_RESULT_ROOT:-}"
./run_campaign.sh sliding-window-battery --run-id "${RUN_ID}"
SCRIPT
  echo $! > "$pidfile"
  echo "[OK] Batería anti-FP en background PID $(cat "$pidfile")"
  echo "  tail -f ${logfile}"
}

show_status() {
  ensure_run_id
  echo "RUN_ID actual: ${RUN_ID}"
  echo "Carpeta: $(run_root)"
  if [[ -f "$(run_root)/run_meta.json" ]]; then
    echo "Meta: $(run_root)/run_meta.json"
  fi
  for name in preflight train eval pipeline sliding_window; do
    local pf
    pf="$(logs_dir)/${name}.pid"
    if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
      echo "  ${name}: RUNNING pid=$(cat "$pf") log=$(logs_dir)/${name}.log"
    elif [[ -f "$(logs_dir)/${name}.log" ]]; then
      echo "  ${name}: log=$(logs_dir)/${name}.log"
    fi
  done
}

parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
strip_run_id_from_extra

case "$CMD" in
  new-run)
    RUN_ID="$("$PY" -c "from campaign_paths import new_run_id; print(new_run_id())")"
    mkdir -p "${SCRIPT_DIR}/artifacts/runs"
    echo "$RUN_ID" > "$CURRENT_RUN_FILE"
    mkdir -p "$(run_root)/logs"
    write_run_meta "created" >/dev/null
    echo "Nuevo RUN_ID: ${RUN_ID}"
    echo "Carpeta: $(run_root)"
    echo "Siguiente: ./run_campaign.sh preflight"
    ;;
  status)
    if [[ -z "$RUN_ID" ]] && [[ ! -f "$CURRENT_RUN_FILE" ]]; then
      echo "No hay RUN_ID activo. Usa: ./run_campaign.sh new-run"
      exit 1
    fi
    show_status
    ;;
  check)
    exec "$PY" validate_campaign.py "${EXTRA_ARGS[@]}"
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
  train-fg)
    run_train_fg
    ;;
  eval|eval-bg)
    run_eval_bg
    ;;
  eval-fg)
    run_eval_fg
    ;;
  export-ensemble-fp)
    ensure_run_id
    exec "$PY" export_ensemble_fp.py --split val --outcomes errors $(run_args) "${EXTRA_ARGS[@]}"
    ;;
  sliding-window|sliding-window-fg)
    SLIDING_MODE="${SLIDING_MODE:-full}"
    run_sliding_window_fg
    ;;
  sliding-window-fp)
    SLIDING_MODE=fp-only
    run_sliding_window_fg
    ;;
  sliding-window-mc|sliding-window-multiclass)
    SLIDING_MODE=multiclass
    run_sliding_window_fg
    ;;
  sliding-window-strict)
    SLIDING_MODE=strict
    run_sliding_window_fg
    ;;
  sliding-window-ensemble-49-57)
    SLIDING_MODE=ensemble-49-57
    SW_CELL=bin_filtered
    run_sliding_window_fg
    ;;
  sliding-window-ensemble-49-57-bg)
    SLIDING_MODE=ensemble-49-57
    SW_CELL=bin_filtered
    run_sliding_window_bg
    ;;
  sliding-window-bg)
    SLIDING_MODE="${SLIDING_MODE:-full}"
    run_sliding_window_bg
    ;;
  sliding-window-battery)
    run_sliding_window_battery
    ;;
  sliding-window-battery-bg)
    run_sliding_window_battery_bg
    ;;
  summary)
    ensure_run_id
    exec "$PY" summarize_campaign.py $(run_args) "${EXTRA_ARGS[@]}"
    ;;
  all)
    run_preflight
    run_train_fg
    run_eval_fg
    ;;
  all-bg|nohup|pipeline-bg)
    run_all_bg
    ;;
  help|*)
    cat <<'EOF'
Uso: ./run_campaign.sh <comando> [args Python...]

Cada RUN_ID aísla artefactos en artifacts/runs/<RUN_ID>/:
  plans/ models/ reports/ logs/ run_meta.json

Variables:
  RUN_ID=campaign_20260621_120000   (opcional; si no, usa .current_run o auto-crea)

Comandos:
  new-run          Crea RUN_ID nuevo y lo deja como run activo
  status           Muestra RUN_ID, PIDs y logs
  check            Validación sin planes
  preflight        Preflight + logs/preflight.log (foreground)
  train            Entrena en BACKGROUND → logs/train.log
  train-fg         Entrena en primer plano (con log)
  eval             Eval + summary en BACKGROUND (espera train si sigue vivo)
  eval-fg          Eval en primer plano
  summary          CSV maestro del RUN_ID activo
  export-ensemble-fp  Re-exporta FP/FN del mejor ensemble (lee best_ensemble.json)
  sliding-window      Ventanas 3s + filtros anti-FP (foreground, modo full)
  sliding-window-fp   Solo clips FP/FN del CSV de errores modelo_12
  sliding-window-mc   Multiclase mc_full + veto 6→3/4/5
  sliding-window-strict  Filtro conservador + barrido
  sliding-window-ensemble-49-57  49|57 mean @0.86 bin_filtered + SW + barrido
  sliding-window-ensemble-49-57-bg  Igual en BACKGROUND
  sliding-window-bg   Igual que sliding-window en BACKGROUND
  sliding-window-battery  Batería bin_full: modelo_12 + ensemble F1 + fp-only (sin mc salvo SW_INCLUDE_MC=1)
  sliding-window-battery-bg  Batería en BACKGROUND (recomendado)

Variables extra (sliding-window):
  SW_CELL=bin_full   SW_MODEL=modelo_12   SW_SPLIT=val   SW_MAX_FP=1
  SLIDING_MODE=full|fp-only|multiclass|strict

Flujo anti-FP (tras eval de campaña yolo26m):
  export GUADIA_DATA_RESULT_ROOT=/home/angel/.../data_yolo26m/data_result
  export RUN_ID=campaign_20260714_164642
  ./run_campaign.sh sliding-window-battery-bg
  tail -f artifacts/runs/$RUN_ID/logs/sliding_window_battery.log

Flujo entrenamiento (60 experimentos, SSH):
  ./run_campaign.sh new-run
  ./run_campaign.sh preflight
  ./run_campaign.sh train          # background
  tail -f artifacts/runs/<RUN_ID>/logs/train.log
  ./run_campaign.sh eval           # background tras train
  ./run_campaign.sh all-bg         # o pipeline completo en background

Pasar --run-id explícito en cualquier fase para reutilizar carpeta.
EOF
    ;;
esac
