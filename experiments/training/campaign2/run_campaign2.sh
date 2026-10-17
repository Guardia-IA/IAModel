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
  validate             Tests rápidos (~5 s, sin GPU ni datos)
  estimate-time        Estimación P1 GPU vs CPU (--data-root = dataset completo; --matrix = matrix-pipeline)
  probe [--run-id ID]  gpu_probe_campaign2.py
  preflight            Inventario + caché 12Hz (añade --p0 para cola baseline)
  train                Ejecuta jobs/queue.json (P0 tras preflight --p0)
  p0-summary           Agrega eval P0 → reports/p0_baseline.json
  summary              Muestra tablas/tiempos del último preflight (sin recomputar)
  p0-pipeline          new-run + probe + preflight --p0 + train + p0-summary
  p1-smoke             preflight --p1-smoke + train (2 épocas, prueba train_v2)
  p1-status            Snapshot de jobs (época, rec@FP, FP)
  p1-status --watch    Refresco en bucle en el mismo terminal (Ctrl+C)
  p1-summary           Tabla final tras train P1
  p1-calibrate         Umbral en split threshold_calib (T) + threshold.json
  p1-eval              Cola eval sobre best.pt + train (paralelo)
  p1-eval-summary      Métricas eval clip-max
  p1-seed-ensemble     Media prob entre semillas v2 (mismo job base)
  p1-promote           Copia mejor checkpoint a models/promoted/
  debug-gate           Comprueba RUN (robos en split, best.pt, etc.)
  debug-pipeline       new-run + probe + preflight --debug --p1 + train + eval + resúmenes
  matrix-pipeline      train v2 (matriz arq×semillas) + calibrate + eval + leaderboard
  debug-matrix         matrix-pipeline en debug (~10 clips, sin P0 legacy)
  launch-debug         alias: validate + debug-matrix (pasa --data-root)
  launch-train         alias: matrix-pipeline full dataset (~19k clips)
  ensemble-leaderboard Mejor par #A+#B por pose_source (tras eval)
  pipeline             new-run + probe + preflight (sin train; Sprint 1)
  smoke-pipeline       new-run + probe + preflight --smoke-test + train

Variables:
  RUN_ID, CAMPAIGN2_MAX_PARALLEL, PYTHON

Ejemplo:
  $0 launch-debug --data-root /media/8TB/.../data_clips/
  $0 launch-train --data-root /media/8TB/.../data_clips/
  $0 new-run
  $0 preflight --data-root /media/8TB/.../data_result
  export CAMPAIGN2_MODEL_41=/ruta/modelo_41.pt CAMPAIGN2_MODEL_48=/ruta/modelo_48.pt
  $0 preflight --p0 --data-root /media/8TB/.../data_result
  $0 probe && $0 train && $0 p0-summary
EOF
    ;;
  new-run)
    mkdir -p "$(dirname "$CURRENT_RUN_FILE")"
    RUN_ID="$("$PY" -c "from campaign2_paths import new_run_id; print(new_run_id())")"
    echo "$RUN_ID" > "$CURRENT_RUN_FILE"
    mkdir -p "${SCRIPT_DIR}/artifacts/runs/${RUN_ID}"
    echo "RUN_ID=${RUN_ID}"
    ;;
  validate)
    "$PY" validate_campaign2.py
    ;;
  estimate-time)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    if [[ -n "${RUN_ID:-}" ]]; then
      "$PY" estimate_run_time.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    else
      "$PY" estimate_run_time.py "${EXTRA_ARGS[@]}"
    fi
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
    echo "[pipeline] Sprint 1 termina aquí (sin train). Usa smoke-pipeline para probar scheduler."
    ;;
  summary)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" show_preflight_summary.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  p0-summary)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" p0_summarize.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  p0-pipeline)
    "$0" new-run
    RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
    export RUN_ID
    "$0" probe --run-id "$RUN_ID"
    "$0" preflight --run-id "$RUN_ID" --p0 "${EXTRA_ARGS[@]}"
    "$0" train --run-id "$RUN_ID"
    "$0" p0-summary --run-id "$RUN_ID"
    ;;
  p1-smoke)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$0" preflight --run-id "$RUN_ID" --p1-smoke --skip-cache "${EXTRA_ARGS[@]}"
    "$0" train --run-id "$RUN_ID"
    "$0" p1-summary --run-id "$RUN_ID"
    ;;
  p1-status)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" p1_status.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  p1-summary)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" p1_summarize.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  p1-calibrate)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" prepare_p1_calibrate.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    "$PY" train_campaign2_parallel.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  p1-eval)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" prepare_p1_eval.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    "$PY" train_campaign2_parallel.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  p1-seed-ensemble)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" p1_seed_ensemble.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  p1-eval-summary)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" p1_eval_summarize.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  debug-pipeline)
    "$0" new-run
    RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
    export RUN_ID
    "$0" probe --run-id "$RUN_ID"
    "$0" preflight --run-id "$RUN_ID" --debug --p1 "${EXTRA_ARGS[@]}"
    "$0" train --run-id "$RUN_ID"
    "$0" p1-summary --run-id "$RUN_ID"
    "$0" p1-eval --run-id "$RUN_ID"
    "$0" p1-eval-summary --run-id "$RUN_ID"
    echo "[debug-pipeline] OK — quita --debug y sube max_clips para campaña real."
    ;;
  matrix-pipeline)
    "$0" new-run
    RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
    export RUN_ID
    "$0" probe --run-id "$RUN_ID"
    "$0" preflight --run-id "$RUN_ID" --matrix "${EXTRA_ARGS[@]}"
    "$0" train --run-id "$RUN_ID"
    "$0" p1-summary --run-id "$RUN_ID"
    "$0" p0-summary --run-id "$RUN_ID" || true
    "$0" p1-calibrate --run-id "$RUN_ID"
    "$0" p1-eval --run-id "$RUN_ID"
    "$0" p1-eval-summary --run-id "$RUN_ID"
    "$0" p1-seed-ensemble --run-id "$RUN_ID" || true
    "$0" ensemble-leaderboard --run-id "$RUN_ID"
    echo "[matrix-pipeline] Ver reports/ensemble_leaderboard.json"
    ;;
  debug-matrix)
    "$0" validate || exit 1
    "$0" new-run
    RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
    export RUN_ID
    "$0" probe --run-id "$RUN_ID"
    "$0" preflight --run-id "$RUN_ID" --matrix --debug "${EXTRA_ARGS[@]}"
    "$0" train --run-id "$RUN_ID"
    "$0" p1-summary --run-id "$RUN_ID"
    "$0" p1-calibrate --run-id "$RUN_ID"
    "$0" p1-eval --run-id "$RUN_ID"
    "$0" p1-eval-summary --run-id "$RUN_ID"
    "$0" p1-seed-ensemble --run-id "$RUN_ID" || true
    "$0" ensemble-leaderboard --run-id "$RUN_ID"
    "$0" p1-promote --run-id "$RUN_ID" || true
    "$0" debug-gate --run-id "$RUN_ID"
    echo "[debug-matrix] OK — para producción: ./run_campaign2.sh launch-train --data-root ..."
    ;;
  launch-debug)
    DATA_OK=0
    for a in "${EXTRA_ARGS[@]}"; do
      [[ "$a" == --data-root || "$a" == --data-root=* ]] && DATA_OK=1
    done
    if [[ "$DATA_OK" -eq 0 ]]; then
      echo "Uso: $0 launch-debug --data-root /media/8TB/.../data_clips/" >&2
      exit 2
    fi
    "$0" debug-matrix "${EXTRA_ARGS[@]}"
    ;;
  launch-train)
    DATA_OK=0
    for a in "${EXTRA_ARGS[@]}"; do
      [[ "$a" == --data-root || "$a" == --data-root=* ]] && DATA_OK=1
    done
    if [[ "$DATA_OK" -eq 0 ]]; then
      echo "Uso: $0 launch-train --data-root /media/8TB/.../data_clips/" >&2
      exit 2
    fi
    "$0" matrix-pipeline "${EXTRA_ARGS[@]}"
    ;;
  debug-gate)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" debug_gate.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  p1-promote)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" p1_promote.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  ensemble-leaderboard)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" ensemble_leaderboard.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    ;;
  smoke-pipeline)
    "$0" new-run
    RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
    export RUN_ID
    "$0" probe --run-id "$RUN_ID"
    "$0" preflight --run-id "$RUN_ID" --smoke-test "${EXTRA_ARGS[@]}"
    "$0" train --run-id "$RUN_ID"
    ;;
  *)
    echo "Comando desconocido: $CMD" >&2
    exit 1
    ;;
esac
