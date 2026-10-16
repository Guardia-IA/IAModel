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
  preflight            Inventario + caché 12Hz (añade --p0 para cola baseline)
  train                Ejecuta jobs/queue.json (P0 tras preflight --p0)
  p0-summary           Agrega eval P0 → reports/p0_baseline.json
  summary              Muestra tablas/tiempos del último preflight (sin recomputar)
  p0-pipeline          new-run + probe + preflight --p0 + train + p0-summary
  p1-smoke             preflight --p1-smoke + train (2 épocas, prueba train_v2)
  p1-status            Progreso de jobs P1 (época, val rec/FP)
  p1-summary           Tabla final tras train P1
  p1-eval              Cola eval sobre best.pt + train (paralelo)
  p1-eval-summary      Métricas eval clip-max
  debug-pipeline       new-run + probe + preflight --debug --p1 + train + eval + resúmenes
  matrix-pipeline      full+filtered, legacy eval + train v2 + eval + ranking ensembles
  debug-matrix         matrix-pipeline en modo debug (~50 clips)
  ensemble-leaderboard Mejor par #A+#B por pose_source (tras eval)
  pipeline             new-run + probe + preflight (sin train; Sprint 1)
  smoke-pipeline       new-run + probe + preflight --smoke-test + train

Variables:
  RUN_ID, CAMPAIGN2_MAX_PARALLEL, PYTHON

Ejemplo:
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
  p1-eval)
    parse_run_id_from_args "${EXTRA_ARGS[@]}" || true
    ensure_run_id
    "$PY" prepare_p1_eval.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
    "$PY" train_campaign2_parallel.py --run-id "$RUN_ID" "${EXTRA_ARGS[@]}"
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
    "$0" p1-eval --run-id "$RUN_ID"
    "$0" p1-eval-summary --run-id "$RUN_ID"
    "$0" ensemble-leaderboard --run-id "$RUN_ID"
    echo "[matrix-pipeline] Ver reports/ensemble_leaderboard.json"
    ;;
  debug-matrix)
    "$0" new-run
    RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
    export RUN_ID
    "$0" probe --run-id "$RUN_ID"
    "$0" preflight --run-id "$RUN_ID" --matrix --debug "${EXTRA_ARGS[@]}"
    "$0" train --run-id "$RUN_ID"
    "$0" p1-eval --run-id "$RUN_ID"
    "$0" ensemble-leaderboard --run-id "$RUN_ID"
    echo "[debug-matrix] OK — para producción: ./run_campaign2.sh matrix-pipeline --data-root ..."
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
