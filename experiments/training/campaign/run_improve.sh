#!/usr/bin/env bash
# Runs de mejora post-campaña — artefactos aislados en artifacts/runs/<RUN_ID>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-python3}"
IMPROVE_CONFIG="${IMPROVE_CONFIG:-campaign_config_improve.json}"
RUN_ID="${RUN_ID:-improve_v1}"
HN_CSV="${HN_CSV:-}"
PROFILE="${PROFILE:-}"
UNIFORM_OPS="${UNIFORM_OPS:-}"

CMD="${1:-help}"
shift || true

common_args=(
  --config "$IMPROVE_CONFIG"
  --run-id "$RUN_ID"
)

if [[ -n "$HN_CSV" ]]; then
  common_args+=(--hard-negative-csv "$HN_CSV")
fi
if [[ -n "$PROFILE" ]]; then
  common_args+=(--improve-profile "$PROFILE")
fi
if [[ -n "$UNIFORM_OPS" ]]; then
  common_args+=(--uniform-ops-per-clip "$UNIFORM_OPS")
fi

case "$CMD" in
  check)
    exec "$PY" validate_campaign.py "${common_args[@]}" "$@"
    ;;
  preflight)
    "$PY" validate_campaign.py "${common_args[@]}" "$@" || exit 1
    exec "$PY" preflight_campaign.py --write-all "${common_args[@]}" "$@"
    ;;
  train)
    "$PY" validate_campaign.py --require-plans "${common_args[@]}" "$@" || exit 1
    exec "$PY" train_campaign.py --all --resume "${common_args[@]}" "$@"
    ;;
  eval)
    exec "$PY" evaluate_campaign.py --all --export-fp-videos "${common_args[@]}" "$@"
    ;;
  summary)
    exec "$PY" summarize_campaign.py "${common_args[@]}" "$@"
    ;;
  all)
    "$PY" validate_campaign.py "${common_args[@]}" "$@" || exit 1
    "$PY" preflight_campaign.py --write-all "${common_args[@]}" "$@"
    "$PY" validate_campaign.py --require-plans "${common_args[@]}" "$@" || exit 1
    "$PY" train_campaign.py --all --resume "${common_args[@]}" "$@"
    "$PY" evaluate_campaign.py --all --export-fp-videos "${common_args[@]}" "$@"
    "$PY" summarize_campaign.py "${common_args[@]}" "$@"
    ;;
  all-bg|nohup)
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG="${SCRIPT_DIR}/artifacts/runs/${RUN_ID}/logs/improve_${TS}.log"
    mkdir -p "$(dirname "$LOG")"
    echo "Lanzando improve run en background..."
    echo "  RUN_ID=${RUN_ID}"
    echo "  Log: ${LOG}"
    nohup "$0" all "$@" >> "${LOG}" 2>&1 &
    echo $! > "${SCRIPT_DIR}/artifacts/runs/${RUN_ID}/logs/improve_${TS}.pid"
    echo "  PID: $(cat "${SCRIPT_DIR}/artifacts/runs/${RUN_ID}/logs/improve_${TS}.pid")"
    echo "  tail -f ${LOG}"
    ;;
  help|*)
    cat <<EOF
Uso: ./run_improve.sh <comando> [args extra para los scripts Python]

Variables de entorno (opcionales):
  RUN_ID=improve_v1          Carpeta artifacts/runs/<RUN_ID>/ (no machaca campaña base)
  HN_CSV=/ruta/fp_val.csv    CSV de FP (export_ensemble_fp.py) → hard negatives + boost augment
  PROFILE=fp_hardened_hn     Override improve_profile del config
  UNIFORM_OPS=4              Experimento augment uniforme ×N por categoría
  IMPROVE_CONFIG=...         Default: campaign_config_improve.json

Comandos: check | preflight | train | eval | summary | all | all-bg

Ejemplo típico (hard negatives desde ensemble baseline):
  export RUN_ID=hn_from_ensemble_v1
  export HN_CSV=artifacts/reports/bin_full_hardened/val_fp_manifest.csv
  export PROFILE=fp_hardened_hn
  ./run_improve.sh preflight
  ./run_improve.sh train
  ./run_improve.sh eval
  ./run_improve.sh summary

Experimento augment uniforme ×4 (A/B):
  RUN_ID=uniform4_ab UNIFORM_OPS=4 PROFILE=fp_hardened_uniform4 ./run_improve.sh all-bg

Artefactos baseline intactos en artifacts/models/, artifacts/plans/, etc.
Mejoras en artifacts/runs/<RUN_ID>/models/, plans/, reports/, ...
EOF
    ;;
esac
