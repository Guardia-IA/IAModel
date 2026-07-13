#!/usr/bin/env bash
# Pipeline completo reducción FP (sin mass augment).
# Etapa 1: bin_filtered_hardened | Etapa 2: bin_verifier_234 | Heurísticas + ensemble conservador
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-fp_pipeline_v1}"
IMPROVE_CONFIG="${IMPROVE_CONFIG:-campaign_config_fp_pipeline.json}"
HN_CSV="${HN_CSV:-}"
SPLIT="${SPLIT:-val}"
STAGE1_CELL="${STAGE1_CELL:-bin_filtered_hardened}"
VERIFIER_CELL="${VERIFIER_CELL:-bin_verifier_234}"

CMD="${1:-help}"
shift || true

run_root() { echo "${SCRIPT_DIR}/artifacts/runs/${RUN_ID}"; }

preflight_args=(--config "$IMPROVE_CONFIG" --run-id "$RUN_ID")
run_args=(--config "$IMPROVE_CONFIG" --run-id "$RUN_ID")

if [[ -n "$HN_CSV" ]]; then
  preflight_args+=(--hard-negative-csv "$HN_CSV")
fi

case "$CMD" in
  preflight)
    "$PY" validate_campaign.py --config "$IMPROVE_CONFIG" --run-id "$RUN_ID" || exit 1
    exec "$PY" preflight_campaign.py --write-all "${preflight_args[@]}" "$@"
    ;;
  train)
    "$PY" validate_campaign.py --config "$IMPROVE_CONFIG" --run-id "$RUN_ID" --require-plans || exit 1
    exec "$PY" train_campaign.py --all --resume "${run_args[@]}" "$@"
    ;;
  eval)
    exec "$PY" evaluate_campaign.py --all --export-fp-videos "${run_args[@]}" "$@"
    ;;
  eval-pipeline)
    # Eval modelos + ensemble + verificador + heurísticas → F1/FN finales
    extra=(--run-id "$RUN_ID" --split "$SPLIT" --stage1-cell "$STAGE1_CELL" --verifier-cell "$VERIFIER_CELL")
    if [[ "${SKIP_MODEL_EVAL:-}" == "1" ]]; then
      extra+=(--skip-model-eval)
    fi
    exec "$PY" evaluate_fp_pipeline.py "${extra[@]}" "$@"
    ;;
  export-stage1)
    # Mejor ensemble etapa 1 (ajusta modelos/umbral tras eval)
    exec "$PY" export_ensemble_fp.py --run-id "$RUN_ID" --cell "$STAGE1_CELL" --split "$SPLIT" \
      --outcomes errors "$@"
    ;;
  merge-verifier)
    ENS_CSV="${2:-}"
    if [[ -z "$ENS_CSV" ]]; then
      echo "Uso: RUN_ID=$RUN_ID $0 merge-verifier /ruta/ensemble.csv"
      exit 1
    fi
    exec "$PY" merge_verifier_probs.py --run-id "$RUN_ID" --split "$SPLIT" \
      --ensemble-csv "$ENS_CSV" "$@"
    ;;
  heuristics-batch)
    exec "$PY" pose_robbery_heuristics.py batch --cell "$STAGE1_CELL" --split "$SPLIT" \
      --run-id "$RUN_ID" "$@"
    ;;
  pipeline-sweep)
    ENS_CSV="${2:-}"
    if [[ -z "$ENS_CSV" ]]; then
      echo "Uso: RUN_ID=$RUN_ID $0 pipeline-sweep /ruta/ensemble_with_verifier.csv [--rule and]"
      exit 1
    fi
    shift || true
    exec "$PY" pose_robbery_heuristics.py pipeline --ensemble-csv "$ENS_CSV" "$@"
    ;;
  all)
    "$0" preflight
    "$0" train
    "$0" eval-pipeline
    ;;
  all-bg|nohup)
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG="$(run_root)/logs/fp_pipeline_${TS}.log"
    mkdir -p "$(run_root)/logs"
    nohup "$0" all "$@" >> "$LOG" 2>&1 &
    echo $! > "$(run_root)/logs/fp_pipeline_${TS}.pid"
    echo "PID $(cat "$(run_root)/logs/fp_pipeline_${TS}.pid") — tail -f $LOG"
    ;;
  help|*)
    cat <<EOF
Uso: ./run_fp_pipeline.sh <comando>

Variables:
  RUN_ID=fp_pipeline_v1
  HN_CSV=/ruta/fp_val.csv     Hard negatives (export_ensemble_fp del run anterior)
  SPLIT=val
  IMPROVE_CONFIG=campaign_config_fp_pipeline.json

Comandos:
  preflight          Planes train (2 celdas, sin mass augment)
  train              Entrena bin_filtered_hardened + bin_verifier_234
  eval               Solo modelos + ensemble grid (sin heurísticas)
  eval-pipeline      Eval completa: modelos + ensemble + verificador + H1–H12
  export-stage1      CSV ensemble etapa 1 (pasar --models --rule --threshold)
  merge-verifier     Añade p_verifier al CSV
  heuristics-batch   Features H1–H12 en todo el split
  pipeline-sweep     Barrido etapa1+2+heurísticas+temporal
  all                preflight → train → eval-pipeline
  all-bg             Igual en background

Ejemplo completo:
  export RUN_ID=fp_pipeline_v1
  export HN_CSV=/ruta/val_ensemble_fp_mean_modelo_36+modelo_40_t0.68.csv
  ./run_fp_pipeline.sh preflight
  ./run_fp_pipeline.sh train
  ./run_fp_pipeline.sh eval-pipeline

  # Si ya corriste eval (solo modelos), reutilízalo:
  SKIP_MODEL_EVAL=1 ./run_fp_pipeline.sh eval-pipeline

  # Pasos manuales (equivalente a eval-pipeline):
  ./run_fp_pipeline.sh export-stage1 --models modelo_36 modelo_40 --rule mean --threshold 0.68 --outcomes all
  ./run_fp_pipeline.sh merge-verifier artifacts/runs/\$RUN_ID/reports/bin_filtered_hardened/val_ensemble_fp_mean_modelo_36+modelo_40_t0.68.csv
  ./run_fp_pipeline.sh pipeline-sweep artifacts/runs/\$RUN_ID/reports/bin_filtered_hardened/val_ensemble_fp_mean_modelo_36+modelo_40_t0.68_with_verifier.csv --rule mean
EOF
    ;;
esac
