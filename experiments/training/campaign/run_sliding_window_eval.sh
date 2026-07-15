#!/usr/bin/env bash
# Batería anti-FP: ventanas 3s + filtros temporales/cinemáticos
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TRAINING_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"

PY="${PYTHON:-python3}"
CURRENT_RUN_FILE="${SCRIPT_DIR}/artifacts/runs/.current_run"

if [[ -z "${RUN_ID:-}" ]] && [[ -f "$CURRENT_RUN_FILE" ]]; then
  RUN_ID="$(tr -d '[:space:]' < "$CURRENT_RUN_FILE")"
fi
RUN_ID="${RUN_ID:-campaign_20260714_164642}"

SPLIT="${SW_SPLIT:-val}"
MODEL="${SW_MODEL:-modelo_12}"
CELL="${SW_CELL:-bin_full}"
MAX_FP="${SW_MAX_FP:-1}"

CMD="${1:-full-both}"
shift || true

common_args=(
  --run-id "$RUN_ID"
  --cell "$CELL"
  --model "$MODEL"
  --split "$SPLIT"
  --sweep
  --max-fp-target "$MAX_FP"
)

case "$CMD" in
  full)
    exec "$PY" sliding_window_eval.py "${common_args[@]}" --predictor single "$@"
    ;;
  full-both|both)
    # modelo_12 + mejor ensemble F1 del grid (típ. 11|12 mean @ 0.50)
    exec "$PY" sliding_window_eval.py "${common_args[@]}" --predictor both "$@"
    ;;
  ensemble|ensemble-f1)
    exec "$PY" sliding_window_eval.py \
      --run-id "$RUN_ID" --cell "$CELL" --split "$SPLIT" \
      --predictor ensemble --ensemble-source best_f1 \
      --sweep --max-fp-target "$MAX_FP" "$@"
    ;;
  ensemble-low-fp)
    # val_best_ensemble.json — conservador (6 FP, muchos FN)
    exec "$PY" sliding_window_eval.py \
      --run-id "$RUN_ID" --cell "$CELL" --split "$SPLIT" \
      --predictor ensemble --ensemble-source best_low_fp \
      --sweep --max-fp-target "$MAX_FP" "$@"
    ;;
  ensemble-49-57|ensemble-49|49-57)
    # 49|57 mean @ 0.86 (bin_filtered) — 6 FP en campaña, F1 ~77.6%
    exec "$PY" sliding_window_eval.py \
      --run-id "$RUN_ID" --cell bin_filtered --split "$SPLIT" \
      --predictor ensemble \
      --ensemble-models modelo_49 modelo_57 \
      --ensemble-rule mean --ensemble-threshold 0.86 \
      --sweep --max-fp-target "$MAX_FP" "$@"
    ;;
  fp-only)
    ERR_CSV="${2:-artifacts/runs/${RUN_ID}/reports/${CELL}/${SPLIT}_errors_${MODEL}.csv}"
    if [[ ! -f "$ERR_CSV" ]]; then
      echo "ERROR: no existe $ERR_CSV" >&2
      exit 1
    fi
    shift || true
    exec "$PY" sliding_window_eval.py "${common_args[@]}" --predictor both --errors-csv "$ERR_CSV" "$@"
    ;;
  multiclass)
    exec "$PY" sliding_window_eval.py \
      --run-id "$RUN_ID" --cell mc_full --model "$MODEL" --split "$SPLIT" \
      --predictor single --sweep --max-fp-target "$MAX_FP" "$@"
    ;;
  strict)
    exec "$PY" sliding_window_eval.py \
      --run-id "$RUN_ID" --cell "$CELL" --model "$MODEL" --split "$SPLIT" \
      --predictor both \
      --min-consecutive-windows 3 --min-s-kin 0.50 --require-conceal \
      --p-window-threshold 0.55 --full-clip-threshold 0.55 \
      --sweep --max-fp-target "$MAX_FP" "$@"
    ;;
  help|*)
    cat <<EOF
Uso: GUADIA_DATA_RESULT_ROOT=... RUN_ID=... $0 <comando>

Comandos:
  full-both     modelo_12 + mejor ensemble F1 + barrido (default batería)
  full          Solo modelo_12
  ensemble-f1   Solo mejor ensemble F1 (grid)
  ensemble-low-fp  Ensemble auto val_best_ensemble.json (6 FP)
  ensemble-49-57  49|57 mean @ 0.86 bin_filtered (6 FP campaña)
  fp-only       FP/FN CSV con single+ensemble
  multiclass    mc_full modelo_12
  strict        both + filtros conservadores

Defaults: RUN_ID=$RUN_ID CELL=$CELL MODEL=$MODEL
EOF
    ;;
esac
