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

CMD="${1:-full}"
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
    exec "$PY" sliding_window_eval.py "${common_args[@]}" "$@"
    ;;
  fp-only)
    ERR_CSV="${2:-artifacts/runs/${RUN_ID}/reports/${CELL}/${SPLIT}_errors_${MODEL}.csv}"
    if [[ ! -f "$ERR_CSV" ]]; then
      echo "ERROR: no existe $ERR_CSV" >&2
      echo "Ejecuta antes: ./run_campaign.sh eval --run-id $RUN_ID --cells $CELL" >&2
      exit 1
    fi
    shift || true
    exec "$PY" sliding_window_eval.py "${common_args[@]}" --errors-csv "$ERR_CSV" "$@"
    ;;
  multiclass)
    exec "$PY" sliding_window_eval.py \
      --run-id "$RUN_ID" \
      --cell mc_full \
      --model "$MODEL" \
      --split "$SPLIT" \
      --sweep \
      --max-fp-target "$MAX_FP" \
      "$@"
    ;;
  strict)
    exec "$PY" sliding_window_eval.py \
      --run-id "$RUN_ID" \
      --cell "$CELL" \
      --model "$MODEL" \
      --split "$SPLIT" \
      --min-consecutive-windows 3 \
      --min-s-kin 0.50 \
      --require-conceal \
      --p-window-threshold 0.55 \
      --full-clip-threshold 0.55 \
      --sweep \
      --max-fp-target "$MAX_FP" \
      "$@"
    ;;
  help|*)
    cat <<EOF
Uso: GUADIA_DATA_RESULT_ROOT=... RUN_ID=... $0 <comando>

Comandos:
  full        Val completo + barrido anti-FP (default)
  fp-only     Solo clips del CSV val_errors_\${MODEL}.csv
  multiclass  mc_full + regla post-compra en ventanas
  strict      Política conservadora + barrido

Defaults:
  RUN_ID=$RUN_ID
  CELL=$CELL  MODEL=$MODEL  SPLIT=$SPLIT  MAX_FP=$MAX_FP
EOF
    ;;
esac
