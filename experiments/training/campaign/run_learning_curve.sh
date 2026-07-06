#!/usr/bin/env bash
# Curva de aprendizaje: preflight → train → eval → export FP/FN → comparativa
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TRAINING_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"

CMD="${1:-help}"
shift || true

PY="${PYTHON:-python3}"
LOG_DIR="${SCRIPT_DIR}/artifacts/logs"
mkdir -p "$LOG_DIR"

COMMON_ARGS=("$@")

# Parse flags: validate_campaign solo entiende subset; el resto va a scripts LC
TRAIN_SIZES=()
LC_CELLS=()
CONFIG_ARG=()
DATA_ROOT_ARG=()
STRICT_ARG=()
SKIP_SMOKE_ARG=()
SPLIT_ARG=()

parse_common_args() {
  local i=0
  while [[ $i -lt ${#COMMON_ARGS[@]} ]]; do
    local a="${COMMON_ARGS[$i]}"
    case "$a" in
      --train-sizes)
        i=$((i + 1))
        while [[ $i -lt ${#COMMON_ARGS[@]} && "${COMMON_ARGS[$i]}" != --* ]]; do
          TRAIN_SIZES+=("${COMMON_ARGS[$i]}")
          i=$((i + 1))
        done
        ;;
      --cells)
        i=$((i + 1))
        while [[ $i -lt ${#COMMON_ARGS[@]} && "${COMMON_ARGS[$i]}" != --* ]]; do
          LC_CELLS+=("${COMMON_ARGS[$i]}")
          i=$((i + 1))
        done
        ;;
      --config)
        CONFIG_ARG=(--config "${COMMON_ARGS[$((i + 1))]}")
        i=$((i + 2))
        ;;
      --data-root)
        DATA_ROOT_ARG=(--data-root "${COMMON_ARGS[$((i + 1))]}")
        i=$((i + 2))
        ;;
      --strict)
        STRICT_ARG=(--strict)
        i=$((i + 1))
        ;;
      --skip-smoke)
        SKIP_SMOKE_ARG=(--skip-smoke)
        i=$((i + 1))
        ;;
      --split)
        SPLIT_ARG=(--split "${COMMON_ARGS[$((i + 1))]}")
        i=$((i + 2))
        ;;
      *)
        i=$((i + 1))
        ;;
    esac
  done
}

validate_args() {
  local extra=("$@")
  local args=("${CONFIG_ARG[@]}" "${DATA_ROOT_ARG[@]}" "${STRICT_ARG[@]}" "${SKIP_SMOKE_ARG[@]}")
  if ((${#LC_CELLS[@]})); then
    args+=(--cells "${LC_CELLS[@]}")
  fi
  args+=("${extra[@]}")
  printf '%s\n' "${args[@]}"
}

lc_py_args() {
  local args=("${CONFIG_ARG[@]}" "${DATA_ROOT_ARG[@]}")
  if ((${#TRAIN_SIZES[@]})); then
    args+=(--train-sizes "${TRAIN_SIZES[@]}")
  fi
  if ((${#LC_CELLS[@]})); then
    args+=(--cells "${LC_CELLS[@]}")
  fi
  printf '%s\n' "${args[@]}"
}

summary_args() {
  local args=("${CONFIG_ARG[@]}" "${SPLIT_ARG[@]}")
  if ((${#TRAIN_SIZES[@]})); then
    args+=(--train-sizes "${TRAIN_SIZES[@]}")
  fi
  if ((${#LC_CELLS[@]})); then
    args+=(--cells "${LC_CELLS[@]}")
  fi
  printf '%s\n' "${args[@]}"
}

parse_common_args

# Resuelve specs (3500 max …) → enteros usando split maestro / manifiesto
lc_resolve_sizes() {
  local cell="$1"
  shift
  "$PY" -c "
import sys
sys.path.insert(0, '.')
from campaign_paths import load_merged_campaign_config
from learning_curve_utils import get_learning_curve_train_sizes
config = load_merged_campaign_config(None)
specs = sys.argv[1:] if len(sys.argv) > 1 else None
sizes = get_learning_curve_train_sizes(
    cli_sizes=specs,
    config=config,
    cell_id=sys.argv[0],
)
print(' '.join(str(n) for n in sizes))
" "$cell" "$@"
}

lc_preflight() {
  mapfile -t _va < <(validate_args)
  mapfile -t _lc < <(lc_py_args)
  "$PY" validate_campaign.py "${_va[@]}" || exit 1
  exec "$PY" preflight_campaign.py --learning-curve --write-all "${_lc[@]}"
}

lc_train() {
  mapfile -t _lc < <(lc_py_args)
  local sizes=("${TRAIN_SIZES[@]}")
  local cells=("${LC_CELLS[@]}")
  if ((${#sizes[@]} == 0)); then
    sizes=(3500 6500 max)
  fi
  if ((${#cells[@]} == 0)); then
    cells=(bin_full_hardened)
  fi
  read -ra resolved <<< "$(lc_resolve_sizes "${cells[0]}" "${sizes[@]}")"
  mapfile -t _va < <(validate_args --require-plans --run-id "lc_${resolved[0]}" --cells "${cells[@]}")
  "$PY" validate_campaign.py "${_va[@]}" || exit 1
  exec "$PY" train_campaign.py --learning-curve --resume --train-sizes "${sizes[@]}" --cells "${cells[@]}" \
    "${CONFIG_ARG[@]}" "${DATA_ROOT_ARG[@]}"
}

lc_eval() {
  mapfile -t _lc < <(lc_py_args)
  exec "$PY" evaluate_campaign.py --learning-curve --export-fp-videos "${_lc[@]}"
}

lc_export_errors() {
  local sizes=("${TRAIN_SIZES[@]}")
  local cells=("${LC_CELLS[@]}")
  local split="val"
  if ((${#SPLIT_ARG[@]})); then
    split="${SPLIT_ARG[1]}"
  fi

  if ((${#cells[@]} == 0)); then
    read -ra cells <<< "$("$PY" -c "
import sys
sys.path.insert(0, '.')
from learning_curve_utils import load_learning_curve_cell_ids
print(' '.join(load_learning_curve_cell_ids()))
")"
  fi

  if ((${#sizes[@]} == 0)); then
    read -ra resolved <<< "$(lc_resolve_sizes "${cells[0]}")"
  else
    read -ra resolved <<< "$(lc_resolve_sizes "${cells[0]}" "${sizes[@]}")"
  fi

  for cell in "${cells[@]}"; do
    for n in "${resolved[@]}"; do
      echo ""
      echo "=== Export errors — cell=$cell train_size=$n run_id=lc_$n ==="
      if [[ "$cell" == bin_* ]]; then
        "$PY" export_ensemble_fp.py \
          "${CONFIG_ARG[@]}" \
          --cell "$cell" \
          --split "$split" \
          --run-id "lc_${n}" \
          --outcomes errors \
          --export-videos
      else
        echo "  (multiclase: FP en reports/$cell/${split}_fp_manifest.csv tras eval)"
      fi
    done
  done
}

case "$CMD" in
  check)
    mapfile -t _va < <(validate_args)
    exec "$PY" validate_campaign.py "${_va[@]}"
    ;;
  preflight)
    lc_preflight
    ;;
  train)
    lc_train
    ;;
  eval)
    lc_eval
    ;;
  export-errors)
    lc_export_errors
    ;;
  summary)
    mapfile -t _sa < <(summary_args)
    exec "$PY" summarize_learning_curve.py "${_sa[@]}"
    ;;
  all)
    mapfile -t _va < <(validate_args)
    mapfile -t _lc < <(lc_py_args)
    mapfile -t _sa < <(summary_args)
    "$PY" validate_campaign.py "${_va[@]}" || exit 1
    "$PY" preflight_campaign.py --learning-curve --write-all "${_lc[@]}"
    "$PY" train_campaign.py --learning-curve --resume "${_lc[@]}"
    "$PY" evaluate_campaign.py --learning-curve --export-fp-videos "${_lc[@]}"
    lc_export_errors
    "$PY" summarize_learning_curve.py "${_sa[@]}"
    ;;
  all-bg|nohup)
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG="${LOG_DIR}/learning_curve_${TS}.log"
    PID_FILE="${LOG_DIR}/learning_curve_${TS}.pid"
    echo "Lanzando curva de aprendizaje en background..."
    echo "  Log:  ${LOG}"
    nohup "$0" all "${COMMON_ARGS[@]}" >> "${LOG}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "  PID:  $(cat "${PID_FILE}")"
    echo ""
    echo "Puedes cerrar SSH; el proceso sigue con nohup."
    echo "Monitorizar: tail -f ${LOG}"
    ;;
  help|*)
    cat <<'EOF'
Uso: ./run_learning_curve.sh <comando> [args]

Curva de aprendizaje: varios entrenamientos con distinto tamaño de train
(val/test fijos). Por defecto 4 celdas en config; puedes limitar con --cells.

Comandos:
  check           Validación previa (sin exigir planes)
  preflight       Plan maestro (_lc_master) + planes lc_<N>
  train           Entrena cada lc_<N> (--resume)
  eval            Evalúa val/test por tamaño
  export-errors   CSV FP+FN del mejor ensemble (tras eval)
  summary         Tabla comparativa + FP por categoría + listas FN/FP
  all             check → preflight → train → eval → export-errors → summary
  all-bg          Igual que all en background (nohup, apto para cerrar SSH)

Ejemplo (binario, 3500 / 6500 / max, 60 arquitecturas):
  ./run_learning_curve.sh all-bg --train-sizes 3500 6500 max --cells bin_full_hardened

Flags (preflight/train/eval/summary):
  --train-sizes 3500 6500 max   'max' = todos los clips train del split maestro
  --cells bin_full_hardened
  --config campaign_config.json
  --data-root /ruta/data_result

Artefactos:
  artifacts/runs/_lc_master/plans/<cell>/     split maestro
  artifacts/runs/lc_3500/ ... lc_<N>/         train/eval por tamaño
  artifacts/reports/_master/learning_curve/   comparativa final
  artifacts/logs/learning_curve_*.log         log de all-bg
EOF
    ;;
esac
