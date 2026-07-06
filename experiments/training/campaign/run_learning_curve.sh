#!/usr/bin/env bash
# Curva de aprendizaje: preflight → train → eval → export FP/FN → comparativa
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CMD="${1:-help}"
shift || true

PY="${PYTHON:-python3}"
LOG_DIR="${SCRIPT_DIR}/artifacts/logs"
mkdir -p "$LOG_DIR"

# Resuelve specs (3000 max …) → enteros usando split maestro / manifiesto
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

# Argumentos comunes (se pasan a los scripts Python)
COMMON_ARGS=("$@")

lc_preflight() {
  "$PY" validate_campaign.py "${COMMON_ARGS[@]}" || exit 1
  exec "$PY" preflight_campaign.py --learning-curve --write-all "${COMMON_ARGS[@]}"
}

lc_train() {
  # Valida el primer run lc_* (todos comparten misma celda/plan structure)
  local sizes=()
  local cell=""
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --train-sizes)
        shift
        while [[ $# -gt 0 && "$1" != --* ]]; do sizes+=("$1"); shift; done
        ;;
      --cells)
        shift; cell="$1"; shift
        ;;
      *)
        extra+=("$1"); shift
        ;;
    esac
  done
  if [[ ${#sizes[@]} -eq 0 ]]; then sizes=(3000 6500 max); fi
  if [[ -z "$cell" ]]; then cell="bin_full_hardened"; fi
  read -ra resolved <<< "$(lc_resolve_sizes "$cell" "${sizes[@]}")"
  "$PY" validate_campaign.py --require-plans --run-id "lc_${resolved[0]}" --cells "$cell" "${extra[@]}" || exit 1
  exec "$PY" train_campaign.py --learning-curve --resume --train-sizes "${sizes[@]}" --cells "$cell" "${extra[@]}"
}

lc_eval() {
  exec "$PY" evaluate_campaign.py --learning-curve --export-fp-videos "${COMMON_ARGS[@]}"
}

lc_export_errors() {
  local sizes=()
  local cells=()
  local split="val"
  local config_arg=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --train-sizes)
        shift
        while [[ $# -gt 0 && "$1" != --* ]]; do sizes+=("$1"); shift; done
        ;;
      --cells)
        shift
        while [[ $# -gt 0 && "$1" != --* ]]; do cells+=("$1"); shift; done
        ;;
      --split)
        shift
        split="$1"
        shift
        ;;
      --config)
        shift
        config_arg=(--config "$1")
        shift
        ;;
      *)
        shift
        ;;
    esac
  done

  if [[ ${#cells[@]} -eq 0 ]]; then
    read -ra cells <<< "$("$PY" -c "
import sys
sys.path.insert(0, '.')
from learning_curve_utils import load_learning_curve_cell_ids
print(' '.join(load_learning_curve_cell_ids()))
")"
  fi

  if [[ ${#sizes[@]} -eq 0 ]]; then
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
          "${config_arg[@]}" \
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
    exec "$PY" validate_campaign.py "${COMMON_ARGS[@]}"
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
    lc_export_errors "${COMMON_ARGS[@]}"
    ;;
  summary)
    exec "$PY" summarize_learning_curve.py "${COMMON_ARGS[@]}"
    ;;
  all)
    "$PY" validate_campaign.py "${COMMON_ARGS[@]}" || exit 1
    "$PY" preflight_campaign.py --learning-curve --write-all "${COMMON_ARGS[@]}"
    "$PY" train_campaign.py --learning-curve --resume "${COMMON_ARGS[@]}"
    "$PY" evaluate_campaign.py --learning-curve --export-fp-videos "${COMMON_ARGS[@]}"
    lc_export_errors "${COMMON_ARGS[@]}"
    "$PY" summarize_learning_curve.py "${COMMON_ARGS[@]}"
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
    echo "Monitorizar: tail -f ${LOG}"
    ;;
  help|*)
    cat <<'EOF'
Uso: ./run_learning_curve.sh <comando> [args]

Curva de aprendizaje: varios entrenamientos con distinto tamaño de train
(val/test fijos). Por defecto 4 celdas: mc_full, mc_filtered, bin_full_hardened, bin_filtered_hardened.
Cada tamaño → artifacts/runs/lc_<N>/ (todas las celdas comparten run_id).

Comandos:
  check           Validación previa (sin exigir planes)
  preflight       Plan maestro (_lc_master) + planes lc_<N>
  train           Entrena cada lc_<N> (--resume)
  eval            Evalúa val/test por tamaño
  export-errors   CSV FP+FN ensemble (MEAN 06+14 @ 0.68 por defecto)
  summary         Tabla comparativa + FP por categoría + listas FN/FP
  all             check → preflight → train → eval → export-errors → summary
  all-bg          Igual que all en background (nohup)

Ejemplo (4 celdas × 3000 / 6500 / max):
  ./run_learning_curve.sh all --train-sizes 3000 6500 max

  # Solo binario full
  ./run_learning_curve.sh preflight --cells bin_full_hardened --train-sizes 3000 max

Flags Python (preflight/train/eval):
  --train-sizes N1 N2 max     'max' = todos los clips train del split maestro
  --cells bin_full_hardened
  --config campaign_config.json
  --data-root /ruta/data_result

Artefactos:
  artifacts/runs/_lc_master/plans/<cell>/     split maestro
  artifacts/runs/lc_3000/ ... lc_<N>/         train/eval por tamaño (N real, p. ej. 9867)
  artifacts/reports/_master/learning_curve/   comparativa final

Config opcional en campaign_config.json → bloque "learning_curve":
  cells, train_sizes, binary_experiment_ids, multiclass_experiment_ids, ensemble
EOF
    ;;
esac
