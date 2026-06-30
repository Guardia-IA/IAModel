#!/usr/bin/env bash
# Orquestador de la campaña de experimentos (preflight → train → eval → summary)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CMD="${1:-help}"
shift || true

PY="${PYTHON:-python3}"
LOG_DIR="${SCRIPT_DIR}/artifacts/logs"
mkdir -p "$LOG_DIR"

run_check() {
  # Tras preflight, exige planes escritos
  local extra=("$@")
  "$PY" validate_campaign.py "${extra[@]}"
}

case "$CMD" in
  check)
    # Validación completa SIN exigir planes (útil antes del primer preflight)
    exec "$PY" validate_campaign.py "$@"
    ;;
  check-ready)
    # Validación + exige training_plan.json por celda (tras preflight --write-all)
    exec "$PY" validate_campaign.py --require-plans "$@"
    ;;
  preflight)
    "$PY" validate_campaign.py "$@" || exit 1
    exec "$PY" preflight_campaign.py --write-all "$@"
    ;;
  train)
    "$PY" validate_campaign.py --require-plans "$@" || exit 1
    exec "$PY" train_campaign.py --all --resume "$@"
    ;;
  eval)
    exec "$PY" evaluate_campaign.py --all --export-fp-videos "$@"
    ;;
  export-ensemble-fp)
    exec "$PY" export_ensemble_fp.py --split val --export-videos "$@"
    ;;
  summary)
    exec "$PY" summarize_campaign.py "$@"
    ;;
  all)
    # Secuencia en primer plano (se corta si cierras SSH)
    "$PY" validate_campaign.py "$@" || exit 1
    "$PY" preflight_campaign.py --write-all "$@"
    "$PY" validate_campaign.py --require-plans "$@" || exit 1
    "$PY" train_campaign.py --all --resume "$@"
    "$PY" evaluate_campaign.py --all --export-fp-videos "$@"
    "$PY" summarize_campaign.py "$@"
    ;;
  all-bg|nohup)
    # Para SSH: sobrevive al cierre de sesión. Log unificado con timestamp.
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG="${LOG_DIR}/campaign_all_${TS}.log"
    PID_FILE="${LOG_DIR}/campaign_all_${TS}.pid"
    echo "Lanzando campaña en background..."
    echo "  Log:  ${LOG}"
    nohup "$0" all "$@" >> "${LOG}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "  PID:  $(cat "${PID_FILE}")"
    echo ""
    echo "Puedes cerrar SSH. Monitorizar con:"
    echo "  tail -f ${LOG}"
    echo "  ps -p \$(cat ${PID_FILE})"
    ;;
  help|*)
    cat <<'EOF'
Uso: ./run_campaign.sh <comando> [args]

Comandos:
  check         Validación previa (imports, datos, sintaxis) — SIN exigir planes
  check-ready   Igual que check pero exige preflight --write-all ya hecho
  preflight     check + genera training_plan + augment por celda
  train         check-ready + entrena (--resume)
  eval          Evalúa val + export FP clips
  export-ensemble-fp  Lista FP ensemble MEAN 06+14 @ 0.68 (val) + symlinks vídeo
  summary       CSV maestro + campaign_gaps.txt
  all           check → preflight → check-ready → train → eval → summary (primer plano)
  all-bg        Igual que all pero con nohup (para SSH — puedes desconectar)
  nohup         Alias de all-bg

SSH (recomendado):
  ./run_campaign.sh all-bg
  tail -f artifacts/logs/campaign_all_*.log

Solo comprobar que no fallará nada:
  ./run_campaign.sh check
  ./run_campaign.sh check --strict          # exige GPU
  ./run_campaign.sh check --data-root /ruta/data_result

Nota: `all` y `all-bg` NO son lo mismo.
  - all     → se para si cierras la sesión SSH
  - all-bg  → sigue en background (nohup incluido)
EOF
    ;;
esac
