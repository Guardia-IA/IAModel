#!/usr/bin/env bash
# Orquestador de la campaña de experimentos (preflight → train → eval → summary)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CMD="${1:-help}"
shift || true

PY="${PYTHON:-python3}"

case "$CMD" in
  preflight)
    exec "$PY" preflight_campaign.py --write-all "$@"
    ;;
  train)
    exec "$PY" train_campaign.py --all --resume "$@"
    ;;
  eval)
    exec "$PY" evaluate_campaign.py --all --export-fp-videos "$@"
    ;;
  summary)
    exec "$PY" summarize_campaign.py "$@"
    ;;
  all)
    "$PY" preflight_campaign.py --write-all
    "$PY" train_campaign.py --all --resume
    "$PY" evaluate_campaign.py --all --export-fp-videos
    "$PY" summarize_campaign.py
    ;;
  help|*)
    cat <<'EOF'
Uso: ./run_campaign.sh <comando> [args]

Comandos:
  preflight   Genera training_plan + augment por celda
  train       Entrena experiment_ids de campaign_config (--resume)
  eval        Evalúa val + export FP clips
  summary     CSV maestro + campaign_gaps.txt
  all         Secuencia completa

Ejemplos:
  nohup ./run_campaign.sh train > artifacts/logs/train.log 2>&1 &
  ./run_campaign.sh eval
  python train_campaign.py --cells bin_full bin_filtered
  python evaluate_campaign.py --cells bin_full --split val
EOF
    ;;
esac
