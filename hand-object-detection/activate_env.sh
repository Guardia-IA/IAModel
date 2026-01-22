#!/bin/bash
# Script para activar el entorno virtual

cd "$(dirname "$0")"
source venv/bin/activate
echo "✓ Entorno virtual activado"
echo "  Para desactivar, ejecuta: deactivate"
