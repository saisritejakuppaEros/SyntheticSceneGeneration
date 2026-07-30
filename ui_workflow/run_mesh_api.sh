#!/usr/bin/env bash
set -eo pipefail

UI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH_ROOT="$(cd "$UI_ROOT/.." && pwd)"
VENV="$MESH_ROOT/.venv"

source "$VENV/bin/activate"

if ! python -c "import fastapi, uvicorn, xatlas" 2>/dev/null; then
  echo "Installing API dependencies into meshcleaning .venv..."
  pip install -q fastapi uvicorn python-multipart xatlas
fi

cd "$UI_ROOT"
exec python -m uvicorn api.mesh_api:app \
  --host "${MESH_API_HOST:-0.0.0.0}" \
  --port "${MESH_API_PORT:-8101}" \
  "$@"
