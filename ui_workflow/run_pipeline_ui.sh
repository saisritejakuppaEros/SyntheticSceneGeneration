#!/usr/bin/env bash
set -eo pipefail

UI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESH_ROOT="$(cd "$UI_ROOT/.." && pwd)"
VENV="$MESH_ROOT/.venv"

source "$VENV/bin/activate"

if ! python -c "import gradio, requests" 2>/dev/null; then
  echo "Installing pipeline UI dependencies..."
  pip install -q gradio requests
fi

cd "$UI_ROOT"
exec python pipeline_app.py "$@"
