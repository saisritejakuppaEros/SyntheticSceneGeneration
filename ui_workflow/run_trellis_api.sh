#!/usr/bin/env bash
set -eo pipefail

UI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$UI_ROOT"

exec "$UI_ROOT/run_trellis_python.sh" -m uvicorn api.trellis_api:app \
  --host "${TRELLIS_API_HOST:-0.0.0.0}" \
  --port "${TRELLIS_API_PORT:-8100}" \
  "$@"
