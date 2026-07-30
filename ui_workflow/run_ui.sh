#!/usr/bin/env bash
set -eo pipefail

UI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$UI_ROOT/run_trellis_python.sh" "$UI_ROOT/app.py" "$@"
