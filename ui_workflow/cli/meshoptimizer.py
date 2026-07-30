#!/usr/bin/env python3
"""CLI: meshoptimizer via meshcleaning scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))

from api.common import mesh_stats
from services.mesh_tools import run_meshoptimizer


def main() -> int:
    parser = argparse.ArgumentParser(description="meshoptimizer CLI wrapper")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--target-tris", type=int, default=30000)
    parser.add_argument("--simplify-error", type=float, default=0.01)
    args = parser.parse_args()

    proc = run_meshoptimizer(
        args.input.resolve(),
        args.output.resolve(),
        target_tris=args.target_tris,
        simplify_error=args.simplify_error,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        return proc.returncode

    print(json.dumps({"output_glb": str(args.output), "stats": mesh_stats(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
