#!/usr/bin/env python3
"""CLI: AutoRemesher via meshcleaning scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))

from api.common import mesh_stats
from services.mesh_tools import run_autoremesher


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoRemesher CLI wrapper")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--target-quads", type=int, default=5000)
    parser.add_argument("--prep-target-tris", type=int, default=0)
    parser.add_argument("--solid-only", action="store_true")
    parser.add_argument("--from-solid", action="store_true")
    args = parser.parse_args()

    proc = run_autoremesher(
        args.input.resolve(),
        args.output.resolve(),
        target_quads=args.target_quads,
        prep_target_tris=args.prep_target_tris,
        solid_only=args.solid_only,
        from_solid=args.from_solid,
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
