#!/usr/bin/env python3
"""CLI: Instant Meshes via meshcleaning scripts (.venv)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))

from api.common import count_obj_quads, mesh_stats, resolve_instant_meshes_output
from services.mesh_tools import run_instant_meshes


def main() -> int:
    parser = argparse.ArgumentParser(description="Instant Meshes CLI wrapper")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--target-quads", type=int, default=5000)
    parser.add_argument("--from-meshopt", action="store_true")
    parser.add_argument("--dominant", action="store_true")
    parser.add_argument("--no-boundaries", action="store_true")
    args = parser.parse_args()

    proc = run_instant_meshes(
        args.input.resolve(),
        args.output.resolve(),
        target_quads=args.target_quads,
        from_meshopt=args.from_meshopt,
        dominant=args.dominant,
        boundaries=not args.no_boundaries,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        return proc.returncode

    output_glb, output_obj = resolve_instant_meshes_output(
        args.input.resolve(),
        args.output.resolve(),
    )
    stats = mesh_stats(output_glb)
    if output_obj:
        stats["obj_quads"] = count_obj_quads(output_obj)

    print(json.dumps({
        "output_glb": str(output_glb),
        "output_obj": str(output_obj) if output_obj else None,
        "stats": stats,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
