#!/usr/bin/env python3
"""CLI: QuadriFlow via meshcleaning scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))

from api.common import count_obj_quads, mesh_stats, resolve_quadriflow_output
from services.mesh_tools import run_quadriflow


def main() -> int:
    parser = argparse.ArgumentParser(description="QuadriFlow CLI wrapper")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--target-quads", type=int, default=5000)
    parser.add_argument("--sharp", action="store_true")
    parser.add_argument("--repair", dest="skip_repair", action="store_false")
    parser.add_argument("--from-meshopt", action="store_true")
    args = parser.parse_args()

    proc = run_quadriflow(
        args.input.resolve(),
        args.output.resolve(),
        target_quads=args.target_quads,
        sharp=args.sharp,
        skip_repair=args.skip_repair,
        from_meshopt=args.from_meshopt,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        return proc.returncode

    output_glb, output_obj = resolve_quadriflow_output(
        args.input.resolve(),
        args.output.resolve(),
    )
    stats = mesh_stats(output_glb)
    if output_obj:
        stats["obj_quads"] = count_obj_quads(output_obj)

    result = {
        "output_glb": str(output_glb),
        "output_obj": str(output_obj) if output_obj else None,
        "stats": stats,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
