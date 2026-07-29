#!/usr/bin/env python3
"""
Step 0 — Fragment cleanup (manual cleanup equivalent).

Drop tiny floating shells and weld coincident vertices so the mesh
has fewer disconnected components before reduction/remesh.

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate
  python scripts/v2/cleanup_fragments.py dataset/rodin_3.obj -o dataset/v2_out/rodin_3_clean.glb
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _paths import DEFAULT_INPUT
from mesh_prep import cleanup_fragments, load_mesh, mesh_stats


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 fragment cleanup")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--min-component-faces", type=int, default=50)
    parser.add_argument("--merge-distance", type=float, default=0.002)
    parser.add_argument("--target-tris", type=int, default=None)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output or (input_path.parent / "v2_out" / f"{input_path.stem}_clean.glb")

    print(f"Loading {input_path}...")
    mesh = load_mesh(input_path)
    before = mesh_stats(mesh)
    print(f"  {before['vertices']} verts, {before['faces']} faces, {before['components']} components")

    print("Cleaning fragments...")
    cleaned = cleanup_fragments(
        mesh,
        min_component_faces=args.min_component_faces,
        merge_distance=args.merge_distance,
        target_tris=args.target_tris,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.export(output_path)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
