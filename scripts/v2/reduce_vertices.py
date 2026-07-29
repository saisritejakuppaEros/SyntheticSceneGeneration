#!/usr/bin/env python3
"""
Step 1 — Vertex reduction (Houdini equivalent).

Uses meshoptimizer gltfpack to simplify triangle count while preserving
embedded textures in the output GLB.

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate
  python scripts/v2/reduce_vertices.py dataset/rodin.obj --target-tris 5000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _paths import DATASET_DIR, DEFAULT_INPUT, GLTFPACK
from mesh_prep import load_mesh, mesh_stats


def target_tris_to_ratio(target_tris: int, face_count: int) -> float:
    if face_count <= 0:
        raise ValueError("Input mesh has no faces")
    return max(0.0001, min(1.0, target_tris / face_count))


def reduce_vertices(
    input_path: Path,
    output_path: Path,
    *,
    target_tris: int = 5000,
    simplify_ratio: float | None = None,
    simplify_error: float = 0.01,
    compressed: bool = False,
    gltfpack: Path = GLTFPACK,
) -> dict:
    gltfpack = gltfpack.resolve()
    if not gltfpack.exists():
        raise FileNotFoundError(
            f"gltfpack not found at {gltfpack}\nBuild: cd meshoptimizer && make gltfpack"
        )

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    face_count = len(load_mesh(input_path).faces)
    si = simplify_ratio if simplify_ratio is not None else target_tris_to_ratio(target_tris, face_count)

    print(f"Input:  {input_path} ({face_count} faces)")
    print(f"Target: {target_tris} tris  ratio={si:.4f}  error={simplify_error}")

    cmd = [
        str(gltfpack),
        "-i", str(input_path),
        "-o", str(output_path),
        "-si", f"{si:.6f}",
        "-se", f"{simplify_error:.6f}",
    ]
    if compressed:
        cmd.append("-c")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"gltfpack failed (exit {proc.returncode})")

    stats = mesh_stats(load_mesh(output_path))
    stats["file_kb"] = round(output_path.stat().st_size / 1024, 1)
    stats["output"] = str(output_path)
    print(f"Output: {output_path}  {stats['vertices']} verts, {stats['faces']} faces, {stats['components']} components")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 vertex reduction via meshoptimizer")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--target-tris", type=int, default=5000)
    parser.add_argument("--simplify-ratio", type=float, default=None)
    parser.add_argument("--simplify-error", type=float, default=0.01)
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--gltfpack", type=Path, default=GLTFPACK)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output or (input_path.parent / "v2_out" / f"{input_path.stem}_reduced.glb")

    reduce_vertices(
        input_path,
        output_path,
        target_tris=args.target_tris,
        simplify_ratio=args.simplify_ratio,
        simplify_error=args.simplify_error,
        compressed=args.compressed,
        gltfpack=args.gltfpack,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
