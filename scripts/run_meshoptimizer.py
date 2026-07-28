#!/usr/bin/env python3
"""
Run meshoptimizer gltfpack on a GLB/OBJ (mesh simplify + optimize).

Preserves embedded textures in the output GLB (single self-contained file).

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate
  python /devwork/teja/meshcleaning/scripts/run_meshoptimizer.py \\
    /devwork/teja/meshcleaning/dataset/sample_2026-07-27T084001.382.glb \\
    --target-tris 30000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import trimesh

from _paths import DATASET_DIR, ROOT_DIR

MESHOPT_DIR = ROOT_DIR / "meshoptimizer"
GLTFPACK = MESHOPT_DIR / "gltfpack"
DEFAULT_INPUT = DATASET_DIR / "sample_2026-07-27T084001.382.glb"


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"No mesh geometry in {path}")
        return trimesh.util.concatenate(parts)
    return loaded


def count_faces(path: Path) -> int:
    return len(load_mesh(path).faces)


def target_tris_to_ratio(target_tris: int, face_count: int) -> float:
    if face_count <= 0:
        raise ValueError("Input mesh has no faces")
    return max(0.0001, min(1.0, target_tris / face_count))


def glb_stats(path: Path) -> dict:
    mesh = load_mesh(path)
    stats = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "file_kb": round(path.stat().st_size / 1024, 1),
    }
    if hasattr(mesh.visual, "material") and mesh.visual.material:
        tex = mesh.visual.material.baseColorTexture
        if tex is not None:
            stats["texture_size"] = getattr(tex, "size", None)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="meshoptimizer gltfpack CLI wrapper")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output GLB (default: dataset/meshoptimizer_out/<stem>_meshopt.glb)",
    )
    parser.add_argument(
        "--target-tris",
        type=int,
        default=30000,
        help="Target triangle count (converted to gltfpack -si ratio)",
    )
    parser.add_argument(
        "--simplify-ratio",
        type=float,
        default=None,
        help="Pass -si directly (overrides --target-tris)",
    )
    parser.add_argument(
        "--simplify-error",
        type=float,
        default=0.01,
        help="gltfpack -se simplification error limit (0..1)",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use -c for EXT_meshopt_compression (smaller, needs decoder in viewer)",
    )
    parser.add_argument(
        "--gltfpack",
        type=Path,
        default=GLTFPACK,
        help="Path to gltfpack binary",
    )
    args = parser.parse_args()

    gltfpack = args.gltfpack.resolve()
    if not gltfpack.exists():
        raise SystemExit(
            f"gltfpack not found at {gltfpack}\n"
            f"Build it: cd {MESHOPT_DIR} && make gltfpack"
        )

    input_path = args.input.resolve()
    stem = input_path.stem
    out_dir = input_path.parent / "meshoptimizer_out"
    output_glb = args.output or out_dir / f"{stem}_meshopt.glb"
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}...")
    face_count = count_faces(input_path)
    si = args.simplify_ratio if args.simplify_ratio is not None else target_tris_to_ratio(
        args.target_tris, face_count
    )
    print(f"  input faces={face_count}")
    print(f"  simplify ratio (-si)={si:.4f}  error (-se)={args.simplify_error}")

    cmd = [
        str(gltfpack),
        "-i",
        str(input_path),
        "-o",
        str(output_glb),
        "-si",
        f"{si:.6f}",
        "-se",
        f"{args.simplify_error:.6f}",
    ]
    if args.compressed:
        cmd.append("-c")

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"gltfpack failed (exit {proc.returncode})")

    stats = glb_stats(output_glb)
    print(
        f"Done.\n"
        f"  output: {output_glb}\n"
        f"  vertices={stats['vertices']}, faces={stats['faces']}, size={stats['file_kb']} KB"
    )
    if "texture_size" in stats:
        print(f"  embedded texture: {stats['texture_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
