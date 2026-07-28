#!/usr/bin/env python3
"""Convert GLB -> OBJ, run QuadriFlow retopology, export result OBJ/GLB."""

import argparse
import subprocess
import sys
from pathlib import Path

import trimesh
from pymeshfix import MeshFix

from _paths import ROOT_DIR
from mesh_prep import align_to_reference, load_mesh, mesh_stats, weld_quantized_mesh


def repair_mesh(mesh: trimesh.Trimesh, light: bool = False) -> trimesh.Trimesh:
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if light:
        return mesh

    fixer = MeshFix(mesh.vertices, mesh.faces)
    fixer.repair()
    repaired = trimesh.Trimesh(vertices=fixer.points, faces=fixer.faces)
    repaired.merge_vertices()
    repaired.update_faces(repaired.nondegenerate_faces())
    return repaired


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QuadriFlow on a GLB/OBJ mesh.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "-f",
        "--faces",
        type=int,
        default=10000,
        help="Target quad face count for QuadriFlow (default: 10000)",
    )
    parser.add_argument(
        "--quadriflow",
        type=Path,
        default=ROOT_DIR / "QuadriFlow" / "build" / "quadriflow",
    )
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument(
        "--light-repair",
        action="store_true",
        help="Merge/clean only; skip pymeshfix (good for gltfpack/meshoptimizer meshes)",
    )
    parser.add_argument(
        "--from-meshopt",
        action="store_true",
        help="Weld gltfpack vertices before QuadriFlow and realign output scale",
    )
    parser.add_argument("--sharp", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    stem = input_path.stem
    work_dir = input_path.parent / "quadriflow_out"
    work_dir.mkdir(parents=True, exist_ok=True)

    input_obj = work_dir / f"{stem}_input.obj"
    output_obj = work_dir / f"{stem}_quadriflow.obj"
    output_glb = args.output or work_dir / f"{stem}_quadriflow.glb"

    print(f"Loading {input_path}...")
    mesh = load_mesh(input_path)
    reference = mesh.copy()
    print(f"  vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, watertight={mesh.is_watertight}")

    if args.from_meshopt:
        mesh = weld_quantized_mesh(mesh)
        reference = mesh.copy()
        s = mesh_stats(mesh)
        print(f"  welded: vertices={s['vertices']}, faces={s['faces']}, components={s['components']}")

    if not args.skip_repair:
        print("Repairing mesh for manifold processing...")
        mesh = repair_mesh(mesh, light=args.light_repair)
        print(f"  after repair: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, watertight={mesh.is_watertight}")

    print(f"Exporting OBJ -> {input_obj}")
    mesh.export(input_obj)

    cmd = [
        str(args.quadriflow),
        "-i",
        str(input_obj),
        "-o",
        str(output_obj),
        "-f",
        str(args.faces),
    ]
    if args.sharp:
        cmd.append("-sharp")

    print("Running QuadriFlow:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"Loading QuadriFlow output from {output_obj}")
    result = trimesh.load(output_obj, force="mesh")
    if isinstance(result, trimesh.Scene):
        result = trimesh.util.concatenate(tuple(result.geometry.values()))

    if args.from_meshopt or result.extents.max() > reference.extents.max() * 1.5:
        before_ext = result.extents
        result = align_to_reference(result, reference)
        print(f"  realigned scale: {before_ext} -> {result.extents}")

    print(f"  result: vertices={len(result.vertices)}, faces={len(result.faces)}")
    print(f"Exporting GLB -> {output_glb}")
    result.export(output_glb)
    print("Done.")
    print(f"OBJ: {output_obj}")
    print(f"GLB: {output_glb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
