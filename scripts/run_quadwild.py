#!/usr/bin/env python3
"""
Run QuadWild (same engine as QRemeshify) without Blender.

Uses the native libraries from the QRemeshify Linux release via ctypes.

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate
  python /devwork/teja/meshcleaning/scripts/run_quadwild.py \\
    /devwork/teja/meshcleaning/dataset/sample_2026-07-27T084001.382.glb
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

from _paths import ROOT_DIR
from mesh_prep import load_mesh, mesh_stats, weld_quantized_mesh

LIB_DIR = ROOT_DIR / "quadwild_release" / "QRemeshify" / "lib"
sys.path.insert(0, str(LIB_DIR))

import importlib.util

_spec = importlib.util.spec_from_file_location("quadwild_lib", LIB_DIR / "__init__.py")
_qw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qw)
Quadwild = _qw.Quadwild
QWException = _qw.QWException


def component_sizes(mesh: trimesh.Trimesh) -> list[int]:
    parent = list(range(len(mesh.faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for a, b in mesh.face_adjacency:
        union(a, b)

    counts: dict[int, int] = defaultdict(int)
    for i in range(len(mesh.faces)):
        counts[find(i)] += 1
    return sorted(counts.values(), reverse=True)


def keep_large_components(mesh: trimesh.Trimesh, min_faces: int) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    kept = [p for p in parts if len(p.faces) >= min_faces]
    if not kept:
        kept = [max(parts, key=lambda p: len(p.faces))]
    print(f"  kept {len(kept)}/{len(parts)} components (min_faces={min_faces})")
    return trimesh.util.concatenate(kept)


def prep_mesh(
    mesh: trimesh.Trimesh,
    merge_distance: float,
    target_tris: int,
    min_component_faces: int,
) -> trimesh.Trimesh:
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    sizes = component_sizes(mesh)
    print(f"  components before filter: {len(sizes)}, largest={sizes[0]}, total faces={len(mesh.faces)}")

    if sizes[0] < min_component_faces:
        parts = mesh.split(only_watertight=False)
        parts.sort(key=lambda p: len(p.faces), reverse=True)
        kept, face_budget = [], 0
        target_faces = int(len(mesh.faces) * 0.8)
        for part in parts:
            kept.append(part)
            face_budget += len(part.faces)
            if face_budget >= target_faces:
                break
        print(f"  kept {len(kept)}/{len(parts)} components (~{face_budget} faces)")
        mesh = trimesh.util.concatenate(kept)
    else:
        mesh = keep_large_components(mesh, min_component_faces)

    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    if len(mesh.faces) > target_tris:
        mesh = mesh.simplify_quadric_decimation(face_count=target_tris)
        print(f"  decimated to {len(mesh.faces)} faces")

    if not mesh.is_winding_consistent:
        trimesh.repair.fix_normals(mesh)
    return mesh


def export_obj(mesh: trimesh.Trimesh, path: Path) -> None:
    mesh.export(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="QuadWild CLI (no Blender)")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output GLB path (default: dataset/quadwild_out/<name>_quadwild.glb)",
    )
    parser.add_argument("--preprocess", action="store_true", default=True)
    parser.add_argument("--no-preprocess", dest="preprocess", action="store_false")
    parser.add_argument("--smoothing", action="store_true", default=True)
    parser.add_argument("--no-smoothing", dest="smoothing", action="store_false")
    parser.add_argument("--sharp-angle", type=float, default=35.0)
    parser.add_argument("--scale-fact", type=float, default=1.0)
    parser.add_argument("--merge-distance", type=float, default=0.002)
    parser.add_argument("--target-tris", type=int, default=100000)
    parser.add_argument("--min-component-faces", type=int, default=500)
    parser.add_argument(
        "--from-meshopt",
        action="store_true",
        help="Input already reduced by meshoptimizer: weld only, keep all components, skip decimation",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip prep_mesh entirely (export input as-is after load)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    stem = input_path.stem
    work_dir = input_path.parent / "quadwild_out"
    work_dir.mkdir(parents=True, exist_ok=True)

    input_obj = work_dir / f"{stem}_input.obj"
    output_glb = args.output or work_dir / f"{stem}_quadwild.glb"

    print(f"Loading {input_path}...")
    mesh = load_mesh(input_path)
    before = mesh_stats(mesh)
    print(
        f"  vertices={before['vertices']}, faces={before['faces']}, "
        f"components={before['components']}, watertight={before['watertight']}"
    )

    min_component_faces = args.min_component_faces
    target_tris = args.target_tris
    if args.from_meshopt:
        min_component_faces = 1
        target_tris = max(target_tris, before["faces"])
        print("Preparing mesh (meshoptimizer mode: weld + keep all components)...")
        mesh = weld_quantized_mesh(mesh)
        after = mesh_stats(mesh)
        print(
            f"  welded: vertices={after['vertices']}, faces={after['faces']}, "
            f"components={after['components']}"
        )
    elif args.skip_prep:
        print("Skipping prep_mesh.")
    else:
        print("Preparing mesh...")
        mesh = prep_mesh(
            mesh,
            merge_distance=args.merge_distance,
            target_tris=target_tris,
            min_component_faces=min_component_faces,
        )
        print(f"  prepared: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

    export_obj(mesh, input_obj)

    print("Running QuadWild pipeline...")
    qw = Quadwild(str(input_obj))
    try:
        qw.remeshAndField(
            remesh=args.preprocess,
            enableSharp=False,
            sharpAngle=args.sharp_angle,
        )
        print(f"  remeshed: {qw.remeshed_path}")
        if not qw.trace():
            raise QWException("trace step returned False")
        print(f"  traced: {qw.traced_path}")
        qw.quadrangulate(
            enableSmoothing=args.smoothing,
            scaleFact=args.scale_fact,
            fixedChartClusters=0,
            alpha=0.005,
            ilpMethod="LEASTSQUARES",
            timeLimit=200,
            gapLimit=0.0,
            minimumGap=0.4,
            isometry=True,
            regularityQuadrilaterals=True,
            regularityNonQuadrilaterals=True,
            regularityNonQuadrilateralsWeight=0.9,
            alignSingularities=True,
            alignSingularitiesWeight=0.1,
            repeatLosingConstraintsIterations=True,
            repeatLosingConstraintsQuads=False,
            repeatLosingConstraintsNonQuads=False,
            repeatLosingConstraintsAlign=True,
            hardParityConstraint=True,
            flowConfig="SIMPLE",
            satsumaConfig="DEFAULT",
            callbackTimeLimit=[3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0],
            callbackGapLimit=[0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3],
        )
        result_path = Path(qw.output_smoothed_path if args.smoothing else qw.output_path)
    finally:
        del qw

    if not result_path.exists():
        raise FileNotFoundError(f"QuadWild output not found near {work_dir}")

    print(f"Loading result from {result_path}")
    result = trimesh.load(result_path, force="mesh")
    if isinstance(result, trimesh.Scene):
        result = trimesh.util.concatenate(tuple(result.geometry.values()))
    print(f"  result: vertices={len(result.vertices)}, faces={len(result.faces)}")

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    result.export(output_glb)
    print(f"Done.\nOBJ: {result_path}\nGLB: {output_glb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
