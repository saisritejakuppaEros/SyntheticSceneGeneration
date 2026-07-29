#!/usr/bin/env python3
"""
Step 2 — Quad flow remesh (ZRemesher equivalent).

Runs QuadWild (same engine as QRemeshify) to retrace quad flow on a
reduced mesh without dropping fragments.

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate
  python scripts/v2/quad_flow.py dataset/v2_out/rodin_reduced.glb
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import trimesh

from _paths import DEFAULT_INPUT, QUADWILD_LIB
from mesh_prep import cleanup_fragments, load_mesh, mesh_stats, weld_quantized_mesh


def _load_quadwild():
    sys.path.insert(0, str(QUADWILD_LIB))
    spec = importlib.util.spec_from_file_location("quadwild_lib", QUADWILD_LIB / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Quadwild, mod.QWException


def run_quad_flow(
    input_path: Path,
    output_glb: Path,
    *,
    work_dir: Path | None = None,
    from_reduced: bool = True,
    preprocess: bool = False,
    smoothing: bool = False,
    sharp_angle: float = 35.0,
    scale_fact: float = 1.2,
    min_component_faces: int = 50,
    target_tris: int = 100000,
    merge_distance: float = 0.002,
) -> dict:
    Quadwild, QWException = _load_quadwild()

    input_path = input_path.resolve()
    output_glb = output_glb.resolve()
    work_dir = (work_dir or input_path.parent / "quad_flow").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    input_obj = work_dir / f"{stem}_input.obj"

    print(f"Loading {input_path}...")
    mesh = load_mesh(input_path)
    before = mesh_stats(mesh)
    print(
        f"  {before['vertices']} verts, {before['faces']} faces, "
        f"{before['components']} components"
    )

    if from_reduced:
        print("Preparing reduced mesh (weld quantized verts)...")
        mesh = weld_quantized_mesh(mesh)
    else:
        print("Preparing mesh (fragment cleanup)...")
        mesh = cleanup_fragments(
            mesh,
            min_component_faces=min_component_faces,
            merge_distance=merge_distance,
            target_tris=target_tris,
        )

    mesh.export(input_obj)

    print("Running QuadWild...")
    qw = Quadwild(str(input_obj))
    try:
        qw.remeshAndField(
            remesh=preprocess,
            enableSharp=False,
            sharpAngle=sharp_angle,
        )
        if not qw.trace():
            raise QWException("trace step returned False")
        qw.quadrangulate(
            enableSmoothing=smoothing,
            scaleFact=scale_fact,
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
        result_path = Path(qw.output_smoothed_path if smoothing else qw.output_path)
    finally:
        del qw

    if not result_path.exists():
        raise FileNotFoundError(f"QuadWild output not found: {result_path}")

    result = trimesh.load(result_path, force="mesh")
    if isinstance(result, trimesh.Scene):
        result = trimesh.util.concatenate(tuple(result.geometry.values()))

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    result.export(output_glb)

    stats = mesh_stats(result)
    stats["obj"] = str(result_path)
    stats["glb"] = str(output_glb)
    print(f"Done: {stats['vertices']} verts, {stats['faces']} faces, {stats['components']} components")
    print(f"  OBJ: {result_path}")
    print(f"  GLB: {output_glb}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 quad flow remesh via QuadWild")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--from-raw", action="store_true", help="Run fragment cleanup instead of weld-only")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--smoothing", action="store_true")
    parser.add_argument("--sharp-angle", type=float, default=35.0)
    parser.add_argument("--scale-fact", type=float, default=1.2)
    parser.add_argument("--min-component-faces", type=int, default=50)
    parser.add_argument("--target-tris", type=int, default=100000)
    parser.add_argument("--merge-distance", type=float, default=0.002)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_glb = args.output or (input_path.parent / f"{input_path.stem}_quadflow.glb")

    run_quad_flow(
        input_path,
        output_glb,
        work_dir=args.work_dir,
        from_reduced=not args.from_raw,
        preprocess=args.preprocess,
        smoothing=args.smoothing,
        sharp_angle=args.sharp_angle,
        scale_fact=args.scale_fact,
        min_component_faces=args.min_component_faces,
        target_tris=args.target_tris,
        merge_distance=args.merge_distance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
