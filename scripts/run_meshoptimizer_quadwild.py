#!/usr/bin/env python3
"""
Pipeline: meshoptimizer (triangle reduction) -> QuadWild (quad flow).

meshoptimizer shrinks the mesh and keeps embedded textures.
QuadWild retraces quad flow without dropping gltfpack fragments.

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate
  python /devwork/teja/meshcleaning/scripts/run_meshoptimizer_quadwild.py \\
    /devwork/teja/meshcleaning/dataset/sample_2026-07-27T084001.382.glb \\
    --target-tris 30000
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from _paths import DATASET_DIR, SCRIPT_DIR
from mesh_prep import load_mesh, mesh_stats

RUN_MESHOPT = SCRIPT_DIR / "run_meshoptimizer.py"
RUN_QUADWILD = SCRIPT_DIR / "run_quadwild.py"
RUN_TRANSFER = SCRIPT_DIR / "transfer_texture.py"
DEFAULT_INPUT = DATASET_DIR / "sample_2026-07-27T084001.382.glb"


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed (exit {proc.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description="meshoptimizer reduce -> QuadWild quad flow")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Final textured GLB (default: <work-dir>/<stem>_mqw.glb)",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--meshopt-input",
        type=Path,
        default=None,
        help="Skip meshoptimizer; use existing reduced GLB/OBJ",
    )
    parser.add_argument("--target-tris", type=int, default=30000)
    parser.add_argument("--simplify-error", type=float, default=0.01)
    parser.add_argument("--scale-fact", type=float, default=1.2)
    parser.add_argument("--no-preprocess", action="store_true", default=True)
    parser.add_argument("--preprocess", dest="no_preprocess", action="store_false")
    parser.add_argument("--no-smoothing", action="store_true", default=True)
    parser.add_argument("--smoothing", dest="no_smoothing", action="store_false")
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Skip texture bake; export raw QuadWild GLB",
    )
    parser.add_argument("--texture-size", type=int, default=128)
    args = parser.parse_args()

    input_path = args.input.resolve()
    work_dir = (args.work_dir or input_path.parent / "meshoptimizer_quadwild_out").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.meshopt_input:
        meshopt_glb = args.meshopt_input.resolve()
        stem = meshopt_glb.stem
        print(f"Using existing meshoptimizer output: {meshopt_glb}")
    else:
        stem = input_path.stem
        meshopt_glb = work_dir / f"{stem}_meshopt.glb"
        run_step(
            [
                sys.executable,
                str(RUN_MESHOPT),
                str(input_path),
                "--target-tris",
                str(args.target_tris),
                "--simplify-error",
                str(args.simplify_error),
                "-o",
                str(meshopt_glb),
            ],
            "Step 1/3: meshoptimizer (triangle reduction)",
        )

    meshopt_stats = mesh_stats(load_mesh(meshopt_glb))
    print(
        f"\nmeshoptimizer output: {meshopt_stats['faces']} faces, "
        f"{meshopt_stats['components']} components"
    )

    quadwild_glb = work_dir / f"{stem}_quadwild.glb"
    final_glb = args.output or work_dir / f"{stem}_mqw.glb"

    qw_cmd = [
        sys.executable,
        str(RUN_QUADWILD),
        str(meshopt_glb),
        "--from-meshopt",
        "--target-tris",
        str(max(args.target_tris, meshopt_stats["faces"])),
        "--scale-fact",
        str(args.scale_fact),
        "-o",
        str(quadwild_glb),
    ]
    if args.no_preprocess:
        qw_cmd.append("--no-preprocess")
    if args.no_smoothing:
        qw_cmd.append("--no-smoothing")

    run_step(qw_cmd, "Step 2/3: QuadWild (quad flow remesh)")

    if args.no_texture:
        if final_glb.resolve() != quadwild_glb.resolve():
            shutil.copy2(quadwild_glb, final_glb)
        print(f"\nDone.\nGLB: {final_glb}")
        return 0

    run_step(
        [
            sys.executable,
            str(RUN_TRANSFER),
            "--source",
            str(meshopt_glb),
            "--target",
            str(quadwild_glb),
            "--mode",
            "texture",
            "--texture-size",
            str(args.texture_size),
            "-o",
            str(final_glb),
        ],
        "Step 3/3: embed texture on quad mesh",
    )

    result = mesh_stats(load_mesh(final_glb))
    print(
        f"\nDone.\n"
        f"  meshopt:   {meshopt_glb}\n"
        f"  quadwild:  {quadwild_glb}\n"
        f"  final GLB: {final_glb}\n"
        f"  final:     {result['vertices']} verts, {result['faces']} faces"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
