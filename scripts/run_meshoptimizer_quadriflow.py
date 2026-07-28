#!/usr/bin/env python3
"""
Pipeline: meshoptimizer (triangle reduction) -> QuadriFlow (quad flow remesh).

meshoptimizer reduces triangle count while keeping embedded textures.
QuadriFlow retraces quad edge flow without further reduction by default
(-f matches the meshoptimizer output triangle count).

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate
  python /devwork/teja/meshcleaning/scripts/run_meshoptimizer_quadriflow.py \\
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
RUN_QUADRIFLOW = SCRIPT_DIR / "run_quadriflow.py"
RUN_TRANSFER = SCRIPT_DIR / "transfer_texture.py"
DEFAULT_INPUT = DATASET_DIR / "sample_2026-07-27T084001.382.glb"


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed (exit {proc.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="meshoptimizer reduce -> QuadriFlow quad remesh (preserve count)"
    )
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Final GLB (default: dataset/meshoptimizer_quadriflow_out/<stem>_mq.glb)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Intermediate/output folder (default: dataset/meshoptimizer_quadriflow_out)",
    )
    parser.add_argument(
        "--meshopt-input",
        type=Path,
        default=None,
        help="Skip meshoptimizer; use this GLB/OBJ as QuadriFlow input",
    )
    parser.add_argument("--target-tris", type=int, default=30000)
    parser.add_argument("--simplify-error", type=float, default=0.01)
    parser.add_argument(
        "--quad-faces",
        type=int,
        default=None,
        help="QuadriFlow -f target (default: triangle count after meshoptimizer)",
    )
    parser.add_argument(
        "--quad-face-scale",
        type=float,
        default=1.0,
        help="Multiply auto quad face count by this factor (default: 1.0 = no extra reduction)",
    )
    parser.add_argument("--sharp", action="store_true", help="QuadriFlow -sharp")
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        default=True,
        help="Skip pymeshfix (default: on; meshfix destroys gltfpack meshes)",
    )
    parser.add_argument(
        "--repair",
        dest="skip_repair",
        action="store_false",
        help="Run pymeshfix repair before QuadriFlow (may collapse quantized meshes)",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Skip texture transfer onto QuadriFlow output",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=128,
        help="Embedded texture atlas size for final GLB (via transfer_texture.py)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    work_dir = (args.work_dir or input_path.parent / "meshoptimizer_quadriflow_out").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.meshopt_input:
        meshopt_glb = args.meshopt_input.resolve()
        if not meshopt_glb.exists():
            raise SystemExit(f"--meshopt-input not found: {meshopt_glb}")
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

    quad_glb = work_dir / f"{stem}_quadriflow.glb"
    final_glb = args.output or work_dir / f"{stem}_mq.glb"

    mesh = load_mesh(meshopt_glb)
    tri_faces = len(mesh.faces)
    quad_faces = args.quad_faces
    if quad_faces is None:
        quad_faces = max(1000, int(tri_faces * args.quad_face_scale))
    print(
        f"\nQuadriFlow target: {quad_faces} quads "
        f"(from {tri_faces} input triangles, scale={args.quad_face_scale})"
    )

    quad_cmd = [
        sys.executable,
        str(RUN_QUADRIFLOW),
        str(meshopt_glb),
        "-f",
        str(quad_faces),
        "-o",
        str(quad_glb),
    ]
    if args.sharp:
        quad_cmd.append("--sharp")
    if args.skip_repair:
        quad_cmd.append("--skip-repair")
    else:
        quad_cmd.append("--light-repair")
    quad_cmd.append("--from-meshopt")

    run_step(quad_cmd, "Step 2/3: QuadriFlow (quad flow remesh)")

    if args.no_texture:
        if final_glb.resolve() != quad_glb.resolve():
            shutil.copy2(quad_glb, final_glb)
        print(f"\nDone (no texture transfer).\nGLB: {final_glb}")
        return 0

    run_step(
        [
            sys.executable,
            str(RUN_TRANSFER),
            "--source",
            str(meshopt_glb),
            "--target",
            str(quad_glb),
            "--mode",
            "texture",
            "--texture-size",
            str(args.texture_size),
            "-o",
            str(final_glb),
        ],
        "Step 3/3: embed texture on quad mesh",
    )

    result = load_mesh(final_glb)
    print(
        f"\nDone.\n"
        f"  meshopt GLB:  {meshopt_glb}\n"
        f"  quadriflow:   {quad_glb}\n"
        f"  final GLB:    {final_glb}\n"
        f"  final stats:  vertices={len(result.vertices)}, faces={len(result.faces)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
