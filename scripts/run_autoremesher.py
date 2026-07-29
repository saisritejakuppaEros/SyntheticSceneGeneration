#!/usr/bin/env python3
"""
Run AutoRemesher headless on a GLB/OBJ mesh.

Two-stage pipeline:
  1. Solidify  — drop junk shells, keep main body, make watertight (shape preserved)
  2. Remesh     — optional gltfpack reduction, then AutoRemesher quad output

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate

  # Full pipeline
  python scripts/run_autoremesher.py dataset/sample.glb --target-quads 5000

  # Solidify only (inspect before remeshing)
  python scripts/run_autoremesher.py dataset/sample.glb --solid-only

  # Remesh from an existing solid GLB
  python scripts/run_autoremesher.py dataset/autoremesher_out/sample_solid.glb --from-solid
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import trimesh
from pymeshfix import MeshFix

from _paths import ROOT_DIR
from mesh_prep import load_mesh, mesh_stats

AUTOREMESHER = ROOT_DIR / ".venv" / "bin" / "autoremesher"
GLTFPACK = ROOT_DIR / "meshoptimizer" / "gltfpack"
DEFAULT_INPUT = ROOT_DIR / "dataset" / "rodin_3.obj"

_v2_spec = importlib.util.spec_from_file_location(
    "v2_mesh_prep", ROOT_DIR / "scripts" / "v2" / "mesh_prep.py"
)
_v2_mesh_prep = importlib.util.module_from_spec(_v2_spec)
_v2_spec.loader.exec_module(_v2_mesh_prep)
cleanup_fragments = _v2_mesh_prep.cleanup_fragments


def print_stats(label: str, mesh: trimesh.Trimesh) -> None:
    stats = mesh_stats(mesh)
    print(
        f"  {label}: {stats['vertices']} verts, {stats['faces']} faces, "
        f"{stats['components']} components, watertight={stats['watertight']}, "
        f"extents={[round(x, 4) for x in stats['extents']]}"
    )


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if not parts:
        raise RuntimeError("Mesh has no geometry")
    main = max(parts, key=lambda p: len(p.faces)).copy()
    main.merge_vertices()
    main.update_faces(main.nondegenerate_faces())
    main.remove_unreferenced_vertices()
    return main


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if mesh_stats(mesh)["components"] != 1:
        raise RuntimeError(
            "pymeshfix repair expects a single connected shell; "
            "use --keep-largest-only or reduce --min-component-faces"
        )
    fixer = MeshFix(mesh.vertices, mesh.faces)
    fixer.repair()
    repaired = trimesh.Trimesh(vertices=fixer.points, faces=fixer.faces)
    repaired.merge_vertices()
    repaired.update_faces(repaired.nondegenerate_faces())
    repaired.remove_unreferenced_vertices()
    return repaired


def solidify_mesh(
    mesh: trimesh.Trimesh,
    *,
    min_component_faces: int,
    merge_distance: float,
    keep_largest_only: bool,
    repair: bool,
) -> trimesh.Trimesh:
    print("Step 1/2: Solidify (drop junk, make watertight, preserve shape)...")
    solid = cleanup_fragments(
        mesh,
        min_component_faces=min_component_faces,
        merge_distance=merge_distance,
        target_tris=None,
    )

    if keep_largest_only:
        before_parts = mesh_stats(solid)["components"]
        solid = keep_largest_component(solid)
        print(f"  kept largest shell ({before_parts} -> 1 component)")

    print_stats("after fragment cleanup", solid)

    if repair:
        print("  running pymeshfix on single shell...")
        solid = repair_mesh(solid)
        print_stats("after repair", solid)

    return solid


def cleanup_remesh_output(
    mesh: trimesh.Trimesh,
    *,
    min_component_faces: int,
    keep_largest_only: bool,
) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if not parts:
        return mesh

    if keep_largest_only:
        cleaned = max(parts, key=lambda p: len(p.faces)).copy()
        dropped = len(parts) - 1
    else:
        kept = [p for p in parts if len(p.faces) >= min_component_faces]
        if not kept:
            kept = [max(parts, key=lambda p: len(p.faces))]
        cleaned = trimesh.util.concatenate(kept)
        dropped = len(parts) - len(kept)

    cleaned.merge_vertices()
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()

    if dropped:
        print(f"  removed {dropped} stray remesh islands")
    return cleaned


def auto_target_quads(triangle_faces: int) -> int:
    """Pick a quad budget that tracks input complexity without going too high."""
    return int(max(5000, min(triangle_faces // 3, 25000)))


def finalize_remesh_input(mesh: trimesh.Trimesh, *, repair: bool) -> trimesh.Trimesh:
    mesh = keep_largest_component(mesh)
    if repair and not mesh_stats(mesh)["watertight"]:
        print("  re-solidifying reduced mesh before AutoRemesher...")
        mesh = repair_mesh(mesh)
    return mesh


def target_tris_to_ratio(target_tris: int, face_count: int) -> float:
    if face_count <= 0:
        raise ValueError("Input mesh has no faces")
    return max(0.0001, min(1.0, target_tris / face_count))


def reduce_for_remesh(
    solid: trimesh.Trimesh,
    *,
    target_tris: int,
    simplify_error: float,
    gltfpack: Path,
    work_dir: Path,
    stem: str,
) -> trimesh.Trimesh:
    if target_tris <= 0 or len(solid.faces) <= target_tris:
        print(f"  skipping reduction ({len(solid.faces)} faces)")
        return solid

    gltfpack = gltfpack.resolve()
    if not gltfpack.exists():
        print(
            f"  gltfpack not found at {gltfpack}; "
            f"falling back to quadric decimation"
        )
        reduced = solid.simplify_quadric_decimation(face_count=target_tris)
        print_stats("after quadric decimation", reduced)
        return reduced

    solid_glb = work_dir / f"{stem}_solid.glb"
    reduced_glb = work_dir / f"{stem}_reduced.glb"
    solid.export(solid_glb)

    ratio = target_tris_to_ratio(target_tris, len(solid.faces))
    cmd = [
        str(gltfpack),
        "-i",
        str(solid_glb),
        "-o",
        str(reduced_glb),
        "-si",
        f"{ratio:.6f}",
        "-se",
        f"{simplify_error:.6f}",
    ]
    print(
        f"  reducing with gltfpack: {len(solid.faces)} -> ~{target_tris} tris "
        f"(ratio={ratio:.4f}, error={simplify_error})"
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"gltfpack failed (exit {proc.returncode})")

    reduced = load_mesh(reduced_glb)
    reduced = finalize_remesh_input(reduced, repair=True)
    print_stats("after gltfpack", reduced)
    return reduced


def prep_for_remesh(
    solid: trimesh.Trimesh,
    *,
    target_tris: int,
    simplify_error: float,
    gltfpack: Path,
    work_dir: Path,
    stem: str,
    repair: bool,
) -> trimesh.Trimesh:
    print("Step 2/2: Prepare solid mesh for AutoRemesher...")
    if target_tris <= 0 or len(solid.faces) <= target_tris:
        print(f"  using solid mesh as-is ({len(solid.faces)} faces)")
        return finalize_remesh_input(solid, repair=repair)
    return reduce_for_remesh(
        solid,
        target_tris=target_tris,
        simplify_error=simplify_error,
        gltfpack=gltfpack,
        work_dir=work_dir,
        stem=stem,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Solidify + AutoRemesher pipeline")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output GLB path (default: dataset/autoremesher_out/<name>_autoremesher.glb)",
    )
    parser.add_argument(
        "--autoremesher",
        type=Path,
        default=AUTOREMESHER,
        help="Path to autoremesher wrapper/binary",
    )
    parser.add_argument(
        "--target-quads",
        type=int,
        default=0,
        help="Target quad count for AutoRemesher (0 = auto from input complexity)",
    )
    parser.add_argument("--edge-scaling", type=float, default=1.0)
    parser.add_argument("--sharp-edge", type=float, default=90.0)
    parser.add_argument("--smooth-normal", type=float, default=0.0)
    parser.add_argument("--adaptivity", type=float, default=1.0)
    parser.add_argument(
        "--prep-target-tris",
        type=int,
        default=0,
        help="Reduce solid mesh before remesh (0 = use solid as-is, recommended)",
    )
    parser.add_argument(
        "--simplify-error",
        type=float,
        default=0.005,
        help="gltfpack simplification error limit (default: 0.005)",
    )
    parser.add_argument(
        "--min-component-faces",
        type=int,
        default=50,
        help="Drop disconnected shells smaller than this (default: 50)",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=0.002,
        help="Vertex merge distance during cleanup (default: 0.002)",
    )
    parser.add_argument(
        "--keep-largest-only",
        action="store_true",
        default=True,
        help="Keep only the largest shell before repair (default: on)",
    )
    parser.add_argument(
        "--keep-all-fragments",
        dest="keep_largest_only",
        action="store_false",
        help="Keep all shells above --min-component-faces (risky for pymeshfix)",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        default=True,
        help="Run pymeshfix after solidify (default: on)",
    )
    parser.add_argument(
        "--no-repair",
        dest="repair",
        action="store_false",
        help="Skip pymeshfix repair",
    )
    parser.add_argument(
        "--solid-output",
        type=Path,
        default=None,
        help="Path for intermediate solid GLB (default: autoremesher_out/<name>_solid.glb)",
    )
    parser.add_argument(
        "--solid-only",
        action="store_true",
        help="Stop after solidify step; do not run AutoRemesher",
    )
    parser.add_argument(
        "--from-solid",
        action="store_true",
        help="Input is already a solid mesh; skip solidify step",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        default=True,
        help="Drop tiny stray islands from AutoRemesher output (default: on)",
    )
    parser.add_argument(
        "--no-clean-output",
        dest="clean_output",
        action="store_false",
        help="Keep all components from AutoRemesher output",
    )
    parser.add_argument(
        "--keep-output-largest-only",
        action="store_true",
        default=True,
        help="Keep only the largest remeshed shell in the output (default: on)",
    )
    parser.add_argument(
        "--keep-all-output-fragments",
        dest="keep_output_largest_only",
        action="store_false",
        help="Keep all remeshed shells above --min-component-faces",
    )
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip both solidify and reduction; send input directly to AutoRemesher",
    )
    args = parser.parse_args()

    if not args.solid_only and not args.autoremesher.exists():
        raise FileNotFoundError(
            f"AutoRemesher not found at {args.autoremesher}. "
            "Build it with: cd autoremesher && qmake && make -j$(nproc)"
        )

    input_path = args.input.resolve()
    stem = input_path.stem.replace("_solid", "")
    if input_path.parent.name == "autoremesher_out":
        work_dir = input_path.parent
    else:
        work_dir = input_path.parent / "autoremesher_out"
    work_dir.mkdir(parents=True, exist_ok=True)

    solid_glb = args.solid_output or work_dir / f"{stem}_solid.glb"
    input_obj = work_dir / f"{stem}_input.obj"
    output_obj = work_dir / f"{stem}_autoremesher.obj"
    report_path = work_dir / f"{stem}_autoremesher_report.txt"
    output_glb = args.output or work_dir / f"{stem}_autoremesher.glb"

    print(f"Loading {input_path}...")
    mesh = load_mesh(input_path)
    print_stats("input", mesh)

    if args.skip_prep:
        prepared = mesh
    elif args.from_solid:
        print("Using input as pre-solidified mesh.")
        prepared = prep_for_remesh(
            mesh,
            target_tris=args.prep_target_tris,
            simplify_error=args.simplify_error,
            gltfpack=GLTFPACK,
            work_dir=work_dir,
            stem=stem,
            repair=args.repair,
        )
    else:
        solid = solidify_mesh(
            mesh,
            min_component_faces=args.min_component_faces,
            merge_distance=args.merge_distance,
            keep_largest_only=args.keep_largest_only,
            repair=args.repair,
        )
        solid_glb.parent.mkdir(parents=True, exist_ok=True)
        solid.export(solid_glb)
        print(f"  solid GLB: {solid_glb}")

        if args.solid_only:
            print("Done (--solid-only). Inspect the solid mesh before remeshing.")
            return 0

        prepared = prep_for_remesh(
            solid,
            target_tris=args.prep_target_tris,
            simplify_error=args.simplify_error,
            gltfpack=GLTFPACK,
            work_dir=work_dir,
            stem=stem,
            repair=args.repair,
        )

    target_quads = args.target_quads or auto_target_quads(len(prepared.faces))
    print(f"AutoRemesher target quads: {target_quads} (from {len(prepared.faces)} input tris)")
    print(f"Exporting OBJ -> {input_obj}")
    prepared.export(input_obj)

    cmd = [
        str(args.autoremesher),
        "--input",
        str(input_obj),
        "--output",
        str(output_obj),
        "--report",
        str(report_path),
        "--target-quads",
        str(target_quads),
        "--edge-scaling",
        str(args.edge_scaling),
        "--sharp-edge",
        str(args.sharp_edge),
        "--smooth-normal",
        str(args.smooth_normal),
        "--adaptivity",
        str(args.adaptivity),
    ]

    print("Running AutoRemesher:", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", f"/tmp/runtime-{os.environ.get('USER', 'unknown')}")
    try:
        subprocess.run(cmd, check=True, env=env, cwd=work_dir)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "AutoRemesher failed. Try --solid-only first to inspect the solid mesh, "
            "or adjust --prep-target-tris / --target-quads."
        ) from exc

    if report_path.exists():
        print(report_path.read_text().rstrip())

    print(f"Loading AutoRemesher output from {output_obj}")
    result = load_mesh(output_obj)
    print_stats("raw result", result)

    if args.clean_output:
        print("Cleaning AutoRemesher output...")
        result = cleanup_remesh_output(
            result,
            min_component_faces=args.min_component_faces,
            keep_largest_only=args.keep_output_largest_only,
        )
        print_stats("cleaned result", result)

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    result.export(output_glb)
    print("Done.")
    print(f"Solid: {solid_glb if not args.skip_prep and not args.from_solid else '(skipped)'}")
    print(f"OBJ:   {output_obj}")
    print(f"GLB:   {output_glb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
