#!/usr/bin/env python3
"""Convert GLB -> OBJ, run Instant Meshes (field-aligned quads), export quad OBJ."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import trimesh

from _paths import ROOT_DIR
from mesh_prep import load_mesh, mesh_stats, weld_quantized_mesh

UI_ROOT = ROOT_DIR / "ui_workflow"
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from services.quad_mesh_io import (  # noqa: E402
    align_vertices_to_reference,
    count_obj_face_types,
    export_obj_mesh,
    load_obj_polygons,
    merge_vertices_polygons,
    quad_face_stats,
    triangulated_copy,
)

DEFAULT_BIN = ROOT_DIR / "instant-meshes" / "build" / "Instant Meshes"


def _np_extents(vertices):
    return vertices.max(axis=0) - vertices.min(axis=0)


def find_instant_meshes_bin(custom: Path | None = None) -> Path:
    if custom and custom.is_file():
        return custom
    candidates = [
        DEFAULT_BIN,
        ROOT_DIR / "instant-meshes" / "Instant Meshes",
        Path("/usr/local/bin/Instant Meshes"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return DEFAULT_BIN


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Instant Meshes on a triangle mesh.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "-f",
        "--faces",
        type=int,
        default=5000,
        help="Target quad face count (Instant Meshes -f)",
    )
    parser.add_argument(
        "--instant-meshes",
        type=Path,
        default=None,
        help="Path to Instant Meshes binary",
    )
    parser.add_argument(
        "--from-meshopt",
        action="store_true",
        help="Weld gltfpack vertices before remeshing",
    )
    parser.add_argument(
        "--dominant",
        action="store_true",
        help="Quad-dominant instead of pure-quad output (-D)",
    )
    parser.add_argument(
        "--boundaries",
        action="store_true",
        default=True,
        help="Align field to open mesh boundaries (-b, default on)",
    )
    parser.add_argument(
        "--no-boundaries",
        dest="boundaries",
        action="store_false",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--rosy", type=int, default=4, choices=[2, 4, 6])
    parser.add_argument("--posy", type=int, default=4, choices=[4, 6])
    args = parser.parse_args()

    instant_bin = find_instant_meshes_bin(args.instant_meshes)
    if not instant_bin.is_file():
        raise FileNotFoundError(
            f"Instant Meshes binary not found at {instant_bin}. "
            "Build with: cd instant-meshes && mkdir -p build && cd build && cmake .. && make -j$(nproc)"
        )

    input_path = args.input.resolve()
    stem = input_path.stem
    work_dir = input_path.parent / "instant_meshes_out"
    work_dir.mkdir(parents=True, exist_ok=True)

    input_obj = work_dir / f"{stem}_input.obj"
    output_obj = work_dir / f"{stem}_instant.obj"
    output_glb = args.output or work_dir / f"{stem}_instant.glb"

    print(f"Loading {input_path}...")
    mesh = load_mesh(input_path)
    reference = mesh.copy()
    print(f"  vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, watertight={mesh.is_watertight}")

    if args.from_meshopt:
        mesh = weld_quantized_mesh(mesh)
        reference = mesh.copy()
        s = mesh_stats(mesh)
        print(f"  welded: vertices={s['vertices']}, faces={s['faces']}, components={s['components']}")

    print(f"Exporting triangle OBJ -> {input_obj}")
    mesh.export(input_obj)

    cmd = [
        str(instant_bin),
        "-o",
        str(output_obj),
        "-f",
        str(args.faces),
        "-r",
        str(args.rosy),
        "-p",
        str(args.posy),
    ]
    if args.boundaries:
        cmd.append("-b")
    if args.dominant:
        cmd.append("-D")
    if args.deterministic:
        cmd.append("-d")
    cmd.append(str(input_obj))

    print("Running Instant Meshes:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"Loading Instant Meshes output from {output_obj}")
    vertices, face_list = load_obj_polygons(output_obj)
    before = quad_face_stats(face_list)
    print(f"  raw OBJ: {before['quads']} quads, {before['triangles']} tris, {len(vertices)} verts")

    if args.from_meshopt or _np_extents(vertices).max() > reference.extents.max() * 1.5:
        before_ext = _np_extents(vertices)
        vertices = align_vertices_to_reference(vertices, reference)
        print(f"  realigned scale: {before_ext} -> {_np_extents(vertices)}")

    vertices, faces = merge_vertices_polygons(vertices, face_list)
    export_obj_mesh(output_obj, vertices, faces)
    after = count_obj_face_types(output_obj)
    print(f"  final OBJ: {after['quads']} quads, {after['triangles']} tris, {len(vertices)} verts")

    print(f"Exporting GLB preview -> {output_glb}")
    tri_preview = triangulated_copy(vertices, faces)
    tri_preview.export(output_glb)
    print("Done.")
    print(f"OBJ (quads): {output_obj}")
    print(f"GLB (preview): {output_glb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
