"""xatlas UV unwrap for remeshed / optimized meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import xatlas
from trimesh.visual import TextureVisuals


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"No mesh geometry in {path}")
        return trimesh.util.concatenate(parts)
    return loaded


def _to_triangles(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if mesh.faces.shape[1] == 3:
        return mesh
    tri = mesh.triangulate()
    if isinstance(tri, trimesh.Trimesh):
        return tri
    return trimesh.Trimesh(vertices=mesh.vertices, faces=tri, process=False)


def unwrap_uv(
    input_path: Path,
    output_path: Path,
    *,
    resolution: int = 2048,
    padding: int = 2,
) -> dict:
    mesh = load_mesh(input_path)
    mesh = _to_triangles(mesh)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    pack = xatlas.PackOptions()
    pack.resolution = int(resolution)
    pack.padding = int(padding)
    pack.bilinear = True
    atlas.generate(pack_options=pack)

    vmapping, indices, uvs = atlas[0]
    new_vertices = vertices[vmapping]
    result = trimesh.Trimesh(vertices=new_vertices, faces=indices, process=False)
    result.visual = TextureVisuals(uv=uvs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.export(output_path)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "vertices": len(result.vertices),
        "faces": len(result.faces),
        "atlas_resolution": resolution,
        "file_kb": round(output_path.stat().st_size / 1024, 1),
    }
