"""Shared mesh prep helpers for gltfpack / meshoptimizer outputs."""

from __future__ import annotations

import numpy as np
import trimesh


def load_mesh(path) -> trimesh.Trimesh:
    from pathlib import Path

    loaded = trimesh.load(Path(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"No mesh geometry in {path}")
        return trimesh.util.concatenate(parts)
    return loaded


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    parts = mesh.split(only_watertight=False)
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "components": len(parts),
        "largest_component_faces": max(len(p.faces) for p in parts) if parts else 0,
        "extents": mesh.extents.tolist(),
        "watertight": mesh.is_watertight,
    }


def weld_quantized_mesh(mesh: trimesh.Trimesh, decimals: int = 4) -> trimesh.Trimesh:
    """Merge gltfpack quantized vertices so downstream tools see fewer fragments."""
    welded = mesh.copy()
    welded.vertices = np.round(welded.vertices, decimals=decimals)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    welded.update_faces(welded.nondegenerate_faces())
    welded.remove_unreferenced_vertices()
    return welded


def align_to_reference(mesh: trimesh.Trimesh, reference: trimesh.Trimesh) -> trimesh.Trimesh:
    """Map mesh AABB to reference AABB (fixes QuadriFlow scale drift)."""
    aligned = mesh.copy()
    src_min, src_max = aligned.bounds
    ref_min, ref_max = reference.bounds
    src_size = np.maximum(src_max - src_min, 1e-8)
    ref_size = np.maximum(ref_max - ref_min, 1e-8)
    aligned.vertices = (aligned.vertices - src_min) / src_size * ref_size + ref_min
    return aligned
