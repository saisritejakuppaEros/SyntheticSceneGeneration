"""Mesh loading, stats, and fragment cleanup for production assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path | str) -> trimesh.Trimesh:
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


def component_face_counts(mesh: trimesh.Trimesh) -> list[int]:
    return sorted((len(p.faces) for p in mesh.split(only_watertight=False)), reverse=True)


def weld_quantized_mesh(mesh: trimesh.Trimesh, decimals: int = 4) -> trimesh.Trimesh:
    """Merge gltfpack quantized vertices so downstream tools see fewer fragments."""
    welded = mesh.copy()
    welded.vertices = np.round(welded.vertices, decimals=decimals)
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    welded.update_faces(welded.nondegenerate_faces())
    welded.remove_unreferenced_vertices()
    return welded


def align_to_reference(mesh: trimesh.Trimesh, reference: trimesh.Trimesh) -> trimesh.Trimesh:
    aligned = mesh.copy()
    src_min, src_max = aligned.bounds
    ref_min, ref_max = reference.bounds
    src_size = np.maximum(src_max - src_min, 1e-8)
    ref_size = np.maximum(ref_max - ref_min, 1e-8)
    aligned.vertices = (aligned.vertices - src_min) / src_size * ref_size + ref_min
    return aligned


def cleanup_fragments(
    mesh: trimesh.Trimesh,
    *,
    min_component_faces: int = 50,
    merge_distance: float = 0.002,
    target_tris: int | None = None,
    face_budget_ratio: float = 0.95,
) -> trimesh.Trimesh:
    """
    Reduce disconnected shell count without dropping major surface area.

    Maps to the manual-cleanup step in a Houdini + ZRemesher pipeline:
    drop tiny floating fragments, weld coincident verts, optionally decimate.
    """
    cleaned = mesh.copy()
    cleaned.merge_vertices(merge_tex=True, merge_norm=True)
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()

    sizes = component_face_counts(cleaned)
    print(
        f"  fragments before cleanup: {len(sizes)}, "
        f"largest={sizes[0] if sizes else 0}, total_faces={len(cleaned.faces)}"
    )

    parts = cleaned.split(only_watertight=False)
    kept = [p for p in parts if len(p.faces) >= min_component_faces]

    if not kept:
        kept = [max(parts, key=lambda p: len(p.faces))]
        print(f"  all fragments below threshold; kept largest ({len(kept[0].faces)} faces)")
    else:
        print(f"  kept {len(kept)}/{len(parts)} fragments (min_faces={min_component_faces})")

    cleaned = trimesh.util.concatenate(kept)
    cleaned.merge_vertices(merge_tex=True, merge_norm=True)
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()

    if merge_distance > 0:
        cleaned.merge_vertices(merge_tex=True, merge_norm=True)
        cleaned.update_faces(cleaned.nondegenerate_faces())
        cleaned.remove_unreferenced_vertices()

    if target_tris and len(cleaned.faces) > target_tris:
        budget = int(len(cleaned.faces) * face_budget_ratio)
        if budget > target_tris:
            cleaned = cleaned.simplify_quadric_decimation(face_count=target_tris)
            print(f"  decimated to {len(cleaned.faces)} faces")

    if not cleaned.is_winding_consistent:
        trimesh.repair.fix_normals(cleaned)

    after = mesh_stats(cleaned)
    print(
        f"  after cleanup: {after['vertices']} verts, {after['faces']} faces, "
        f"{after['components']} components"
    )
    return cleaned
