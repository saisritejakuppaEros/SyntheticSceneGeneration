"""Bake TRELLIS textures onto remeshed geometry; preserve quads via OBJ export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import xatlas
from PIL import Image
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point as closest_point_on_triangles
from trimesh.triangles import points_to_barycentric
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from services.quad_mesh_io import (
    attach_texture,
    export_obj_with_uv,
    load_quad_target,
    merge_vertices_polygons,
    quad_face_stats,
    triangulated_copy,
)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"No mesh geometry in {path}")
        return trimesh.util.concatenate(parts)
    return loaded


def align_to_reference(mesh: trimesh.Trimesh, reference: trimesh.Trimesh) -> trimesh.Trimesh:
    aligned = mesh.copy()
    src_min, src_max = aligned.bounds
    ref_min, ref_max = reference.bounds
    src_size = np.maximum(src_max - src_min, 1e-8)
    ref_size = np.maximum(ref_max - ref_min, 1e-8)
    aligned.vertices = (aligned.vertices - src_min) / src_size * ref_size + ref_min
    return aligned


def _face_triangles(face: np.ndarray) -> list[np.ndarray]:
    """Split a polygon face into triangles for rasterization."""
    if len(face) == 3:
        return [face]
    if len(face) == 4:
        return [face[[0, 1, 2]], face[[0, 2, 3]]]
    # fan triangulation for n-gons
    return [face[[0, i, i + 1]] for i in range(1, len(face) - 1)]


def box_project_uv(vertices: np.ndarray) -> np.ndarray:
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    size = np.maximum(bounds[1] - bounds[0], 1e-8)
    norm = (vertices - bounds[0]) / size
    return np.column_stack([norm[:, 0], norm[:, 2]])


def unwrap_with_box_projection(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Box-project UVs without triangulating or changing face topology."""
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    uvs = box_project_uv(mesh.vertices)
    mesh.visual = TextureVisuals(uv=uvs)
    return mesh


def _to_triangles(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if mesh.faces.shape[1] == 3:
        return mesh
    tri = mesh.triangulate()
    if isinstance(tri, trimesh.Trimesh):
        return tri
    return trimesh.Trimesh(vertices=mesh.vertices, faces=tri, process=False)


def unwrap_with_xatlas(mesh: trimesh.Trimesh, *, padding: int = 2) -> trimesh.Trimesh:
    mesh = _to_triangles(mesh)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)

    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    pack = xatlas.PackOptions()
    pack.padding = int(padding)
    pack.bilinear = True
    atlas.generate(pack_options=pack)

    vmapping, indices, uvs = atlas[0]
    result = trimesh.Trimesh(vertices=vertices[vmapping], faces=indices, process=False)
    result.visual = TextureVisuals(uv=uvs)
    return result


def texture_image(source: trimesh.Trimesh) -> np.ndarray:
    mat = source.visual.material
    if mat is None or mat.baseColorTexture is None:
        raise RuntimeError("Source mesh has no baseColorTexture")
    img = mat.baseColorTexture
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGBA"))
    return np.asarray(img)


def sample_image(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    u = np.clip(uv[:, 0], 0.0, 1.0) * (w - 1)
    v = (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (h - 1)

    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (u - x0)[..., None]
    wy = (v - y0)[..., None]

    c00 = img[y0, x0].astype(np.float32)
    c10 = img[y0, x1].astype(np.float32)
    c01 = img[y1, x0].astype(np.float32)
    c11 = img[y1, x1].astype(np.float32)
    top = c00 * (1 - wx) + c10 * wx
    bot = c01 * (1 - wx) + c11 * wx
    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


class SourceSampler:
    def __init__(self, mesh: trimesh.Trimesh, img: np.ndarray, k: int = 16):
        if mesh.visual.uv is None:
            raise RuntimeError("Source mesh has no UV coordinates")
        self.triangles = mesh.triangles
        self.faces = mesh.faces
        self.uvs = mesh.visual.uv
        self.img = img
        self.k = min(k, len(self.triangles))
        centroids = self.triangles.mean(axis=1)
        self.tree = cKDTree(centroids)

    def sample(self, points: np.ndarray, chunk: int = 4096) -> np.ndarray:
        colors = np.zeros((len(points), 4), dtype=np.uint8)
        for start in range(0, len(points), chunk):
            end = min(start + chunk, len(points))
            batch = points[start:end]
            _, tri_idx = self.tree.query(batch, k=self.k)
            if self.k == 1:
                tri_idx = tri_idx[:, None]

            best_dist = np.full(len(batch), np.inf)
            best_uv = np.zeros((len(batch), 2), dtype=np.float64)

            for j in range(self.k):
                cand_tris = self.triangles[tri_idx[:, j]]
                closest = closest_point_on_triangles(cand_tris, batch)
                dist = np.linalg.norm(closest - batch, axis=1)
                bary = points_to_barycentric(cand_tris, closest)
                better = dist < best_dist
                if not np.any(better):
                    continue
                face_idx = self.faces[tri_idx[better, j]]
                tri_uv = self.uvs[face_idx]
                interp_uv = (bary[better][..., None] * tri_uv).sum(axis=1)
                best_dist[better] = dist[better]
                best_uv[better] = interp_uv

            colors[start:end] = sample_image(self.img, best_uv)
        return colors


def _rasterize_uv_face(
    atlas: np.ndarray,
    mask: np.ndarray,
    uv_tri: np.ndarray,
    colors: np.ndarray,
) -> None:
    size = atlas.shape[0]
    uv_px = uv_tri.copy()
    uv_px[:, 0] *= size - 1
    uv_px[:, 1] = (1.0 - uv_px[:, 1]) * (size - 1)

    min_x = max(int(np.floor(uv_px[:, 0].min())), 0)
    max_x = min(int(np.ceil(uv_px[:, 0].max())), size - 1)
    min_y = max(int(np.floor(uv_px[:, 1].min())), 0)
    max_y = min(int(np.ceil(uv_px[:, 1].max())), size - 1)
    if min_x > max_x or min_y > max_y:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float64)
    ys = np.arange(min_y, max_y + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    v0, v1, v2 = uv_px
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = pts - v0
    dot00 = np.dot(v0v2, v0v2)
    dot01 = np.dot(v0v2, v0v1)
    dot02 = (v0v2 * v0p).sum(axis=1)
    dot11 = np.dot(v0v1, v0v1)
    dot12 = (v0v1 * v0p).sum(axis=1)
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    if not np.any(inside):
        return

    bary = np.column_stack([1 - u - v, v, u])[inside]
    rgb = (bary @ colors.astype(np.float64)).astype(np.uint8)
    px = pts[inside].astype(int)
    atlas[px[:, 1], px[:, 0]] = rgb
    mask[px[:, 1], px[:, 0]] = True


def _fill_atlas_gaps(atlas: np.ndarray, mask: np.ndarray, passes: int = 4) -> None:
    """Dilate baked colors into empty texels to hide small UV padding gaps."""
    for _ in range(passes):
        empty = ~mask
        if not empty.any():
            break
        filled = atlas.copy()
        filled_mask = mask.copy()
        h, w = mask.shape
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
            nx = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
            neighbor_mask = mask[ny, nx]
            write = empty & neighbor_mask
            if not write.any():
                continue
            filled[write] = atlas[ny, nx][write]
            filled_mask[write] = True
        atlas[:] = filled
        mask[:] = filled_mask


def bake_texture(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    *,
    target_obj_path: Path | None = None,
    texture_size: int = 512,
    uv_padding: int = 4,
    sample_k: int = 16,
    mode: str = "texture",
    uv_method: str = "box",
    output_format: str = "both",
) -> dict:
    source_raw = load_mesh(source_path)
    quad_source = target_obj_path if target_obj_path and target_obj_path.is_file() else target_path
    vertices, faces, target_ref = load_quad_target(quad_source)
    work_vertices, work_faces = merge_vertices_polygons(vertices, faces)
    face_stats = quad_face_stats(work_faces)

    if uv_method == "box":
        uvs = box_project_uv(work_vertices)
    else:
        tri_mesh = unwrap_with_xatlas(triangulated_copy(work_vertices, work_faces), padding=uv_padding)
        work_vertices = tri_mesh.vertices
        work_faces = tri_mesh.faces
        uvs = tri_mesh.visual.uv
        face_stats = quad_face_stats(work_faces)

    source = align_to_reference(
        source_raw.copy(),
        target_ref or triangulated_copy(work_vertices, work_faces),
    )

    output_obj = output_path.with_suffix(".obj")
    output_glb = output_path if output_path.suffix.lower() == ".glb" else output_path.with_suffix(".glb")
    texture_png = output_path.with_suffix(".png")

    if mode == "vertex":
        sampler = SourceSampler(source, texture_image(source), k=sample_k)
        rgba = sampler.sample(work_vertices)
        export_obj_with_uv(
            output_obj,
            work_vertices,
            work_faces,
            uvs,
            vertex_colors=rgba,
        )
        result_glb = None
        if output_format in {"glb", "both"}:
            result_glb = triangulated_copy(work_vertices, work_faces)
            result_glb.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)
            trimesh.Scene(result_glb).export(output_glb, file_type="glb")
    else:
        sampler = SourceSampler(source, texture_image(source), k=sample_k)
        atlas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        mask = np.zeros((texture_size, texture_size), dtype=bool)

        for face in work_faces:
            face_arr = np.asarray(face, dtype=np.int64)
            for tri in _face_triangles(face_arr):
                uv_tri = uvs[tri]
                tri_colors = sampler.sample(work_vertices[tri])
                _rasterize_uv_face(atlas, mask, uv_tri, tri_colors[:, :3])

        if not mask.any():
            raise RuntimeError("Texture bake produced an empty atlas")

        _fill_atlas_gaps(atlas, mask)
        Image.fromarray(atlas, mode="RGB").save(texture_png)
        export_obj_with_uv(
            output_obj,
            work_vertices,
            work_faces,
            uvs,
            texture_path=texture_png,
        )

        result_glb = None
        if output_format in {"glb", "both"}:
            result_glb = attach_texture(triangulated_copy(work_vertices, work_faces), uvs, atlas)
            trimesh.Scene(result_glb).export(output_glb, file_type="glb")

    primary = output_obj if output_format in {"obj", "both"} else output_glb
    primary.parent.mkdir(parents=True, exist_ok=True)

    return {
        "source": str(source_path),
        "target": str(target_path),
        "target_obj": str(target_obj_path) if target_obj_path else None,
        "output_obj": str(output_obj) if output_obj.is_file() else None,
        "output_glb": str(output_glb) if output_glb.is_file() else None,
        "output": str(primary),
        "vertices": len(work_vertices),
        "faces": face_stats.get("faces", len(work_faces)),
        "quads": quad_face_stats(work_faces).get("quads", face_stats.get("quads", 0)),
        "triangles": quad_face_stats(work_faces).get("triangles", face_stats.get("triangles", 0)),
        "texture_size": texture_size,
        "uv_method": uv_method,
        "output_format": output_format,
        "file_kb": round(primary.stat().st_size / 1024, 1) if primary.is_file() else 0,
    }
