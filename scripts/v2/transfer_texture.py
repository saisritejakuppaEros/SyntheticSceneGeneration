#!/usr/bin/env python3
"""
Step 3 — Texture transfer.

Bake color/texture from the reduced source mesh onto the quad-flow target.

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate
  python scripts/v2/transfer_texture.py \\
    --source dataset/v2_out/rodin_reduced.glb \\
    --target dataset/v2_out/rodin_quadflow.glb \\
    -o dataset/v2_out/rodin_final.glb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from trimesh.triangles import closest_point as closest_point_on_triangles
from trimesh.triangles import points_to_barycentric
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from _paths import DATASET_DIR
from mesh_prep import align_to_reference, load_mesh


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
    def __init__(self, mesh: trimesh.Trimesh, img: np.ndarray, k: int = 8):
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


def box_project_uv(vertices: np.ndarray) -> np.ndarray:
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    size = np.maximum(bounds[1] - bounds[0], 1e-8)
    norm = (vertices - bounds[0]) / size
    return np.column_stack([norm[:, 0], norm[:, 2]])


def rasterize_uv_face(atlas: np.ndarray, mask: np.ndarray, uv_tri: np.ndarray, colors: np.ndarray) -> None:
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


def transfer_texture(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    *,
    mode: str = "texture",
    texture_size: int = 128,
) -> Path:
    source_raw = load_mesh(source_path)
    target = load_mesh(target_path)
    source = align_to_reference(source_raw.copy(), target)

    print(f"Source: {source_path}  ({len(source.vertices)} verts)")
    print(f"Target: {target_path}  ({len(target.vertices)} verts)")

    if mode == "vertex":
        sampler = SourceSampler(source, texture_image(source))
        rgba = sampler.sample(target.vertices)
        result = target.copy()
        result.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)
    else:
        sampler = SourceSampler(source, texture_image(source))
        uvs = box_project_uv(target.vertices)
        atlas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        mask = np.zeros((texture_size, texture_size), dtype=bool)
        for face in target.faces:
            uv_tri = uvs[face]
            tri_colors = sampler.sample(target.vertices[face])
            rasterize_uv_face(atlas, mask, uv_tri, tri_colors[:, :3])
        if not mask.any():
            raise RuntimeError("Texture bake produced an empty atlas")
        result = target.copy()
        material = PBRMaterial(baseColorTexture=Image.fromarray(atlas, mode="RGB"))
        result.visual = TextureVisuals(uv=uvs, material=material)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Scene(result).export(output_path, file_type="glb")
    print(f"Saved: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 texture transfer")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["vertex", "texture"], default="texture")
    parser.add_argument("--texture-size", type=int, default=128)
    args = parser.parse_args()

    transfer_texture(
        args.source.resolve(),
        args.target.resolve(),
        args.output.resolve(),
        mode=args.mode,
        texture_size=args.texture_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
