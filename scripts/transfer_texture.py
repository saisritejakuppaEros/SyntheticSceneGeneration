#!/usr/bin/env python3
"""Transfer color/texture from a textured source mesh onto a remeshed target."""

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

DEFAULT_SOURCE = DATASET_DIR / "sample_2026-07-27T084001.382.glb"
DEFAULT_TARGET = DATASET_DIR / "quadwild_ablation" / "tris30000_scale1p20.glb"
DEFAULT_OUTPUT = DATASET_DIR / "quadwild_ablation" / "tris30000_scale1p20_textured.glb"


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"No mesh geometry in {path}")
        return trimesh.util.concatenate(parts)
    return loaded


def align_bbox(source: trimesh.Trimesh, target: trimesh.Trimesh) -> trimesh.Trimesh:
    """Map source AABB to target AABB (scale + translate)."""
    aligned = source.copy()
    src_min, src_max = aligned.bounds
    tgt_min, tgt_max = target.bounds
    src_size = np.maximum(src_max - src_min, 1e-8)
    tgt_size = np.maximum(tgt_max - tgt_min, 1e-8)
    aligned.vertices = (aligned.vertices - src_min) / src_size * tgt_size + tgt_min
    return aligned


def texture_image(source: trimesh.Trimesh) -> np.ndarray:
    mat = source.visual.material
    if mat is None or mat.baseColorTexture is None:
        raise RuntimeError("Source mesh has no baseColorTexture")
    img = mat.baseColorTexture
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGBA"))
    return np.asarray(img)


def sample_image(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample RGBA image with UV in [0, 1]."""
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
    """Closest-point sampler on a textured mesh (scipy cKDTree, no rtree)."""

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


def build_sampler(source: trimesh.Trimesh) -> SourceSampler:
    return SourceSampler(source, texture_image(source))


def transfer_vertex_colors(source: trimesh.Trimesh, target: trimesh.Trimesh) -> trimesh.Trimesh:
    sampler = build_sampler(source)
    rgba = sampler.sample(target.vertices)
    out = target.copy()
    out.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)
    return out


def box_project_uv(vertices: np.ndarray) -> np.ndarray:
    """Simple XZ box projection into [0, 1]."""
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    size = np.maximum(bounds[1] - bounds[0], 1e-8)
    norm = (vertices - bounds[0]) / size
    return np.column_stack([norm[:, 0], norm[:, 2]])


def rasterize_uv_face(
    atlas: np.ndarray,
    mask: np.ndarray,
    uv_tri: np.ndarray,
    colors: np.ndarray,
) -> None:
    """Rasterize one UV triangle into a low-res atlas."""
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


def export_embedded_glb(mesh: trimesh.Trimesh, path: Path) -> None:
    """Export a single self-contained GLB (texture/colors embedded, no sidecar files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene(mesh)
    scene.export(path, file_type="glb")


def transfer_lowres_texture(
    source: trimesh.Trimesh,
    target: trimesh.Trimesh,
    texture_size: int,
) -> trimesh.Trimesh:
    sampler = build_sampler(source)
    uvs = box_project_uv(target.vertices)
    atlas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    mask = np.zeros((texture_size, texture_size), dtype=bool)

    for face in target.faces:
        uv_tri = uvs[face]
        tri_colors = sampler.sample(target.vertices[face])
        rasterize_uv_face(atlas, mask, uv_tri, tri_colors[:, :3])

    if not mask.any():
        raise RuntimeError("Texture bake produced an empty atlas")

    out = target.copy()
    material = PBRMaterial(baseColorTexture=Image.fromarray(atlas, mode="RGB"))
    out.visual = TextureVisuals(uv=uvs, material=material)
    return out


def batch_targets(ablation_dir: Path) -> list[Path]:
    skip = {"textured", "tex", "embedded", "viewer"}
    out = []
    for path in sorted(ablation_dir.glob("tris*_scale*.glb")):
        stem = path.stem.lower()
        if any(tag in stem for tag in skip):
            continue
        out.append(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Transfer texture/color onto remeshed mesh")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Textured source GLB")
    parser.add_argument("--target", type=Path, default=None, help="Remeshed target GLB/OBJ")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="Process all ablation GLBs in this folder; writes <name>_embedded.glb",
    )
    parser.add_argument(
        "--mode",
        choices=["vertex", "texture"],
        default="texture",
        help="texture = embedded UV atlas in GLB (best for online viewers), vertex = COLOR_0 only",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=128,
        help="Atlas resolution for --mode texture (embedded inside GLB)",
    )
    parser.add_argument(
        "--save-texture",
        type=Path,
        default=None,
        help="Optional debug PNG export (not needed for viewing; GLB is self-contained)",
    )
    args = parser.parse_args()

    print(f"Loading source {args.source}...")
    source_raw = load_mesh(args.source)
    print(f"  vertices={len(source_raw.vertices)}, faces={len(source_raw.faces)}")

    targets: list[tuple[Path, Path]] = []
    if args.batch_dir:
        ablation_dir = args.batch_dir.resolve()
        for target_path in batch_targets(ablation_dir):
            out_path = ablation_dir / f"{target_path.stem}_embedded.glb"
            targets.append((target_path, out_path))
        if not targets:
            raise SystemExit(f"No ablation GLBs found in {ablation_dir}")
    else:
        target_path = (args.target or DEFAULT_TARGET).resolve()
        out_path = (args.output or DEFAULT_OUTPUT).resolve()
        targets = [(target_path, out_path)]

    for target_path, out_path in targets:
        print(f"\nLoading target {target_path}...")
        target = load_mesh(target_path)
        print(f"  vertices={len(target.vertices)}, faces={len(target.faces)}")

        source = align_bbox(source_raw.copy(), target)
        print("  aligned source to target bounds")

        if args.mode == "vertex":
            print("  transferring vertex colors (embedded in GLB)...")
            result = transfer_vertex_colors(source, target)
        else:
            print(f"  baking {args.texture_size}x{args.texture_size} embedded texture...")
            result = transfer_lowres_texture(source, target, args.texture_size)

        export_embedded_glb(result, out_path)
        print(f"  -> {out_path}")

        if args.save_texture and args.mode == "texture":
            img = result.visual.material.baseColorTexture
            args.save_texture.parent.mkdir(parents=True, exist_ok=True)
            img.save(args.save_texture)
            print(f"  (debug PNG: {args.save_texture})")

    print("\nDone. Upload any *_embedded.glb directly — no separate PNG needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
