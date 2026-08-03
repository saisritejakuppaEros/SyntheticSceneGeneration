"""Load and export polygon meshes (quads) without trimesh triangulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import SimpleMaterial
from trimesh.visual.texture import TextureVisuals


def count_obj_face_types(path: Path) -> dict[str, int]:
    quads = tris = other = 0
    with path.open() as handle:
        for line in handle:
            if not line.startswith("f "):
                continue
            n = len(line.split()) - 1
            if n == 4:
                quads += 1
            elif n == 3:
                tris += 1
            else:
                other += 1
    return {"quads": quads, "triangles": tris, "other": other}


def load_obj_polygons(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                face = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
                faces.append(face)
    if not vertices or not faces:
        raise RuntimeError(f"No mesh geometry in {path}")
    return np.asarray(vertices, dtype=np.float64), faces


def _uniform_face_width(faces: list[list[int]]) -> int | None:
    widths = {len(face) for face in faces}
    if len(widths) == 1:
        return widths.pop()
    return None


def faces_to_array(faces: list[list[int]]) -> np.ndarray:
    width = _uniform_face_width(faces)
    if width is None:
        return np.asarray(faces, dtype=object)
    return np.asarray(faces, dtype=np.int64)


def quad_face_stats(faces: np.ndarray | list[list[int]]) -> dict[str, int]:
    if isinstance(faces, list):
        quads = sum(1 for face in faces if len(face) == 4)
        tris = sum(1 for face in faces if len(face) == 3)
        return {"quads": quads, "triangles": tris, "faces": len(faces)}
    if faces.dtype == object:
        return quad_face_stats(faces.tolist())
    if faces.shape[1] == 4:
        return {"quads": len(faces), "triangles": 0, "faces": len(faces)}
    if faces.shape[1] == 3:
        return {"quads": 0, "triangles": len(faces), "faces": len(faces)}
    return {"quads": 0, "triangles": 0, "faces": len(faces)}


def merge_vertices_polygons(
    vertices: np.ndarray,
    faces: np.ndarray | list[list[int]],
    *,
    decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Weld vertices and remap faces without triangulating polygons."""
    rounded = np.round(vertices, decimals=decimals)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)

    if isinstance(faces, np.ndarray) and faces.dtype != object:
        remapped: np.ndarray | list = inverse[faces]
    else:
        face_list = faces.tolist() if isinstance(faces, np.ndarray) else list(faces)
        remapped = [[int(inverse[i]) for i in face] for face in face_list]
        remapped = faces_to_array(remapped)

    return unique, remapped


def align_vertices_to_reference(vertices: np.ndarray, reference: trimesh.Trimesh) -> np.ndarray:
    """Scale/translate vertices to match reference AABB."""
    src_min, src_max = vertices.min(axis=0), vertices.max(axis=0)
    ref_min, ref_max = reference.bounds
    src_size = np.maximum(src_max - src_min, 1e-8)
    ref_size = np.maximum(ref_max - ref_min, 1e-8)
    return (vertices - src_min) / src_size * ref_size + ref_min


def recover_quads_from_tris(faces: np.ndarray) -> np.ndarray:
    """Merge adjacent triangle pairs into quads (fixes trimesh triangulation)."""
    if faces.ndim != 2 or faces.shape[1] != 3:
        return faces

    from collections import defaultdict

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for v0, v1 in ((a, b), (b, c), (c, a)):
            edge_to_faces[tuple(sorted((v0, v1)))].append(fi)

    used: set[int] = set()
    quads: list[list[int]] = []
    orphan_tris: list[list[int]] = []

    for face_indices in edge_to_faces.values():
        if len(face_indices) != 2:
            continue
        f1, f2 = face_indices
        if f1 in used or f2 in used:
            continue
        verts = sorted(set(map(int, faces[f1])) | set(map(int, faces[f2])))
        if len(verts) != 4:
            continue

        # Order the four corners by walking shared edge then opposite verts
        shared = set(faces[f1]) & set(faces[f2])
        if len(shared) != 2:
            continue
        v0, v1 = sorted(shared)
        extra = [v for v in verts if v not in shared]
        if len(extra) != 2:
            continue
        # Walk f1 to get consistent winding
        tri = list(map(int, faces[f1]))
        if v0 in tri and v1 in tri:
            i0, i1 = tri.index(v0), tri.index(v1)
            if (i0 + 1) % 3 == i1:
                ordered = [v0, v1, extra[0], extra[1]]
            elif (i1 + 1) % 3 == i0:
                ordered = [v1, v0, extra[0], extra[1]]
            else:
                ordered = [v0, extra[0], v1, extra[1]]
        else:
            ordered = [v0, extra[0], v1, extra[1]]

        used.add(f1)
        used.add(f2)
        quads.append(ordered)

    for fi, face in enumerate(faces):
        if fi not in used:
            orphan_tris.append(list(map(int, face)))

    if not quads:
        return faces
    if not orphan_tris:
        return np.asarray(quads, dtype=np.int64)
    return np.asarray(quads + orphan_tris, dtype=object)


def ensure_quad_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return quad faces; recover from tris when trimesh already triangulated."""
    stats = quad_face_stats(faces)
    if stats["quads"] > 0 and stats["triangles"] == 0:
        return faces
    if stats["triangles"] > 0 and faces.ndim == 2 and faces.shape[1] == 3:
        recovered = recover_quads_from_tris(faces)
        if quad_face_stats(recovered)["quads"] > 0:
            return recovered
    return faces


def export_obj_mesh(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray | list[list[int]],
) -> None:
    """Write geometry-only OBJ preserving quad/tri faces (no UVs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for x, y, z in vertices:
            out.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        face_rows = faces.tolist() if isinstance(faces, np.ndarray) and faces.dtype != object else faces
        for face in face_rows:
            parts = " ".join(str(idx + 1) for idx in face)
            out.write(f"f {parts}\n")


def load_quad_target(path: Path) -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh | None]:
    """Load quad/polygon target; returns (vertices, faces, trimesh for bounds only)."""
    if path.suffix.lower() == ".obj":
        vertices, face_list = load_obj_polygons(path)
        faces = ensure_quad_faces(vertices, faces_to_array(face_list))
        reference = triangulated_copy(vertices, faces)
        return vertices, faces, reference

    reference = trimesh.load(path, force="mesh")
    if isinstance(reference, trimesh.Scene):
        parts = [g for g in reference.geometry.values() if isinstance(g, trimesh.Trimesh)]
        reference = trimesh.util.concatenate(parts) if parts else None
    if reference is None:
        raise RuntimeError(f"No mesh geometry in {path}")

    obj_sidecar = path.parent / "quadriflow_out" / f"{path.stem.replace('_quadriflow', '')}_quadriflow.obj"
    if not obj_sidecar.is_file():
        obj_sidecar = path.parent / "instant_meshes_out" / f"{path.stem}_instant.obj"
    if not obj_sidecar.is_file():
        work_dir = path.parent / "instant_meshes_out"
        if work_dir.is_dir():
            matches = sorted(work_dir.glob("*_instant.obj"), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                obj_sidecar = matches[0]
    if not obj_sidecar.is_file():
        work_dir = path.parent / "quadriflow_out"
        if work_dir.is_dir():
            matches = sorted(work_dir.glob("*_quadriflow.obj"), key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                obj_sidecar = matches[0]
    if obj_sidecar.is_file():
        vertices, face_list = load_obj_polygons(obj_sidecar)
        return vertices, faces_to_array(face_list), reference

    vertices = reference.vertices.copy()
    faces = ensure_quad_faces(vertices, reference.faces.copy())
    return vertices, faces, reference


def export_obj_with_uv(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray | list[list[int]],
    uvs: np.ndarray,
    *,
    texture_path: Path | None = None,
    vertex_colors: np.ndarray | None = None,
) -> None:
    """Write Wavefront OBJ with quad/tri faces and optional MTL texture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mtl_name = None
    if texture_path is not None:
        mtl_name = path.with_suffix(".mtl").name
        tex_name = texture_path.name
        mtl_path = path.with_suffix(".mtl")
        mtl_path.write_text(
            "\n".join([
                "newmtl material_0",
                "Ka 1.0 1.0 1.0",
                "Kd 1.0 1.0 1.0",
                "Ks 0.0 0.0 0.0",
                "d 1.0",
                f"map_Kd {tex_name}",
                "",
            ])
        )
        if texture_path.resolve().parent != path.parent.resolve():
            import shutil
            shutil.copy2(texture_path, path.parent / tex_name)

    with path.open("w") as out:
        if mtl_name:
            out.write(f"mtllib {mtl_name}\n")
        for x, y, z in vertices:
            out.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        if vertex_colors is not None:
            for rgba in vertex_colors:
                out.write(f"vc {rgba[0]/255:.6f} {rgba[1]/255:.6f} {rgba[2]/255:.6f}\n")
        for u, v in uvs:
            out.write(f"vt {u:.6f} {v:.6f}\n")
        if mtl_name:
            out.write("usemtl material_0\n")
        face_rows = faces.tolist() if isinstance(faces, np.ndarray) and faces.dtype != object else faces
        for face in face_rows:
            parts = " ".join(f"{idx + 1}/{idx + 1}" for idx in face)
            out.write(f"f {parts}\n")


def triangulated_copy(vertices: np.ndarray, faces: np.ndarray | list[list[int]]) -> trimesh.Trimesh:
    """Build a triangle mesh for GLB export."""
    if isinstance(faces, np.ndarray) and faces.dtype != object and faces.shape[1] == 3:
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    triangles: list[list[int]] = []
    face_rows = faces.tolist() if isinstance(faces, np.ndarray) else faces
    for face in face_rows:
        face = list(face)
        if len(face) == 3:
            triangles.append(face)
        elif len(face) == 4:
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        else:
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i + 1]])
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(triangles, dtype=np.int64), process=False)


def attach_texture(mesh: trimesh.Trimesh, uvs: np.ndarray, atlas: np.ndarray) -> trimesh.Trimesh:
    from PIL import Image

    material = SimpleMaterial(image=Image.fromarray(atlas, mode="RGB"))
    mesh.visual = TextureVisuals(uv=uvs, material=material)
    return mesh
