"""Shared helpers for ui_workflow API servers."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import UploadFile

UI_ROOT = Path(__file__).resolve().parents[1]
MAX_TARGET_QUADS = 100_000  # 1 lakh upper limit
JOBS_ROOT = UI_ROOT / "jobs"
MESH_ROOT = UI_ROOT.parent
SCRIPTS_DIR = MESH_ROOT / "scripts"
VENV_PYTHON = MESH_ROOT / ".venv" / "bin" / "python"
QUADRIFLOW_BIN = MESH_ROOT / "QuadriFlow" / "build" / "quadriflow"
INSTANT_MESHES_BIN = MESH_ROOT / "instant-meshes" / "build" / "Instant Meshes"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id(prefix: str = "") -> str:
    token = uuid.uuid4().hex[:12]
    return f"{prefix}{token}" if prefix else token


def job_dir(job_id: str) -> Path:
    path = JOBS_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


async def save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return dest


def write_manifest(job_id: str, payload: dict[str, Any]) -> Path:
    manifest = job_dir(job_id) / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2))
    return manifest


def mesh_stats(path: Path) -> dict[str, Any]:
    import trimesh

    if not path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        parts = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(parts) if parts else None
    if mesh is None:
        return {"vertices": 0, "faces": 0, "file_kb": round(path.stat().st_size / 1024, 1)}
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "file_kb": round(path.stat().st_size / 1024, 1),
    }


def resolve_autoremesher_output(
    input_path: Path,
    output_path: Path,
    *,
    solid_only: bool,
) -> Path:
    """Ensure API output path exists; autoremesher may write under autoremesher_out/."""
    if output_path.is_file():
        return output_path

    work_dir = input_path.parent / "autoremesher_out"
    stem = input_path.stem.replace("_solid", "")

    candidates: list[Path] = []
    if solid_only:
        candidates.extend([
            work_dir / f"{stem}_solid.glb",
            input_path.parent / f"{stem}_solid.glb",
        ])
    candidates.extend([
        work_dir / f"{stem}_autoremesher.glb",
        output_path,
    ])

    for candidate in candidates:
        if candidate.is_file():
            if candidate.resolve() != output_path.resolve():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, output_path)
            return output_path

    raise FileNotFoundError(
        f"AutoRemesher produced no output at {output_path}. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def resolve_quadriflow_output(input_path: Path, output_path: Path) -> tuple[Path, Path | None]:
    """Ensure API output GLB exists; quadriflow writes under quadriflow_out/."""
    work_dir = input_path.parent / "quadriflow_out"
    stem = input_path.stem
    obj_path = work_dir / f"{stem}_quadriflow.obj"

    if not output_path.is_file():
        candidates = [
            work_dir / f"{stem}_quadriflow.glb",
            output_path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if candidate.resolve() != output_path.resolve():
                    shutil.copy2(candidate, output_path)
                break
        else:
            raise FileNotFoundError(
                f"QuadriFlow produced no output at {output_path}. "
                f"Checked: {[str(c) for c in candidates]}"
            )

    output_obj = output_path.with_suffix(".obj")
    if obj_path.is_file():
        if output_obj.resolve() != obj_path.resolve():
            shutil.copy2(obj_path, output_obj)
        return output_path, output_obj

    return output_path, None


def resolve_instant_meshes_output(input_path: Path, output_path: Path) -> tuple[Path, Path | None]:
    """Ensure API output GLB exists; Instant Meshes writes under instant_meshes_out/."""
    work_dir = input_path.parent / "instant_meshes_out"
    stem = input_path.stem
    obj_path = work_dir / f"{stem}_instant.obj"

    if not output_path.is_file():
        candidates = [
            work_dir / f"{stem}_instant.glb",
            output_path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if candidate.resolve() != output_path.resolve():
                    shutil.copy2(candidate, output_path)
                break
        else:
            raise FileNotFoundError(
                f"Instant Meshes produced no output at {output_path}. "
                f"Checked: {[str(c) for c in candidates]}"
            )

    output_obj = output_path.with_suffix(".obj")
    if obj_path.is_file():
        if output_obj.resolve() != obj_path.resolve():
            shutil.copy2(obj_path, output_obj)
        return output_path, output_obj

    return output_path, None


def count_obj_quads(path: Path) -> dict[str, int]:
    """Count quad vs triangle faces in an OBJ (trimesh GLB loads triangulate quads)."""
    quads = tris = 0
    with path.open() as handle:
        for line in handle:
            if not line.startswith("f "):
                continue
            verts = len(line.split()) - 1
            if verts == 4:
                quads += 1
            elif verts == 3:
                tris += 1
    return {"quads": quads, "triangles": tris}
