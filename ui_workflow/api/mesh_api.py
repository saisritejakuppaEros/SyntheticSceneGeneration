#!/usr/bin/env python3
"""FastAPI server for mesh processing (.venv): autoremesher, meshoptimizer, xatlas."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

UI_ROOT = Path(__file__).resolve().parents[1]
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from api.common import (
    JOBS_ROOT,
    MAX_TARGET_QUADS,
    job_dir,
    mesh_stats,
    new_job_id,
    resolve_autoremesher_output,
    save_upload,
    utc_now,
    write_manifest,
)
from services.mesh_tools import run_autoremesher, run_meshoptimizer
from services.transfer_texture import bake_texture
from services.xatlas_uv import unwrap_uv

app = FastAPI(title="Mesh Processing API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok", "service": "mesh", "time": utc_now()}


def _job_file(job_id: str, filename: str) -> Path:
    path = JOBS_ROOT / job_id / filename
    if not path.is_file():
        raise HTTPException(404, f"{filename} not found for job {job_id}")
    return path


def _fail(proc, label: str) -> None:
    detail = (proc.stderr or proc.stdout or f"{label} failed").strip()
    raise HTTPException(500, detail)


@app.post("/v1/autoremesher")
async def autoremesher(
    mesh: UploadFile = File(...),
    target_quads: int = Form(5000),
    prep_target_tris: int = Form(0),
    solid_only: bool = Form(False),
    from_solid: bool = Form(False),
) -> dict:
    if target_quads < 1 or target_quads > MAX_TARGET_QUADS:
        raise HTTPException(
            400,
            f"target_quads must be between 1 and {MAX_TARGET_QUADS:,}",
        )
    job_id = new_job_id("autoremesh_")
    out_dir = job_dir(job_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(mesh.filename or "input.glb").suffix or ".glb"
    input_path = out_dir / f"input_{timestamp}{suffix}"
    output_path = out_dir / f"output_{timestamp}.glb"

    await save_upload(mesh, input_path)
    proc = run_autoremesher(
        input_path,
        output_path,
        target_quads=target_quads,
        prep_target_tris=prep_target_tris,
        solid_only=solid_only,
        from_solid=from_solid,
    )
    if proc.returncode != 0:
        _fail(proc, "AutoRemesher")

    try:
        output_path = resolve_autoremesher_output(
            input_path,
            output_path,
            solid_only=solid_only,
        )
    except FileNotFoundError as exc:
        log = (proc.stdout or "") + (proc.stderr or "")
        raise HTTPException(500, f"{exc}\n\n{log.strip()}") from exc

    payload = {
        "job_id": job_id,
        "status": "completed",
        "input": str(input_path),
        "output_glb": str(output_path),
        "solid_only": solid_only,
        "from_solid": from_solid,
        "target_quads": target_quads,
        "stats": mesh_stats(output_path),
        "log": (proc.stdout or "")[-4000:],
        "time": utc_now(),
    }
    write_manifest(job_id, payload)
    return payload


@app.post("/v1/meshoptimizer")
async def meshoptimizer(
    mesh: UploadFile = File(...),
    target_tris: int = Form(30000),
    simplify_error: float = Form(0.01),
) -> dict:
    job_id = new_job_id("meshopt_")
    out_dir = job_dir(job_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(mesh.filename or "input.glb").suffix or ".glb"
    input_path = out_dir / f"input_{timestamp}{suffix}"
    output_path = out_dir / f"output_{timestamp}.glb"

    await save_upload(mesh, input_path)
    proc = run_meshoptimizer(
        input_path,
        output_path,
        target_tris=target_tris,
        simplify_error=simplify_error,
    )
    if proc.returncode != 0:
        _fail(proc, "meshoptimizer")

    payload = {
        "job_id": job_id,
        "status": "completed",
        "input": str(input_path),
        "output_glb": str(output_path),
        "target_tris": target_tris,
        "stats": mesh_stats(output_path),
        "log": (proc.stdout or "")[-4000:],
        "time": utc_now(),
    }
    write_manifest(job_id, payload)
    return payload


@app.post("/v1/transfer-texture")
async def transfer_texture(
    source: UploadFile = File(..., description="Textured source mesh (e.g. TRELLIS GLB)"),
    target: UploadFile = File(..., description="Remeshed target mesh"),
    texture_size: int = Form(512),
    uv_padding: int = Form(4),
    mode: str = Form("texture"),
) -> dict:
    if mode not in {"texture", "vertex"}:
        raise HTTPException(400, "mode must be 'texture' or 'vertex'")
    if texture_size < 32 or texture_size > 4096:
        raise HTTPException(400, "texture_size must be between 32 and 4096")

    job_id = new_job_id("texture_")
    out_dir = job_dir(job_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_path = out_dir / f"source_{timestamp}.glb"
    target_path = out_dir / f"target_{timestamp}.glb"
    output_path = out_dir / f"output_{timestamp}.glb"

    await save_upload(source, source_path)
    await save_upload(target, target_path)

    try:
        bake_stats = bake_texture(
            source_path,
            target_path,
            output_path,
            texture_size=texture_size,
            uv_padding=uv_padding,
            mode=mode,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    payload = {
        "job_id": job_id,
        "status": "completed",
        "source": str(source_path),
        "target": str(target_path),
        "output_glb": str(output_path),
        "texture_size": texture_size,
        "uv_padding": uv_padding,
        "uv_method": "xatlas",
        "mode": mode,
        "stats": mesh_stats(output_path),
        "bake_stats": bake_stats,
        "time": utc_now(),
    }
    write_manifest(job_id, payload)
    return payload


@app.post("/v1/xatlas")
async def xatlas_uv(
    mesh: UploadFile = File(...),
    resolution: int = Form(2048),
    padding: int = Form(2),
) -> dict:
    job_id = new_job_id("xatlas_")
    out_dir = job_dir(job_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = Path(mesh.filename or "input.glb").suffix or ".glb"
    input_path = out_dir / f"input_{timestamp}{suffix}"
    output_path = out_dir / f"output_{timestamp}.glb"

    await save_upload(mesh, input_path)
    try:
        stats = unwrap_uv(input_path, output_path, resolution=resolution, padding=padding)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    payload = {
        "job_id": job_id,
        "status": "completed",
        "input": str(input_path),
        "output_glb": str(output_path),
        "resolution": resolution,
        "padding": padding,
        "stats": stats,
        "time": utc_now(),
    }
    write_manifest(job_id, payload)
    return payload


@app.get("/v1/jobs/{job_id}/download")
def download_output(job_id: str):
    manifest_path = JOBS_ROOT / job_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Job not found")
    import json

    manifest = json.loads(manifest_path.read_text())
    output = manifest.get("output_glb") or manifest.get("glb")
    if not output:
        raise HTTPException(404, "No output file recorded")
    path = Path(output)
    if not path.is_file():
        raise HTTPException(404, "Output file missing")
    return FileResponse(path, media_type="model/gltf-binary", filename=path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.mesh_api:app",
        host=os.environ.get("MESH_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("MESH_API_PORT", "8101")),
        reload=False,
    )
