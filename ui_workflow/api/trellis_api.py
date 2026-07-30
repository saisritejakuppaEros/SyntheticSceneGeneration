#!/usr/bin/env python3
"""FastAPI server for TRELLIS.2 image-to-3D (runs in TRELLIS conda env)."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

UI_ROOT = Path(__file__).resolve().parents[1]
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

from api.common import JOBS_ROOT, job_dir, mesh_stats, new_job_id, save_upload, utc_now, write_manifest
from inference import PIPELINE_TYPE_MAP, generate_from_image, load_envmap, load_pipeline

app = FastAPI(title="TRELLIS.2 API", version="1.0.0")

pipeline = None
envmap = None


@app.on_event("startup")
def _load_models() -> None:
    global pipeline, envmap
    print("Loading TRELLIS.2 pipeline...")
    pipeline = load_pipeline()
    envmap = load_envmap()
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    print("TRELLIS.2 API ready.")


@app.get("/v1/health")
def health() -> dict:
    import torch

    return {
        "status": "ok",
        "service": "trellis",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "time": utc_now(),
    }


@app.post("/v1/trellis/generate")
async def trellis_generate(
    image: UploadFile = File(...),
    resolution: str = Form("512"),
    skip_video: bool = Form(False),
) -> dict:
    if resolution not in PIPELINE_TYPE_MAP:
        raise HTTPException(400, f"resolution must be one of {list(PIPELINE_TYPE_MAP)}")

    job_id = new_job_id("trellis_")
    out_dir = job_dir(job_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = out_dir / f"input_{timestamp}.png"
    output_name = f"model_{timestamp}"

    await save_upload(image, input_path)
    pil_image = Image.open(input_path).convert("RGBA")

    glb_path, video_path = generate_from_image(
        pil_image,
        out_dir,
        output_name,
        pipeline=pipeline,
        envmap=envmap,
        resolution=resolution,
        skip_video=skip_video,
    )

    payload = {
        "job_id": job_id,
        "status": "completed",
        "input": str(input_path),
        "glb": str(glb_path),
        "video": str(video_path) if video_path else None,
        "resolution": resolution,
        "stats": mesh_stats(glb_path),
        "time": utc_now(),
    }
    write_manifest(job_id, payload)
    return payload


@app.get("/v1/jobs/{job_id}/glb")
def download_glb(job_id: str):
    manifest_path = JOBS_ROOT / job_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Job not found")
    import json

    manifest = json.loads(manifest_path.read_text())
    glb_path = Path(manifest["glb"])
    if not glb_path.is_file():
        raise HTTPException(404, "GLB not found")
    return FileResponse(glb_path, media_type="model/gltf-binary", filename=glb_path.name)


@app.get("/v1/jobs/{job_id}/video")
def download_video(job_id: str):
    manifest_path = JOBS_ROOT / job_id / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Job not found")
    import json

    manifest = json.loads(manifest_path.read_text())
    video = manifest.get("video")
    if not video:
        raise HTTPException(404, "Video not available for this job")
    video_path = Path(video)
    if not video_path.is_file():
        raise HTTPException(404, "Video file missing")
    return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.trellis_api:app",
        host=os.environ.get("TRELLIS_API_HOST", "0.0.0.0"),
        port=int(os.environ.get("TRELLIS_API_PORT", "8100")),
        reload=False,
    )
