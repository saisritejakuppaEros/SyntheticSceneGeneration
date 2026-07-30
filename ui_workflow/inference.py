#!/usr/bin/env python3
"""
TRELLIS.2 image-to-3D inference helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

TRELLIS_ROOT = Path(os.environ.get("TRELLIS_ROOT", "/home/parth_h200/parth/TRELLIS.2"))
if str(TRELLIS_ROOT) not in sys.path:
    sys.path.insert(0, str(TRELLIS_ROOT))

MODELS_DIR = os.environ.get("TRELLIS_MODEL_CACHE", str(TRELLIS_ROOT / "models"))
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", MODELS_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(MODELS_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(MODELS_DIR, "transformers"))
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import o_voxel  # noqa: E402
from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: E402
from trellis2.renderers import EnvMap  # noqa: E402
from trellis2.utils import render_utils  # noqa: E402

PIPELINE_TYPE_MAP = {
    "512": "512",
    "1024": "1024_cascade",
    "1536": "1536_cascade",
}


def load_exr_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[..., :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    import Imath  # noqa: WPS433
    import OpenEXR  # noqa: WPS433

    exr = OpenEXR.InputFile(path)
    dw = exr.header()["dataWindow"]
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = exr.header()["channels"]

    def read_channel(name: str) -> np.ndarray:
        if name not in channels:
            raise KeyError(f"Missing channel {name} in {path}")
        h = dw.max.y - dw.min.y + 1
        w = dw.max.x - dw.min.x + 1
        return np.frombuffer(exr.channel(name, pixel_type), dtype=np.float32).reshape(h, w)

    r, g, b = read_channel("R"), read_channel("G"), read_channel("B")
    return np.stack([r, g, b], axis=-1)


def load_pipeline() -> Trellis2ImageTo3DPipeline:
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    return pipeline


def load_envmap() -> EnvMap:
    hdri_path = TRELLIS_ROOT / "assets/hdri/forest.exr"
    return EnvMap(
        torch.tensor(load_exr_rgb(str(hdri_path)), dtype=torch.float32, device="cuda")
    )


def generate_from_image(
    image: Image.Image,
    output_dir: Path,
    output_name: str,
    *,
    pipeline: Trellis2ImageTo3DPipeline,
    envmap: EnvMap,
    resolution: str = "512",
    skip_video: bool = False,
) -> tuple[Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_path = output_dir / f"{output_name}.glb"
    video_path = output_dir / f"{output_name}.mp4"
    pipeline_type = PIPELINE_TYPE_MAP.get(resolution, resolution)

    mesh = pipeline.run(image, pipeline_type=pipeline_type)[0]
    mesh.simplify(16777216)

    if not skip_video:
        video = render_utils.make_pbr_vis_frames(
            render_utils.render_video(mesh, envmap=envmap)
        )
        imageio.mimsave(str(video_path), video, fps=15)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=100000,
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(str(glb_path), extension_webp=True)
    torch.cuda.empty_cache()
    return glb_path, video_path if not skip_video else None
