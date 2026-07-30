#!/usr/bin/env python3
"""
Gradio UI: upload an image, run TRELLIS.2 image-to-3D, inspect the GLB interactively.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

from inference import generate_from_image, load_envmap, load_pipeline

UI_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = UI_ROOT / "outputs"
TMP_DIR = UI_ROOT / "tmp"


def _session_dir(request: gr.Request) -> Path:
    path = TMP_DIR / str(request.session_hash)
    path.mkdir(parents=True, exist_ok=True)
    return path


def start_session(request: gr.Request) -> None:
    _session_dir(request)


def end_session(request: gr.Request) -> None:
    path = TMP_DIR / str(request.session_hash)
    if path.exists():
        shutil.rmtree(path)


def image_to_3d(
    image: Image.Image | None,
    resolution: str,
    skip_video: bool,
    request: gr.Request,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str | None, str | None, str | None, str | None]:
    if image is None:
        raise gr.Error("Upload an image first.")

    session_dir = _session_dir(request)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"model_{timestamp}"

    glb_path, video_path = generate_from_image(
        image.convert("RGBA"),
        session_dir,
        output_name,
        pipeline=pipeline,
        envmap=envmap,
        resolution=resolution,
        skip_video=skip_video,
    )

    glb_out = str(glb_path)
    video_out = str(video_path) if video_path and video_path.is_file() else None
    return glb_out, glb_out, video_out, glb_out


with gr.Blocks(title="Image to 3D") as demo:
    gr.Markdown(
        """
        ## Image to 3D
        Upload an image to generate a textured 3D model with
        [TRELLIS.2](https://microsoft.github.io/TRELLIS.2), then rotate and inspect it below.
        """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_input = gr.Image(
                label="Input image",
                type="pil",
                format="png",
                image_mode="RGBA",
                height=400,
            )
            resolution = gr.Radio(
                ["512", "1024", "1536"],
                label="Resolution",
                value="512",
                info="Higher resolution is slower and uses more GPU memory.",
            )
            skip_video = gr.Checkbox(label="Skip preview video (faster)", value=False)
            generate_btn = gr.Button("Generate 3D", variant="primary")

        with gr.Column(scale=1.4):
            with gr.Tabs():
                with gr.Tab("Textured"):
                    model_solid = gr.Model3D(
                        label="Solid preview",
                        height=520,
                        display_mode="solid",
                        clear_color=(0.15, 0.15, 0.15, 1.0),
                    )
                with gr.Tab("Wireframe"):
                    model_wireframe = gr.Model3D(
                        label="Wireframe preview",
                        height=520,
                        display_mode="wireframe",
                        clear_color=(0.15, 0.15, 0.15, 1.0),
                    )
            video_output = gr.Video(label="Turntable preview", height=280)
            download_btn = gr.DownloadButton(label="Download GLB")

    demo.load(start_session)
    demo.unload(end_session)

    generate_btn.click(
        image_to_3d,
        inputs=[image_input, resolution, skip_video],
        outputs=[model_solid, model_wireframe, video_output, download_btn],
    )


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TRELLIS.2 pipeline...")
    pipeline = load_pipeline()
    envmap = load_envmap()
    print("Pipeline ready.")

    demo.launch(
        server_name=os.environ.get("GRADIO_ADDRESS", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
    )
