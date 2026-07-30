#!/usr/bin/env python3
"""
Gradio pipeline UI — image → TRELLIS → solidify → AutoRemesher → texture transfer.

Calls HTTP APIs on localhost:8100 (TRELLIS) and :8101 (mesh tools).
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

from api_client import (
    ApiError,
    autoremesher,
    format_stats,
    health_check,
    transfer_texture,
    trellis_generate,
)

UI_ROOT = Path(__file__).resolve().parent
VIEWER_HEIGHT = 320
VIEWER_KWARGS = dict(clear_color=(0.15, 0.15, 0.15, 1.0))

try:
    from api.common import MAX_TARGET_QUADS
except ImportError:
    MAX_TARGET_QUADS = 100_000


def _status_line(step: str, payload: dict) -> str:
    stats = payload.get("stats") or {}
    path = payload.get("output_glb") or payload.get("glb") or "?"
    return f"**{step}** ✓ `{path}` — {format_stats(stats)}"


def _pair(glb: str | None) -> tuple[str | None, str | None]:
    return glb, glb


def run_pipeline(
    image: Image.Image | None,
    resolution: str,
    skip_video: bool,
    target_quads: int,
    texture_size: int,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[
    str | None, str | None,
    str | None, str | None,
    str | None, str | None,
    str | None, str | None,
    str | None,
    str,
]:
    if image is None:
        raise gr.Error("Upload an image first.")

    health = health_check()
    if not health["trellis_ok"]:
        err = health.get("trellis_err") or "not reachable"
        raise gr.Error(f"TRELLIS API down ({health['trellis_url']}): {err}")
    if not health["mesh_ok"]:
        err = health.get("mesh_err") or "not reachable"
        raise gr.Error(f"Mesh API down ({health['mesh_url']}): {err}")

    log_lines = [f"Pipeline started {datetime.now().strftime('%H:%M:%S')}"]
    trellis_glb: str | None = None
    solid_glb: str | None = None
    remesh_glb: str | None = None
    final_glb: str | None = None
    video_path: str | None = None

    with tempfile.TemporaryDirectory(prefix="pipeline_ui_") as tmp:
        input_path = Path(tmp) / "input.png"
        image.convert("RGBA").save(input_path)

        progress(0.08, desc="TRELLIS image→3D…")
        try:
            trellis = trellis_generate(
                input_path,
                resolution=resolution,
                skip_video=skip_video,
            )
        except ApiError as exc:
            raise gr.Error(str(exc)) from exc

        trellis_glb = trellis["glb"]
        video_path = trellis.get("video")
        log_lines.append(_status_line("1/4 TRELLIS", trellis))

        progress(0.28, desc="Solidify mesh…")
        try:
            solid = autoremesher(
                Path(trellis_glb),
                target_quads=int(target_quads),
                solid_only=True,
            )
        except ApiError as exc:
            raise gr.Error(str(exc)) from exc

        solid_glb = solid["output_glb"]
        log_lines.append(_status_line("2/4 Solid (pre-remesh)", solid))

        progress(0.52, desc="AutoRemesher…")
        try:
            remesh = autoremesher(
                Path(solid_glb),
                target_quads=int(target_quads),
                from_solid=True,
            )
        except ApiError as exc:
            raise gr.Error(str(exc)) from exc

        remesh_glb = remesh["output_glb"]
        log_lines.append(_status_line("3/4 AutoRemesher", remesh))

        progress(0.72, desc="xatlas UV + texture bake…")
        try:
            textured = transfer_texture(
                Path(trellis_glb),
                Path(remesh_glb),
                texture_size=int(texture_size),
                uv_padding=4,
                mode="texture",
            )
        except ApiError as exc:
            raise gr.Error(str(exc)) from exc

        final_glb = textured["output_glb"]
        log_lines.append(_status_line("4/4 Texture transfer", textured))

        # --- meshoptimizer (disabled for now) ---
        # progress(0.90, desc="meshoptimizer…")
        # try:
        #     from api_client import meshoptimizer
        #     optimized = meshoptimizer(Path(final_glb), target_tris=30000)
        # except ApiError as exc:
        #     raise gr.Error(str(exc)) from exc
        # final_glb = optimized["output_glb"]
        # log_lines.append(_status_line("meshoptimizer", optimized))

    progress(1.0, desc="Done")
    log_lines.append(f"\n**Final GLB:** `{final_glb}`")
    log = "\n\n".join(log_lines)

    video_out = video_path if video_path and Path(video_path).is_file() else None

    return (
        *_pair(trellis_glb),
        *_pair(solid_glb),
        *_pair(remesh_glb),
        *_pair(final_glb),
        video_out,
        final_glb,
        log,
    )


def refresh_api_status() -> str:
    health = health_check()
    trellis = "🟢 online" if health["trellis_ok"] else "🔴 offline"
    mesh = "🟢 online" if health["mesh_ok"] else "🔴 offline"
    lines = [
        f"- **TRELLIS API** ({health['trellis_url']}): {trellis}",
        f"- **Mesh API** ({health['mesh_url']}): {mesh}",
    ]
    if health.get("trellis_err"):
        lines.append(f"  - trellis error: `{health['trellis_err']}`")
    if health.get("mesh_err"):
        lines.append(f"  - mesh error: `{health['mesh_err']}`")
    if not health["trellis_ok"] or not health["mesh_ok"]:
        lines.append(
            "\nStart servers:\n"
            "```bash\n"
            "./run_trellis_api.sh   # port 8100\n"
            "./run_mesh_api.sh      # port 8101\n"
            "```"
        )
    return "\n".join(lines)


def _viewer_column(title: str, solid_label: str, wire_label: str):
    gr.Markdown(title)
    with gr.Tabs():
        with gr.Tab("Solid"):
            solid = gr.Model3D(
                label=solid_label,
                height=VIEWER_HEIGHT,
                display_mode="solid",
                **VIEWER_KWARGS,
            )
        with gr.Tab("Wireframe"):
            wire = gr.Model3D(
                label=wire_label,
                height=VIEWER_HEIGHT,
                display_mode="wireframe",
                **VIEWER_KWARGS,
            )
    return solid, wire


with gr.Blocks(title="Mesh Pipeline") as demo:
    gr.Markdown(
        """
        ## Mesh Pipeline
        **TRELLIS 2.0** → **Solidify** → **AutoRemesher** → **Texture transfer**

        Four outputs side-by-side (solid + wireframe each). Final column is textured remesh.
        """
    )

    api_status = gr.Markdown(refresh_api_status())
    refresh_btn = gr.Button("Refresh API status", size="sm")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            image_input = gr.Image(
                label="Input image",
                type="pil",
                format="png",
                image_mode="RGBA",
                height=300,
            )
            resolution = gr.Radio(["512", "1024", "1536"], label="TRELLIS resolution", value="512")
            skip_video = gr.Checkbox(label="Skip TRELLIS preview video", value=True)
            target_quads = gr.Slider(
                1000,
                MAX_TARGET_QUADS,
                value=5000,
                step=500,
                label="AutoRemesher target quads",
                info=f"Max {MAX_TARGET_QUADS:,} (1 lakh)",
            )
            texture_size = gr.Slider(
                128,
                2048,
                value=512,
                step=128,
                label="Texture atlas size",
                info="xatlas UVs + baked atlas (512+ recommended)",
            )
            run_btn = gr.Button("Run pipeline", variant="primary")

        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column():
                    trellis_solid, trellis_wire = _viewer_column(
                        "### 1. TRELLIS",
                        "TRELLIS solid",
                        "TRELLIS wireframe",
                    )
                with gr.Column():
                    solid_solid, solid_wire = _viewer_column(
                        "### 2. Solid",
                        "Solid solid",
                        "Solid wireframe",
                    )
                with gr.Column():
                    remesh_solid, remesh_wire = _viewer_column(
                        "### 3. Remeshed",
                        "Remesh solid",
                        "Remesh wireframe",
                    )
                with gr.Column():
                    final_solid, final_wire = _viewer_column(
                        "### 4. Final textured",
                        "Textured solid",
                        "Textured wireframe",
                    )

            video_output = gr.Video(label="TRELLIS turntable preview", height=180)
            download_btn = gr.DownloadButton(label="Download final textured GLB")
            pipeline_log = gr.Markdown(label="Pipeline log")

    refresh_btn.click(refresh_api_status, outputs=api_status)

    run_btn.click(
        run_pipeline,
        inputs=[image_input, resolution, skip_video, target_quads, texture_size],
        outputs=[
            trellis_solid, trellis_wire,
            solid_solid, solid_wire,
            remesh_solid, remesh_wire,
            final_solid, final_wire,
            video_output,
            download_btn,
            pipeline_log,
        ],
    )

    demo.load(refresh_api_status, outputs=api_status)


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_ADDRESS", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
    )
