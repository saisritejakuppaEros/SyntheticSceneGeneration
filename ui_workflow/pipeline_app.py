#!/usr/bin/env python3
"""
Gradio pipeline UI — image → TRELLIS → solidify → meshoptimizer → Instant Meshes → texture transfer.

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
    instant_meshes,
    meshoptimizer,
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
    path = (
        payload.get("output_obj")
        or payload.get("output_glb")
        or payload.get("glb")
        or payload.get("output")
        or "?"
    )
    quad_note = ""
    if stats.get("quads"):
        quad_note = f", quads={stats['quads']}"
    elif stats.get("obj_quads", {}).get("quads"):
        quad_note = f", quads={stats['obj_quads']['quads']}"
    return f"**{step}** ✓ `{path}` — {format_stats(stats)}{quad_note}"


def _pair(model: str | None) -> tuple[str | None, str | None]:
    return model, model


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
    remesh_obj: str | None = None
    final_glb: str | None = None
    final_obj: str | None = None
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
        log_lines.append(_status_line("1/5 TRELLIS", trellis))

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
        log_lines.append(_status_line("2/5 Solid (pre-remesh)", solid))

        progress(0.40, desc="meshoptimizer (triangle prep)…")
        try:
            target_tris = max(int(target_quads) * 2, 2000)
            reduced = meshoptimizer(
                Path(solid_glb),
                target_tris=target_tris,
            )
        except ApiError as exc:
            raise gr.Error(str(exc)) from exc

        reduced_glb = reduced["output_glb"]
        log_lines.append(_status_line("3/5 meshoptimizer", reduced))

        progress(0.58, desc="Instant Meshes (quad remesh)…")
        try:
            remesh = instant_meshes(
                Path(reduced_glb),
                target_quads=int(target_quads),
                from_meshopt=True,
            )
        except ApiError as exc:
            raise gr.Error(str(exc)) from exc

        remesh_glb = remesh["output_glb"]
        remesh_obj = remesh.get("output_obj")
        quad_info = remesh.get("stats", {}).get("obj_quads")
        step_label = "4/5 Instant Meshes"
        if quad_info:
            step_label += f" ({quad_info.get('quads', '?')} quads)"
        log_lines.append(_status_line(step_label, remesh))

        progress(0.78, desc="Texture bake → quad OBJ…")
        try:
            textured = transfer_texture(
                Path(trellis_glb),
                Path(remesh_obj or remesh_glb),
                target_obj_path=Path(remesh_obj) if remesh_obj else None,
                texture_size=int(texture_size),
                uv_padding=4,
                mode="texture",
                uv_method="box",
                output_format="both",
            )
        except ApiError as exc:
            raise gr.Error(str(exc)) from exc

        final_obj = textured.get("output_obj") or textured.get("output")
        final_glb = textured.get("output_glb")
        log_lines.append(_status_line("5/5 Texture transfer (quad OBJ)", textured))

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
    if final_obj:
        log_lines.append(f"\n**Final quad OBJ:** `{final_obj}`")
    if final_glb:
        log_lines.append(f"**Preview GLB:** `{final_glb}`")
    log = "\n\n".join(log_lines)

    video_out = video_path if video_path and Path(video_path).is_file() else None
    preview_model = final_obj or final_glb

    return (
        *_pair(trellis_glb),
        *_pair(solid_glb),
        *_pair(remesh_obj or remesh_glb),
        *_pair(preview_model),
        video_out,
        final_obj,
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
        **TRELLIS 2.0** → **Solidify** → **meshoptimizer** → **Instant Meshes** → **Texture transfer**

        Quad remesh via Instant Meshes (.venv). **Download the quad OBJ** — real `f v1 v2 v3 v4` faces.
        GLB is preview-only (triangulated).
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
                label="Instant Meshes target quads",
                info=f"meshoptimizer prep ≈ 2× tris; max {MAX_TARGET_QUADS:,}",
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
                        "### 3. Instant Meshes",
                        "Quad mesh solid",
                        "Quad mesh wireframe",
                    )
                with gr.Column():
                    final_solid, final_wire = _viewer_column(
                        "### 4. Final textured",
                        "Textured solid",
                        "Textured wireframe",
                    )

            video_output = gr.Video(label="TRELLIS turntable preview", height=180)
            download_obj_btn = gr.DownloadButton(label="Download quad OBJ (primary output)", variant="primary")
            download_glb_btn = gr.DownloadButton(label="Download preview GLB")
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
            download_obj_btn,
            download_glb_btn,
            pipeline_log,
        ],
    )

    demo.load(refresh_api_status, outputs=api_status)


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_ADDRESS", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
    )
