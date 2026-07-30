"""HTTP client for mesh pipeline APIs."""

from __future__ import annotations

import os
from pathlib import Path

import requests

TRELLIS_API_URL = os.environ.get("TRELLIS_API_URL", "http://localhost:8100").rstrip("/")
MESH_API_URL = os.environ.get("MESH_API_URL", "http://localhost:8101").rstrip("/")
DEFAULT_TIMEOUT = int(os.environ.get("PIPELINE_API_TIMEOUT", "3600"))


class ApiError(RuntimeError):
    def __init__(self, service: str, detail: str, status: int | None = None):
        self.service = service
        self.status = status
        super().__init__(f"{service}: {detail}")


def _check(resp: requests.Response, service: str) -> dict:
    if resp.ok:
        return resp.json()
    detail = resp.text.strip() or resp.reason
    raise ApiError(service, detail, resp.status_code)


def health_check() -> dict[str, bool | str | None]:
    trellis_ok = mesh_ok = False
    trellis_err = mesh_err = None
    try:
        trellis_ok = requests.get(f"{TRELLIS_API_URL}/v1/health", timeout=5).ok
    except requests.RequestException as exc:
        trellis_err = str(exc)
    try:
        mesh_ok = requests.get(f"{MESH_API_URL}/v1/health", timeout=5).ok
    except requests.RequestException as exc:
        mesh_err = str(exc)
    return {
        "trellis_ok": trellis_ok,
        "mesh_ok": mesh_ok,
        "trellis_url": TRELLIS_API_URL,
        "mesh_url": MESH_API_URL,
        "trellis_err": trellis_err,
        "mesh_err": mesh_err,
    }


def trellis_generate(
    image_path: Path,
    *,
    resolution: str = "512",
    skip_video: bool = False,
) -> dict:
    with image_path.open("rb") as handle:
        resp = requests.post(
            f"{TRELLIS_API_URL}/v1/trellis/generate",
            files={"image": (image_path.name, handle, "image/png")},
            data={"resolution": resolution, "skip_video": str(skip_video).lower()},
            timeout=DEFAULT_TIMEOUT,
        )
    return _check(resp, "TRELLIS")


def autoremesher(
    mesh_path: Path,
    *,
    target_quads: int = 5000,
    prep_target_tris: int = 0,
    solid_only: bool = False,
    from_solid: bool = False,
) -> dict:
    with mesh_path.open("rb") as handle:
        resp = requests.post(
            f"{MESH_API_URL}/v1/autoremesher",
            files={"mesh": (mesh_path.name, handle, "model/gltf-binary")},
            data={
                "target_quads": str(target_quads),
                "prep_target_tris": str(prep_target_tris),
                "solid_only": str(solid_only).lower(),
                "from_solid": str(from_solid).lower(),
            },
            timeout=DEFAULT_TIMEOUT,
        )
    return _check(resp, "AutoRemesher")


def meshoptimizer(
    mesh_path: Path,
    *,
    target_tris: int = 30000,
    simplify_error: float = 0.01,
) -> dict:
    with mesh_path.open("rb") as handle:
        resp = requests.post(
            f"{MESH_API_URL}/v1/meshoptimizer",
            files={"mesh": (mesh_path.name, handle, "model/gltf-binary")},
            data={
                "target_tris": str(target_tris),
                "simplify_error": str(simplify_error),
            },
            timeout=DEFAULT_TIMEOUT,
        )
    return _check(resp, "meshoptimizer")


def xatlas_uv(
    mesh_path: Path,
    *,
    resolution: int = 2048,
    padding: int = 2,
) -> dict:
    with mesh_path.open("rb") as handle:
        resp = requests.post(
            f"{MESH_API_URL}/v1/xatlas",
            files={"mesh": (mesh_path.name, handle, "model/gltf-binary")},
            data={"resolution": str(resolution), "padding": str(padding)},
            timeout=DEFAULT_TIMEOUT,
        )
    return _check(resp, "xatlas")


def transfer_texture(
    source_path: Path,
    target_path: Path,
    *,
    texture_size: int = 512,
    uv_padding: int = 4,
    mode: str = "texture",
) -> dict:
    with source_path.open("rb") as src, target_path.open("rb") as tgt:
        resp = requests.post(
            f"{MESH_API_URL}/v1/transfer-texture",
            files={
                "source": (source_path.name, src, "model/gltf-binary"),
                "target": (target_path.name, tgt, "model/gltf-binary"),
            },
            data={
                "texture_size": str(texture_size),
                "uv_padding": str(uv_padding),
                "mode": mode,
            },
            timeout=DEFAULT_TIMEOUT,
        )
    return _check(resp, "transfer_texture")


def format_stats(stats: dict | None) -> str:
    if not stats:
        return ""
    return (
        f"verts={stats.get('vertices', '?')}, "
        f"faces={stats.get('faces', '?')}, "
        f"size={stats.get('file_kb', '?')} KB"
    )
