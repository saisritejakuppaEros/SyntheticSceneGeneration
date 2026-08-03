"""Run existing meshcleaning scripts via subprocess."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from api.common import MESH_ROOT, SCRIPTS_DIR, VENV_PYTHON


def _python() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return sys.executable


def run_script(script: Path, args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [_python(), str(script), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd or MESH_ROOT),
        check=False,
    )


def run_autoremesher(
    input_path: Path,
    output_path: Path,
    *,
    target_quads: int = 5000,
    prep_target_tris: int = 0,
    solid_only: bool = False,
    from_solid: bool = False,
) -> subprocess.CompletedProcess[str]:
    args = [
        str(input_path),
        "-o",
        str(output_path),
        "--target-quads",
        str(target_quads),
        "--prep-target-tris",
        str(prep_target_tris),
    ]
    if solid_only:
        args.extend(["--solid-only", "--solid-output", str(output_path)])
    elif from_solid:
        args.append("--from-solid")
    return run_script(SCRIPTS_DIR / "run_autoremesher.py", args)


def run_meshoptimizer(
    input_path: Path,
    output_path: Path,
    *,
    target_tris: int = 30000,
    simplify_error: float = 0.01,
) -> subprocess.CompletedProcess[str]:
    args = [
        str(input_path),
        "-o",
        str(output_path),
        "--target-tris",
        str(target_tris),
        "--simplify-error",
        str(simplify_error),
    ]
    return run_script(SCRIPTS_DIR / "run_meshoptimizer.py", args)


def run_instant_meshes(
    input_path: Path,
    output_path: Path,
    *,
    target_quads: int = 5000,
    from_meshopt: bool = False,
    dominant: bool = False,
    boundaries: bool = True,
) -> subprocess.CompletedProcess[str]:
    args = [
        str(input_path),
        "-o",
        str(output_path),
        "-f",
        str(target_quads),
    ]
    if from_meshopt:
        args.append("--from-meshopt")
    if dominant:
        args.append("--dominant")
    if boundaries:
        args.append("--boundaries")
    else:
        args.append("--no-boundaries")
    return run_script(SCRIPTS_DIR / "run_instant_meshes.py", args)


def run_quadriflow(
    input_path: Path,
    output_path: Path,
    *,
    target_quads: int = 5000,
    sharp: bool = False,
    skip_repair: bool = True,
    from_meshopt: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Legacy QuadriFlow wrapper (prefer run_instant_meshes)."""
    args = [
        str(input_path),
        "-o",
        str(output_path),
        "-f",
        str(target_quads),
    ]
    if sharp:
        args.append("--sharp")
    if skip_repair:
        args.append("--skip-repair")
    else:
        args.append("--light-repair")
    if from_meshopt:
        args.append("--from-meshopt")
    return run_script(SCRIPTS_DIR / "run_quadriflow.py", args)


def run_transfer_texture(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    *,
    texture_size: int = 512,
    uv_padding: int = 4,
    mode: str = "texture",
) -> dict:
    from services.transfer_texture import bake_texture

    return bake_texture(
        source_path,
        target_path,
        output_path,
        texture_size=texture_size,
        uv_padding=uv_padding,
        mode=mode,
    )
