#!/usr/bin/env python3
"""
Production mesh pipeline v2
============================

Maps the film/game asset workflow to automated steps:

  Step 0  cleanup_fragments   manual cleanup — drop tiny shells, weld verts
  Step 1  reduce_vertices     Houdini — triangle reduction via meshoptimizer
  Step 2  quad_flow            ZRemesher — quad flow remesh via QuadWild
  Step 3  transfer_texture     bake embedded texture onto final mesh

Usage:
  source /devwork/teja/meshcleaning/.venv/bin/activate

  # Full pipeline
  python scripts/v2/pipeline.py dataset/rodin.obj

  # Custom targets
  python scripts/v2/pipeline.py dataset/rodin_3.obj \\
    --target-tris 5000 \\
    --min-component-faces 50 \\
    --scale-fact 1.2 \\
    --texture-size 128 \\
    -o dataset/v2_out/rodin_3_final.glb

  # Run individual steps
  python scripts/v2/cleanup_fragments.py dataset/rodin_3.obj
  python scripts/v2/reduce_vertices.py dataset/v2_out/rodin_3_clean.glb
  python scripts/v2/quad_flow.py dataset/v2_out/rodin_3_clean_reduced.glb
  python scripts/v2/transfer_texture.py --source ... --target ... -o ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from _paths import DEFAULT_INPUT
from mesh_prep import cleanup_fragments, load_mesh, mesh_stats
from quad_flow import run_quad_flow
from reduce_vertices import reduce_vertices
from transfer_texture import transfer_texture


@dataclass
class PipelineResult:
    input: str
    work_dir: str
    clean_glb: str
    reduced_glb: str
    quadflow_glb: str
    final_glb: str
    stats: dict = field(default_factory=dict)


def run_pipeline(
    input_path: Path,
    *,
    output: Path | None = None,
    work_dir: Path | None = None,
    min_component_faces: int = 50,
    merge_distance: float = 0.002,
    target_tris: int = 5000,
    simplify_error: float = 0.01,
    scale_fact: float = 1.2,
    texture_size: int = 128,
    skip_cleanup: bool = False,
    skip_texture: bool = False,
    skip_quad_flow: bool = False,
) -> PipelineResult:
    input_path = input_path.resolve()
    stem = input_path.stem
    work_dir = (work_dir or input_path.parent / "v2_out" / stem).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    clean_glb = work_dir / f"{stem}_clean.glb"
    reduced_glb = work_dir / f"{stem}_reduced.glb"
    quadflow_glb = work_dir / f"{stem}_quadflow.glb"
    final_glb = output or work_dir / f"{stem}_final.glb"

    stats: dict = {"input": mesh_stats(load_mesh(input_path))}

    # Step 0 — fragment cleanup
    if skip_cleanup:
        print("\n" + "=" * 72)
        print("Step 0/4: SKIPPED (using input directly)")
        print("=" * 72)
        step_input = input_path
    else:
        print("\n" + "=" * 72)
        print("Step 0/4: Fragment cleanup")
        print("=" * 72)
        mesh = load_mesh(input_path)
        cleaned = cleanup_fragments(
            mesh,
            min_component_faces=min_component_faces,
            merge_distance=merge_distance,
        )
        cleaned.export(clean_glb)
        stats["clean"] = mesh_stats(cleaned)
        step_input = clean_glb
        print(f"  -> {clean_glb}")

    # Step 1 — vertex reduction
    print("\n" + "=" * 72)
    print("Step 1/4: Vertex reduction (meshoptimizer)")
    print("=" * 72)
    reduce_vertices(
        step_input,
        reduced_glb,
        target_tris=target_tris,
        simplify_error=simplify_error,
    )
    stats["reduced"] = mesh_stats(load_mesh(reduced_glb))

    if skip_quad_flow:
        print("\n" + "=" * 72)
        print("Step 2/4: SKIPPED")
        print("=" * 72)
        if final_glb.resolve() != reduced_glb.resolve():
            import shutil
            shutil.copy2(reduced_glb, final_glb)
        return PipelineResult(
            input=str(input_path),
            work_dir=str(work_dir),
            clean_glb=str(clean_glb if not skip_cleanup else step_input),
            reduced_glb=str(reduced_glb),
            quadflow_glb=str(reduced_glb),
            final_glb=str(final_glb),
            stats=stats,
        )

    # Step 2 — quad flow
    print("\n" + "=" * 72)
    print("Step 2/4: Quad flow remesh (QuadWild)")
    print("=" * 72)
    run_quad_flow(
        reduced_glb,
        quadflow_glb,
        work_dir=work_dir / "quad_flow",
        from_reduced=True,
        scale_fact=scale_fact,
    )
    stats["quadflow"] = mesh_stats(load_mesh(quadflow_glb))

    # Step 3 — texture
    if skip_texture:
        print("\n" + "=" * 72)
        print("Step 3/4: SKIPPED (no texture bake)")
        print("=" * 72)
        if final_glb.resolve() != quadflow_glb.resolve():
            import shutil
            shutil.copy2(quadflow_glb, final_glb)
    else:
        print("\n" + "=" * 72)
        print("Step 3/4: Texture transfer")
        print("=" * 72)
        try:
            transfer_texture(
                reduced_glb,
                quadflow_glb,
                final_glb,
                texture_size=texture_size,
            )
            stats["final"] = mesh_stats(load_mesh(final_glb))
        except RuntimeError as exc:
            if "baseColorTexture" in str(exc):
                print(f"  No embedded texture on source — exporting quad mesh as final.")
                import shutil
                shutil.copy2(quadflow_glb, final_glb)
                stats["final"] = stats["quadflow"]
            else:
                raise

    summary_path = work_dir / f"{stem}_summary.json"
    result = PipelineResult(
        input=str(input_path),
        work_dir=str(work_dir),
        clean_glb=str(clean_glb if not skip_cleanup else step_input),
        reduced_glb=str(reduced_glb),
        quadflow_glb=str(quadflow_glb),
        final_glb=str(final_glb),
        stats=stats,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"\nSummary: {summary_path}")

    final = stats.get("final", stats.get("quadflow", stats.get("reduced", {})))
    print(
        f"\nDone.\n"
        f"  input:    {input_path}\n"
        f"  final:    {final_glb}\n"
        f"  verts:    {final.get('vertices', '?')}\n"
        f"  faces:    {final.get('faces', '?')}\n"
        f"  shells:   {final.get('components', '?')}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="v2 production pipeline: cleanup -> reduce -> quad flow -> texture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--min-component-faces", type=int, default=50)
    parser.add_argument("--merge-distance", type=float, default=0.002)
    parser.add_argument("--target-tris", type=int, default=5000)
    parser.add_argument("--simplify-error", type=float, default=0.01)
    parser.add_argument("--scale-fact", type=float, default=1.2)
    parser.add_argument("--texture-size", type=int, default=128)
    parser.add_argument("--skip-cleanup", action="store_true")
    parser.add_argument("--skip-texture", action="store_true")
    parser.add_argument("--skip-quad-flow", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        args.input,
        output=args.output,
        work_dir=args.work_dir,
        min_component_faces=args.min_component_faces,
        merge_distance=args.merge_distance,
        target_tris=args.target_tris,
        simplify_error=args.simplify_error,
        scale_fact=args.scale_fact,
        texture_size=args.texture_size,
        skip_cleanup=args.skip_cleanup,
        skip_texture=args.skip_texture,
        skip_quad_flow=args.skip_quad_flow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
