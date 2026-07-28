#!/usr/bin/env python3
"""One-at-a-time ablation sweeps for QuadWild hyperparameters."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from _paths import DATASET_DIR, SCRIPT_DIR

RUN_SCRIPT = SCRIPT_DIR / "run_quadwild.py"
DEFAULT_INPUT = DATASET_DIR / "quadwild_out" / "sample_voxel_clean.obj"
DEFAULT_OUT_DIR = DATASET_DIR / "quadwild_ablation"


def run_one(
    input_path: Path,
    out_dir: Path,
    target_tris: int,
    scale_fact: float,
    extra_args: list[str],
) -> dict:
    label = f"tris{target_tris}_scale{scale_fact:.2f}".replace(".", "p")
    output = out_dir / f"{label}.glb"
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        str(input_path),
        "--no-preprocess",
        "--no-smoothing",
        "--target-tris",
        str(target_tris),
        "--scale-fact",
        str(scale_fact),
        "-o",
        str(output),
        *extra_args,
    ]
    print(f"\n{'=' * 72}\nRunning: {' '.join(cmd)}\n{'=' * 72}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    ok = proc.returncode == 0 and output.exists()
    record = {
        "target_tris": target_tris,
        "scale_fact": scale_fact,
        "output": str(output),
        "ok": ok,
        "elapsed_s": round(elapsed, 1),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if ok:
        print(f"Finished {label} in {elapsed:.1f}s -> {output}", flush=True)
    else:
        print(f"FAILED {label} (exit {proc.returncode})", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="QuadWild one-at-a-time ablation")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fixed-tris", type=int, default=30000)
    parser.add_argument("--fixed-scale", type=float, default=1.2)
    parser.add_argument(
        "--tris-values",
        type=int,
        nargs="+",
        default=[10000, 20000, 30000, 50000, 80000],
    )
    parser.add_argument(
        "--scale-values",
        type=float,
        nargs="+",
        default=[0.6, 0.9, 1.2, 1.5, 2.0],
    )
    parser.add_argument(
        "--only",
        choices=["tris", "scale", "all"],
        default="all",
        help="Run only target_tris sweep, only scale_fact sweep, or both",
    )
    args, extra = parser.parse_known_args()

    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    if args.only in ("tris", "all"):
        print(f"\n### Ablation: target_tris (fixed scale_fact={args.fixed_scale}) ###")
        for tris in args.tris_values:
            results.append(
                run_one(
                    input_path,
                    out_dir,
                    target_tris=tris,
                    scale_fact=args.fixed_scale,
                    extra_args=extra,
                )
            )

    if args.only in ("scale", "all"):
        print(f"\n### Ablation: scale_fact (fixed target_tris={args.fixed_tris}) ###")
        for scale in args.scale_values:
            results.append(
                run_one(
                    input_path,
                    out_dir,
                    target_tris=args.fixed_tris,
                    scale_fact=scale,
                    extra_args=extra,
                )
            )

    summary_path = out_dir / "ablation_summary.json"
    summary = {
        "input": str(input_path),
        "fixed_tris": args.fixed_tris,
        "fixed_scale": args.fixed_scale,
        "tris_values": args.tris_values,
        "scale_values": args.scale_values,
        "runs": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {summary_path}")
    print("\nResults:")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        print(
            f"  [{status}] tris={r['target_tris']:>6}  scale={r['scale_fact']:.2f}  ->  {r['output']}"
        )
    failed = sum(1 for r in results if not r["ok"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
