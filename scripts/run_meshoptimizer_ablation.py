#!/usr/bin/env python3
"""One-at-a-time ablation sweeps for meshoptimizer / gltfpack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from _paths import DATASET_DIR, SCRIPT_DIR

RUN_SCRIPT = SCRIPT_DIR / "run_meshoptimizer.py"
DEFAULT_INPUT = DATASET_DIR / "sample_2026-07-27T084001.382.glb"
DEFAULT_OUT_DIR = DATASET_DIR / "meshoptimizer_ablation"


def run_one(
    input_path: Path,
    out_dir: Path,
    target_tris: int | None,
    simplify_ratio: float | None,
    simplify_error: float,
    extra_args: list[str],
) -> dict:
    if target_tris is not None:
        label = f"tris{target_tris}_se{simplify_error:.3f}".replace(".", "p")
    else:
        label = f"si{simplify_ratio:.4f}_se{simplify_error:.3f}".replace(".", "p")
    output = out_dir / f"{label}.glb"

    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        str(input_path),
        "--simplify-error",
        str(simplify_error),
        "-o",
        str(output),
        *extra_args,
    ]
    if target_tris is not None:
        cmd.extend(["--target-tris", str(target_tris)])
    if simplify_ratio is not None:
        cmd.extend(["--simplify-ratio", str(simplify_ratio)])

    print(f"\n{'=' * 72}\nRunning: {' '.join(cmd)}\n{'=' * 72}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    ok = proc.returncode == 0 and output.exists()
    record = {
        "target_tris": target_tris,
        "simplify_ratio": simplify_ratio,
        "simplify_error": simplify_error,
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
    print(f"{'OK' if ok else 'FAIL'} {label} ({elapsed:.1f}s)", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="meshoptimizer one-at-a-time ablation")
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fixed-se", type=float, default=0.01)
    parser.add_argument("--fixed-tris", type=int, default=30000)
    parser.add_argument(
        "--tris-values",
        type=int,
        nargs="+",
        default=[15000, 20000, 30000, 50000, 80000],
    )
    parser.add_argument(
        "--se-values",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.1],
    )
    parser.add_argument(
        "--only",
        choices=["tris", "se", "all"],
        default="all",
    )
    args, extra = parser.parse_known_args()

    input_path = args.input.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    if args.only in ("tris", "all"):
        print(f"\n### Ablation: target_tris (fixed simplify_error={args.fixed_se}) ###")
        for tris in args.tris_values:
            results.append(
                run_one(
                    input_path,
                    out_dir,
                    target_tris=tris,
                    simplify_ratio=None,
                    simplify_error=args.fixed_se,
                    extra_args=extra,
                )
            )

    if args.only in ("se", "all"):
        print(f"\n### Ablation: simplify_error (fixed target_tris={args.fixed_tris}) ###")
        for se in args.se_values:
            results.append(
                run_one(
                    input_path,
                    out_dir,
                    target_tris=args.fixed_tris,
                    simplify_ratio=None,
                    simplify_error=se,
                    extra_args=extra,
                )
            )

    summary_path = out_dir / "ablation_summary.json"
    summary = {
        "input": str(input_path),
        "fixed_se": args.fixed_se,
        "fixed_tris": args.fixed_tris,
        "tris_values": args.tris_values,
        "se_values": args.se_values,
        "runs": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary: {summary_path}")
    print("\nResults (upload any .glb directly — textures embedded):")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        tris = r["target_tris"] if r["target_tris"] is not None else "-"
        print(f"  [{status}] tris={tris!s:>6}  se={r['simplify_error']:.3f}  ->  {r['output']}")
    failed = sum(1 for r in results if not r["ok"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
