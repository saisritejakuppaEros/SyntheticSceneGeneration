#!/usr/bin/env python3
"""CLI: xatlas UV unwrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))

from services.xatlas_uv import unwrap_uv


def main() -> int:
    parser = argparse.ArgumentParser(description="xatlas UV unwrap")
    parser.add_argument("input", type=Path, help="Input GLB/OBJ")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output GLB")
    parser.add_argument("--resolution", type=int, default=2048, help="Atlas resolution")
    parser.add_argument("--padding", type=int, default=2, help="Chart padding in texels")
    args = parser.parse_args()

    stats = unwrap_uv(args.input.resolve(), args.output.resolve(), resolution=args.resolution, padding=args.padding)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
