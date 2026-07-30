#!/usr/bin/env python3
"""CLI: transfer texture with xatlas UVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UI_ROOT))

from services.transfer_texture import bake_texture


def main() -> int:
    parser = argparse.ArgumentParser(description="Transfer texture with xatlas UV unwrap")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--texture-size", type=int, default=512)
    parser.add_argument("--uv-padding", type=int, default=4)
    parser.add_argument("--mode", choices=["texture", "vertex"], default="texture")
    args = parser.parse_args()

    stats = bake_texture(
        args.source.resolve(),
        args.target.resolve(),
        args.output.resolve(),
        texture_size=args.texture_size,
        uv_padding=args.uv_padding,
        mode=args.mode,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
