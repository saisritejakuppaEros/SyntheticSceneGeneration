#!/usr/bin/env python3
"""
CLI wrapper for TRELLIS.2 image-to-3D.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from inference import generate_from_image, load_envmap, load_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="TRELLIS.2 image-to-3D")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Directory for GLB/MP4 outputs")
    parser.add_argument("--output-name", default="result", help="Base name for output files")
    parser.add_argument("--pipeline-type", default="512", help="512, 1024, or 1536")
    parser.add_argument("--skip-video", action="store_true", help="Skip MP4 preview render")
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipeline = load_pipeline()
    envmap = load_envmap()
    glb_path, video_path = generate_from_image(
        Image.open(args.image),
        Path(args.output_dir).resolve(),
        args.output_name,
        pipeline=pipeline,
        envmap=envmap,
        resolution=args.pipeline_type,
        skip_video=args.skip_video,
    )
    print(f"GLB={glb_path}")
    if video_path:
        print(f"MP4={video_path}")


if __name__ == "__main__":
    main()
