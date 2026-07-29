"""Path helpers for the v2 production pipeline."""

from pathlib import Path

V2_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = V2_DIR.parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "dataset"
MESHOPT_DIR = ROOT_DIR / "meshoptimizer"
GLTFPACK = MESHOPT_DIR / "gltfpack"
QUADWILD_LIB = ROOT_DIR / "quadwild_release" / "QRemeshify" / "lib"

DEFAULT_INPUT = DATASET_DIR / "rodin.obj"
