"""Project path helpers for meshcleaning scripts."""

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "dataset"
AUTOREMESHER = ROOT_DIR / ".venv" / "bin" / "autoremesher"
AUTOREMESHER_BIN = ROOT_DIR / "autoremesher" / "autoremesher"
