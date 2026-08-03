#!/usr/bin/env bash
# Build Instant Meshes binary (one-time). Python/API uses meshcleaning .venv.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/instant-meshes/build"

if ! dpkg -s libxrandr-dev >/dev/null 2>&1; then
  echo "Installing X11 dev packages for Instant Meshes..."
  sudo apt-get update -qq
  sudo apt-get install -y libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev xorg-dev
fi

cd "$ROOT/instant-meshes"
git submodule update --init --recursive
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)" InstantMeshes

echo "Built: $BUILD/Instant Meshes"
echo "Test with meshcleaning .venv:"
echo "  source $ROOT/.venv/bin/activate"
echo "  python $ROOT/scripts/run_instant_meshes.py $ROOT/QuadriFlow/examples/Gargoyle_input.obj -f 1200 -o /tmp/im.glb"
