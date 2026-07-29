# v2 production pipeline

Production film/game asset cleanup, mapped to automated steps.

| Production step | v2 script | Tool |
|---|---|---|
| Manual cleanup | `cleanup_fragments.py` | trimesh |
| Houdini (reduce verts) | `reduce_vertices.py` | meshoptimizer gltfpack |
| ZRemesher (quad flow) | `quad_flow.py` | QuadWild |
| Texture bake | `transfer_texture.py` | trimesh + scipy |
| Full pipeline | `pipeline.py` | all of the above |

## Setup

```bash
cd /devwork/teja/meshcleaning
source .venv/bin/activate

# Build gltfpack if needed
cd meshoptimizer && make gltfpack && cd ..
```

## Full pipeline (recommended)

```bash
python scripts/v2/pipeline.py dataset/rodin.obj

# Heavily fragmented mesh (e.g. rodin_3 with 1265 shells)
python scripts/v2/pipeline.py dataset/rodin_3.obj \
  --min-component-faces 50 \
  --target-tris 5000 \
  --scale-fact 1.2
```

Outputs land in `dataset/v2_out/<name>/`:

- `<name>_clean.glb` — fragments culled
- `<name>_reduced.glb` — triangle-reduced
- `<name>_quadflow.glb` — quad flow remesh
- `<name>_final.glb` — textured final asset
- `<name>_summary.json` — stats at each stage

## Individual steps

```bash
# Step 0: drop tiny shells
python scripts/v2/cleanup_fragments.py dataset/rodin_3.obj

# Step 1: reduce triangle count
python scripts/v2/reduce_vertices.py dataset/v2_out/rodin_3/rodin_3_clean.glb --target-tris 5000

# Step 2: quad flow remesh
python scripts/v2/quad_flow.py dataset/v2_out/rodin_3/rodin_3_reduced.glb

# Step 3: bake texture (skip if source has no texture)
python scripts/v2/transfer_texture.py \
  --source dataset/v2_out/rodin_3/rodin_3_reduced.glb \
  --target dataset/v2_out/rodin_3/rodin_3_quadflow.glb \
  -o dataset/v2_out/rodin_3/rodin_3_final.glb
```

## Pipeline flags

| Flag | Default | Purpose |
|---|---|---|
| `--min-component-faces` | 50 | Drop shells smaller than this |
| `--merge-distance` | 0.002 | Weld threshold (fraction of bbox) |
| `--target-tris` | 5000 | Triangle budget after reduction |
| `--simplify-error` | 0.01 | gltfpack error limit |
| `--scale-fact` | 1.2 | QuadWild chart scale |
| `--texture-size` | 128 | Embedded atlas resolution |
| `--skip-cleanup` | off | Skip fragment culling |
| `--skip-quad-flow` | off | Stop after reduction |
| `--skip-texture` | off | Skip texture bake |

## vs v1 scripts

v1 scripts in `scripts/` are unchanged. v2 is a cleaner layout with explicit
step names matching the production workflow and a single `pipeline.py` entry point.
