## Pipeline UI

Full Gradio app that chains all four APIs:

```bash
# Terminal 1
./run_trellis_api.sh

# Terminal 2
./run_mesh_api.sh

# Terminal 3
./run_pipeline_ui.sh
```

Open http://localhost:7860


Two HTTP servers power the workflow:

| Service | Env | Port | Launcher |
|---------|-----|------|----------|
| TRELLIS.2 image→3D | TRELLIS conda | 8100 | `run_trellis_api.sh` |
| AutoRemesher / meshoptimizer / xatlas | meshcleaning `.venv` | 8101 | `run_mesh_api.sh` |

## Start servers

```bash
# Terminal 1 — TRELLIS (conda)
/devwork/teja/meshcleaning/ui_workflow/run_trellis_api.sh

# Terminal 2 — mesh tools (.venv)
/devwork/teja/meshcleaning/ui_workflow/run_mesh_api.sh
```

Install mesh API deps once:

```bash
source /devwork/teja/meshcleaning/.venv/bin/activate
pip install -r /devwork/teja/meshcleaning/ui_workflow/requirements-api.txt
```

Install TRELLIS API deps once (if needed):

```bash
/devwork/teja/meshcleaning/ui_workflow/run_trellis_python.sh -m pip install fastapi uvicorn python-multipart
```

---

## 1. TRELLIS.2 — image to 3D

**Health**

```bash
curl -s http://localhost:8100/v1/health | jq
```

**Generate GLB from image**

```bash
curl -s -X POST http://localhost:8100/v1/trellis/generate \
  -F "image=@/path/to/input.png" \
  -F "resolution=512" \
  -F "skip_video=false" | jq
```

Response:

```json
{
  "job_id": "trellis_abc123",
  "status": "completed",
  "glb": "/devwork/teja/meshcleaning/ui_workflow/jobs/trellis_abc123/model_....glb",
  "video": "...mp4",
  "stats": { "vertices": 12345, "faces": 24680, "file_kb": 512.3 }
}
```

**Download GLB**

```bash
JOB_ID=trellis_abc123
curl -o out.glb "http://localhost:8100/v1/jobs/${JOB_ID}/glb"
```

**CLI (no HTTP)**

```bash
/devwork/teja/meshcleaning/ui_workflow/run_trellis_python.sh \
  /devwork/teja/meshcleaning/ui_workflow/generate_3d.py \
  --image /path/to/input.png \
  --output-dir /tmp/out \
  --output-name model \
  --pipeline-type 512
```

---

## 2. AutoRemesher (solidify prep)

Used by the pipeline for **solidify only** (watertight cleanup before remesh). Quad remesh is done by Instant Meshes (section 4).

**API**

```bash
curl -s -X POST http://localhost:8101/v1/autoremesher \
  -F "mesh=@/path/to/input.glb" \
  -F "target_quads=5000" \
  -F "solid_only=true" | jq
```

**CLI**

```bash
source /devwork/teja/meshcleaning/.venv/bin/activate
python /devwork/teja/meshcleaning/ui_workflow/cli/autoremesher.py \
  /path/to/input.glb \
  -o /path/to/solid.glb \
  --solid-only
```

---

## 3. meshoptimizer (triangle prep)

Reduce triangle count before Instant Meshes. Typical prep: `target_tris ≈ 2 × target_quads`.

**API**

```bash
curl -s -X POST http://localhost:8101/v1/meshoptimizer \
  -F "mesh=@/path/to/solid.glb" \
  -F "target_tris=10000" \
  -F "simplify_error=0.01" | jq
```

**CLI**

```bash
source /devwork/teja/meshcleaning/.venv/bin/activate
python /devwork/teja/meshcleaning/ui_workflow/cli/meshoptimizer.py \
  /path/to/solid.glb \
  -o /path/to/reduced.glb \
  --target-tris 10000
```

---

## 4. Instant Meshes (triangle → quad remesh)

Field-aligned **pure quad** remesh from a triangle mesh. Output includes GLB (preview) and OBJ (real quads).

Build the binary once (system deps + cmake):

```bash
/devwork/teja/meshcleaning/scripts/build_instant_meshes.sh
```

All Python/API calls use **meshcleaning `.venv`**:

```bash
source /devwork/teja/meshcleaning/.venv/bin/activate
```

**API**

```bash
curl -s -X POST http://localhost:8101/v1/instant-meshes \
  -F "mesh=@/path/to/reduced.glb" \
  -F "target_quads=5000" \
  -F "from_meshopt=true" \
  -F "boundaries=true" | jq
```

Response includes `output_glb` and `output_obj`. `stats.obj_quads` reports quad count.

**CLI (.venv)**

```bash
source /devwork/teja/meshcleaning/.venv/bin/activate
python /devwork/teja/meshcleaning/ui_workflow/cli/instant_meshes.py \
  /path/to/reduced.glb \
  -o /path/to/quad.glb \
  --target-quads 5000 \
  --from-meshopt
```

**Legacy:** `/v1/quadriflow` still available if QuadriFlow is built.

---

## 5. Texture transfer (final step)

Bake TRELLIS textures onto the Instant Meshes quad OBJ. Default `uv_method=box` and `output_format=both`.

**Primary output is `output_obj`** — pure quad faces (`f v1 v2 v3 v4`). GLB is preview-only.

**API**

```bash
curl -s -X POST http://localhost:8101/v1/transfer-texture \
  -F "source=@/path/to/trellis.glb" \
  -F "target_obj=@/path/to/quadriflow.obj" \
  -F "target=@/path/to/quadriflow.glb" \
  -F "texture_size=512" \
  -F "uv_method=box" \
  -F "output_format=both" \
  -F "mode=texture" | jq
```

Download quad OBJ:

```bash
curl -o out.obj "http://localhost:8101/v1/jobs/${JOB_ID}/download?format=obj"
```

**CLI**

```bash
source /devwork/teja/meshcleaning/.venv/bin/activate
python /devwork/teja/meshcleaning/ui_workflow/cli/transfer_texture.py \
  --source /path/to/trellis.glb \
  --target /path/to/remeshed.glb \
  -o /path/to/final_textured.glb \
  --texture-size 128
```

---

## 6. xatlas UV (optional) unwrap

Run **after** autoremesher and meshoptimizer to generate clean UVs.

**API**

```bash
curl -s -X POST http://localhost:8101/v1/xatlas \
  -F "mesh=@/path/to/remeshed.glb" \
  -F "resolution=2048" \
  -F "padding=2" | jq
```

**CLI**

```bash
source /devwork/teja/meshcleaning/.venv/bin/activate
python /devwork/teja/meshcleaning/ui_workflow/cli/xatlas_uv.py \
  /path/to/input.glb \
  -o /path/to/output_uv.glb \
  --resolution 2048
```

**Download mesh API output**

```bash
JOB_ID=autoremesh_abc123
curl -o out.glb "http://localhost:8101/v1/jobs/${JOB_ID}/download"
```

---

## Full pipeline example

```bash
BASE=http://localhost:8100
MESH=http://localhost:8101
IMG=/path/to/photo.png

# 1) TRELLIS
TRELLIS=$(curl -s -X POST "$BASE/v1/trellis/generate" -F "image=@${IMG}" -F "resolution=512")
JOB1=$(echo "$TRELLIS" | jq -r .job_id)
GLB1=$(echo "$TRELLIS" | jq -r .glb)

# 2) Solidify
SOLID=$(curl -s -X POST "$MESH/v1/autoremesher" \
  -F "mesh=@${GLB1}" -F "target_quads=5000" \
  -F "solid_only=true")
GLB_SOLID=$(echo "$SOLID" | jq -r .output_glb)
echo "SOLID: $GLB_SOLID"

# 3) meshoptimizer prep
REDUCED=$(curl -s -X POST "$MESH/v1/meshoptimizer" \
  -F "mesh=@${GLB_SOLID}" -F "target_tris=10000")
GLB_REDUCED=$(echo "$REDUCED" | jq -r .output_glb)
echo "REDUCED: $GLB_REDUCED"

# 4) Instant Meshes
IM=$(curl -s -X POST "$MESH/v1/instant-meshes" \
  -F "mesh=@${GLB_REDUCED}" -F "target_quads=5000" \
  -F "from_meshopt=true" -F "boundaries=true")
GLB_IM=$(echo "$IM" | jq -r .output_glb)
OBJ_IM=$(echo "$IM" | jq -r .output_obj)
echo "INSTANT MESHES GLB: $GLB_IM"
echo "INSTANT MESHES OBJ: $OBJ_IM"

# 5) Texture transfer
FINAL=$(curl -s -X POST "$MESH/v1/transfer-texture" \
  -F "source=@${GLB1}" -F "target_obj=@${OBJ_IM}" -F "target=@${GLB_IM}" \
  -F "texture_size=512" -F "uv_method=box" -F "output_format=both")
echo "$FINAL" | jq
```

Job artifacts are stored under `ui_workflow/jobs/<job_id>/`.
