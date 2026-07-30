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

## 2. AutoRemesher

**API**

```bash
curl -s -X POST http://localhost:8101/v1/autoremesher \
  -F "mesh=@/path/to/input.glb" \
  -F "target_quads=5000" \
  -F "prep_target_tris=0" \
  -F "solid_only=false" \
  -F "from_solid=false" | jq
```

**CLI**

```bash
source /devwork/teja/meshcleaning/.venv/bin/activate
python /devwork/teja/meshcleaning/ui_workflow/cli/autoremesher.py \
  /path/to/input.glb \
  -o /path/to/output.glb \
  --target-quads 5000
```

---

## 3. meshoptimizer

**API**

```bash
curl -s -X POST http://localhost:8101/v1/meshoptimizer \
  -F "mesh=@/path/to/input.glb" \
  -F "target_tris=30000" \
  -F "simplify_error=0.01" | jq
```

**CLI**

```bash
source /devwork/teja/meshcleaning/.venv/bin/activate
python /devwork/teja/meshcleaning/ui_workflow/cli/meshoptimizer.py \
  /path/to/input.glb \
  -o /path/to/output.glb \
  --target-tris 30000
```

---

## 4. Texture transfer (final step)

Bake TRELLIS textures onto the remeshed mesh using `scripts/transfer_texture.py`.

**API**

```bash
curl -s -X POST http://localhost:8101/v1/transfer-texture \
  -F "source=@/path/to/trellis.glb" \
  -F "target=@/path/to/remeshed.glb" \
  -F "texture_size=128" \
  -F "mode=texture" | jq
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

## 5. xatlas UV (optional) unwrap

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

# 2) AutoRemesher
AR=$(curl -s -X POST "$MESH/v1/autoremesher" -F "mesh=@${GLB1}" -F "target_quads=5000")
GLB2=$(echo "$AR" | jq -r .output_glb)

# 3) meshoptimizer
MO=$(curl -s -X POST "$MESH/v1/meshoptimizer" -F "mesh=@${GLB2}" -F "target_tris=30000")
GLB3=$(echo "$MO" | jq -r .output_glb)

# 4) xatlas UV
UV=$(curl -s -X POST "$MESH/v1/xatlas" -F "mesh=@${GLB3}" -F "resolution=2048")
echo "$UV" | jq
```

Job artifacts are stored under `ui_workflow/jobs/<job_id>/`.
