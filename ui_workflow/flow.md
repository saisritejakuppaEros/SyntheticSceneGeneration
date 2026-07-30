# Mesh Pipeline — API Flow

Document for chaining **image → production mesh** via HTTP APIs in `/devwork/teja/meshcleaning/ui_workflow`.

---

## Overview

```
Input image (PNG/JPG)
       │
       ▼
┌──────────────────┐   port 8100   TRELLIS conda env
│  TRELLIS 2.0 API │   POST /v1/trellis/generate
└────────┬─────────┘
         │ textured GLB (high-poly, ~100k tris)
         ▼
┌──────────────────┐   port 8101   meshcleaning .venv
│  AutoRemesher    │   POST /v1/autoremesher  (solid_only=true)
│  — solidify      │
└────────┬─────────┘
         │ solid GLB (watertight, junk removed)
         ▼
┌──────────────────┐   port 8101
│  AutoRemesher    │   POST /v1/autoremesher  (from_solid=true)
│  — quad remesh   │
└────────┬─────────┘
         │ remeshed GLB (quads, NO texture — geometry only)
         ▼
┌──────────────────┐   port 8101
│  Texture transfer│   POST /v1/transfer-texture
│  (scripts/       │   source=TRELLIS GLB, target=remeshed GLB
│   transfer_      │
│   texture.py)    │
└────────┬─────────┘
         │ final textured GLB
         ▼
      DONE
```

**Optional / not in active Gradio pipeline (commented out or manual):**
- `POST /v1/meshoptimizer` — triangle reduction via gltfpack
- `POST /v1/xatlas` — UV unwrap before texture bake (recommended for better textures)

---

## Servers

| Server | Port | Environment | Start script |
|--------|------|-------------|--------------|
| TRELLIS API | 8100 | `/home/parth_h200/parth/TRELLIS.2/trellis` conda | `./run_trellis_api.sh` |
| Mesh API | 8101 | `/devwork/teja/meshcleaning/.venv` | `./run_mesh_api.sh` |
| Gradio UI | 7860 | meshcleaning `.venv` | `./run_pipeline_ui.sh` |

```bash
cd /devwork/teja/meshcleaning/ui_workflow

# Terminal 1
./run_trellis_api.sh

# Terminal 2
./run_mesh_api.sh

# Terminal 3 (optional UI)
./run_pipeline_ui.sh
```

Health checks:
```bash
curl -s http://localhost:8100/v1/health | jq
curl -s http://localhost:8101/v1/health | jq
```

---

## Active pipeline (4 steps)

This matches `pipeline_app.py` / Gradio UI.

### Step 1 — TRELLIS 2.0 (image → 3D)

**Env:** TRELLIS conda  
**Backend:** `inference.py` → `Trellis2ImageTo3DPipeline`  
**Input:** image file  
**Output:** textured GLB (+ optional MP4 turntable)

```bash
curl -s -X POST http://localhost:8100/v1/trellis/generate \
  -F "image=@/path/to/input.png" \
  -F "resolution=512" \
  -F "skip_video=true" | jq
```

| Param | Values | Notes |
|-------|--------|-------|
| `resolution` | `512`, `1024`, `1536` | Higher = slower, more VRAM |
| `skip_video` | `true` / `false` | MP4 preview |

**Response fields:**
```json
{
  "job_id": "trellis_xxxx",
  "glb": "/devwork/teja/meshcleaning/ui_workflow/jobs/trellis_xxxx/model_....glb",
  "video": "...mp4 or null",
  "stats": { "vertices": N, "faces": N, "file_kb": N }
}
```

Save as `GLB_TRELLIS` for later steps.

---

### Step 2 — Solidify (pre-remesh)

**Env:** meshcleaning `.venv`  
**Backend:** `scripts/run_autoremesher.py` with `--solid-only`  
**Input:** TRELLIS GLB  
**Output:** solid watertight GLB

```bash
curl -s -X POST http://localhost:8101/v1/autoremesher \
  -F "mesh=@${GLB_TRELLIS}" \
  -F "target_quads=5000" \
  -F "solid_only=true" \
  -F "from_solid=false" | jq
```

What it does:
- Drop tiny mesh fragments
- Keep largest shell
- pymeshfix repair → watertight solid
- **Does NOT remesh yet**

| Param | Default | Max |
|-------|---------|-----|
| `target_quads` | 5000 | 100000 (1 lakh) |

**Response:** `output_glb` → save as `GLB_SOLID`

---

### Step 3 — AutoRemesher (quad remesh)

**Env:** meshcleaning `.venv`  
**Backend:** `scripts/run_autoremesher.py` + `autoremesher` binary  
**Input:** solid GLB from step 2  
**Output:** quad-dominant remeshed GLB (**no texture / no UV**)

```bash
curl -s -X POST http://localhost:8101/v1/autoremesher \
  -F "mesh=@${GLB_SOLID}" \
  -F "target_quads=5000" \
  -F "solid_only=false" \
  -F "from_solid=true" | jq
```

**Important:** AutoRemesher exports **geometry only** (`v` + `f` in OBJ). It does **not** transfer textures. Internal UVs exist for preview but are not written to output.

**Response:** `output_glb` → save as `GLB_REMESH`

---

### Step 4 — Texture transfer (final)

**Env:** meshcleaning `.venv`  
**Backend:** `services/transfer_texture.py` (xatlas UV + bake)  
**Input:**
- `source` = **TRELLIS GLB** (original textured high-poly)
- `target` = **remeshed GLB** from step 3

```bash
curl -s -X POST http://localhost:8101/v1/transfer-texture \
  -F "source=@${GLB_TRELLIS}" \
  -F "target=@${GLB_REMESH}" \
  -F "texture_size=512" \
  -F "uv_padding=4" \
  -F "mode=texture" | jq
```

| Param | Default | Notes |
|-------|---------|-------|
| `texture_size` | 512 | Baked atlas resolution (128–2048) |
| `uv_padding` | 4 | xatlas chart padding in texels |
| `mode` | `texture` | `vertex` = vertex colors only (debug) |

Method (improved):
1. **xatlas** unwrap on remesh target (proper UV charts, no box projection)
2. Align TRELLIS source bbox to target
3. Closest-point sample colors from TRELLIS (k=16 neighbors)
4. Rasterize into atlas using xatlas UVs
5. Fill small UV gaps via dilation

**Response:** `output_glb` → **final asset**

---

## Full curl chain (copy-paste)

```bash
BASE=http://localhost:8100
MESH=http://localhost:8101
IMG=/path/to/photo.png

# 1) TRELLIS
TRELLIS=$(curl -s -X POST "$BASE/v1/trellis/generate" \
  -F "image=@${IMG}" -F "resolution=512" -F "skip_video=true")
GLB_TRELLIS=$(echo "$TRELLIS" | jq -r .glb)
echo "TRELLIS: $GLB_TRELLIS"

# 2) Solidify
SOLID=$(curl -s -X POST "$MESH/v1/autoremesher" \
  -F "mesh=@${GLB_TRELLIS}" -F "target_quads=5000" \
  -F "solid_only=true" -F "from_solid=false")
GLB_SOLID=$(echo "$SOLID" | jq -r .output_glb)
echo "SOLID: $GLB_SOLID"

# 3) Remesh
REMESH=$(curl -s -X POST "$MESH/v1/autoremesher" \
  -F "mesh=@${GLB_SOLID}" -F "target_quads=5000" \
  -F "solid_only=false" -F "from_solid=true")
GLB_REMESH=$(echo "$REMESH" | jq -r .output_glb)
echo "REMESH: $GLB_REMESH"

# 4) Texture transfer
FINAL=$(curl -s -X POST "$MESH/v1/transfer-texture" \
  -F "source=@${GLB_TRELLIS}" -F "target=@${GLB_REMESH}" \
  -F "texture_size=512" -F "mode=texture")
GLB_FINAL=$(echo "$FINAL" | jq -r .output_glb)
echo "FINAL: $GLB_FINAL"
```

---

## Optional steps (API exists, not in active UI)

### meshoptimizer — triangle reduction

```bash
curl -s -X POST http://localhost:8101/v1/meshoptimizer \
  -F "mesh=@${GLB_REMESH}" \
  -F "target_tris=30000" \
  -F "simplify_error=0.01" | jq
```

Backend: `scripts/run_meshoptimizer.py` → `gltfpack`

### xatlas — UV unwrap (run BEFORE texture transfer for better results)

```bash
curl -s -X POST http://localhost:8101/v1/xatlas \
  -F "mesh=@${GLB_REMESH}" \
  -F "resolution=2048" \
  -F "padding=2" | jq
```

Backend: `services/xatlas_uv.py` (Python `xatlas` package)

Suggested improved flow:
```
remesh → xatlas UV → transfer_texture (with higher texture_size)
```

---

## Repo layout

```
ui_workflow/
├── run_trellis_api.sh      # :8100 TRELLIS conda
├── run_mesh_api.sh         # :8101 .venv
├── run_pipeline_ui.sh      # :7860 Gradio
├── run_trellis_python.sh   # conda wrapper for TRELLIS scripts
├── pipeline_app.py         # Gradio — chains all 4 active steps
├── api_client.py           # HTTP client used by Gradio
├── inference.py            # TRELLIS inference (used by trellis API)
├── api/
│   ├── trellis_api.py      # :8100 FastAPI
│   ├── mesh_api.py         # :8101 FastAPI
│   └── common.py           # jobs dir, mesh_stats, MAX_TARGET_QUADS=100000
├── services/
│   ├── mesh_tools.py       # subprocess wrappers → scripts/
│   └── xatlas_uv.py
├── cli/                    # standalone CLI wrappers
└── jobs/                   # API job outputs (gitignored)
    └── <job_id>/
        ├── manifest.json
        └── output_*.glb

scripts/                      # underlying mesh tools (meshcleaning root)
├── run_autoremesher.py       # solidify + autoremesher
├── run_meshoptimizer.py
└── transfer_texture.py       # texture bake source→target
```

---

## API response shape (mesh endpoints)

All mesh API POST endpoints return:
```json
{
  "job_id": "autoremesh_xxxx",
  "status": "completed",
  "input": "/path/to/input.glb",
  "output_glb": "/path/to/output.glb",
  "stats": { "vertices": N, "faces": N, "file_kb": N },
  "log": "...",
  "time": "ISO8601"
}
```

Download:
```bash
curl -o out.glb "http://localhost:8101/v1/jobs/${JOB_ID}/download"
```

---

## Gradio UI outputs

`pipeline_app.py` shows **4 columns**, each with Solid + Wireframe tabs:

| Column | Stage | Source |
|--------|-------|--------|
| 1 | TRELLIS | step 1 |
| 2 | Solid | step 2 |
| 3 | Remeshed | step 3 (untextured) |
| 4 | Final textured | step 4 |

Download button = final textured GLB from step 4.

---

## Known limitations

1. **AutoRemesher has no texture transfer** — output is bare geometry; texturing is always step 4.
2. **Texture quality** — use `texture_size=512+`; step 4 now uses **xatlas UVs** (not box projection).
3. **solid_only output path** — mesh API resolves solid GLB from `autoremesher_out/*_solid.glb` if needed (`resolve_autoremesher_output` in `api/common.py`).
4. **meshoptimizer** — implemented but commented out in Gradio pipeline.
5. **Two envs required** — TRELLIS needs GPU + conda; mesh tools use `.venv` only.

---

## Environment variables

| Variable | Default |
|----------|---------|
| `TRELLIS_API_URL` | `http://localhost:8100` |
| `MESH_API_URL` | `http://localhost:8101` |
| `TRELLIS_API_PORT` | `8100` |
| `MESH_API_PORT` | `8101` |
| `GRADIO_PORT` | `7860` |
| `PIPELINE_API_TIMEOUT` | `3600` (seconds) |

---

## Dependencies

**TRELLIS conda** (`run_trellis_api.sh`):
- PyTorch, TRELLIS.2, fastapi, uvicorn, python-multipart

**Mesh .venv** (`run_mesh_api.sh`):
```bash
pip install -r /devwork/teja/meshcleaning/ui_workflow/requirements-api.txt
# fastapi, uvicorn, xatlas, gradio, requests, trimesh, scipy, pillow
```

AutoRemesher binary: `/devwork/teja/meshcleaning/.venv/bin/autoremesher`  
gltfpack: `/devwork/teja/meshcleaning/meshoptimizer/gltfpack`
