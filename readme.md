# tk_vision

Web-served, SAM3-driven semi-automatic segmentation pipeline for Robocup vision. Captures clips from a RealSense (or a folder of frames), seeds masks from text prompts via SAM3, propagates them through the clip with the SAM3 video tracker, lets you review/edit in a browser, then exports a YOLO-seg dataset and trains/tests a YOLO11 model — all without leaving the SPA.

The legacy CLI scripts (`yolo_tuning/`) still work for the original cv2-window flow; new work happens through the web UI under `web/` + `server/`.

## Requirements

- Python 3.10
- A CUDA GPU for SAM3 (bfloat16 recommended; tested on RTX 5070 Ti / sm_120, PyTorch 2.11+cu128)
- Intel RealSense (optional — folder import works without one)
- Node 20+ if rebuilding the SPA

Install Python deps via `pip install -e ./server` (preferred — pulls FastAPI, ultralytics, transformers, albumentations, etc. from `pyproject.toml`) or `pip install -r requirements.txt` for the legacy scripts.

### SAM3 weights

The default checkpoint at `sam3_checkpoint_hf/` was converted from `sam3.pt` and has trained `tracker_neck` weights stored under `tracker_model.tracker_neck.*`; `Sam3Engine.load` aliases them to top-level `tracker_neck.*` at startup. If `tk_vision serve` errors with `tracker_neck patch loaded N/22 weights`, install a clean checkpoint:

```
tk_vision fetch-weights --source local                              # verify the on-disk copy
tk_vision fetch-weights --source hf --repo facebook/sam3 --yes      # pull from Hugging Face
tk_vision fetch-weights --source url --url https://… --sha256 … --yes
```

`--source hf|url` requires `--yes` (or `TK_VISION_ALLOW_DOWNLOAD=1`); the canary refuses to install unless all 22 tracker_neck weights load.

## Pipeline

```
tk_vision serve                  # FastAPI on :28000, SPA at http://localhost:28000
```

Then in the browser:

1. **Clips page** — record a live RealSense clip, import a folder of images, or pick an existing clip.
2. **Label page** — `Seed first frame` runs SAM3 against the ontology; click on a track to refine with positive/negative points; press `Propagate` to run the SAM3 video tracker forward through the clip in chunks. Scrub frames, mark bad frames `Delete`/`Restore`/`Prune from here`, re-seed problem frames.
3. **Export YOLO-seg** writes `data/runs/<run_id>/{images,labels}/{train,val}/` + `data.yaml`. Per-clip split is the default to avoid temporal leakage; multi-contour polygons are preserved.
4. **Augment** applies the Albumentations ops in `configs/default.yaml > augment.ops` plus optional copy-paste, materializing `<stem>_aug{0..N-1}.jpg`/`.txt` in place.
5. **Train** spawns `python -m tk_vision._train_runner` as a subprocess running `model.train(...)`; stdout streams to the SPA via WebSocket. Cancel via SIGTERM; `metrics.json` is written from `results.csv` on success.
6. **Test → /test/<run_id>/<clip_id>** loads a `.pt` weights path and runs `model.predict` on every non-deleted frame, persisting predictions as JSON. The frame scrubber overlays predicted polygons + class scores against the original frame.

### Ontology guidance

`resource/ontology.json` is `{prompt: label}`:

```json
{"<text prompt>": "<label>"}
```

The keys feed SAM3's open-vocab text head, which was trained on short noun phrases (`"cat"`, `"remote"`). Keep keys to ≤6 tokens, simple visual descriptors, no task jargon. Long phrases collapse SAM3's `presence_logits` and seed returns nothing. Values become YOLO class names — keep them stable.

### Score mode

`configs/default.yaml > sam3.score_mode` (or the SPA toggle on the Label page) selects:

- `native` — Meta default: `score = sigmoid(pred_logits) * sigmoid(presence_logits)`. Use with short prompts.
- `per_query` — drops the presence multiplier. Use when you keep verbose / domain-specific prompts and `native` returns nothing.

## Configuration

`configs/default.yaml` controls server bind, capture device, SAM3 checkpoint path/dtype, propagation chunk size, augmentation ops, and training hyperparameters. Field-level docs live in `server/tk_vision/config.py`. Override via `tk_vision serve --config <path>` or by editing the YAML in place.

## Background-job model

Long operations (propagate, train, infer) follow a uniform pattern:

- `POST /api/.../{op}` — start. Returns `{job_id, ...}` and starts an asyncio task.
- `GET /api/.../{op}/{job_id}` — poll status.
- `DELETE /api/.../{op}/{job_id}` — cancel (SIGTERM for subprocess jobs, `cancel.set()` for in-process loops).
- `WS /ws/{op}/.../{job_id}` — stream `log`/`frame`/`done`/`error`/`cancelled` events.

Job status is one of `pending | running | done | error | cancelled` (`JobStatus` in `web/src/api/rest.ts`).pip install "httpx[socks]"

## Running tests

```
cd server
PYTHONPATH=. python -m pytest tests/ -q --override-ini "addopts="
```

The GPU smoke test (`test_propagate_smoke.py`) and the tracker_neck patch test require a working CUDA + SAM3 checkpoint and skip otherwise.

## Legacy scripts (yolo_tuning/)

Still functional for the original cv2-window flow; tagged `legacy/v0` in git. See git history for the pre-web README.
