# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

YOLO fine-tuning pipeline for Robocup competition object identification. Captures images from an Intel RealSense camera, auto-labels them with GroundingDINO (boxes) or LangSAM + SAM (masks), then fine-tunes a YOLO11 model. Both detection and segmentation flows exist as parallel `*` / `*_seg` scripts.

## Environment

- Python 3.10. Conda env name is `visionTrain` (see `environment.yml`); `pip install -r requirements.txt` is the alternative.
- `lang-sam` and `segment-anything` install from GitHub (see `requirements.txt` `-e git+...` lines). When using the conda yml file these must be installed manually from GitHub.
- `sam_vit_b_01ec64.pth` must be downloaded to the repo root (gitignored). `create_dataset_seg.py` looks for it via `SAM_CHECKPOINT_PATH` env var, falling back to `<repo>/sam_vit_b_01ec64.pth`.
- All training/dataset scripts must be run from inside `yolo_tuning/` (they resolve `.env` and ontology paths relative to CWD).

## Configuration via `yolo_tuning/.env`

Every entry point (`load_dotenv()` early in each script) reads:

- `DATASET_DIR` — output dir for `images/`, `labels/`. **Must be empty before a fresh capture run** or new samples will mix with stale data.
- `ONTOLOGY_PATH` — JSON map of `"<GroundingDINO/LangSAM prompt>": "<class_label>"`. Keys are prompts, values become YOLO class names. Order of values defines class IDs.
- `CHECKPOINT_DIR` — Ultralytics `project=` dir for training runs.
- `BEST_MODEL_PATH` — path used by `test_new_model*.py` to load the trained model.

## Pipeline (run from `yolo_tuning/`)

Detection flow:
```
python -m create_dataset       # capture + GroundingDINO auto-label boxes (RealSense required)
python -m prepare_dataset      # 80/20 train/val split (in-place under DATASET_DIR)
python -m tune_YOLOv11         # trains yolo11s.pt → ./yolo_finetuned_best.pt
python -m test_new_model       # live inference from RealSense
```

Segmentation flow (parallel scripts):
```
python -m create_dataset_seg          # LangSAM auto-masks; supports manual SAM-prompt mode
python -m create_dataset_seg_video    # video/sequence variant using SAM2 propagation
python -m tune_YOLOv11_seg            # trains yolo11m-seg.pt → ./yolo_seg_finetuned_best.pt
python -m test_new_model_seg          # live segmentation inference
python -m test_new_model_seg_inst     # variant that displays via matplotlib instead of cv2
```

`combine.py` merges two dataset directories, suffixing copied files with `_2` to avoid name clashes — edit the hardcoded `source_dataset`/`target_dataset` paths at the bottom before running.

## Architecture notes

- The training scripts call `prepare_dataset.split_dataset()` themselves if `images/train` doesn't exist yet, and they auto-generate `<DATASET_DIR>/data.yaml` from the ontology each run. You don't separately invoke `prepare_dataset` unless you want to split without training.
- `tune_YOLOv11.py` hardcodes the base weights (`yolo11s.pt`) and 50 epochs; `tune_YOLOv11_seg.py` uses `yolo11m-seg.pt` and 100 epochs. Adjust at the `model = YOLO(...)` line and the `model.train(...)` call.
- After training, the best weights are copied from `<CHECKPOINT_DIR>/yolo_finetuned/weights/best.pt` to `./yolo_finetuned_best.pt` (or `./yolo_seg_finetuned_best.pt`). Loss curves + confusion matrix are combined into `training_plots.png` in the run directory.
- `create_dataset.py` interaction loop (per frame): `↑/↓` cycle through detections, `d` deletes the highlighted one, `s` saves the kept set as YOLO labels, `space` skips, `q` quits. The display image is padded 50 px on all sides; predictions are deep-copied and offset for display while the originals remain in source coordinates for label writing.
- `create_dataset_seg.py` adds a manual mode (`m`): clicked points are fed to `SamPredictor` to generate a mask preview; `Enter` confirms the segment, `↑/↓` chooses its class, `a` adds it. It uses `thefuzz` to fuzzy-match LangSAM's returned phrases back to ontology prompts (≥80 score required).
- Segmentation labels are written in YOLO-seg polygon format (`class x1 y1 x2 y2 ...` normalized).

## Things that bite

- The RealSense pipeline opens `rs.stream.color` at 640×480 for capture/detection scripts but 1280×720 for `test_new_model_seg*.py`. Camera must be plugged in via USB before running any capture/test script.
- `create_dataset.py` mutates `self.base_model.dino_model.model` to move it to CUDA — this depends on `autodistill-groundingdino` internals and will silently fall back to CPU if the attribute path changes.
- Several files set `os.environ["HF_DATASETS_OFFLINE"] = "1"` at top-of-module; the transformers offline flag is intentionally commented out so `bert-base-uncased` can pre-cache. First run needs network access (HuggingFace).
- `*dataset*/`, `*.pt`, `runs*/`, and `sam_vit_b_01ec64.pth` are gitignored — don't commit captured data or weights.
