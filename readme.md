# YOLO Finetuning
Package for collecting data, splitting datasets, and training YOLO (boxes + masks) for competition items.

## Requirements
- Python 3.10; install with `pip install -r requirements.txt`.
- Download `sam_vit_b_01ec64.pth` to the repo root. If you use the conda yml, install LangSAM and SAM2 manually from GitHub.

## Ontology
Edit `yolo_tuning/resource/ontology.json` with GroundingDINO/LangSAM prompts mapped to class labels:
```json
{"<prompt>": "label"}
```

## Quick workflow (all commands run from repo root)
Environment variables (optional): `DATASET_DIR` (det), `DATASET_SEG_DIR` (seg), `ONTOLOGY_PATH`, `CHECKPOINT_DIR`, `YOLO_BASE_WEIGHTS`, `YOLO_SEG_WEIGHTS`, `VISION_TRAIN_SEED`.

### 1) Collect data (RealSense live)
- Boxes only: `python -m yolo_tuning.vision_tuning.cli create-bbox --dataset-dir dataset_det`
- Seg masks: `python -m yolo_tuning.vision_tuning.cli create-seg --dataset-dir dataset_seg`
- Seg masks via continuous stream (not from file): `python -m yolo_tuning.vision_tuning.cli create-seg-stream --dataset-dir dataset_seg`

Controls are unchanged from the original scripts:
- Boxes: up/down to select, `d` delete, `s` save, space skip, `q` quit (fix prompts if many errors).
- Seg: space capture, up/down to select masks, `d` delete, `m` manual mode (click points then Enter, choose label with arrows, `a` add), `s` save, `esc` discard.

### 2) Split raw dataset
If your dataset is not already split:  
`python -m yolo_tuning.vision_tuning.cli split --dataset-dir dataset_det --train-ratio 0.8`

### 3) Train
- YOLO detector: `python -m yolo_tuning.vision_tuning.cli train-bbox --dataset-dir dataset_det`
- YOLO-seg: `python -m yolo_tuning.vision_tuning.cli train-seg --dataset-dir dataset_seg`

After training, best weights are copied to `yolo_finetuned_best.pt` or `yolo_seg_finetuned_best.pt`; full logs live in `CHECKPOINT_DIR/<run>/`.

### 4) Live test
- Detector: `python -m yolo_tuning.vision_tuning.cli test-bbox --model-path yolo_finetuned_best.pt`
- Segmenter: `python -m yolo_tuning.vision_tuning.cli test-seg --model-path yolo_seg_finetuned_best.pt`

## Legacy entrypoints (still work)
- Dataset split: `python -m yolo_tuning.prepare_dataset`
- Training: `python -m yolo_tuning.tune_YOLOv11` or `python -m yolo_tuning.tune_YOLOv11_seg`
- Live tests: `python -m yolo_tuning.test_new_model` or `python -m yolo_tuning.test_new_model_seg`

## Package layout (developer facing)
- `yolo_tuning/vision_tuning/config.py` – central configuration (paths, weights, device).
- `yolo_tuning/vision_tuning/ontology.py` – load ontology JSON and emit YOLO data.yaml.
- `yolo_tuning/vision_tuning/datasets/` – bbox/seg/stream collectors and dataset splitter.
- `yolo_tuning/vision_tuning/training/` – YOLO training helpers for detector/segmenter.
- `yolo_tuning/vision_tuning/testing/` – wrappers for live inference.
- `yolo_tuning/vision_tuning/cli.py` – single CLI entry with subcommands for all tasks.
