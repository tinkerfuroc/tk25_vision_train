# YOLO Finetuning
Utilities for RealSense data collection, YOLO dataset preparation, and YOLO11 fine-tuning (detection + segmentation).

## Environment and Setup
- Python 3.10
- Install dependencies: `pip install -r requirements.txt`
- Optional conda environment: `conda env create -f environment.yml` (env name: `visionTrain`)
- Put SAM3 checkpoint at `model/sam3/sam3.pt` (or set `SAM3_CHECKPOINT_PATH`)

All commands below are run from repo root.

### venv quickstart
```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
python -m yolo_tuning.vision_tuning.cli -h
```

Full guide: `docs/venv_setup.md`

## Configuration (Optional)
Environment variables used by `VisionConfig`:
- `DATASET_DIR` (default detection dataset: `dataset`)
- `DATASET_SEG_DIR` (default segmentation dataset: `dataset_seg`)
- `ONTOLOGY_PATH` (default: `yolo_tuning/resource/ontology.json`)
- `CHECKPOINT_DIR` (default: `runs`)
- `YOLO_BASE_WEIGHTS` (default: `yolo11s.pt`)
- `YOLO_SEG_WEIGHTS` (default: `yolo11s-seg.pt`)
- `VISION_TRAIN_SEED` (default: `42`)
- `SAM3_CHECKPOINT_PATH` (default: `model/sam3/sam3.pt`)

## Ontology
Edit `yolo_tuning/resource/ontology.json` with prompt-to-label mapping:
```json
{
  "<GroundingDINO or LangSAM prompt>": "label"
}
```

## Workflow A: Canonical CLI
Primary entrypoint:
```bash
python -m yolo_tuning.vision_tuning.cli <subcommand>
```

CLI architecture:
- Parser definitions: `yolo_tuning/vision_tuning/commands/parser.py`
- Command dispatcher: `yolo_tuning/vision_tuning/commands/runner.py`
- Backward-compatible entry: `yolo_tuning/vision_tuning/cli.py`

### 1) Collect data (RealSense live)
```bash
python -m yolo_tuning.vision_tuning.cli create-bbox --dataset-dir dataset_det
python -m yolo_tuning.vision_tuning.cli create-seg --dataset-dir dataset_seg
python -m yolo_tuning.vision_tuning.cli create-seg-stream --dataset-dir dataset_seg_stream
python -m yolo_tuning.vision_tuning.cli create-seg --dataset-dir dataset_seg --input-mode images --source-path /path/to/images
python -m yolo_tuning.vision_tuning.cli create-seg-stream --dataset-dir dataset_seg --input-mode video --source-path /path/to/video.mp4
```

Common options (most subcommands):
- `--device` (for example `cpu`, `cuda`, `cuda:0`)
- `--ontology-path`
- `--checkpoint-dir`
- `--input-mode` (`realsense|images|video`) for segmentation commands
- `--source-path` for `images`/`video` modes
- `--enable-crop-augment --crop-variants --crop-scale-min --crop-scale-max`

Key controls in data collection:
- BBox: up/down select, `d` delete, `s` save, space skip, `q` quit
- Seg/Seg-stream: SAM3-driven collection supports RealSense, image folders, and video input modes

### 2) Split dataset
```bash
python -m yolo_tuning.vision_tuning.cli split --dataset-dir dataset_det --train-ratio 0.8
python -m yolo_tuning.vision_tuning.cli split --dataset-dir dataset_seg --train-ratio 0.8
```

Expected raw structure before split:
```text
<dataset>/
  images/
  labels/
```

Result after split:
```text
<dataset>/
  images/train  images/val
  labels/train  labels/val
```

Note: split uses move semantics, not copy.

### 3) Train
```bash
python -m yolo_tuning.vision_tuning.cli train-bbox --dataset-dir dataset_det --epochs 50 --batch 8 --imgsz 640
python -m yolo_tuning.vision_tuning.cli train-seg --dataset-dir dataset_seg --epochs 250 --batch 4 --imgsz 640
```

Training behavior:
- Auto-splits if `images/train` is missing
- Writes run artifacts under `CHECKPOINT_DIR/<run_name>/`
- Copies best checkpoints to repo root:
  - `yolo_finetuned_best.pt`
  - `yolo_seg_finetuned_best.pt`

### 4) Live test
```bash
python -m yolo_tuning.vision_tuning.cli test-bbox --model-path yolo_finetuned_best.pt
python -m yolo_tuning.vision_tuning.cli test-seg --model-path yolo_seg_finetuned_best.pt
```

## Workflow B: Legacy Scripts (Also Supported)
Legacy commands remain usable and are wrapped by newer modules in parts of the codebase.

### Data collection
```bash
python -m yolo_tuning.create_dataset
python -m yolo_tuning.create_dataset_seg
python -m yolo_tuning.create_dataset_seg_stream
```

### Split
```bash
python -m yolo_tuning.prepare_dataset
```

Legacy training and live-test scripts were removed during the restructuring.
Use the canonical CLI commands:
- `python -m yolo_tuning.vision_tuning.cli train-bbox ...`
- `python -m yolo_tuning.vision_tuning.cli train-seg ...`
- `python -m yolo_tuning.vision_tuning.cli test-bbox ...`
- `python -m yolo_tuning.vision_tuning.cli test-seg ...`

## Operational Caveats and Troubleshooting
- RealSense camera and GUI access are required for live collection and live testing.
- First run may download tokenizer/model assets from Hugging Face and other upstream sources.
- If startup fails, check:
  - RealSense connectivity and permissions (`pyrealsense2`/udev issues)
  - `model/sam3/sam3.pt` path availability (or `SAM3_CHECKPOINT_PATH`)
  - Ontology file path validity (`yolo_tuning/resource/ontology.json` or `--ontology-path`)
  - CUDA/device availability when forcing `--device`

## Project Layout (Developer)
- `yolo_tuning/vision_tuning/endpoints/` - callable service endpoints (collect/train/test/split)
- `yolo_tuning/vision_tuning/commands/` - CLI parser and command runner
- `yolo_tuning/vision_tuning/cli.py` - unified CLI
- `yolo_tuning/vision_tuning/config.py` - configuration defaults and env overrides
- `yolo_tuning/vision_tuning/ontology.py` - ontology loading and `data.yaml` export
- `yolo_tuning/vision_tuning/data_collection/` - bbox/seg/seg-stream collectors and splitter
- `yolo_tuning/vision_tuning/training/specs.py` - training job definitions
- `yolo_tuning/vision_tuning/training/workflow.py` - structured training workflows
- `yolo_tuning/vision_tuning/training/yolo.py` - backward-compatible training wrappers
- `yolo_tuning/vision_tuning/testing/specs.py` - live test job definitions
- `yolo_tuning/vision_tuning/testing/workflow.py` - structured live-test workflows
- `yolo_tuning/vision_tuning/testing/live.py` - backward-compatible live-test wrappers

## Endpoint Reference (for integration)
- `collect_bbox(config, dataset_dir=...)`
- `collect_seg(config, dataset_dir=..., input_mode=..., source_path=..., enable_crop_augment=...)`
- `collect_seg_stream(config, dataset_dir=..., input_mode=..., source_path=..., enable_crop_augment=...)`
- `split_dataset(config, dataset_dir=..., train_ratio=..., seed=...)`
- `train_bbox(config, dataset_dir=..., epochs=..., batch=..., imgsz=..., train_ratio=...)`
- `train_seg(config, dataset_dir=..., epochs=..., batch=..., imgsz=..., train_ratio=...)`
- `test_bbox(model_path=...)`
- `test_seg(model_path=...)`
