# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO fine-tuning pipeline for collecting datasets and training YOLO detectors/segmenters. Uses GroundingDINO for auto-labeling bounding boxes and SAM3 (with LangSAM/SAM fallback) for segmentation mask collection. Designed for RealSense camera integration.

## Environment Setup

- Python 3.10 required
- Install dependencies: `pip install -r requirements.txt`
- Conda environment available: `conda env create -f environment.yml` (creates `visionTrain` env)
- Quick venv setup: `bash scripts/setup_venv.sh && source .venv/bin/activate`
- SAM3 checkpoint required at `model/sam3/sam3.pt` (or set `SAM3_CHECKPOINT_PATH`)

## CLI Commands (Main Entry Point)

All commands run via: `python -m yolo_tuning.vision_tuning.cli <subcommand>`

### Data Collection
```bash
# Bounding boxes (RealSense only)
python -m yolo_tuning.vision_tuning.cli create-bbox --dataset-dir <path>

# Segmentation (multiple input modes)
python -m yolo_tuning.vision_tuning.cli create-seg --dataset-dir <path> --input-mode realsense
python -m yolo_tuning.vision_tuning.cli create-seg --dataset-dir <path> --input-mode images --source-path /path/to/images
python -m yolo_tuning.vision_tuning.cli create-seg-stream --dataset-dir <path> --input-mode video --source-path /path/to/video.mp4
```

Key controls in bbox collection: up/down select, `d` delete, `s` save, space skip, `q` quit.

### Dataset Splitting
```bash
python -m yolo_tuning.vision_tuning.cli split --dataset-dir <path> --train-ratio 0.8
```
Note: Split uses move semantics, not copy. Training auto-splits if `images/train` is missing.

### Training
```bash
python -m yolo_tuning.vision_tuning.cli train-bbox --dataset-dir <path> --epochs 50 --batch 8 --imgsz 640
python -m yolo_tuning.vision_tuning.cli train-seg --dataset-dir <path> --epochs 250 --batch 4 --imgsz 640
```
Best checkpoints are copied to repo root: `yolo_finetuned_best.pt`, `yolo_seg_finetuned_best.pt`

### Testing
```bash
python -m yolo_tuning.vision_tuning.cli test-bbox --model-path yolo_finetuned_best.pt
python -m yolo_tuning.vision_tuning.cli test-seg --model-path yolo_seg_finetuned_best.pt
```

### Running Tests
```bash
python -m pytest yolo_tuning/tests/ -v
# Or run a single test file
python -m pytest yolo_tuning/tests/test_commands_and_endpoints.py -v
```

## Architecture

### Package Structure
- `yolo_tuning/vision_tuning/` - Main package with modular components
  - `cli.py` - Thin wrapper importing from commands module
  - `commands/parser.py` - argparse definitions for all subcommands
  - `commands/runner.py` - CLI dispatcher that calls endpoints
  - `config.py` - `VisionConfig` dataclass for paths, weights, device settings
  - `ontology.py` - `Ontology` class for prompt-to-label mapping and YOLO data.yaml generation
  - `endpoints/` - Callable service functions (collect, train, evaluate)
  - `data_collection/` - Collectors for bbox/seg/stream modes, SAM3 backend, input sources
  - `training/` - Training specs, workflow, and YOLO wrappers
  - `testing/` - Live test specs, workflow, and wrappers

### Data Flow
1. Ontology JSON (`yolo_tuning/resource/ontology.json`) defines text prompts mapped to class labels
2. Collectors use GroundingDINO (bbox) or SAM3 (seg) for auto-labeling
3. Raw datasets split into `images/train`, `images/val`, `labels/train`, `labels/val`
4. Training generates `data.yaml` from ontology, runs YOLO, copies best weights to cwd

### Configuration
Environment variables (all optional):
- `DATASET_DIR`, `DATASET_SEG_DIR` - default dataset paths
- `ONTOLOGY_PATH` - default: `yolo_tuning/resource/ontology.json`
- `CHECKPOINT_DIR` - default: `runs`
- `YOLO_BASE_WEIGHTS`, `YOLO_SEG_WEIGHTS` - default: `yolo11s.pt`, `yolo11s-seg.pt`
- `SAM3_CHECKPOINT_PATH` - default: `model/sam3/sam3.pt`
- `VISION_TRAIN_SEED` - default: `42`

Override defaults via `VisionConfig.override()` or CLI args.

## Ontology Format

JSON file mapping text prompts (for GroundingDINO/LangSAM) to class labels:
```json
{
  "a glossy green bag with dark green stripe": "QingKaiLing",
  "white dotted tray": "Tray"
}
```

## Development Notes

- Legacy scripts (`create_dataset.py`, `create_dataset_seg.py`, etc.) remain in `yolo_tuning/` for backward compatibility but CLI is preferred
- Training uses Ultralytics YOLO11 by default
- Data collection and live testing require RealSense camera hardware and GUI access
- First run may download tokenizer/model assets from Hugging Face
