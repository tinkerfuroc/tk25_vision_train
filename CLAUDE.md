# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO fine-tuning pipeline for collecting datasets and training YOLO detectors/segmenters. Uses GroundingDINO for auto-labeling bounding boxes and LangSAM/SAM for segmentation mask collection. Designed for RealSense camera integration.

## Environment Setup

- Python 3.10 required
- Install dependencies: `pip install -r requirements.txt`
- Conda environment available: `conda env create -f environment.yml` (creates `visionTrain` env)
- Download SAM weights: `sam_vit_b_01ec64.pth` to repo root
- LangSAM and SAM are installed from GitHub as editable packages

## CLI Commands (Main Entry Point)

All commands run via: `python -m yolo_tuning.vision_tuning.cli <subcommand>`

### Data Collection
- **Bounding boxes**: `python -m yolo_tuning.vision_tuning.cli create-bbox --dataset-dir <path>`
- **Segmentation masks**: `python -m yolo_tuning.vision_tuning.cli create-seg --dataset-dir <path>`
- **Segmentation stream**: `python -m yolo_tuning.vision_tuning.cli create-seg-stream --dataset-dir <path>`

### Dataset Splitting
`python -m yolo_tuning.vision_tuning.cli split --dataset-dir <path> --train-ratio 0.8`

### Training
- **Detector**: `python -m yolo_tuning.vision_tuning.cli train-bbox --dataset-dir <path> --epochs 50`
- **Segmenter**: `python -m yolo_tuning.vision_tuning.cli train-seg --dataset-dir <path> --epochs 250`

### Testing
- **Detector**: `python -m yolo_tuning.vision_tuning.cli test-bbox --model-path yolo_finetuned_best.pt`
- **Segmenter**: `python -m yolo_tuning.vision_tuning.cli test-seg --model-path yolo_seg_finetuned_best.pt`

## Architecture

### Package Structure
- `yolo_tuning/vision_tuning/` - Main package with modular components
  - `cli.py` - Single entry point with argparse subcommands
  - `config.py` - `VisionConfig` dataclass for paths, weights, device settings
  - `ontology.py` - `Ontology` class for prompt-to-label mapping and YOLO data.yaml generation
  - `data_collection/` - Collectors for bbox/seg/stream modes, splitter utility
  - `training/yolo.py` - `train_detector()` and `train_segmenter()` functions
  - `testing/live.py` - Live inference wrappers

### Data Flow
1. Ontology JSON (`yolo_tuning/resource/ontology.json`) defines text prompts mapped to class labels
2. Collectors use GroundingDINO (bbox) or LangSAM+SAM (seg) for auto-labeling with RealSense camera
3. Raw datasets split into `images/train`, `images/val`, `labels/train`, `labels/val`
4. Training generates `data.yaml` from ontology, runs YOLO, copies best weights to cwd

### Configuration
Environment variables (all optional): `DATASET_DIR`, `DATASET_SEG_DIR`, `ONTOLOGY_PATH`, `CHECKPOINT_DIR`, `YOLO_BASE_WEIGHTS`, `YOLO_SEG_WEIGHTS`, `VISION_TRAIN_SEED`

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

- The `groundingdino/` directory is a git submodule for GroundingDINO model
- Legacy scripts (`tune_YOLOv11.py`, `create_dataset.py`, etc.) remain for backward compatibility but CLI is preferred
- Training uses Ultralytics YOLO11 by default (`yolo11s.pt`, `yolo11s-seg.pt`)
- Data collection requires RealSense camera hardware

## Code Style (from .cursor/general_rules.mdc)

- Verify information from context before presenting
- Make changes file by file
- Preserve existing code; don't remove unrelated functionality
- Use explicit variable names
- Include unit tests for new/modified code
- Implement robust error handling where necessary
- Consider edge cases and use assertions to validate assumptions
