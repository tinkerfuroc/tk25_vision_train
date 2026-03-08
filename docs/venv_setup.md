# Python venv Setup

This project supports both Conda and standard Python virtual environments.
This guide covers `venv`.

## Prerequisites
- Python 3.10 available as `python3.10` (recommended)
- `pip` available
- `git` installed (required by git-based dependencies in `requirements.txt`)

## Option A (Recommended): One-command setup script
From repository root:

```bash
bash scripts/setup_venv.sh
```

Optional overrides:
```bash
VENV_DIR=.venv310 PYTHON_BIN=python3.10 REQUIREMENTS_FILE=requirements.txt bash scripts/setup_venv.sh
```

## Option B: Manual setup

### Linux / macOS
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Verify installation
```bash
python --version
python -m yolo_tuning.vision_tuning.cli -h
```

## Notes
- Install your official SAM3 runtime so Python can import `sam3`.
- If you need GPU builds of PyTorch/Ultralytics, install matching CUDA wheels before running heavy training.
- RealSense collection and live tests require hardware access and GUI support.
- SAM3 checkpoint default path is `model/sam3/sam3.pt` (override with `SAM3_CHECKPOINT_PATH`).
