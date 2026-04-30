"""Subprocess entry point for YOLO-seg training.

Invoked from `train_service` as a separate process so we can stream stdout
+ kill it on cancel without taking down the API server. Writes
`<train_dir>/metrics.json` on success.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _summarize_results(results_dir: Path) -> dict:
    out: dict = {"results_dir": str(results_dir)}
    csv = results_dir / "results.csv"
    if not csv.exists():
        return out
    try:
        import pandas as pd
    except ImportError:
        return out
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    last = df.iloc[-1].to_dict() if len(df) else {}
    out["epochs"] = int(df["epoch"].max()) if "epoch" in df else 0
    for k, v in last.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml path")
    ap.add_argument("--project", required=True, help="project dir for run output")
    ap.add_argument("--name", default="yolo_seg_run")
    ap.add_argument("--base-weights", default="yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    print(f"[train] data={args.data}", flush=True)
    print(f"[train] base_weights={args.base_weights}", flush=True)
    print(
        f"[train] epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} patience={args.patience}",
        flush=True,
    )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[train] ImportError: {e}", flush=True)
        traceback.print_exc()
        return 3

    try:
        model = YOLO(args.base_weights)
        kwargs = dict(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            project=args.project,
            name=args.name,
            verbose=True,
            # Memory optimization for GPUs with limited VRAM
            workers=2,           # Reduce dataloader workers (default is 8)
            cache=False,         # Don't cache images in RAM
            amp=True,            # Mixed precision (saves VRAM)
            close_mosaic=10,     # Disable mosaic augmentation for last 10 epochs
        )
        if args.device:
            kwargs["device"] = args.device
        results = model.train(**kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[train] FAILED: {e}", flush=True)
        traceback.print_exc()
        return 2

    results_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    summary = _summarize_results(results_dir)
    summary["base_weights"] = args.base_weights
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"[train] metrics → {metrics_path}", flush=True)
    print("[train] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
