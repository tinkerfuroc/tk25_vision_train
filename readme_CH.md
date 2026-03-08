# YOLO 微调工具
用于 RealSense 数据采集、YOLO 数据集整理，以及 YOLO11 检测/分割模型微调。

## 环境与安装
- Python 3.10
- 安装依赖：`pip install -r requirements.txt`
- 可选 conda 环境：`conda env create -f environment.yml`（环境名：`visionTrain`）
- 将 SAM3 权重放在 `model/sam3/sam3.pt`（或设置 `SAM3_CHECKPOINT_PATH`）

以下命令均在仓库根目录运行。

## 配置（可选）
`VisionConfig` 支持以下环境变量：
- `DATASET_DIR`（检测数据集默认目录：`dataset`）
- `DATASET_SEG_DIR`（分割数据集默认目录：`dataset_seg`）
- `ONTOLOGY_PATH`（默认：`yolo_tuning/resource/ontology.json`）
- `CHECKPOINT_DIR`（默认：`runs`）
- `YOLO_BASE_WEIGHTS`（默认：`yolo11s.pt`）
- `YOLO_SEG_WEIGHTS`（默认：`yolo11s-seg.pt`）
- `VISION_TRAIN_SEED`（默认：`42`）
- `SAM3_CHECKPOINT_PATH`（默认：`model/sam3/sam3.pt`）

## Ontology 配置
编辑 `yolo_tuning/resource/ontology.json`，格式如下：
```json
{
  "<GroundingDINO 或 LangSAM prompt>": "label"
}
```

## 工作流 A：标准 CLI（推荐）
统一入口：
```bash
python -m yolo_tuning.vision_tuning.cli <subcommand>
```

CLI 结构：
- 参数定义：`yolo_tuning/vision_tuning/commands/parser.py`
- 命令分发：`yolo_tuning/vision_tuning/commands/runner.py`
- 兼容入口：`yolo_tuning/vision_tuning/cli.py`

### 1) 采集数据（RealSense 实时）
```bash
python -m yolo_tuning.vision_tuning.cli create-bbox --dataset-dir dataset_det
python -m yolo_tuning.vision_tuning.cli create-seg --dataset-dir dataset_seg
python -m yolo_tuning.vision_tuning.cli create-seg-stream --dataset-dir dataset_seg_stream
python -m yolo_tuning.vision_tuning.cli create-seg --dataset-dir dataset_seg --input-mode images --source-path /path/to/images
python -m yolo_tuning.vision_tuning.cli create-seg-stream --dataset-dir dataset_seg --input-mode video --source-path /path/to/video.mp4
```

常用参数（多数子命令支持）：
- `--device`（例如 `cpu`、`cuda`、`cuda:0`）
- `--ontology-path`
- `--checkpoint-dir`
- 分割相关命令支持 `--input-mode`（`realsense|images|video`）
- `images`/`video` 模式需提供 `--source-path`
- 可选增强参数：`--enable-crop-augment --crop-variants --crop-scale-min --crop-scale-max`

采集时主要按键：
- BBox：上下选择，`d` 删除，`s` 保存，空格跳过，`q` 退出
- Seg/Seg-stream：基于 SAM3，可在 RealSense、图片目录、视频三种输入模式下采集

### 2) 划分数据集
```bash
python -m yolo_tuning.vision_tuning.cli split --dataset-dir dataset_det --train-ratio 0.8
python -m yolo_tuning.vision_tuning.cli split --dataset-dir dataset_seg --train-ratio 0.8
```

划分前目录结构：
```text
<dataset>/
  images/
  labels/
```

划分后目录结构：
```text
<dataset>/
  images/train  images/val
  labels/train  labels/val
```

注意：`split` 使用移动（move）而非复制（copy）。

### 3) 训练
```bash
python -m yolo_tuning.vision_tuning.cli train-bbox --dataset-dir dataset_det --epochs 50 --batch 8 --imgsz 640
python -m yolo_tuning.vision_tuning.cli train-seg --dataset-dir dataset_seg --epochs 250 --batch 4 --imgsz 640
```

训练行为说明：
- 若缺少 `images/train`，会自动先做 train/val 划分
- 训练日志与产物在 `CHECKPOINT_DIR/<run_name>/`
- 最优权重会复制到仓库根目录：
  - `yolo_finetuned_best.pt`
  - `yolo_seg_finetuned_best.pt`

### 4) 实时测试
```bash
python -m yolo_tuning.vision_tuning.cli test-bbox --model-path yolo_finetuned_best.pt
python -m yolo_tuning.vision_tuning.cli test-seg --model-path yolo_seg_finetuned_best.pt
```

## 工作流 B：Legacy 脚本（同样支持）
旧命令仍可使用，且代码中部分模块仍通过 wrapper 调用它们。

### 数据采集
```bash
python -m yolo_tuning.create_dataset
python -m yolo_tuning.create_dataset_seg
python -m yolo_tuning.create_dataset_seg_stream
```

### 划分
```bash
python -m yolo_tuning.prepare_dataset
```

重构后已移除旧的训练与实时测试脚本。
请使用标准 CLI：
- `python -m yolo_tuning.vision_tuning.cli train-bbox ...`
- `python -m yolo_tuning.vision_tuning.cli train-seg ...`
- `python -m yolo_tuning.vision_tuning.cli test-bbox ...`
- `python -m yolo_tuning.vision_tuning.cli test-seg ...`

## 运行注意事项与排错
- 实时采集和实时测试依赖 RealSense 相机与 GUI 显示环境。
- 首次运行可能会从 Hugging Face 或其他上游源下载 tokenizer/模型资源。
- 启动失败时优先检查：
  - RealSense 连接和权限（`pyrealsense2`/udev 相关问题）
  - `model/sam3/sam3.pt` 路径是否正确（或 `SAM3_CHECKPOINT_PATH`）
  - Ontology 路径是否有效（`yolo_tuning/resource/ontology.json` 或 `--ontology-path`）
  - 强制 `--device` 时 CUDA/设备是否可用

## 项目结构（开发者）
- `yolo_tuning/vision_tuning/endpoints/` - 可直接调用的服务端点（采集/训练/测试/划分）
- `yolo_tuning/vision_tuning/commands/` - CLI 参数与命令调度
- `yolo_tuning/vision_tuning/cli.py` - 统一 CLI
- `yolo_tuning/vision_tuning/config.py` - 默认配置与环境变量覆盖
- `yolo_tuning/vision_tuning/ontology.py` - Ontology 读取与 `data.yaml` 导出
- `yolo_tuning/vision_tuning/data_collection/` - bbox/seg/seg-stream 采集与划分
- `yolo_tuning/vision_tuning/training/specs.py` - 训练任务定义
- `yolo_tuning/vision_tuning/training/workflow.py` - 结构化训练流程
- `yolo_tuning/vision_tuning/training/yolo.py` - 兼容旧接口的训练封装
- `yolo_tuning/vision_tuning/testing/specs.py` - 实时测试任务定义
- `yolo_tuning/vision_tuning/testing/workflow.py` - 结构化实时测试流程
- `yolo_tuning/vision_tuning/testing/live.py` - 兼容旧接口的实时测试封装

## Endpoint 参考（用于集成）
- `collect_bbox(config, dataset_dir=...)`
- `collect_seg(config, dataset_dir=..., input_mode=..., source_path=..., enable_crop_augment=...)`
- `collect_seg_stream(config, dataset_dir=..., input_mode=..., source_path=..., enable_crop_augment=...)`
- `split_dataset(config, dataset_dir=..., train_ratio=..., seed=...)`
- `train_bbox(config, dataset_dir=..., epochs=..., batch=..., imgsz=..., train_ratio=...)`
- `train_seg(config, dataset_dir=..., epochs=..., batch=..., imgsz=..., train_ratio=...)`
- `test_bbox(model_path=...)`
- `test_seg(model_path=...)`
