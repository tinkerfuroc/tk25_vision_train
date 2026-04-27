# tk_vision

基于 web 的 SAM3 半自动分割流水线，用于 Robocup 物品识别。从 RealSense（或图片文件夹）采集 clip，用文本 prompt 通过 SAM3 生成 mask，再用 SAM3 video tracker 沿 clip 传播；浏览器里复核与编辑；导出 YOLO-seg 数据集，训练，测试 — 全在 SPA 内完成。

旧版的 cv2-window 脚本（`yolo_tuning/`）仍可运行，新流程通过 `web/` + `server/` 提供。

## 依赖

- Python 3.10
- CUDA GPU（推荐 bfloat16；测试环境 RTX 5070 Ti / sm_120, PyTorch 2.11+cu128）
- Intel RealSense（可选 — 文件夹导入无需相机）
- 重新构建 SPA 需要 Node 20+

`pip install -e ./server`（推荐，从 `pyproject.toml` 拉取 FastAPI/ultralytics/transformers/albumentations 等），或者用 `requirements.txt` 跑旧脚本。

### SAM3 权重

`sam3_checkpoint_hf/` 是从 `sam3.pt` 转换的默认 checkpoint，里面训练好的 `tracker_neck` 权重存在但前缀是 `tracker_model.tracker_neck.*` 而不是顶层 `tracker_neck.*`；`Sam3Engine.load` 启动时手动 alias。若 `tk_vision serve` 报 `tracker_neck patch loaded N/22 weights`，装一个完整 checkpoint：

```
tk_vision fetch-weights --source local
tk_vision fetch-weights --source hf --repo facebook/sam3 --yes
tk_vision fetch-weights --source url --url https://… --sha256 … --yes
```

`--source hf|url` 要求 `--yes`（或 `TK_VISION_ALLOW_DOWNLOAD=1`）；canary 会校验 22/22 tracker_neck 权重加载，未通过则拒绝写入目标目录。

## 流程

```
tk_vision serve                  # FastAPI 跑在 :28000，SPA 在 http://localhost:28000
```

浏览器内：

1. **Clips 页** — 录制 RealSense clip / 导入图片文件夹 / 选已有 clip。
2. **Label 页** — `Seed first frame` 按 ontology 跑 SAM3；点 track 用正负点 refine；按 `Propagate` 用 SAM3 video tracker 沿 clip 分块前向传播。可滑动帧、用 `Delete`/`Restore`/`Prune from here` 标记坏帧、对单帧重新 seed。
3. **Export YOLO-seg** 写入 `data/runs/<run_id>/{images,labels}/{train,val}/` + `data.yaml`；默认按 clip 切分以避免时间相邻泄漏；保留多 contour 的 polygon。
4. **Augment** 跑 `configs/default.yaml > augment.ops` 中的 Albumentations 操作 + 可选 copy-paste，原地写出 `<stem>_aug{0..N-1}.jpg`/`.txt`。
5. **Train** 启动 `python -m tk_vision._train_runner` 子进程跑 `model.train(...)`，stdout 通过 WebSocket 流式回传 SPA。SIGTERM 取消；成功后从 `results.csv` 写出 `metrics.json`。
6. **Test → /test/<run_id>/<clip_id>** 输入 `.pt` 权重路径，对 clip 中所有未删除的帧跑 `model.predict`，并把预测落地为 JSON。帧滑块叠加预测多边形 + 类别分数。

### Ontology 指南

`resource/ontology.json` 形如 `{prompt: label}`：

```json
{"<text prompt>": "<label>"}
```

key 喂给 SAM3 open-vocab text head，训练时见的是短名词短语（`"cat"`、`"remote"`）。key 控制在 ≤6 个 token，简洁视觉描述，避免任务术语。长 phrase 会让 `presence_logits` 塌掉，seed 返回空。value 是 YOLO class 名 — 保持稳定。

### Score mode

`configs/default.yaml > sam3.score_mode`（或 Label 页 SPA toggle）：

- `native` — Meta 默认：`score = sigmoid(pred_logits) * sigmoid(presence_logits)`。配短 prompt 用。
- `per_query` — 去掉 presence。冗长 / 领域特定 prompt 且 `native` 不返回 mask 时用。

## 配置

`configs/default.yaml` 控制 server bind、采集设备、SAM3 checkpoint 路径与 dtype、propagation 分块、augment 操作、training 超参。字段说明在 `server/tk_vision/config.py`。可以 `tk_vision serve --config <path>` 或者直接改 YAML。

## 后台任务模型

长操作（propagate / train / infer）走统一模式：

- `POST /api/.../{op}` — 启动，返回 `{job_id, ...}`，起 asyncio task。
- `GET /api/.../{op}/{job_id}` — 查询状态。
- `DELETE /api/.../{op}/{job_id}` — 取消（子进程 job 用 SIGTERM；进程内循环用 `cancel.set()`）。
- `WS /ws/{op}/.../{job_id}` — 推 `log`/`frame`/`done`/`error`/`cancelled` 事件。

job 状态 `pending | running | done | error | cancelled`（`JobStatus` 在 `web/src/api/rest.ts`）。

## 跑测试

```
cd server
PYTHONPATH=. python -m pytest tests/ -q --override-ini "addopts="
```

GPU smoke test (`test_propagate_smoke.py`) 与 tracker_neck patch test 需要 CUDA + 完整 SAM3 checkpoint；否则跳过。

## 旧脚本 (yolo_tuning/)

cv2-window 旧流程仍可用，git 中打了 `legacy/v0` tag。pre-web 的 README 见 git 历史。
