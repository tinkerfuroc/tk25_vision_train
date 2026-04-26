"""Browser-based live preview and review UI for data collection."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv


@dataclass
class _ReviewState:
    frame: Optional[np.ndarray] = None
    detections: Optional[sv.Detections] = None
    class_ids: Optional[np.ndarray] = None
    kept_indices: Optional[List[int]] = None
    selected_idx: int = 0
    frame_count: int = 0
    saved_count: int = 0
    message: str = "Waiting for detections..."
    closed: bool = False


class WebReviewCollector:
    """FastAPI-backed review UI that shows live and captured frames together."""

    def __init__(
        self,
        class_names: List[str],
        on_save: Callable,
        *,
        mode: str,
        live_frame_provider: Callable[[], Optional[np.ndarray]],
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        if mode not in {"bbox", "seg"}:
            raise ValueError(f"Unsupported review mode: {mode}")

        self.class_names = class_names
        self.on_save = on_save
        self.mode = mode
        self.live_frame_provider = live_frame_provider
        self.host = host
        self.port = port

        self._state = _ReviewState()
        self._lock = threading.RLock()
        self._server = None
        self._server_thread: Optional[threading.Thread] = None

        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.highlight_annotator = sv.BoxAnnotator(thickness=4, color=sv.Color.RED)
        self.mask_annotator = sv.MaskAnnotator()
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_position=sv.Position.BOTTOM_LEFT,
        )

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        app = self._build_app()

        import uvicorn

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(target=self._server.run, daemon=True)
        self._server_thread.start()

        deadline = time.time() + 5
        while not self._server.started and time.time() < deadline:
            time.sleep(0.05)

        print(f"[WebReview] Open {self.url} in your browser.")

    def update_review(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        class_ids: Optional[np.ndarray] = None,
    ) -> bool:
        with self._lock:
            if self._state.closed:
                return False

            self._state.frame_count += 1

            if len(detections) > 0 and self._state.frame is None:
                copied = copy.deepcopy(detections)
                if class_ids is not None:
                    copied.class_id = class_ids.astype(int).copy()
                self._state.frame = frame.copy()
                self._state.detections = copied
                self._state.class_ids = class_ids.copy() if class_ids is not None else copied.class_id.copy()
                self._state.kept_indices = list(range(len(copied)))
                self._state.selected_idx = 0
                self._state.message = f"Reviewing {len(copied)} detection(s)"

            return True

    def is_alive(self) -> bool:
        with self._lock:
            return not self._state.closed

    def has_pending_review(self) -> bool:
        with self._lock:
            return self._state.frame is not None

    def stop(self) -> None:
        with self._lock:
            self._state.closed = True
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)

    def _build_app(self):
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

        app = FastAPI(title="Vision Collection Review")

        @app.get("/")
        def index():
            return HTMLResponse(self._html())

        @app.get("/api/status")
        def status():
            with self._lock:
                detections = self._current_detection_rows_locked()
                return {
                    "mode": self.mode,
                    "frame_count": self._state.frame_count,
                    "saved_count": self._state.saved_count,
                    "message": self._state.message,
                    "has_review": self._state.frame is not None,
                    "selected_idx": self._state.selected_idx,
                    "kept_count": len(self._state.kept_indices or []),
                    "detections": detections,
                    "closed": self._state.closed,
                }

        @app.post("/api/action/{action}")
        def action(action: str):
            ok, message = self._handle_action(action)
            return JSONResponse({"ok": ok, "message": message})

        @app.get("/stream/live")
        def live_stream():
            return StreamingResponse(
                self._mjpeg_stream(self._render_live_frame),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @app.get("/stream/review")
        def review_stream():
            return StreamingResponse(
                self._mjpeg_stream(self._render_review_frame),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        return app

    def _handle_action(self, action: str) -> tuple[bool, str]:
        with self._lock:
            if action == "quit":
                self._state.closed = True
                return True, "Stopping collection"

            kept = self._state.kept_indices or []
            if self._state.frame is None or self._state.detections is None:
                return False, "No frame to review"

            if action in {"prev", "next"} and kept:
                delta = -1 if action == "prev" else 1
                self._state.selected_idx = (self._state.selected_idx + delta) % len(kept)
                self._state.message = f"Selected detection {self._state.selected_idx + 1}/{len(kept)}"
                return True, self._state.message

            if action == "delete" and kept:
                kept.pop(self._state.selected_idx)
                if not kept:
                    self._clear_current_locked("Deleted all detections")
                    return True, self._state.message
                if self._state.selected_idx >= len(kept):
                    self._state.selected_idx = len(kept) - 1
                self._state.message = f"Deleted. {len(kept)} detection(s) kept."
                return True, self._state.message

            if action == "skip":
                self._clear_current_locked("Skipped")
                return True, "Skipped"

            if action == "save":
                if not kept:
                    return False, "No detections kept"

                frame = self._state.frame.copy()
                detections = self._state.detections[kept]
                class_ids = self._state.class_ids[kept] if self._state.class_ids is not None else detections.class_id

                if self.mode == "seg":
                    self.on_save(frame, detections, class_ids)
                else:
                    self.on_save(frame, detections)

                self._state.saved_count += 1
                self._clear_current_locked("Saved")
                return True, "Saved"

            return False, f"Unsupported action: {action}"

    def _clear_current_locked(self, message: str) -> None:
        self._state.frame = None
        self._state.detections = None
        self._state.class_ids = None
        self._state.kept_indices = []
        self._state.selected_idx = 0
        self._state.message = message

    def _current_detection_rows_locked(self) -> list[dict]:
        if self._state.detections is None or not self._state.kept_indices:
            return []

        rows = []
        detections = self._state.detections
        class_ids = self._state.class_ids if self._state.class_ids is not None else detections.class_id
        for display_idx, detection_idx in enumerate(self._state.kept_indices):
            cid = int(class_ids[detection_idx]) if class_ids is not None else 0
            conf = detections.confidence[detection_idx] if detections.confidence is not None else 1.0
            rows.append(
                {
                    "display_idx": display_idx,
                    "detection_idx": int(detection_idx),
                    "class_name": self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}",
                    "confidence": float(conf),
                    "selected": display_idx == self._state.selected_idx,
                }
            )
        return rows

    def _mjpeg_stream(self, renderer: Callable[[], np.ndarray]):
        while self.is_alive():
            image = renderer()
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            time.sleep(0.1)

    def _render_live_frame(self) -> np.ndarray:
        frame = self.live_frame_provider()
        if frame is None:
            return self._placeholder("Waiting for RealSense live frame")
        return self._fit_canvas(frame, (640, 480))

    def _render_review_frame(self) -> np.ndarray:
        with self._lock:
            frame = None if self._state.frame is None else self._state.frame.copy()
            detections = copy.deepcopy(self._state.detections)
            kept = list(self._state.kept_indices or [])
            selected_idx = self._state.selected_idx
            message = self._state.message

        if frame is None or detections is None:
            return self._placeholder(message)

        if kept:
            display = detections[kept]
            labels = self._labels_for_detections(display)
            if self.mode == "seg" and display.mask is not None:
                annotated = self._annotate_segmentation(frame.copy(), display, labels, selected_idx)
            else:
                annotated = self.box_annotator.annotate(scene=frame.copy(), detections=display)
                annotated = self.label_annotator.annotate(scene=annotated, detections=display, labels=labels)

            if self.mode == "bbox" and 0 <= selected_idx < len(display):
                selected = display[selected_idx:selected_idx + 1]
                annotated = self.highlight_annotator.annotate(scene=annotated, detections=selected)
        else:
            annotated = frame.copy()
            cv2.putText(annotated, "No detections kept", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return self._fit_canvas(annotated, (740, 580))

    def _annotate_segmentation(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        labels: list[str],
        selected_idx: int,
    ) -> np.ndarray:
        annotated = image.copy()
        masks = detections.mask
        boxes = detections.xyxy
        if masks is None:
            return annotated

        palette = [
            (42, 157, 244),
            (76, 175, 80),
            (156, 89, 209),
            (0, 188, 212),
            (255, 193, 7),
            (244, 67, 54),
        ]

        for idx, mask in enumerate(masks):
            mask = self._mask_for_image(mask, annotated.shape[:2])
            if mask is None:
                continue

            color = palette[idx % len(palette)]
            alpha = 0.42 if idx == selected_idx else 0.30
            overlay = annotated.copy()
            overlay[mask.astype(bool)] = color
            annotated = cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0)

            mask_uint8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

            if boxes is not None and len(boxes) > idx:
                x1, y1, x2, y2 = np.rint(boxes[idx]).astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(annotated.shape[1] - 1, x2), min(annotated.shape[0] - 1, y2)
                thickness = 3 if idx == selected_idx else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                label_x, label_y = x1, max(22, y1 - 8)
            else:
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                label_x, label_y = int(xs.min()), max(22, int(ys.min()) - 8)

            label = labels[idx] if idx < len(labels) else f"#{idx}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            cv2.rectangle(
                annotated,
                (label_x, label_y - text_h - baseline - 6),
                (label_x + text_w + 10, label_y + baseline),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (label_x + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    @staticmethod
    def _mask_for_image(mask: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Return a 2D boolean mask matching the image, resizing model output if needed."""
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_arr = np.squeeze(mask_arr)
            if mask_arr.ndim == 3:
                if mask_arr.shape[2] <= 4:
                    mask_arr = np.max(mask_arr, axis=2)
                else:
                    mask_arr = np.max(mask_arr, axis=0)
        if mask_arr.ndim != 2:
            return None

        image_h, image_w = image_shape
        if mask_arr.shape == (image_w, image_h):
            mask_arr = mask_arr.T
        if mask_arr.shape != (image_h, image_w):
            mask_arr = cv2.resize(mask_arr.astype(np.float32), (image_w, image_h), interpolation=cv2.INTER_NEAREST)

        return mask_arr.astype(bool) if mask_arr.dtype == np.bool_ else mask_arr > 0.5

    def _labels_for_detections(self, detections: sv.Detections) -> list[str]:
        labels = []
        confidences = detections.confidence if detections.confidence is not None else np.ones(len(detections))
        for idx, (cid, conf) in enumerate(zip(detections.class_id, confidences)):
            name = self.class_names[int(cid)] if int(cid) < len(self.class_names) else f"class_{int(cid)}"
            labels.append(f"#{idx} {name} {float(conf):.2f}")
        return labels

    def _fit_canvas(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        canvas_w, canvas_h = size
        h, w = image.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas

    def _placeholder(self, text: str) -> np.ndarray:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(canvas, text, (32, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        return canvas

    def _html(self) -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vision Collection Review</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #111418; color: #eef2f6; }
    header { padding: 14px 18px; border-bottom: 1px solid #2b313a; display: flex; justify-content: space-between; gap: 16px; align-items: center; }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    main { display: grid; grid-template-columns: minmax(320px, 640px) minmax(360px, 740px) 280px; gap: 14px; padding: 14px; align-items: start; }
    section { min-width: 0; }
    h2 { font-size: 13px; color: #9fb0c3; margin: 0 0 8px; font-weight: 600; }
    img { width: 100%; background: #050607; border: 1px solid #2b313a; display: block; }
    .panel { border: 1px solid #2b313a; padding: 10px; background: #171b21; }
    .buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    button { border: 1px solid #3a4655; background: #242b34; color: #f6f8fb; padding: 9px 10px; font-size: 14px; cursor: pointer; }
    button.primary { background: #1f6f4a; border-color: #2e9f6d; }
    button.warn { background: #6b2d35; border-color: #98414c; }
    .status { display: grid; gap: 6px; margin-bottom: 12px; font-size: 13px; color: #d7dee8; }
    .detections { display: grid; gap: 6px; margin-top: 12px; max-height: 360px; overflow: auto; }
    .det { display: flex; justify-content: space-between; gap: 8px; border: 1px solid #303844; padding: 7px; font-size: 13px; }
    .det.selected { border-color: #e25c5c; color: #ffffff; }
    @media (max-width: 1100px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>Vision Collection Review</h1>
    <div id="message">Starting...</div>
  </header>
  <main>
    <section>
      <h2>Live RealSense</h2>
      <img src="/stream/live" alt="Live RealSense stream">
    </section>
    <section>
      <h2>Captured Review Frame</h2>
      <img src="/stream/review" alt="Captured review frame">
    </section>
    <aside class="panel">
      <div class="status">
        <div>Processed: <strong id="frameCount">0</strong></div>
        <div>Saved: <strong id="savedCount">0</strong></div>
        <div>Kept: <strong id="keptCount">0</strong></div>
      </div>
      <div class="buttons">
        <button onclick="sendAction('prev')">Prev</button>
        <button onclick="sendAction('next')">Next</button>
        <button class="warn" onclick="sendAction('delete')">Delete</button>
        <button onclick="sendAction('skip')">Skip</button>
        <button class="primary" onclick="sendAction('save')">Save</button>
        <button onclick="sendAction('quit')">Quit</button>
      </div>
      <div class="detections" id="detections"></div>
    </aside>
  </main>
  <script>
    async function sendAction(action) {
      await fetch(`/api/action/${action}`, { method: 'POST' });
      await refreshStatus();
    }

    async function refreshStatus() {
      const res = await fetch('/api/status', { cache: 'no-store' });
      const data = await res.json();
      document.getElementById('message').textContent = data.message;
      document.getElementById('frameCount').textContent = data.frame_count;
      document.getElementById('savedCount').textContent = data.saved_count;
      document.getElementById('keptCount').textContent = data.kept_count;
      const list = document.getElementById('detections');
      list.innerHTML = '';
      for (const det of data.detections) {
        const row = document.createElement('div');
        row.className = `det${det.selected ? ' selected' : ''}`;
        row.innerHTML = `<span>#${det.display_idx} ${det.class_name}</span><span>${det.confidence.toFixed(2)}</span>`;
        list.appendChild(row);
      }
    }

    document.addEventListener('keydown', (event) => {
      const key = event.key.toLowerCase();
      if (key === 'arrowup' || key === 'k') sendAction('prev');
      if (key === 'arrowdown' || key === 'j') sendAction('next');
      if (key === 'd') sendAction('delete');
      if (key === 's') sendAction('save');
      if (key === ' ') { event.preventDefault(); sendAction('skip'); }
      if (key === 'q' || key === 'escape') sendAction('quit');
    });

    setInterval(refreshStatus, 500);
    refreshStatus();
  </script>
</body>
</html>"""


__all__ = ["WebReviewCollector"]
