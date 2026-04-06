"""Tkinter-based GUI for data collection and review.

Uses Tkinter instead of OpenCV to avoid CUDA+Qt/GTK conflicts.
Features dual-window mode: live preview + review window with detections.
"""

import copy
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageTk


@dataclass
class FrameAnnotation:
    """Serializable frame annotation for review."""
    image: np.ndarray
    xyxy: np.ndarray
    mask: Optional[np.ndarray]
    confidence: Optional[np.ndarray]
    class_id: np.ndarray


class DualWindowBBoxCollector:
    """Dual-window BBox collector: live preview + review window.

    - Live window: shows real-time camera feed without annotations
    - Review window: shows detections with interactive controls
    """

    def __init__(
        self,
        class_names: List[str],
        on_save: Callable[[np.ndarray, sv.Detections], None],
    ):
        self.class_names = class_names
        self.on_save = on_save

        self._closed = False
        self._current_frame: Optional[np.ndarray] = None
        self._current_predictions: Optional[sv.Detections] = None
        self._kept_indices: List[int] = []
        self._selected_idx = 0
        self._frame_count = 0
        self._saved_count = 0
        self._has_detections = False

        # Tkinter widgets
        self.root: Optional[tk.Tk] = None
        self.live_window: Optional[tk.Toplevel] = None
        self.live_canvas: Optional[tk.Canvas] = None
        self.review_canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[tk.Label] = None
        self.detection_info_label: Optional[tk.Label] = None

        # Image references to prevent garbage collection
        self._live_img_tk = None
        self._review_img_tk = None

        # Camera for live preview
        self._camera = None
        self._live_update_id = None

        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_position=sv.Position.BOTTOM_LEFT
        )
        self.highlight_annotator = sv.BoxAnnotator(thickness=4, color=sv.Color.RED)

    def start(self) -> None:
        """Start both windows and live preview loop."""
        self.root = tk.Tk()
        self.root.title("Review Window - Detections")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # === Review Window (main) ===
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(main_frame, text="=== Review Window (Detections) ===", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.review_canvas = tk.Canvas(main_frame, width=740, height=580, bg="gray20")
        self.review_canvas.grid(row=1, column=0, pady=5)

        # Initial message on canvas
        self.review_canvas.create_text(
            370, 290, text="Waiting for detections...", fill="white", font=("Arial", 14), tags="message"
        )

        self.detection_info_label = ttk.Label(main_frame, text="No detections yet")
        self.detection_info_label.grid(row=2, column=0, sticky="w")

        # Status and controls
        self.status_label = ttk.Label(main_frame, text="Frame: 0 | Saved: 0")
        self.status_label.grid(row=3, column=0, sticky="w", pady=2)

        ttk.Label(main_frame, text="Controls: ↑/↓ Select | d Delete | s Save | Space Skip", font=("Arial", 9)).grid(
            row=4, column=0, sticky="w"
        )

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, pady=5)

        ttk.Button(btn_frame, text="↑ Prev", command=self._prev_detection, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="↓ Next", command=self._next_detection, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete (d)", command=self._delete_detection, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Save (s)", command=self._save_current, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Skip", command=self._skip_frame, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Quit (q)", command=self._on_close, width=8).pack(side=tk.LEFT, padx=2)

        # === Live Preview Window (separate) ===
        self.live_window = tk.Toplevel(self.root)
        self.live_window.title("Live Preview - RealSense Camera")
        self.live_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close on live window

        live_frame = ttk.Frame(self.live_window, padding="5")
        live_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(live_frame, text="=== Live Preview (Real-time) ===", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.live_canvas = tk.Canvas(live_frame, width=640, height=480, bg="gray20")
        self.live_canvas.grid(row=1, column=0, pady=5)

        # Initial message
        self.live_canvas.create_text(
            320, 240, text="Starting camera...", fill="white", font=("Arial", 14), tags="message"
        )

        self.live_status_label = ttk.Label(live_frame, text="Initializing...")
        self.live_status_label.grid(row=2, column=0, sticky="w")

        # Key bindings
        self.root.bind("<Up>", lambda e: self._prev_detection())
        self.root.bind("<Down>", lambda e: self._next_detection())
        self.root.bind("d", lambda e: self._delete_detection())
        self.root.bind("s", lambda e: self._save_current())
        self.root.bind("<space>", lambda e: self._skip_frame())
        self.root.bind("q", lambda e: self._on_close())
        self.root.bind("<Escape>", lambda e: self._on_close())

        # Position windows side by side
        self.root.update_idletasks()
        self.live_window.geometry("+0+0")
        self.root.geometry("+660+0")

        # Start camera and live preview loop
        self._init_camera()

        self.root.update()

    def _init_camera(self) -> None:
        """Initialize shared camera for live preview."""
        try:
            from yolo_tuning.vision_tuning.data_collection.input_sources import get_shared_camera
            self._camera = get_shared_camera()
            self._camera.start()
            # Start live preview update loop
            self._schedule_live_update()
        except Exception as e:
            print(f"[GUI] Failed to init camera: {e}")
            if self.live_status_label:
                self.live_status_label.config(text=f"Camera error: {e}")

    def _schedule_live_update(self) -> None:
        """Schedule next live preview update."""
        if self._closed or self.root is None:
            return
        self._update_live_from_camera()
        # Schedule next update in 33ms (~30fps)
        self._live_update_id = self.root.after(33, self._schedule_live_update)

    def _update_live_from_camera(self) -> None:
        """Update live preview from shared camera."""
        if self._camera is None or self.live_canvas is None:
            return

        frame = self._camera.get_frame()
        if frame is None:
            return

        # Convert to RGB and create image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self._live_img_tk = img_tk  # Keep reference to prevent GC

        # Clear and draw new image
        self.live_canvas.delete("all")
        self.live_canvas.create_image(320, 240, anchor=tk.CENTER, image=img_tk)

        if self.live_status_label:
            h, w = frame.shape[:2]
            frame_count = self._camera.get_frame_count()
            self.live_status_label.config(text=f"Live: {frame_count} | Resolution: {w}x{h}")

    def _on_close(self) -> None:
        self._closed = True
        if self._live_update_id:
            try:
                self.root.after_cancel(self._live_update_id)
            except:
                pass
        if self.live_window:
            self.live_window.destroy()
        if self.root:
            self.root.destroy()

    def _prev_detection(self) -> None:
        if self._kept_indices:
            self._selected_idx = (self._selected_idx - 1 + len(self._kept_indices)) % len(self._kept_indices)
            self._refresh_review_display()

    def _next_detection(self) -> None:
        if self._kept_indices:
            self._selected_idx = (self._selected_idx + 1) % len(self._kept_indices)
            self._refresh_review_display()

    def _delete_detection(self) -> None:
        if self._kept_indices:
            self._kept_indices.pop(self._selected_idx)
            if self._kept_indices and self._selected_idx >= len(self._kept_indices):
                self._selected_idx = len(self._kept_indices) - 1
            self._refresh_review_display()
            self._update_detection_info()

    def _save_current(self) -> None:
        if self._current_frame is not None and self._kept_indices:
            final_predictions = self._current_predictions[self._kept_indices]
            self.on_save(self._current_frame, final_predictions)
            self._saved_count += 1
            self._update_status("Saved!")
            self._clear_current()

    def _skip_frame(self) -> None:
        self._clear_current()
        self._update_status("Skipped")

    def _clear_current(self) -> None:
        self._current_frame = None
        self._current_predictions = None
        self._kept_indices = []
        self._selected_idx = 0
        self._has_detections = False
        self._refresh_review_display()

    def _update_status(self, msg: str = "") -> None:
        if self.status_label:
            status = f"Frame: {self._frame_count} | Saved: {self._saved_count}"
            if msg:
                status += f" | {msg}"
            self.status_label.config(text=status)

    def _update_detection_info(self) -> None:
        if self.detection_info_label:
            if self._kept_indices:
                info = f"Detections: {len(self._kept_indices)} | Selected: {self._selected_idx + 1}"
            else:
                info = "No detections"
            self.detection_info_label.config(text=info)

    def _refresh_review_display(self) -> None:
        """Refresh the review canvas with detections."""
        self.review_canvas.delete("all")

        if self._current_frame is None:
            self.review_canvas.create_text(
                370, 290, text="Waiting for detections...", fill="white", font=("Arial", 14)
            )
            return

        # Get frame dimensions
        h, w = self._current_frame.shape[:2]

        if not self._kept_indices:
            # Show frame without detections, scaled to fit canvas
            img_rgb = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self._review_img_tk = img_tk  # Keep reference

            self.review_canvas.create_image(370, 290, anchor=tk.CENTER, image=img_tk)
            self._update_detection_info()
            return

        # Prepare display with padding
        padding = (50, 50, 50, 50)
        pad_top, pad_bottom, pad_left, pad_right = padding

        padded_image = cv2.copyMakeBorder(
            self._current_frame, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0],
        )

        # Offset predictions for display
        display_predictions = copy.deepcopy(self._current_predictions)
        display_predictions.xyxy = display_predictions.xyxy.copy()
        display_predictions.xyxy[:, [0, 2]] += pad_left
        display_predictions.xyxy[:, [1, 3]] += pad_top

        detections_to_show = display_predictions[self._kept_indices]

        labels = [
            f"{self.class_names[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(detections_to_show.class_id, detections_to_show.confidence)
        ]

        annotated = self.box_annotator.annotate(scene=padded_image, detections=detections_to_show)
        annotated = self.label_annotator.annotate(scene=annotated, detections=detections_to_show, labels=labels)

        # Highlight selected detection
        selected_detection = detections_to_show[self._selected_idx:self._selected_idx + 1]
        annotated = self.highlight_annotator.annotate(scene=annotated, detections=selected_detection)

        # Convert to Tkinter
        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self._review_img_tk = img_tk  # Keep reference

        self.review_canvas.create_image(370, 290, anchor=tk.CENTER, image=img_tk)
        self._update_detection_info()

    def update_review(self, frame: np.ndarray, predictions: sv.Detections) -> bool:
        """Update the review window with detection results."""
        if self._closed or self.root is None:
            return False

        self._frame_count += 1

        if len(predictions) > 0 and not self._has_detections:
            self._current_frame = frame.copy()
            self._current_predictions = copy.deepcopy(predictions)
            self._kept_indices = list(range(len(predictions)))
            self._selected_idx = 0
            self._has_detections = True
            self._refresh_review_display()

        self._update_status()
        return True

    def is_alive(self) -> bool:
        return not self._closed and self.root is not None

    def stop(self) -> None:
        self._closed = True
        if self._live_update_id and self.root:
            try:
                self.root.after_cancel(self._live_update_id)
            except:
                pass
        if self.live_window:
            self.live_window.destroy()
        if self.root:
            self.root.destroy()
        self._closed = True


class DualWindowSegCollector:
    """Dual-window segmentation collector: live preview + review window."""

    def __init__(
        self,
        class_names: List[str],
        on_save: Callable[[np.ndarray, sv.Detections, np.ndarray], None],
    ):
        self.class_names = class_names
        self.on_save = on_save

        self._closed = False
        self._current_frame: Optional[np.ndarray] = None
        self._current_detections: Optional[sv.Detections] = None
        self._current_class_ids: Optional[np.ndarray] = None
        self._frame_count = 0
        self._saved_count = 0
        self._has_detections = False

        # Tkinter widgets
        self.root: Optional[tk.Tk] = None
        self.live_window: Optional[tk.Toplevel] = None
        self.live_canvas: Optional[tk.Canvas] = None
        self.review_canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[tk.Label] = None

        # Image references to prevent garbage collection
        self._live_img_tk = None
        self._review_img_tk = None

        # Camera for live preview
        self._camera = None
        self._live_update_id = None

        # Annotators
        self.mask_annotator = sv.MaskAnnotator()
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_position=sv.Position.BOTTOM_LEFT
        )

    def start(self) -> None:
        """Start both windows and live preview loop."""
        self.root = tk.Tk()
        self.root.title("Review Window - Segmentation")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # === Review Window (main) ===
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(main_frame, text="=== Review Window (Segmentation) ===", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.review_canvas = tk.Canvas(main_frame, width=740, height=580, bg="gray20")
        self.review_canvas.grid(row=1, column=0, pady=5)

        # Initial message
        self.review_canvas.create_text(
            370, 290, text="Waiting for detections...", fill="white", font=("Arial", 14)
        )

        self.detection_info_label = ttk.Label(main_frame, text="No detections yet")
        self.detection_info_label.grid(row=2, column=0, sticky="w")

        self.status_label = ttk.Label(main_frame, text="Frame: 0 | Saved: 0")
        self.status_label.grid(row=3, column=0, sticky="w", pady=2)

        ttk.Label(main_frame, text="Controls: s Save | Space Skip | q Quit", font=("Arial", 9)).grid(
            row=4, column=0, sticky="w"
        )

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, pady=5)

        ttk.Button(btn_frame, text="Save (s)", command=self._save_current, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Skip", command=self._skip_frame, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Quit (q)", command=self._on_close, width=10).pack(side=tk.LEFT, padx=5)

        # === Live Preview Window ===
        self.live_window = tk.Toplevel(self.root)
        self.live_window.title("Live Preview - RealSense Camera")
        self.live_window.protocol("WM_DELETE_WINDOW", lambda: None)

        live_frame = ttk.Frame(self.live_window, padding="5")
        live_frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(live_frame, text="=== Live Preview (Real-time) ===", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        self.live_canvas = tk.Canvas(live_frame, width=640, height=480, bg="gray20")
        self.live_canvas.grid(row=1, column=0, pady=5)

        # Initial message
        self.live_canvas.create_text(
            320, 240, text="Starting camera...", fill="white", font=("Arial", 14)
        )

        self.live_status_label = ttk.Label(live_frame, text="Initializing...")
        self.live_status_label.grid(row=2, column=0, sticky="w")

        # Key bindings
        self.root.bind("s", lambda e: self._save_current())
        self.root.bind("<space>", lambda e: self._skip_frame())
        self.root.bind("q", lambda e: self._on_close())
        self.root.bind("<Escape>", lambda e: self._on_close())

        # Position windows
        self.root.update_idletasks()
        self.live_window.geometry("+0+0")
        self.root.geometry("+660+0")

        # Start camera and live preview loop
        self._init_camera()

        self.root.update()

    def _init_camera(self) -> None:
        """Initialize shared camera for live preview."""
        try:
            from yolo_tuning.vision_tuning.data_collection.input_sources import get_shared_camera
            self._camera = get_shared_camera()
            self._camera.start()
            # Start live preview update loop
            self._schedule_live_update()
        except Exception as e:
            print(f"[GUI] Failed to init camera: {e}")
            if self.live_status_label:
                self.live_status_label.config(text=f"Camera error: {e}")

    def _schedule_live_update(self) -> None:
        """Schedule next live preview update."""
        if self._closed or self.root is None:
            return
        self._update_live_from_camera()
        # Schedule next update in 33ms (~30fps)
        self._live_update_id = self.root.after(33, self._schedule_live_update)

    def _update_live_from_camera(self) -> None:
        """Update live preview from shared camera."""
        if self._camera is None or self.live_canvas is None:
            return

        frame = self._camera.get_frame()
        if frame is None:
            return

        # Convert to RGB and create image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self._live_img_tk = img_tk  # Keep reference to prevent GC

        # Clear and draw new image
        self.live_canvas.delete("all")
        self.live_canvas.create_image(320, 240, anchor=tk.CENTER, image=img_tk)

        if self.live_status_label:
            h, w = frame.shape[:2]
            frame_count = self._camera.get_frame_count()
            self.live_status_label.config(text=f"Live: {frame_count} | Resolution: {w}x{h}")

    def _on_close(self) -> None:
        self._closed = True
        if self._live_update_id and self.root:
            try:
                self.root.after_cancel(self._live_update_id)
            except:
                pass
        if self.live_window:
            self.live_window.destroy()
        if self.root:
            self.root.destroy()

    def _save_current(self) -> None:
        if self._current_frame is not None and self._current_detections is not None:
            self.on_save(self._current_frame, self._current_detections, self._current_class_ids)
            self._saved_count += 1
            self._update_status("Saved!")
            self._clear_current()

    def _skip_frame(self) -> None:
        self._clear_current()
        self._update_status("Skipped")

    def _clear_current(self) -> None:
        self._current_frame = None
        self._current_detections = None
        self._current_class_ids = None
        self._has_detections = False
        self._refresh_review_display()

    def _update_status(self, msg: str = "") -> None:
        if self.status_label:
            status = f"Processed: {self._frame_count} | Saved: {self._saved_count}"
            if msg:
                status += f" | {msg}"
            self.status_label.config(text=status)

    def _pad_masks(self, masks: np.ndarray, image_shape: Tuple[int, int], padding: Tuple[int, int, int, int]) -> np.ndarray:
        top, bottom, left, right = padding
        img_h, img_w = image_shape
        mask_h, mask_w = masks.shape[1], masks.shape[2]

        if mask_h != img_h or mask_w != img_w:
            return masks

        padded_masks = []
        for mask in masks:
            padded_mask = np.zeros((img_h + top + bottom, img_w + left + right), dtype=bool)
            padded_mask[top:top + img_h, left:left + img_w] = mask
            padded_masks.append(padded_mask)
        return np.array(padded_masks)

    def _refresh_review_display(self) -> None:
        self.review_canvas.delete("all")

        if self._current_frame is None or self._current_detections is None:
            self.review_canvas.create_text(
                370, 290, text="Waiting for detections...", fill="white", font=("Arial", 14)
            )
            return

        padding = (50, 50, 50, 50)
        pad_top, pad_bottom, pad_left, pad_right = padding

        padded_image = cv2.copyMakeBorder(
            self._current_frame, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0],
        )

        if len(self._current_detections) > 0:
            display_detections = copy.deepcopy(self._current_detections)
            if display_detections.xyxy is not None and len(display_detections.xyxy) > 0:
                display_detections.xyxy = display_detections.xyxy.copy()
                display_detections.xyxy += np.array([pad_left, pad_top, pad_left, pad_top])

            if display_detections.mask is not None and len(display_detections.mask) > 0:
                display_detections.mask = self._pad_masks(display_detections.mask, self._current_frame.shape[:2], padding)

            annotated = self.mask_annotator.annotate(scene=padded_image, detections=display_detections)

            labels = []
            for idx, (cid, conf) in enumerate(zip(display_detections.class_id, display_detections.confidence)):
                class_name = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
                labels.append(f"#{idx} {class_name} {conf:.2f}")
            annotated = self.label_annotator.annotate(scene=annotated, detections=display_detections, labels=labels)
        else:
            annotated = padded_image
            cv2.putText(annotated, "No detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convert to Tkinter
        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self._review_img_tk = img_tk  # Keep reference

        self.review_canvas.create_image(370, 290, anchor=tk.CENTER, image=img_tk)

        if self.detection_info_label:
            count = len(self._current_detections) if self._current_detections else 0
            self.detection_info_label.config(text=f"Masks: {count}")

    def update_review(self, frame: np.ndarray, detections: sv.Detections, class_ids: np.ndarray) -> bool:
        """Update the review window with detection results."""
        if self._closed or self.root is None:
            return False

        self._frame_count += 1

        if len(detections) > 0 and not self._has_detections:
            self._current_frame = frame.copy()
            self._current_detections = copy.deepcopy(detections)
            self._current_class_ids = class_ids.copy()
            self._has_detections = True
            self._refresh_review_display()

        self._update_status()
        return True

    def is_alive(self) -> bool:
        return not self._closed and self.root is not None

    def stop(self) -> None:
        self._closed = True
        if self._live_update_id and self.root:
            try:
                self.root.after_cancel(self._live_update_id)
            except:
                pass
        if self.live_window:
            self.live_window.destroy()
        if self.root:
            self.root.destroy()
        self._closed = True


class FrameReviewGUI:
    """Tkinter-based GUI for reviewing collected frames."""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.status_label: Optional[tk.Label] = None

        self.annotations: List[FrameAnnotation] = []
        self.kept_indices: List[int] = []
        self.cursor = 0
        self.result: Optional[List[int]] = None

        # Annotators
        self.mask_annotator = sv.MaskAnnotator()
        self.label_annotator = sv.LabelAnnotator(
            text_scale=0.5, text_thickness=1, text_position=sv.Position.BOTTOM_LEFT
        )

    def run(self, annotations: List[FrameAnnotation]) -> Optional[List[int]]:
        """Run the review GUI and return kept indices, or None if cancelled."""
        if not annotations:
            return []

        self.annotations = annotations
        self.kept_indices = list(range(len(annotations)))
        self.cursor = 0
        self.result = None

        self.root = tk.Tk()
        self.root.title("Frame Review")
        self.root.protocol("WM_DELETE_WINDOW", self._on_cancel)

        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(main_frame, width=740, height=580, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=4)

        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=1, column=0, columnspan=4, sticky="w", pady=5)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, columnspan=4, pady=5)

        ttk.Button(btn_frame, text="Prev (←/k)", command=self._prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Next (→/j)", command=self._next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete (d)", command=self._delete_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save All (s)", command=self._save_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel (q)", command=self._on_cancel).pack(side=tk.LEFT, padx=5)

        self.root.bind("<Left>", lambda e: self._prev_frame())
        self.root.bind("<Right>", lambda e: self._next_frame())
        self.root.bind("k", lambda e: self._prev_frame())
        self.root.bind("j", lambda e: self._next_frame())
        self.root.bind("d", lambda e: self._delete_frame())
        self.root.bind("s", lambda e: self._save_all())
        self.root.bind("q", lambda e: self._on_cancel())
        self.root.bind("<Escape>", lambda e: self._on_cancel())

        self._update_display()
        self.root.mainloop()

        return self.result

    def _update_display(self) -> None:
        if not self.kept_indices:
            self.canvas.delete("all")
            self.canvas.create_text(370, 290, text="No frames left", fill="white", font=("Arial", 20))
            self.status_label.config(text="No frames to review")
            return

        self.cursor = self.cursor % len(self.kept_indices)
        current_idx = self.kept_indices[self.cursor]
        entry = self.annotations[current_idx]

        detections = sv.Detections(
            xyxy=entry.xyxy,
            mask=entry.mask,
            confidence=entry.confidence,
            class_id=entry.class_id,
        )

        padding = (50, 50, 50, 50)
        pad_top, pad_bottom, pad_left, pad_right = padding
        image = entry.image

        padded_image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0],
        )

        display_detections = copy.deepcopy(detections)
        if display_detections.xyxy is not None and len(display_detections.xyxy) > 0:
            display_detections.xyxy = display_detections.xyxy.copy()
            display_detections.xyxy += np.array([pad_left, pad_top, pad_left, pad_top])

        if display_detections.mask is not None and len(display_detections.mask) > 0:
            display_detections.mask = self._pad_masks(display_detections.mask, image.shape[:2], padding)

        annotated = self.mask_annotator.annotate(scene=padded_image, detections=display_detections)

        labels = []
        for idx_i, (cid, conf) in enumerate(zip(display_detections.class_id, display_detections.confidence)):
            name = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
            labels.append(f"F{current_idx}#{idx_i} {name} {conf:.2f}")
        annotated = self.label_annotator.annotate(scene=annotated, detections=display_detections, labels=labels)

        info = f"Frame {self.cursor + 1}/{len(self.kept_indices)} (original #{current_idx}) | {len(detections)} detection(s)"
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

        self.status_label.config(text=f"Viewing frame {self.cursor + 1} of {len(self.kept_indices)}")

    def _pad_masks(self, masks: np.ndarray, image_shape: Tuple[int, int], padding: Tuple[int, int, int, int]) -> np.ndarray:
        top, bottom, left, right = padding
        img_h, img_w = image_shape
        mask_h, mask_w = masks.shape[1], masks.shape[2]

        if mask_h != img_h or mask_w != img_w:
            return masks

        padded_masks = []
        for mask in masks:
            padded_mask = np.zeros((img_h + top + bottom, img_w + left + right), dtype=bool)
            padded_mask[top:top + img_h, left:left + img_w] = mask
            padded_masks.append(padded_mask)
        return np.array(padded_masks)

    def _prev_frame(self) -> None:
        if self.kept_indices:
            self.cursor = (self.cursor - 1 + len(self.kept_indices)) % len(self.kept_indices)
            self._update_display()

    def _next_frame(self) -> None:
        if self.kept_indices:
            self.cursor = (self.cursor + 1) % len(self.kept_indices)
            self._update_display()

    def _delete_frame(self) -> None:
        if self.kept_indices:
            removed = self.kept_indices.pop(self.cursor)
            print(f"[Review] Removed frame #{removed}. {len(self.kept_indices)} remaining.")
            if self.cursor >= len(self.kept_indices):
                self.cursor = 0
            self._update_display()

    def _save_all(self) -> None:
        self.result = self.kept_indices
        self.root.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.root.destroy()
