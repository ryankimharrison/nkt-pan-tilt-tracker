"""
detector.py — YOLOv8 + TensorRT person detector wrapper.

Note: In the current architecture, tracker.py owns its own YOLO model and
calls model.track() directly (which handles detection + ByteTrack in one call).
This module is kept for standalone testing / future use.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import config


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


class Detector:
    """Wraps a YOLOv8 model for person-only detection."""

    def __init__(self):
        from ultralytics import YOLO
        print(f"[Detector] Loading model: {config.YOLO_MODEL_PATH}")
        self._model = YOLO(config.YOLO_MODEL_PATH)
        print("[Detector] Model loaded.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(
            frame,
            classes=[0],
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(Detection(x1, y1, x2, y2, conf))
        return detections
