"""Thin Ultralytics YOLO wrapper returning :class:`Detection` objects.

Works with ``.pt`` weights as well as exported ``.onnx`` / NCNN / OpenVINO
models (ultralytics picks the backend from the path).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .types import Detection


class YoloDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.4,
        iou: float = 0.45,
        device: str = "cpu",
        half: bool = False,
        classes: Optional[Sequence[int]] = None,
    ):
        from ultralytics import YOLO  # imported lazily: heavy

        self.model = YOLO(str(weights), task="detect")
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.half = bool(half)
        self.classes = list(classes) if classes is not None else None

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        res = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            classes=self.classes,
            verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        names = res.names
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(label=names[int(k)], conf=float(c), box=tuple(float(v) for v in b))
            for b, c, k in zip(xyxy, conf, cls)
        ]
