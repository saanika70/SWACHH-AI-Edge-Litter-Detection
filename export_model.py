"""Export trained weights for fast CPU inference on the Raspberry Pi.

    python training/export_model.py --weights best.pt --format ncnn --imgsz 640

NCNN (ARM-optimised) or ONNX are usually much faster than PyTorch on a Pi.
Point `models.litter.weights` in config.yaml at the resulting file/folder.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="ncnn", choices=["ncnn", "onnx", "openvino", "tflite"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    path = YOLO(args.weights).export(format=args.format, imgsz=args.imgsz, half=args.half)
    print("Exported to:", path)


if __name__ == "__main__":
    main()
