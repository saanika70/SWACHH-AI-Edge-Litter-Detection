"""Fine-tune YOLOv8n on the garbage dataset.

    python training/train.py --data data/data.yaml --epochs 100 --imgsz 640

Built-in Ultralytics augmentation (mosaic, HSV, flips, scale, translate, mixup)
is enabled below; installing `albumentations` adds blur / CLAHE / grayscale.
Use training/augment_offline.py first if you also want an expanded dataset on disk.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1 or cpu (default: auto)")
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="swachh_yolov8n")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        cos_lr=True,
        # ---- augmentation ------------------------------------------------
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,      # lighting / colour variation (day, dusk, night)
        degrees=8.0, translate=0.1, scale=0.5, shear=2.0,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.1, close_mosaic=10,
        plots=True,
    )
    print("Best weights:", model.trainer.best)


if __name__ == "__main__":
    main()
