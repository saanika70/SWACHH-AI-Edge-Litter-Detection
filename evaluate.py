"""Evaluate weights and dump metrics (per class) to JSON + a Markdown table.

    python training/evaluate.py --weights runs/train/swachh_yolov8n/weights/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device, plots=True,
                  project=args.out, name=f"eval_{args.split}", exist_ok=True)

    names = m.names
    per_class = {}
    for i, cid in enumerate(m.box.ap_class_index):
        per_class[names[int(cid)]] = {
            "precision": float(m.box.p[i]),
            "recall": float(m.box.r[i]),
            "mAP50": float(m.box.ap50[i]),
            "mAP50-95": float(m.box.ap[i]),
        }
    summary = {
        "split": args.split,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "per_class": per_class,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{args.split}.json").write_text(json.dumps(summary, indent=2))

    lines = ["| Class | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for k, v in per_class.items():
        lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50-95']:.3f} |")
    lines.append(f"| **all** | {summary['precision']:.3f} | {summary['recall']:.3f} | "
                 f"{summary['mAP50']:.3f} | {summary['mAP50-95']:.3f} |")
    table = "\n".join(lines)
    (out / f"metrics_{args.split}.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
