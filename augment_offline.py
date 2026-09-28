"""Offline (on-disk) augmentation for a YOLO-format dataset using Albumentations.

Only the *train* split is expanded - val/test are copied untouched to avoid
leaking augmented copies of the same image across splits.

    python training/augment_offline.py --src datasets/swachh_garbage \
        --dst datasets/swachh_garbage_aug --copies 2
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import albumentations as A
import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_pipeline() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(p=0.4),
            A.CLAHE(p=0.2),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.GaussNoise(p=0.2),
            A.RandomShadow(p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def read_labels(path: Path):
    boxes, classes = [], []
    if path.is_file():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:  # skip polygons / malformed rows
                continue
            c, x, y, w, h = int(p[0]), *map(float, p[1:])
            clamp = lambda v: min(max(v, 1e-6), 1.0)
            boxes.append([clamp(x), clamp(y), clamp(w), clamp(h)])
            classes.append(c)
    return boxes, classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--copies", type=int, default=2, help="augmented copies per training image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    pipeline = build_pipeline()

    for split in ("val", "test"):
        for kind in ("images", "labels"):
            s = src / kind / split
            if s.is_dir():
                shutil.copytree(s, dst / kind / split, dirs_exist_ok=True)

    (dst / "images/train").mkdir(parents=True, exist_ok=True)
    (dst / "labels/train").mkdir(parents=True, exist_ok=True)

    n_img = n_aug = 0
    for img_path in sorted((src / "images/train").iterdir()):
        if img_path.suffix.lower() not in IMG_EXT:
            continue
        lbl_path = src / "labels/train" / f"{img_path.stem}.txt"
        shutil.copy2(img_path, dst / "images/train" / img_path.name)
        if lbl_path.is_file():
            shutil.copy2(lbl_path, dst / "labels/train" / lbl_path.name)
        n_img += 1

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, classes = read_labels(lbl_path)
        for k in range(args.copies):
            try:
                out = pipeline(image=image, bboxes=boxes, class_labels=classes)
            except Exception as exc:
                print(f"skip {img_path.name}: {exc}")
                continue
            if boxes and not out["bboxes"]:
                continue  # every object was cropped out
            stem = f"{img_path.stem}_aug{k}"
            cv2.imwrite(str(dst / "images/train" / f"{stem}.jpg"), cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR))
            with open(dst / "labels/train" / f"{stem}.txt", "w") as fh:
                for (x, y, w, h), c in zip(out["bboxes"], out["class_labels"]):
                    fh.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            n_aug += 1
    print(f"{n_img} original + {n_aug} augmented training images -> {dst}")


if __name__ == "__main__":
    main()
