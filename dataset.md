# Dataset & training notes

The dataset is **not** included in this repository. Build yours from your own footage and public sources
(e.g. TACO, Roboflow Universe litter datasets — check each licence) and annotate in **YOLO format**:

```
datasets/swachh_garbage/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt      # class x_center y_center width height (normalised)
```

Edit the class list in `data/data.yaml` to match your labels.

## Good practice

- **Split by scene / location / session, not by random image**, so near-duplicate frames don't leak between train and test.
- Include the deployment conditions: your camera height and angle, day, dusk, night (IR), rain, motion blur, and small/far objects.
- Add **negative images** (clean streets, bins, shadows) to cut false positives.
- Include items *in hands* and *in flight* if you want the detector to help the throw logic.
- Keep class definitions strict and consistent across annotators.

## Augmentation

Two layers, both included:

1. **Offline** – `training/augment_offline.py` (Albumentations): flip, affine (scale/rotate/translate), brightness/contrast, HSV, CLAHE, motion blur, noise, shadows. Only the training split is expanded.
2. **Online** – Ultralytics during training (`training/train.py`): mosaic, mixup, HSV jitter, scale/translate/shear, flips.

## Reporting results

Report metrics from the **held-out test split** (`--split test`), state the dataset size/splits, and include per-class numbers — classes with few instances can show inflated mAP.
