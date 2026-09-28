"""Measure inference latency / FPS of a model on this machine (e.g. the Pi).

    python scripts/benchmark.py --weights models/swachh_litter.pt --imgsz 640 --frames 100
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from swachh_ai.detector import YoloDetector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    det = YoloDetector(args.weights, imgsz=args.imgsz, device=args.device)
    frame = (np.random.rand(480, 640, 3) * 255).astype("uint8")
    for _ in range(5):  # warm-up
        det(frame)
    t0 = time.perf_counter()
    for _ in range(args.frames):
        det(frame)
    dt = (time.perf_counter() - t0) / args.frames
    print(f"{args.weights}: {dt * 1000:.1f} ms/frame  ->  {1 / dt:.1f} FPS  (imgsz={args.imgsz}, device={args.device})")


if __name__ == "__main__":
    main()
