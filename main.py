"""CLI entry point:  python -m swachh_ai.main --config config/config.yaml"""
from __future__ import annotations

import argparse
import logging

from .config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="SWACHH-AI edge litter-deterrence system")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--source", help="override camera source: index, video file path or RTSP URL")
    ap.add_argument("--show", action="store_true", help="show a live debug window (needs a display)")
    ap.add_argument("--mock-hw", action="store_true", help="do not touch GPIO (laptop / video testing)")
    ap.add_argument("--no-notify", action="store_true", help="disable MQTT / Telegram / webhook")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    cfg = load_config(args.config)
    if args.source is not None:
        cfg.setdefault("camera", {})["source"] = args.source

    from .pipeline import SwachhPipeline  # imported late: pulls in torch/opencv

    SwachhPipeline(cfg, show=args.show, force_mock_hw=args.mock_hw, disable_notify=args.no_notify).run()


if __name__ == "__main__":
    main()
