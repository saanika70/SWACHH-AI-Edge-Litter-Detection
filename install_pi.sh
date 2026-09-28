#!/usr/bin/env bash
# One-shot setup for Raspberry Pi OS (Bookworm, 64-bit). Run from the repo root.
set -euo pipefail

sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv python3-picamera2 python3-lgpio git

# --system-site-packages exposes the apt-provided picamera2 / lgpio to the venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data models
[ -f .env ] || cp .env.example .env

echo
echo "Done. Test with:  source .venv/bin/activate && pytest -q && python scripts/demo_synthetic.py"
echo "Install the service:  sudo cp scripts/swachh-ai.service /etc/systemd/system/ && sudo systemctl enable --now swachh-ai"
