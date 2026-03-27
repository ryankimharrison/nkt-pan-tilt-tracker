"""
export_engine.py — One-time TensorRT engine build for the YOLO pose model.

Run this once on the target machine before starting the tracker:
    python export_engine.py

The engine is device-specific (tied to your exact GPU and TensorRT version).
If you upgrade drivers, reinstall CUDA, or switch GPUs, re-run this script.

Requirements:
    - NVIDIA GPU with CUDA installed
    - TensorRT installed (comes with nvidia-tensorrt or via torch extras)
    - pip install ultralytics

Output: yolov8n-pose.engine (same directory as this script)
"""

import os
import sys

# Read model path from config so this stays in sync with the tracker
sys.path.insert(0, os.path.dirname(__file__))
import config

pt_path = config.YOLO_MODEL_PATH
if pt_path.endswith(".engine"):
    print(f"[Export] config already points to an engine file: {pt_path}")
    print("[Export] Set YOLO_MODEL_PATH to the .pt file to export from.")
    sys.exit(1)

if not os.path.isfile(pt_path):
    print(f"[Export] Model not found: {pt_path}")
    sys.exit(1)

engine_path = os.path.splitext(pt_path)[0] + ".engine"
print(f"[Export] Source : {pt_path}")
print(f"[Export] Output : {engine_path}")
print(f"[Export] imgsz  : {config.CAMERA_WIDTH} (square, matches ultralytics .track())")
print("[Export] This may take several minutes on first run ...")

from ultralytics import YOLO
model = YOLO(pt_path)
model.export(
    format="engine",
    imgsz=config.CAMERA_WIDTH,   # square — ultralytics .track(imgsz=N) pads to NxN
    half=True,       # FP16 — 2x throughput on Turing/Ampere/Ada GPUs with no accuracy loss
    device=0,        # GPU 0
    verbose=True,
)

if os.path.isfile(engine_path):
    size_mb = os.path.getsize(engine_path) / 1024 / 1024
    print(f"\n[Export] Done. Engine: {engine_path} ({size_mb:.1f} MB)")
    print("[Export] The tracker will automatically use this engine on next launch.")
else:
    print("\n[Export] WARNING: engine file not found after export — check output above for errors.")
