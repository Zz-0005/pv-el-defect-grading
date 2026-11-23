# -*- coding: utf-8 -*-
"""用当前演示模型（E3_three）复测：热力图是否落在缺陷暗区上。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pvinspect import compat  # noqa: E402

compat.configure_dirs()
compat.patch_ultralytics()

from ultralytics import YOLO  # noqa: E402
from pvinspect.cam import GradCAM, preprocess  # noqa: E402


def main() -> int:
    w = str(ROOT / "artifacts" / "models" / "E3_three.pt")
    model = YOLO(w)
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gcam = GradCAM(model.model)
    model.model.to(dev)

    test = ROOT / "artifacts" / "ds3" / "test"
    print(f"{'类别':8s} {'样本':12s} {'预测':6s} {'暗/亮比':>8s}")
    ratios: dict[str, list[float]] = {}
    for cls in names:
        for p in sorted((test / cls).glob("*.png"))[:6]:
            gray = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            r = model.predict(gray, imgsz=224, device=dev, verbose=False)[0]
            pi = int(r.probs.data.cpu().numpy().argmax())
            x = preprocess(gray, 224).to(dev)
            cam, _, _ = gcam(x, pi)
            cam = cv2.resize(cam, (gray.shape[1], gray.shape[0]))
            dark = gray <= np.percentile(gray, 10)
            bright = gray >= np.percentile(gray, 50)
            ratio = float(cam[dark].mean() / max(cam[bright].mean(), 1e-6))
            ratios.setdefault(cls, []).append(ratio)
            print(f"{cls:8s} {p.name:12s} {names[pi]:6s} {ratio:8.2f}")

    print("\n分档汇总（暗/亮比，>2 才算热力图真的指向缺陷）：")
    for cls, v in ratios.items():
        print(f"  {cls:8s} n={len(v)}  中位 {np.median(v):.2f}  范围 {min(v):.2f}~{max(v):.2f}")
    gcam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
