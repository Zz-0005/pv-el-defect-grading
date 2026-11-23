# -*- coding: utf-8 -*-
"""验证 Grad-CAM 与手动预处理是否和 ultralytics 的推理结果一致。"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvinspect import compat  # noqa: E402

compat.configure_dirs()
compat.patch_ultralytics()

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from pvinspect.cam import GradCAM, overlay, preprocess  # noqa: E402

w = ROOT / "artifacts" / "models" / "E1_grouped.pt"
model = YOLO(str(w))
names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
print("classes:", names)

test_dir = ROOT / "artifacts" / "ds4" / "test"
samples = []
for cls in names:
    files = sorted((test_dir / cls).glob("*.png"))[:2]
    samples.extend(files)
print("测试样本:", len(samples))

gcam = GradCAM(model.model)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.model.to(dev)          # CAM 走原生 torch 前向，必须与输入同设备
out_dir = ROOT / "artifacts" / "cam_check"
out_dir.mkdir(parents=True, exist_ok=True)
ok = 0
for p in samples:
    gray = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    r = model.predict(gray, imgsz=224, device=dev, verbose=False)[0]
    ref = r.probs.data.cpu().numpy()
    ref_idx = int(ref.argmax())

    x = preprocess(gray, 224).to(dev)
    cam, idx, probs = gcam(x, None)

    diff = float(np.abs(ref - probs).max())
    same = idx == ref_idx
    ok += int(same and diff < 2e-3)
    print(f"  {p.parent.name:8s} {p.name:14s} predict={names[ref_idx]:8s} cam={names[idx]:8s} "
          f"一致={same} 概率最大偏差={diff:.2e} cam范围=[{cam.min():.2f},{cam.max():.2f}]")

    rgb = overlay(gray, cam)
    Image.fromarray(rgb).save(out_dir / f"cam_{p.parent.name}_{p.stem}.png")

    # 量化验证热力图是否有意义：EL 缺陷表现为暗区，
    # 若 CAM 主要落在最暗的 10% 像素上，说明它看的是缺陷而不是边框/背景。
    # 注意 CAM 是在 224×224 的输入空间算出来的，需要缩回原图尺寸才能对齐像素。
    import cv2  # noqa: E402

    cam_full = cv2.resize(cam, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    thr = np.percentile(gray, 10)
    dark = gray <= thr
    bright = gray >= np.percentile(gray, 50)
    cam_dark = float(cam_full[dark].mean())
    cam_bright = float(cam_full[bright].mean())
    print(f"        CAM 暗区均值={cam_dark:.3f} 亮区均值={cam_bright:.3f} "
          f"比值={cam_dark / max(cam_bright, 1e-6):.2f}")

gcam.close()
print(f"\n{ok}/{len(samples)} 与 ultralytics 推理一致")
print(f"CAM 叠加图：{out_dir}")
