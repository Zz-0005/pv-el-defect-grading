# -*- coding: utf-8 -*-
"""层选择实验：比较不同深度的特征图作为 CAM 目标层时的"缺陷对齐度"。

指标：CAM 在最暗 10% 像素上的均值 / 在最亮 50% 像素上的均值。
EL 图像里缺陷是暗的，比值明显 > 1 才说明热力图落在缺陷上。
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pvinspect import compat  # noqa: E402

compat.configure_dirs()
compat.patch_ultralytics()

from ultralytics import YOLO  # noqa: E402
from pvinspect.cam import preprocess  # noqa: E402

w = ROOT / "artifacts" / "models" / "E1_grouped.pt"
model = YOLO(str(w))
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = model.model.to(dev).eval()
net.requires_grad_(True)

acts: list[tuple[int, torch.Tensor]] = []
handles = []


def mk_hook(idx):
    def hook(mod, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4 and out.requires_grad:
            out.retain_grad()
            acts.append((idx, out))
    return hook


for i, m in enumerate(net.model):
    handles.append(m.register_forward_hook(mk_hook(i)))

samples = sorted((ROOT / "artifacts" / "ds4" / "test" / "G3_严重").glob("*.png"))[:4]

for p in samples:
    gray = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
    acts.clear()
    x = preprocess(gray, 224).to(dev)
    net.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = net(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    idx = int(out[0].argmax())
    out[0, idx].backward()

    print(f"\n{p.name}  预测类 {idx}")
    thr = np.percentile(gray, 10)
    dark = gray <= thr
    bright = gray >= np.percentile(gray, 50)

    for li, act in acts:
        g = act.grad
        if g is None:
            continue
        wgt = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((wgt * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        if cam.max() - cam.min() < 1e-9:
            continue
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam_full = cv2.resize(cam, (gray.shape[1], gray.shape[0]))
        ratio = float(cam_full[dark].mean() / max(cam_full[bright].mean(), 1e-6))
        size = tuple(act.shape[-2:])
        print(f"  层{li:3d}  特征图 {str(size):10s}  暗/亮比 {ratio:.2f}")

for h in handles:
    h.remove()
