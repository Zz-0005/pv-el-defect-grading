# -*- coding: utf-8 -*-
"""抓取 model.predict 实际送入模型的张量，与 classify_transforms 的输出逐元素比对。"""
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
from ultralytics.data.augment import classify_transforms, DEFAULT_MEAN, DEFAULT_STD  # noqa: E402

print("DEFAULT_MEAN", DEFAULT_MEAN, "DEFAULT_STD", DEFAULT_STD)

w = ROOT / "artifacts" / "models" / "E1_grouped.pt"
model = YOLO(str(w))
dev = "cuda" if torch.cuda.is_available() else "cpu"

p = sorted((ROOT / "artifacts" / "ds4" / "test").rglob("*.png"))[0]
pil = Image.fromarray(np.asarray(Image.open(p).convert("L"))).convert("RGB")

captured = {}


def pre_hook(mod, args):
    captured["x"] = args[0].detach().clone()


h = model.model.register_forward_pre_hook(pre_hook)
r = model.predict(pil, imgsz=224, device=dev, verbose=False)[0]
h.remove()
px = captured["x"]
print("predict 送入张量:", tuple(px.shape), px.dtype, "范围", float(px.min()), float(px.max()))

tf = classify_transforms(224)
tx = tf(pil).unsqueeze(0).to(px.device)
print("我的张量      :", tuple(tx.shape), tx.dtype, "范围", float(tx.min()), float(tx.max()))
print("最大逐元素差:", float((tx.float() - px.float()).abs().max()))

# 用 predict 的真实张量走一遍前向
model.model.to(px.device).eval()
with torch.inference_mode():
    lg = model.model(px)
    if isinstance(lg, (list, tuple)):
        lg = lg[0]
    pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
print("用 predict 张量前向:", np.round(pr, 4))
print("predict 返回概率   :", np.round(r.probs.data.cpu().numpy(), 4))

# 用我的张量前向
with torch.inference_mode():
    lg2 = model.model(tx.float())
    if isinstance(lg2, (list, tuple)):
        lg2 = lg2[0]
    pr2 = torch.softmax(lg2.float(), 1)[0].cpu().numpy()
print("用我的张量前向     :", np.round(pr2, 4))
