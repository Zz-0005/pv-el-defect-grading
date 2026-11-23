# -*- coding: utf-8 -*-
"""对照：不同输入形式送进 model.predict，哪种与 classify_transforms 等价。"""
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
from ultralytics.data.augment import classify_transforms  # noqa: E402

w = ROOT / "artifacts" / "models" / "E1_grouped.pt"
model = YOLO(str(w))
dev = "cuda" if torch.cuda.is_available() else "cpu"

p = sorted((ROOT / "artifacts" / "ds4" / "test").rglob("*.png"))[0]
gray = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
rgb = np.stack([gray] * 3, axis=-1)
pil = Image.fromarray(gray).convert("RGB")

# 参考：直接用 classify_transforms 张量做前向
tf = classify_transforms(224)
x = tf(pil).unsqueeze(0).to(dev)
model.model.to(dev).eval()
with torch.no_grad():
    logits = model.model(x)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    ref = torch.softmax(logits, 1)[0].cpu().numpy()

variants = {
    "numpy 2D 灰度": gray,
    "numpy 3 通道 (RGB)": rgb,
    "PIL RGB": pil,
    "文件路径": str(p),
}
print(f"参考（classify_transforms 张量）：{np.round(ref, 4)}")
for name, inp in variants.items():
    r = model.predict(inp, imgsz=224, device=dev, verbose=False)[0]
    pr = r.probs.data.cpu().numpy()
    print(f"  {name:20s} {np.round(pr, 4)}  最大偏差 {np.abs(pr - ref).max():.2e}")
