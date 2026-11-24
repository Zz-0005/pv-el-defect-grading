# -*- coding: utf-8 -*-
"""
为交付包的演示图预生成 Grad-CAM 热力图（这一步需要 torch，只在制作包时跑）。

产物：交付包 web/assets/cam/<name>.png（单通道热力图，0~255）
交付包运行时不需要 torch —— 热力图是预先算好的静态资源。

为什么不在交付包里现场算：CAM 要反向传播，必须依赖 torch；
而交付包的定位是「装 4 个包、双击就能跑」，所以把重依赖留在制作阶段。
"""
import sys
from pathlib import Path

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

PKG = ROOT.parent / "光伏电池片EL缺陷分级_交付包" / "pv-demo"
OUT = PKG / "web" / "assets" / "cam"
W = ROOT / "artifacts" / "models" / "E3_three.pt"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(W))
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gcam = GradCAM(model.model)
    model.model.to(dev)

    targets = []
    for sub in ("demo", "demo2", "fails"):
        for p in sorted((PKG / sub / "images").glob("*.png")):
            targets.append((sub, p))

    print(f"生成热力图 {len(targets)} 张 -> {OUT}")
    for i, (sub, p) in enumerate(targets, 1):
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        r = model.predict(gray, imgsz=224, device=dev, verbose=False)[0]
        pi = int(r.probs.data.cpu().numpy().argmax())
        x = preprocess(gray, 224).to(dev)
        cam, _, _ = gcam(x, pi)
        Image.fromarray((cam * 255).astype(np.uint8), mode="L").save(OUT / f"{p.stem}.png")
        if i % 20 == 0:
            print(f"  {i}/{len(targets)}", flush=True)

    gcam.close()
    print(f"完成：{len(list(OUT.glob('*.png')))} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
