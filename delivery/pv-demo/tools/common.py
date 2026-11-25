# -*- coding: utf-8 -*-
"""交付包内各脚本共用的小工具：路径、读图、预处理、字体、标签。

设计约束：这里**不允许 import torch / ultralytics**。
整个交付包只用 numpy + onnxruntime + Pillow 就能跑，装包约 120 MB。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]          # pv-demo/
MODEL_DIR = ROOT / "models"
OUT = ROOT / "out"
WEB = ROOT / "web"
ASSETS = WEB / "assets"

IMGSZ = 224

# 类别顺序必须与训练时一致：ONNX 输出的第 i 个分量对应 CLASSES[i]
CLASSES = ["P0_合格", "P1_待复核", "P2_判废"]
CLASS_CN = {"P0_合格": "合格", "P1_待复核": "待复核", "P2_判废": "判废"}
CLASS_ACTION = {"P0_合格": "正常流转", "P1_待复核": "返修 / 降级", "P2_判废": "判废"}
CLASS_COLOR = {"P0_合格": (26, 127, 55), "P1_待复核": (200, 130, 10), "P2_判废": (192, 57, 43)}

# 业务代价矩阵（与训练评估一致）。只用于算"加权代价"，不参与训练。
COST = {
    ("P2_判废", "P0_合格"): 20.0, ("P2_判废", "P1_待复核"): 5.0,
    ("P1_待复核", "P0_合格"): 4.0, ("P1_待复核", "P2_判废"): 2.0,
    ("P0_合格", "P2_判废"): 2.5, ("P0_合格", "P1_待复核"): 1.0,
}
DEFAULT_COST = 1.0

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def imread_gray(path: Path) -> np.ndarray:
    """Windows 上中文路径用 cv2.imread 会失败，这里统一用 Pillow。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def preprocess(gray: np.ndarray, size: int = IMGSZ) -> np.ndarray:
    """
    与训练/导出时**完全一致**的预处理，返回 (1,3,size,size) float32。

    训练用的是 ultralytics classify_transforms：
        Resize(短边) → CenterCrop(size) → ToTensor(/255) → Normalize(mean=0, std=1)
    这里必须逐项对齐，否则精度会掉（本项目实测差 0.5 个点以上）。
    """
    im = Image.fromarray(gray, mode="L").convert("RGB")
    w, h = im.size
    short = size
    if w <= h:
        im = im.resize((short, max(short, int(round(h * short / w)))), Image.BILINEAR)
    else:
        im = im.resize((max(short, int(round(w * short / h))), short), Image.BILINEAR)

    # CenterCrop
    w, h = im.size
    left = (w - size) // 2
    top = (h - size) // 2
    im = im.crop((left, top, left + size, top + size))

    a = np.asarray(im, dtype=np.float32) / 255.0
    return a.transpose(2, 0, 1)[None, ...]


def read_label(path: Path) -> str:
    txt = path.read_text(encoding="utf-8").strip().splitlines()
    return txt[0].strip() if txt else ""


def iter_split(split: str):
    """遍历 train/valid/test 里的 (图片路径, 标签)。"""
    base = ROOT / split
    img_dir = base / "images"
    lbl_dir = base / "labels"
    for p in sorted(img_dir.glob("*.png")):
        yield p, read_label(lbl_dir / f"{p.stem}.txt"), split


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cost_of(true: str, pred: str) -> float:
    if true == pred:
        return 0.0
    return COST.get((true, pred), DEFAULT_COST)
