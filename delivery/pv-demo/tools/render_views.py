# -*- coding: utf-8 -*-
"""
把演示图渲染成网页用的三视图：原图 / 数据集真值 / 模型预测。

    python tools/render_views.py --set demo
    python tools/render_views.py --set demo2

产物（web/assets/）：
    orig_<name>.jpg   原图（2 倍放大，带图名）
    gt_<name>.jpg     原图 + 真值标签条
    pred_<name>.jpg   原图 + Grad-CAM 热力叠加 + 预测标签条与置信度

热力图来自 web/assets/cam/<name>.png（制作包时用 torch 预生成），
所以本脚本**不需要 torch**。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ASSETS, CLASS_ACTION, CLASS_CN, CLASS_COLOR, CLASSES, ROOT, font, imread_gray, load_json,
)

SCALE = 2
BAR_H = 46


def _label_bar(w: int, text: str, color) -> Image.Image:
    bar = Image.new("RGB", (w, BAR_H), color)
    d = ImageDraw.Draw(bar)
    f = font(24)
    tw = d.textlength(text, font=f)
    d.text(((w - tw) / 2, (BAR_H - 28) / 2), text, font=f, fill=(255, 255, 255))
    return bar


def overlay(gray: np.ndarray, cam: np.ndarray | None, alpha: float = 0.45) -> np.ndarray:
    base = np.stack([gray] * 3, axis=-1).astype(np.float32)
    if cam is None:
        return base.astype(np.uint8)
    h, w = gray.shape
    c = np.asarray(Image.fromarray(cam, mode="L").resize((w, h), Image.BILINEAR),
                   dtype=np.float32) / 255.0
    r = np.clip(1.5 * c - 0.3, 0, 1)
    g = np.clip(1.6 * np.minimum(c, 1 - c) * 2, 0, 1)
    b = np.clip(1.2 - 1.8 * c, 0, 1)
    heat = np.stack([r, g, b], axis=-1) * 255.0
    return np.clip(base * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="demo", choices=["demo", "demo2", "fails"])
    a = ap.parse_args(argv)

    split_dir = ROOT / a.set
    pred_json = ROOT / "out" / f"pred_{a.set}.json"
    if not pred_json.exists():
        print(f"缺少 {pred_json}，先跑 onnx_predict.py")
        return 1
    preds = {r["image"]: r for r in load_json(pred_json)}

    ASSETS.mkdir(parents=True, exist_ok=True)
    cam_dir = ASSETS / "cam"
    n = 0
    for p in sorted((split_dir / "images").glob("*.png")):
        rec = preds.get(p.name)
        if rec is None:
            continue
        gray = imread_gray(p)
        up = Image.fromarray(gray, mode="L").resize(
            (gray.shape[1] * SCALE, gray.shape[0] * SCALE), Image.LANCZOS).convert("RGB")
        w, h = up.size

        true = rec["true"] or ""
        pred = rec["pred"]
        conf = rec["conf"]

        # ① 原图
        orig = up.copy()
        d = ImageDraw.Draw(orig)
        d.text((8, 8), p.stem, font=font(20), fill=(120, 200, 255))
        orig.save(ASSETS / f"orig_{p.stem}.jpg", quality=88)

        # ② 真值
        gt = Image.new("RGB", (w, h + BAR_H))
        gt.paste(up, (0, 0))
        if true in CLASS_CN:
            gt.paste(_label_bar(w, f"数据集真值：{CLASS_CN[true]}（{CLASS_ACTION[true]}）",
                                CLASS_COLOR[true]), (0, h))
        else:
            gt.paste(_label_bar(w, "无标签", (110, 120, 130)), (0, h))
        gt.save(ASSETS / f"gt_{p.stem}.jpg", quality=88)

        # ③ 预测（热力图叠加）
        cam = None
        cam_p = cam_dir / f"{p.stem}.png"
        if cam_p.exists():
            with Image.open(cam_p) as im:
                cam = np.asarray(im.convert("L"), dtype=np.uint8)
        rgb = overlay(gray, cam)
        bg = Image.fromarray(rgb, mode="RGB").resize((w, h), Image.LANCZOS)
        pr = Image.new("RGB", (w, h + BAR_H))
        pr.paste(bg, (0, 0))
        ok = "✓" if pred == true else "✗"
        pr.paste(_label_bar(w, f"模型预测：{CLASS_CN.get(pred, pred)}  {conf:.2f}  {ok}",
                            CLASS_COLOR.get(pred, (90, 90, 90))), (0, h))
        pr.save(ASSETS / f"pred_{p.stem}.jpg", quality=88)
        n += 1

    print(f"{a.set}: 渲染 {n} 组三视图 -> {ASSETS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
