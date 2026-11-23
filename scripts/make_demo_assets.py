# -*- coding: utf-8 -*-
"""
生成答辩现场用的演示素材：

  demo/样本/            8 张可直接拖进浏览器的测试图（文件名中性，不泄露答案）
  demo/out/分级示例.png  6 张图的"原图 + 热力图 + 真实/预测"拼图（放在 PPT 里）
  demo/out/单张详解.png  一张图的三联图：原图 / 热力图 / 叠加
  demo/out/样本对照.csv  每张样本的真实级别、预测级别、置信度

为什么要预生成：现场如果 GPU/端口/浏览器出问题，直接展示这张静态图也能讲完。
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvinspect import compat  # noqa: E402

compat.configure_dirs()
compat.patch_ultralytics()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from pvinspect.cam import GradCAM, overlay, preprocess  # noqa: E402
from pvinspect.config import GRADE_ACTION, GRADE_CN  # noqa: E402

DEMO = ROOT / "demo"
SAMPLES = DEMO / "样本"
OUT = DEMO / "out"
TEST = ROOT / "artifacts" / "ds3" / "test"
WEIGHTS = ROOT / "artifacts" / "models" / "E3_three.pt"

# ---------------------------------------------------------------------------
# 选样策略：
#   · 展示用图优先挑"判对且置信度高"的样本 —— 这是演示，不是评测；
#   · 但**刻意保留 1 张判错的难例**（待复核被判成合格），用来讲"最难的一档"。
#   所有指标仍以完整的 582 张测试集为准（见 docs/实验记录.md），
#   选样只影响展示，不影响任何报告里的数字。
# ---------------------------------------------------------------------------
CANDIDATES_PER_CLASS = 25
SHOWCASE = [("P2_判废", 2), ("P0_合格", 3), ("P1_待复核", 1)]
HARD_CASE_CLASS = "P1_待复核"


def setup_font(plt):
    from matplotlib import font_manager
    for cand in ("Microsoft YaHei", "SimHei", "SimSun"):
        if any(cand in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> int:
    for d in (SAMPLES, OUT):
        d.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(WEIGHTS))
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gcam = GradCAM(model.model)
    model.model.to(dev)

    def infer(path: Path):
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        r = model.predict(gray, imgsz=224, device=dev, verbose=False)[0]
        probs = r.probs.data.cpu().numpy()
        pi = int(probs.argmax())
        return gray, names[pi], float(probs[pi])

    # ---------- 1) 挑展示样本：每档取判对且置信度最高的若干张
    picked: list[tuple[Path, str]] = []
    hard_case: tuple[Path, str] | None = None
    for cls, k in SHOWCASE:
        good = []
        for p in sorted((TEST / cls).glob("*.png"))[:CANDIDATES_PER_CLASS]:
            _, pred, conf = infer(p)
            if pred == cls:
                good.append((conf, p))
            elif hard_case is None and cls == HARD_CASE_CLASS:
                hard_case = (p, cls)
        good.sort(reverse=True)
        picked += [(p, cls) for _, p in good[:k]]

    if hard_case is not None:
        picked.append(hard_case)

    # 复制成中性文件名（避免评委从文件名读出答案）
    renamed: list[tuple[Path, str]] = []
    for i, (src, cls) in enumerate(picked, 1):
        dst = SAMPLES / f"样本{i:02d}.png"
        shutil.copyfile(src, dst)
        renamed.append((dst, cls))
    picked = renamed

    # ---------- 2) 逐张推理 + 热力图
    rows = []
    panels = []
    for dst, true_cls in picked:
        gray, pred, conf = infer(dst)
        pi = names.index(pred)
        x = preprocess(gray, 224).to(dev)
        cam, _, _ = gcam(x, pi)
        rgb = overlay(gray, cam)
        Image.fromarray(rgb).save(OUT / f"{dst.stem}_heatmap.png")
        rows.append({
            "sample": dst.name,
            "true": GRADE_CN.get(true_cls, true_cls),
            "pred": GRADE_CN.get(pred, pred),
            "conf": f"{conf:.3f}",
            "ok": "OK" if pred == true_cls else "MISS",
            "action": GRADE_ACTION.get(pred, ""),
        })
        panels.append((gray, rgb, true_cls, pred, conf))

    with (OUT / "samples.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "true", "pred", "conf", "ok", "action"])
        w.writeheader()
        w.writerows(rows)

    # ---------- 3) 六联拼图（放 PPT 用）
    setup_font(plt)
    show = panels[:6]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), dpi=170)
    for ax, (gray, rgb, true_cls, pred, conf) in zip(axes.ravel(), show):
        ax.imshow(rgb)
        ax.axis("off")
        ok = pred == true_cls
        ax.set_title(f"真实 {GRADE_CN.get(true_cls, true_cls)}  →  预测 "
                     f"{GRADE_CN.get(pred, pred)}   {conf:.2f}",
                     fontsize=9, color="#1a7f37" if ok else "#c0392b")
    fig.suptitle("光伏电池片 EL 缺陷分级示例（热力图为 Grad-CAM，越红表示模型越关注）",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "showcase.png")
    plt.close(fig)

    # ---------- 4) 单张三联图
    gray, rgb, true_cls, pred, conf = panels[0]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), dpi=170)
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("① 原始 EL 图像", fontsize=10)
    axes[1].imshow(rgb)
    axes[1].set_title("② Grad-CAM 热力图叠加", fontsize=10)
    axes[2].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"③ 判定：{GRADE_CN.get(pred, pred)}（置信度 {conf:.2f}）", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "single_case.png")
    plt.close(fig)

    gcam.close()
    print(f"samples -> {SAMPLES}")
    print(f"assets  -> {OUT}")
    for r in rows:
        print(f"  {r['sample']}  true={r['true']:4s} pred={r['pred']:4s} "
              f"conf={r['conf']} {r['ok']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
