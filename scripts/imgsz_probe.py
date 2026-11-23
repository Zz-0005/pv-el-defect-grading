# -*- coding: utf-8 -*-
"""同一权重在不同输入分辨率下的测试集表现 —— 判断"细节是否被缩放抹掉"。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvinspect import compat  # noqa: E402

compat.configure_dirs()
compat.patch_ultralytics()

from pvinspect import evaluate  # noqa: E402
from pvinspect.config import GRADES, WORK  # noqa: E402

w = WORK / "models" / "E1_grouped.pt"
test = WORK / "ds4" / "test"
print(f"权重 {w.name}，测试集 {test}")
print(f"{'imgsz':>6} {'准确率':>8} {'宏F1':>8} {'G1召回':>8} {'G2召回':>8}")
for imgsz in (160, 224, 288, 300, 320):
    probs, truth, names = evaluate.predict_dir(w, test, imgsz=imgsz)
    m = evaluate.metrics_from_probs(probs, truth, names)
    g1 = m["per_class"].get("G1_轻微", {}).get("recall", float("nan"))
    g2 = m["per_class"].get("G2_中度", {}).get("recall", float("nan"))
    print(f"{imgsz:6d} {m['accuracy']:8.4f} {m['macro_f1']:8.4f} {g1:8.3f} {g2:8.3f}")
