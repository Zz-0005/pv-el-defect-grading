# -*- coding: utf-8 -*-
"""
离线试探 2：带预算的不确定性拒识。

上一版（reject_probe.py）按代价最小化搜索阈值，结果退化成"几乎全判待复核"——
因为代价矩阵里"合格→待复核"只记 1 分，"判废→合格"记 20 分，
纯按代价优化时，把所有人都标成待复核就能刷出很低的代价。这不是功能，是钻指标空子。

真实产线有预算约束：复核岗位一天看不了那么多片。
所以这里改成**固定拒识比例 q%**：只把最没把握的 q% 转人工，其余照常自动判定。
q 由业务定（复核产能），不再由代价最优决定。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvinspect import compat  # noqa: E402

compat.configure_dirs()
compat.patch_ultralytics()

from pvinspect import evaluate  # noqa: E402
from pvinspect.config import WORK  # noqa: E402

WEIGHTS = WORK / "models" / "E3_three.pt"
DS = WORK / "ds3"
SAFE = "P1_待复核"


def route(probs: np.ndarray, names: list[str], q: float) -> np.ndarray:
    """把置信度最低的 q 比例样本改判为「待复核」。q=0 即纯 argmax。"""
    order = np.argsort(-probs, axis=1)
    pred = order[:, 0].copy()
    if q <= 0:
        return pred
    conf = probs[np.arange(len(probs)), order[:, 0]]
    n = int(round(len(probs) * q))
    if n <= 0:
        return pred
    idx = np.argsort(conf)[:n]
    pred[idx] = names.index(SAFE)
    return pred


def main() -> int:
    names = None
    data = {}
    for split in ("val", "test"):
        p, t, nm = evaluate.predict_dir(WEIGHTS, DS / split, imgsz=224)
        names, data[split] = nm, (p, t)

    print("带预算的拒识：固定 q% 转人工，其余自动判定")
    print(f"{'q':>5} {'val准确率':>9} {'test准确率':>10} {'test宏F1':>9} "
          f"{'test代价':>9} {'待复核召回':>10}")
    for q in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        row = [q]
        for split in ("val", "test"):
            probs, truth = data[split]
            pred = route(probs, names, q)
            cm = evaluate.confusion(pred, truth, len(names))
            prf = evaluate.per_class_prf(cm)
            if split == "val":
                row.append((pred == truth).mean())
            else:
                row += [(pred == truth).mean(), evaluate.macro_f1(cm),
                        evaluate.weighted_cost(cm, names),
                        prf[names.index(SAFE)]["recall"]]
        print(f"{row[0]:>5.2f} {row[1]:>9.4f} {row[2]:>10.4f} {row[3]:>9.4f} "
              f"{row[4]:>9.1f} {row[5]:>10.3f}")

    print("\n注：q 越大，误判越少但人工负担越重。q=0.15 表示“每 100 片转 15 片给人工复核”。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
