# -*- coding: utf-8 -*-
"""
离线试探：不确定性拒识（reject option）能不能改善三档模型？

规则：当模型"没把握"时，不输出它猜的那一档，而是输出安全档「待复核」。
  A 间隔规则：top1 与 top2 的概率差 < δ   → 待复核
  B 置信规则：最大概率 < θ                → 待复核
  C 两者任一满足                          → 待复核

δ 与 θ 只在**验证集**上按加权代价选，测试集只用来报告最终结果。
"""
from __future__ import annotations

import json
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


def apply_rule(probs: np.ndarray, names: list[str], delta: float = 0.0, theta: float = 0.0) -> np.ndarray:
    """返回预测索引；满足不确定性条件时强制输出 SAFE 类。"""
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    p1 = probs[np.arange(len(probs)), order[:, 0]]
    p2 = probs[np.arange(len(probs)), order[:, 1]]
    uncertain = np.zeros(len(probs), dtype=bool)
    if delta > 0:
        uncertain |= (p1 - p2) < delta
    if theta > 0:
        uncertain |= p1 < theta
    pred = top1.copy()
    pred[uncertain] = names.index(SAFE)
    return pred


def score(pred: np.ndarray, truth: np.ndarray, names: list[str]) -> dict:
    k = len(names)
    cm = evaluate.confusion(pred, truth, k)
    prf = evaluate.per_class_prf(cm)
    return {
        "acc": float((pred == truth).mean()),
        "macro_f1": evaluate.macro_f1(cm),
        "cost": evaluate.weighted_cost(cm, names),
        "recall_safe": prf[names.index(SAFE)]["recall"],
        "cm": cm,
    }


def main() -> int:
    names = None
    probs_val = probs_test = None
    truth_val = truth_test = None
    for split in ("val", "test"):
        p, t, nm = evaluate.predict_dir(WEIGHTS, DS / split, imgsz=224)
        names = nm
        if split == "val":
            probs_val, truth_val = p, t
        else:
            probs_test, truth_test = p, t

    assert names is not None
    safe_i = names.index(SAFE)
    print(f"类别: {names}")
    print(f"验证 {len(truth_val)} 张 / 测试 {len(truth_test)} 张\n")

    base_val = score(apply_rule(probs_val, names), truth_val, names)
    base_test = score(apply_rule(probs_test, names), truth_test, names)
    print(f"基线（纯 argmax）  val: acc={base_val['acc']:.4f} cost={base_val['cost']:.1f} "
          f"待复核召回={base_val['recall_safe']:.3f}")
    print(f"                   test: acc={base_test['acc']:.4f} cost={base_test['cost']:.1f} "
          f"待复核召回={base_test['recall_safe']:.3f}\n")

    # ---- 在验证集上网格搜索 δ / θ
    best = None
    grid = np.arange(0.0, 0.95, 0.05)
    for rule in ("margin", "conf", "either"):
        for d in (grid if rule in ("margin", "either") else [0.0]):
            for th in (grid if rule in ("conf", "either") else [0.0]):
                if rule == "margin" and th > 0:
                    continue
                if rule == "conf" and d > 0:
                    continue
                if d == 0 and th == 0:
                    continue
                pred = apply_rule(probs_val, names, d, th)
                m = score(pred, truth_val, names)
                if best is None or m["cost"] < best[0]:
                    best = (m["cost"], rule, float(d), float(th))

    if best is None:
        print("没有找到更好的规则")
        return 0

    _, rule, d, th = best
    print(f"验证集最优规则: {rule}  δ={d:.2f}  θ={th:.2f}  (val cost {best[0]:.1f})")

    for tag, pb, tb in (("验证", probs_val, truth_val), ("测试", probs_test, truth_test)):
        pred = apply_rule(pb, names, d, th)
        m = score(pred, tb, names)
        base = base_val if tag == "验证" else base_test
        delta_cost = (m["cost"] - base["cost"]) / base["cost"] * 100
        print(f"\n{tag}集：")
        print(f"  准确率   {base['acc']:.4f} → {m['acc']:.4f}")
        print(f"  宏F1     {base['macro_f1']:.4f} → {m['macro_f1']:.4f}")
        print(f"  加权代价 {base['cost']:.1f} → {m['cost']:.1f}  ({delta_cost:+.1f}%)")
        print(f"  待复核召回 {base['recall_safe']:.3f} → {m['recall_safe']:.3f}")
        cm = m["cm"]
        for i, n in enumerate(names):
            print(f"    {n}: " + " ".join(f"{v:4d}" for v in cm[i]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
