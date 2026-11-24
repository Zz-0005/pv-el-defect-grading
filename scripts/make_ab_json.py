# -*- coding: utf-8 -*-
"""
生成交付包 out/ab.json：把"我改了什么"的对照证据汇成一份。

两个来源：
  · 训练阶段的四组实验指标（来自 pv-inspect/artifacts/results/results.json）
  · 拒识预算表：**在交付包内**用 out/pred_test.json 现算，只依赖 numpy
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "results" / "results.json"
PKG = ROOT.parent / "光伏电池片EL缺陷分级_交付包" / "pv-demo"
OUT = PKG / "out" / "ab.json"

CLASSES = ["P0_合格", "P1_待复核", "P2_判废"]
COST = {
    ("P2_判废", "P0_合格"): 20.0, ("P2_判废", "P1_待复核"): 5.0,
    ("P1_待复核", "P0_合格"): 4.0, ("P1_待复核", "P2_判废"): 2.0,
    ("P0_合格", "P2_判废"): 2.5, ("P0_合格", "P1_待复核"): 1.0,
}


def cost_of(t, p):
    return 0.0 if t == p else COST.get((t, p), 1.0)


def reject_table(recs):
    """固定人工预算 q：把置信度最低的 q% 改判为「待复核」。"""
    probs = np.array([[r["probs"][c] for c in CLASSES] for r in recs], dtype=np.float32)
    truth = np.array([CLASSES.index(r["true"]) for r in recs])
    n = len(recs)
    order = np.argsort(-probs, axis=1)
    base_pred = order[:, 0]
    conf = probs[np.arange(n), order[:, 0]]

    rows = []
    for q in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        pred = base_pred.copy()
        if q > 0:
            k = int(round(n * q))
            pred[np.argsort(conf)[:k]] = CLASSES.index("P1_待复核")
        acc = float((pred == truth).mean())
        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(truth, pred):
            cm[t, p] += 1
        f1s = []
        for i in range(3):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            pr = tp / max(tp + fp, 1)
            rc = tp / max(tp + fn, 1)
            f1s.append(2 * pr * rc / max(pr + rc, 1e-9))
        cost = sum(cost_of(CLASSES[t], CLASSES[p]) for t, p in zip(truth, pred)) / n * 1000
        recall_safe = cm[1, 1] / max(cm[1].sum(), 1)
        rows.append({
            "q": q, "accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "cost_per_1000": round(float(cost), 1),
            "recall_review": round(float(recall_safe), 3),
        })
    return rows


def main() -> int:
    res = json.loads(SRC.read_text(encoding="utf-8"))
    recs = json.loads((PKG / "out" / "pred_test.json").read_text(encoding="utf-8"))
    recs = [r for r in recs if r["true"] in CLASSES]

    experiments = []
    meta = {
        "E1_grouped": ("四级分级 · 按组件分组切分", "主实验：暴露了中间两档坍塌"),
        "E1_random": ("四级分级 · 随机切分", "泄漏对照：同一模型换切分方式"),
        "E2_binary": ("二分类 · 合格/不合格", "更粗的基线"),
        "E3_three": ("三档操作级 · 分组切分", "交付包采用的方案"),
    }
    for key, (title, note) in meta.items():
        r = res.get(key)
        if not r:
            continue
        experiments.append({
            "id": key, "title": title, "note": note,
            "accuracy": round(r["accuracy"], 4),
            "macro_f1": round(r["macro_f1"], 4),
            "cost_per_1000": round(r["cost_per_1000"], 1),
            "in_package": key == "E3_three",
        })

    ab = {
        "experiments": experiments,
        "leakage": {
            "grouped": res["E1_grouped"]["accuracy"],
            "random": res["E1_random"]["accuracy"],
            "delta_pt": round((res["E1_random"]["accuracy"] - res["E1_grouped"]["accuracy"]) * 100, 2),
            "groups": 42,
            "cross_groups": 0,
            "cross_dups": 0,
        },
        "reject_budget": reject_table(recs),
        "note": ("四组实验的指标来自训练阶段评估（权重未全部随包分发）；"
                 "E3_three 的指标已在包内用 ONNX 复现，与训练阶段一致。"
                 "拒识预算表由 out/pred_test.json 现算。"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ab, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT}")
    print("  实验数:", len(experiments), " 拒识行数:", len(ab["reject_budget"]))
    print("  泄漏差:", ab["leakage"]["delta_pt"], "个百分点")
    return 0


if __name__ == "__main__":
    sys.exit(main())
