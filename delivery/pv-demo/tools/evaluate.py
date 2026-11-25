# -*- coding: utf-8 -*-
"""
评估：把 out/pred_*.json 与真值比对，算出网页里所有指标。

    python tools/evaluate.py --pred out/pred_test.json --json out/metrics_test.json

指标口径（写进网页，避免"指标好看但没用"）：
  · 全部在冻结测试集上算，测试集从未参与训练或调参；
  · 除 accuracy 外单独报宏平均 F1（待复核只占 9.8%，会被 accuracy 掩盖）；
  · 单独报"判废漏判率"（真实判废被判成合格/待复核）；
  · 报加权代价/千张：判废→合格记 20 分，合格→待复核记 1 分。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, ROOT, cost_of, dump_json, load_json  # noqa: E402


def confusion(pred, true, classes=CLASSES):
    k = len(classes)
    cm = np.zeros((k, k), dtype=np.int64)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(true, pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def per_class(cm, classes=CLASSES):
    out = {}
    for i, c in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out[c] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            "support": int(cm[i].sum()),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="out/pred_test.json")
    ap.add_argument("--json", default="out/metrics_test.json")
    a = ap.parse_args(argv)

    recs = load_json(ROOT / a.pred)
    recs = [r for r in recs if r["true"] in CLASSES]
    true = [r["true"] for r in recs]
    pred = [r["pred"] for r in recs]

    cm = confusion(pred, true)
    prf = per_class(cm)
    acc = float(np.mean([p == t for p, t in zip(pred, true)]))
    macro_f1 = float(np.mean([v["f1"] for v in prf.values()]))

    total_cost = sum(cost_of(t, p) for t, p in zip(true, pred))
    cost_per_1000 = total_cost / max(len(recs), 1) * 1000

    # 判废漏判：真实 P2 被判成 P0/P1
    i2 = CLASSES.index("P2_判废")
    n2 = int(cm[i2].sum())
    miss = int(cm[i2, 0] + cm[i2, 1])
    severe_miss = miss / max(n2, 1)

    ms = np.array([r["ms"] for r in recs])
    metrics = {
        "n": len(recs),
        "classes": CLASSES,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion": cm.tolist(),
        "per_class": prf,
        "cost_per_1000": round(cost_per_1000, 1),
        "severe_miss_rate": round(severe_miss, 4),
        "severe_support": n2,
        "latency_ms": {
            "median": round(float(np.median(ms)), 2),
            "p95": round(float(np.percentile(ms, 95)), 2),
            "max": round(float(ms.max()), 2),
        },
        "source": a.pred,
    }
    dump_json(ROOT / a.json, metrics)

    print(f"样本 {len(recs)}  准确率 {acc:.4f}  宏F1 {macro_f1:.4f}  "
          f"判废漏判率 {severe_miss:.4f}  代价/千张 {cost_per_1000:.1f}")
    for c in CLASSES:
        m = prf[c]
        print(f"  {c:8s} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} n={m['support']}")
    print(f"-> {ROOT / a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
