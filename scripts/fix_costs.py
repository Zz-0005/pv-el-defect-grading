# -*- coding: utf-8 -*-
"""
从已存的混淆矩阵重算加权代价，回写 results.json。

为什么要单独做：代价矩阵改过两次（补 P* 与 normal/defect 条目），
但预测结果（混淆矩阵）没变。重跑一次 eval 要 10 分钟推理，纯浪费。
这里只做矩阵乘法级别的重算。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvinspect.config import DEFAULT_COST, MISCLASS_COST, WORK  # noqa: E402
from pvinspect.evaluate import weighted_cost, report  # noqa: E402


def main() -> int:
    p = WORK / "results" / "results.json"
    results = json.loads(p.read_text(encoding="utf-8"))
    print(f"{'实验':20s} {'旧代价':>9s} {'新代价':>9s}")
    for name, r in results.items():
        cm = np.array(r["confusion"], dtype=np.int64)
        old = r["cost_per_1000"]
        new = weighted_cost(cm, r["classes"])
        r["cost_per_1000"] = new
        flag = "" if abs(new - old) < 1e-6 else "  <- 变"
        print(f"{name:20s} {old:9.1f} {new:9.1f}{flag}")

    p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report(results, WORK / "results" / "results.csv", p)
    print(f"\n已回写 {p}，并重新生成 results.csv")

    # 检查是否还有没定义的组合（会静默退化成默认代价）
    missing = []
    for name, r in results.items():
        for a in r["classes"]:
            for b in r["classes"]:
                if a != b and (a, b) not in MISCLASS_COST:
                    missing.append((name, a, b))
    if missing:
        print(f"\n[警告] 以下组合未定义代价，按默认 {DEFAULT_COST} 计：")
        for m in missing[:10]:
            print("   ", m)
    else:
        print("\n所有类间组合都已定义代价")
    return 0


if __name__ == "__main__":
    sys.exit(main())
