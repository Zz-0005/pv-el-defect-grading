# -*- coding: utf-8 -*-
"""
挑出失败案例并落盘，供网页「十、失败案例」使用。

    python tools/build_failures.py --top 9

排序依据是**业务代价**，不是置信度：
判废被判成合格（记 20 分）远比合格被判成待复核（记 1 分）严重，
按代价排才能看到真正危险的错。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, ROOT, cost_of, dump_json, load_json  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="out/pred_test.json")
    ap.add_argument("--json", default="out/failures.json")
    ap.add_argument("--top", type=int, default=9)
    a = ap.parse_args(argv)

    recs = load_json(ROOT / a.pred)
    recs = [r for r in recs if r["true"] in CLASSES]

    # 各错误类型的计数
    kinds: dict[str, int] = {}
    for r in recs:
        if r["true"] != r["pred"]:
            k = f"{r['true']}→{r['pred']}"
            kinds[k] = kinds.get(k, 0) + 1

    errs = [r for r in recs if r["true"] != r["pred"]]
    errs.sort(key=lambda r: -cost_of(r["true"], r["pred"]))

    dst = ROOT / "fails"
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    picked = []
    for i, r in enumerate(errs[: a.top], 1):
        src = ROOT / "test" / "images" / r["image"]
        if not src.exists():
            continue
        name = f"fail_{i:02d}.png"
        shutil.copyfile(src, dst / "images" / name)
        (dst / "labels" / f"fail_{i:02d}.txt").write_text(f"{r['true']}\n", encoding="utf-8")
        picked.append({
            "image": name,
            "original": r["image"],
            "true": r["true"], "pred": r["pred"],
            "conf": round(r["conf"], 4),
            "cost": cost_of(r["true"], r["pred"]),
        })

    dump_json(ROOT / a.json, {
        "total_errors": len(errs),
        "error_rate": round(len(errs) / max(len(recs), 1), 4),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "picked": picked,
    })
    print(f"错误 {len(errs)} 张（{len(errs)/len(recs)*100:.1f}%），挑选 {len(picked)} 张")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"   {k}: {v}")
    print(f"-> {ROOT / a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
