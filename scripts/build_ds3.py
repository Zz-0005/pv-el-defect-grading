# -*- coding: utf-8 -*-
"""
只构建三档数据集 artifacts/ds3，不动其他目录。

为什么单独写：`prepare` 会重建 ds4/ds2/ds4_rand，而此刻训练正在读 ds4，
重建会把正在训练的数据集删掉。这个脚本只读 samples.csv 并写 ds3。
"""
from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvinspect.config import GRADE3_OF, GRADES3, WORK  # noqa: E402

SRC = ROOT / "data" / "elpv" / "src" / "elpv_dataset" / "data" / "images"
OVER = {"P1_待复核": 2}


def main() -> int:
    rows = list(csv.DictReader((WORK / "samples.csv").open(encoding="utf-8-sig")))
    out = WORK / "ds3"
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        for c in GRADES3:
            (out / split / c).mkdir(parents=True, exist_ok=True)

    for r in rows:
        cls = GRADE3_OF[r["grade"]]
        shutil.copyfile(SRC / Path(r["file"]).name, out / r["split"] / cls / Path(r["file"]).name)

    # 只对 train 的少数档过采样
    for cls, k in OVER.items():
        d = out / "train" / cls
        originals = sorted(p for p in d.glob("*.png") if "_os" not in p.stem)
        for n in range(1, k):
            for p in originals:
                shutil.copyfile(p, d / f"{p.stem}_os{n}.png")

    for split in ("train", "val", "test"):
        c = Counter({k: len(list((out / split / k).glob('*.png'))) for k in GRADES3})
        print(f"  {split:5s} {dict(c)}")
    print(f"ds3 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
