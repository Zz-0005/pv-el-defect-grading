# -*- coding: utf-8 -*-
"""摸清 ELPV 数据：概率分布、单/多晶比例、图像尺寸、重复与近似重复。"""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "elpv" / "src" / "elpv_dataset" / "data"

rows = []
for line in (DATA / "labels.csv").read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) >= 3:
        rows.append((parts[0], float(parts[1]), parts[2]))
print(f"样本数: {len(rows)}")

probs = np.array([r[1] for r in rows])
types = Counter(r[2] for r in rows)
print("晶型:", dict(types))
print(f"概率: min={probs.min():.2f} max={probs.max():.2f} mean={probs.mean():.3f}")

print("\n概率分布（0.1 一档）:")
hist, edges = np.histogram(probs, bins=np.arange(0, 1.101, 0.1))
for h, e in zip(hist, edges):
    print(f"  [{e:.1f}-{e+0.1:.1f})  {h:5d}  {'#' * int(h / 12)}")

for thr in (0.2, 0.35, 0.5, 0.7, 0.8, 0.9):
    print(f"  p > {thr:.2f}: {(probs > thr).sum():5d} ({(probs > thr).mean()*100:5.1f}%)")

# 4 级分档
def grade(p):
    if p <= 0.2: return 0
    if p <= 0.5: return 1
    if p <= 0.8: return 2
    return 3

g = Counter(grade(p) for p in probs)
print("\n四级分档（0:≤0.2, 1:≤0.5, 2:≤0.8, 3:>0.8）:")
for k in sorted(g):
    print(f"  G{k}: {g[k]:5d} ({g[k]/len(probs)*100:5.1f}%)")

sizes = Counter()
for r in rows[:200]:
    p = DATA / r[0]
    with Image.open(p) as im:
        sizes[(im.size, im.mode)] += 1
print("\n图像尺寸/模式（抽样 200）:", dict(sizes))

# 近似重复：用 8x8 缩略图的平均哈希粗查
def ahash(p):
    with Image.open(p) as im:
        a = np.asarray(im.resize((8, 8), Image.BILINEAR), dtype=np.float32)
    return (a > a.mean()).ravel()

hashes = np.stack([ahash(DATA / r[0]) for r in rows])
dup = 0
for i in range(len(hashes) - 1):
    same = (hashes[i + 1:] == hashes[i]).all(axis=1).sum()
    dup += same
print(f"\n64 位平均哈希完全相同的近似重复对: {dup}（说明相邻编号可能来自同一块组件）")
print("结论：不能随机切分，需按相似性分组切分。")
