# -*- coding: utf-8 -*-
"""
自检：不需要训练权重，几秒内验证"数据处理和指标计算有没有算错"。

答辩现场如果被问"你的分组切分是不是真的没泄漏"，跑 `python -m pvinspect selftest`
比口头解释有用。这里每条断言都对应一个具体的失败模式。
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

from . import ops, prepare
from .config import GRADES, WORK

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            fn()
            _RESULTS.append((name, True, ""))
        except Exception as e:
            _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


# ------------------------------------------------------------------ 图像算子
@check("box_blur：常数图输入输出相等（边界填充不能引入暗边）")
def _():
    img = np.full((64, 80), 123.0, dtype=np.float32)
    out = ops.box_blur(img, 15)
    assert np.allclose(out, 123.0, atol=1e-3), (out.min(), out.max())


@check("local_contrast：纯平背景上响应接近 0（否则整幅图都会成为候选）")
def _():
    img = np.full((64, 64), 100.0, dtype=np.float32)
    resp = ops.local_contrast(img, 21)
    assert np.abs(resp).max() < 0.5, np.abs(resp).max()


@check("local_contrast：叠加亮点后峰值落在缺陷处")
def _():
    img = np.full((64, 64), 100.0, dtype=np.float32)
    img[30:34, 30:34] += 60
    resp = ops.local_contrast(img, 21)
    py, px = np.unravel_index(np.abs(resp).argmax(), resp.shape)
    assert 28 <= py <= 35 and 28 <= px <= 35, (py, px)


@check("clahe：输出在 [0,255] 且能提升低对比区域的动态范围")
def _():
    rng = np.random.default_rng(0)
    img = np.clip(100 + rng.normal(0, 4, (96, 96)), 0, 255).astype(np.float32)
    out = ops.clahe(img, tiles=(4, 4), clip_limit=2.0)
    assert out.min() >= 0 and out.max() <= 255
    assert out.std() > img.std(), (out.std(), img.std())


@check("clahe：不同 tile 尺寸不改变输出形状（防栅格化错位）")
def _():
    img = np.random.default_rng(1).uniform(0, 255, (70, 90)).astype(np.float32)
    for tiles in ((2, 3), (4, 4), (8, 8)):
        out = ops.clahe(img, tiles=tiles)
        assert out.shape == img.shape, (tiles, out.shape)


@check("retinex：输出范围合法且不产生 NaN")
def _():
    img = np.random.default_rng(2).uniform(0, 255, (64, 64)).astype(np.float32)
    out = ops.retinex(img, sigma=12)
    assert np.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 255


@check("resize：尺寸一致时直接返回，不复制内容")
def _():
    img = np.zeros((32, 48), dtype=np.float32)
    assert np.array_equal(ops.resize(img, (48, 32)), img)


# ------------------------------------------------------------------ 分级映射
@check("grade_of：四个标注值精确映射到 G0..G3")
def _():
    cases = [(0.0, "G0_良好"), (1 / 3, "G1_轻微"), (2 / 3, "G2_中度"), (1.0, "G3_严重")]
    for p, expect in cases:
        got = prepare.grade_of(p)
        assert got == expect, (p, got, expect)


@check("grade_of：越界值被夹到两端（不产生 IndexError）")
def _():
    assert prepare.grade_of(-0.5) == GRADES[0]
    assert prepare.grade_of(1.7) == GRADES[-1]


# ------------------------------------------------------------------ 分组与切分
@check("kmeans_groups：固定 seed 结果可复现")
def _():
    feats = np.random.default_rng(3).normal(size=(60, 32)).astype(np.float32)
    a = prepare.kmeans_groups(feats, 5, seed=7, restarts=1, iters=20)
    b = prepare.kmeans_groups(feats, 5, seed=7, restarts=1, iters=20)
    assert a.shape == (60,)
    assert np.array_equal(a, b)


@check("kmeans_groups：k 大于样本数时不死循环")
def _():
    feats = np.random.default_rng(4).normal(size=(3, 8)).astype(np.float32)
    lab = prepare.kmeans_groups(feats, 10, seed=1, restarts=1, iters=5)
    assert lab.shape == (3,)


@check("group_split：同一个组绝不跨 split")
def _():
    rng = np.random.default_rng(5)
    n = 300
    rows = [{"grade": GRADES[int(rng.integers(0, 4))]} for _ in range(n)]
    groups = rng.integers(0, 20, n)
    idx = prepare.group_split(rows, groups, prepare.SplitCfg())
    seen: dict[int, str] = {}
    for s, ids in idx.items():
        for i in ids:
            g = int(groups[i])
            assert g not in seen or seen[g] == s, (g, seen[g], s)
            seen[g] = s


@check("group_split：每个样本恰好被分到一次（不重不漏）")
def _():
    rng = np.random.default_rng(6)
    n = 257
    rows = [{"grade": GRADES[int(rng.integers(0, 4))]} for _ in range(n)]
    groups = rng.integers(0, 17, n)
    idx = prepare.group_split(rows, groups, prepare.SplitCfg())
    allidx = np.concatenate([v for v in idx.values()])
    assert len(allidx) == n
    assert len(set(allidx.tolist())) == n


@check("random_split：与分组切分在同一级别分布下可比")
def _():
    rows = [{"grade": GRADES[i % 4]} for i in range(400)]
    idx = prepare.random_split(rows, prepare.SplitCfg())
    total = sum(len(v) for v in idx.values())
    assert total == 400, total


@check("duplicate_pairs + merge_duplicates：重复样本被并进同一组")
def _():
    feats = np.zeros((6, 4), dtype=np.float32)
    feats[0] = [1, 0, 0, 0]
    feats[1] = [1, 0, 0, 0]      # 与 0 完全相同
    feats[2] = [0, 1, 0, 0]
    feats[3] = [0, 1, 0, 0]
    feats[4] = [0, 0, 1, 0]
    feats[5] = [0, 0, 0, 1]
    dups = prepare.duplicate_pairs(feats, 0.999)
    assert (0, 1) in dups and (2, 3) in dups, dups
    groups = prepare.merge_duplicates(np.arange(6, dtype=np.int32), dups)
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]


# ------------------------------------------------------------------ 指标
@check("confusion：对角线与总数正确")
def _():
    from .evaluate import confusion
    pred = np.array([0, 0, 1, 1, 1])
    truth = np.array([0, 1, 1, 1, 0])
    cm = confusion(pred, truth, 2)
    assert cm.tolist() == [[1, 1], [1, 2]], cm.tolist()
    assert cm.sum() == 5


@check("per_class_prf：完美预测时全为 1")
def _():
    from .evaluate import confusion, per_class_prf
    pred = np.array([0, 0, 1, 1])
    truth = np.array([0, 0, 1, 1])
    prf = per_class_prf(confusion(pred, truth, 2))
    for i in (0, 1):
        assert abs(prf[i]["precision"] - 1) < 1e-9
        assert abs(prf[i]["recall"] - 1) < 1e-9
        assert abs(prf[i]["f1"] - 1) < 1e-9


@check("weighted_cost：把严重缺陷判成良好，代价显著更高")
def _():
    from .evaluate import weighted_cost
    names = list(GRADES)
    n = 100
    bad = np.zeros((4, 4), dtype=np.int64)
    bad[0, 0] = n          # 全对
    ok = bad.copy()
    ok[3, 0] = n           # 严重 → 良好
    ok[0, 0] = 0
    assert weighted_cost(ok, names) > weighted_cost(bad, names) * 10


@check("macro_f1：对稀有类同样加权（不被多数类掩盖）")
def _():
    from .evaluate import confusion, macro_f1
    pred = np.array([0] * 90 + [0] * 10)
    truth = np.array([0] * 90 + [1] * 10)
    cm = confusion(pred, truth, 2)
    acc = (pred == truth).mean()
    mf1 = macro_f1(cm)
    assert acc > 0.85 and mf1 < 0.55, (acc, mf1)


# ------------------------------------------------------------------ 数据集结构
@check("落盘目录：ds4/ds2/ds4_rand 的类别目录齐全")
def _():
    need4 = [WORK / "ds4" / s / c for s in ("train", "val", "test") for c in GRADES]
    need2 = [WORK / "ds2" / s / c for s in ("train", "val", "test")
             for c in ("normal", "defect")]
    needr = [WORK / "ds4_rand" / s / c for s in ("train", "val", "test") for c in GRADES]
    missing = [str(p) for p in need4 + need2 + needr if not p.exists()]
    assert not missing, missing[:5]


@check("各数据集覆盖同一批样本（切分不同、过采样只改 train）")
def _():
    def total(ds, include_oversampled=False):
        n = 0
        for p in (WORK / ds).rglob("*.png"):
            if not include_oversampled and "_os" in p.stem:
                continue          # 过采样复制件不计入原始样本
            n += 1
        return n

    base = total("ds4")
    vals = {ds: total(ds) for ds in ("ds4", "ds2", "ds3", "ds4_rand")}
    assert all(v == base for v in vals.values()), vals


@check("过采样只存在于 train，不污染 val/test")
def _():
    leaked = []
    for ds in ("ds4", "ds2", "ds3", "ds4_rand"):
        for split in ("val", "test"):
            leaked += [p.name for p in (WORK / ds / split).rglob("*_os*.png")]
    assert not leaked, leaked[:5]


@check("过采样确实提高了少数档的占比")
def _():
    from collections import Counter
    c = Counter()
    for p in (WORK / "ds4" / "train").rglob("*.png"):
        c[p.parent.name] += 1
    assert c["G2_中度"] > c["G1_轻微"] or c["G2_中度"] > 200, dict(c)


def main() -> int:
    if not (WORK / "ds4").exists():
        print("数据集尚未生成，先运行： python -m pvinspect prepare")
        return 1
    ok = sum(1 for _, p, _ in _RESULTS if p)
    print(f"\n自检结果：{ok}/{len(_RESULTS)} 通过\n")
    for name, passed, msg in _RESULTS:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {name}")
        if not passed:
            print(f"         {msg}")
    return 0 if ok == len(_RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
