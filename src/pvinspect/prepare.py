# -*- coding: utf-8 -*-
"""
数据准备：把 ELPV 的"缺陷概率"转成分级标签，并按**组件分组**切分。

为什么必须分组切分
------------------
ELPV 是 44 块真实组件上切出来的 2624 个电池片。同一块组件上的相邻电池片，
成像条件几乎一致（同一台 EL 相机、同一次通电、同一位置）。若按图片随机切分，
训练集和测试集里会同时出现"同一块组件的孪生样本"，模型只要记住这块组件的
底色就能答对，测试指标虚高。

ELPV 本身**没有提供组件 ID**（labels.csv 只有 path/probability/type）。
所以这里用感知哈希把视觉上高度相似的图像聚成一组，近似还原"同一块组件"。
组数是否合理，用数据集论文给的 44 块组件作为校准点：
自动搜索一个 Hamming 阈值，使聚类数最接近 44。
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    DATA_RAW, GRADE3_OF, GRADES, GRADES3, IMAGES, LABELS, WORK, SplitCfg,
)

SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------ 读标签
def read_labels() -> list[dict]:
    rows = []
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        rel, prob, cell_type = parts[0], float(parts[1]), parts[2]
        rows.append({
            "file": rel,
            "path": DATA_RAW / rel,
            "prob": prob,
            "type": cell_type,
            "grade": grade_of(prob),
        })
    return rows


def grade_of(prob: float) -> str:
    """概率→分级。0.0→G0, 1/3→G1, 2/3→G2, 1.0→G3。"""
    idx = int(round(prob * 3))
    return GRADES[max(0, min(3, idx))]


# ------------------------------------------------------------------ 感知哈希
def average_hash(path: Path, size: int = 8) -> np.uint64:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    bits = (a > a.mean()).ravel()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return np.uint64(v)


def popcount64(x: np.ndarray) -> np.ndarray:
    """uint64 数组的按位 1 计数（无 numpy.bitwise_count 时的可移植写法）。"""
    x = x.copy()
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cluster(hashes: np.ndarray, thr: int) -> np.ndarray:
    """按 Hamming 距离 ≤ thr 做连通分量聚类，返回每个样本的组号。"""
    n = len(hashes)
    dsu = _DSU(n)
    for i in range(n - 1):
        dist = popcount64(hashes[i] ^ hashes[i + 1:])
        for j in np.flatnonzero(dist <= thr):
            dsu.union(i, i + 1 + int(j))
    roots = {}
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        r = dsu.find(i)
        if r not in roots:
            roots[r] = len(roots)
        out[i] = roots[r]
    return out


def _appearance_vectors(paths: list[Path], size: int = 16) -> np.ndarray:
    """把图像缩到 size×size 并做零均值单位方差归一化，作为组件聚类的特征。"""
    feats = np.empty((len(paths), size * size), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
        feats[i] = a.ravel()
    feats -= feats.mean(axis=1, keepdims=True)
    feats /= (feats.std(axis=1, keepdims=True) + 1e-6)
    return feats


def kmeans_groups(feats: np.ndarray, k: int, seed: int = 0, iters: int = 60,
                  restarts: int = 3) -> np.ndarray:
    """
    对外观特征做 k-means，固定 k 组。

    为什么不用"Hamming 阈值 + 连通分量"：阈值稍大就会链式合并
    （A 像 B、B 像 C，则 A/C 也被并成一组），组数崩塌、组内混杂，
    切分后测试集几乎没有稀有级别。k-means 直接指定组数，没有链式问题。
    """
    rng = np.random.default_rng(seed)
    n = len(feats)
    k = min(k, n)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        # k-means++ 初始化
        centers = np.empty((k, feats.shape[1]), dtype=np.float32)
        centers[0] = feats[rng.integers(n)]
        d2 = ((feats - centers[0]) ** 2).sum(1)
        for c in range(1, k):
            probs = d2 / max(d2.sum(), 1e-12)
            centers[c] = feats[rng.choice(n, p=probs)]
            d2 = np.minimum(d2, ((feats - centers[c]) ** 2).sum(1))
        for _ in range(iters):
            dist = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(2)
            lab = dist.argmin(1)
            new = np.array([feats[lab == c].mean(0) if (lab == c).any() else centers[c]
                            for c in range(k)], dtype=np.float32)
            if np.allclose(new, centers, atol=1e-4):
                centers = new
                break
            centers = new
        dist = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        lab = dist.argmin(1)
        inertia = float(dist[np.arange(n), lab].sum())
        if inertia < best_inertia:
            best, best_inertia = lab.astype(np.int32), inertia
    return best


def merge_duplicates(groups: np.ndarray, dup_pairs: list[tuple[int, int]]) -> np.ndarray:
    """把存在重复关系的样本并进同一组，返回重编号后的组号。"""
    n = len(groups)
    dsu = _DSU(n)
    for i, j in dup_pairs:
        dsu.union(i, j)
    # 先把原始组号与 DSU 根都纳入同一套并查集
    root_of_group: dict[int, int] = {}
    for i in range(n):
        g = int(groups[i])
        if g not in root_of_group:
            root_of_group[g] = dsu.find(i)
        else:
            dsu.union(root_of_group[g], i)
    mapping: dict[int, int] = {}
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        r = dsu.find(i)
        if r not in mapping:
            mapping[r] = len(mapping)
        out[i] = mapping[r]
    return out


def duplicate_pairs(feats: np.ndarray, thr: float = 0.995) -> list[tuple[int, int]]:
    """
    基于外观向量的余弦相似度判定重复样本。

    不用感知哈希：64 位 aHash 只保留"高于均值"的位，大量不同电池片会撞哈希
    （实测 Hamming=0 的样本对仍有 670 对，其中 13 对其实来自不同 split）。
    16×16 归一化向量的余弦相似度区分力更强，且与 k-means 分组用的是同一空间，
    两处口径一致才能保证"分组已覆盖重复"。
    """
    n = len(feats)
    norm = np.linalg.norm(feats, axis=1, keepdims=True)
    unit = feats / np.maximum(norm, 1e-8)
    sim = unit @ unit.T
    pairs = []
    for i in range(n - 1):
        for j in np.flatnonzero(sim[i, i + 1:] >= thr):
            pairs.append((i, i + 1 + int(j)))
    return pairs


# ------------------------------------------------------------------ 分组切分
def random_split(rows: list[dict], cfg: SplitCfg) -> dict[str, np.ndarray]:
    """
    对照组：按图片随机切分（忽略组归属）。
    这是文献里最常见、也是最有问题的一种做法 —— 同一块组件的相邻电池片
    会同时落进训练集和测试集。保留它，是为了量化"泄漏"到底能抬高多少指标。
    """
    rng = np.random.default_rng(cfg.seed + 1)
    idx = np.arange(len(rows))
    # 按级别分层后再随机打散，保证与分组切分的类别分布可比
    out: dict[str, list[int]] = {s: [] for s in SPLITS}
    by_grade: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_grade[r["grade"]].append(i)
    for grade, ids in by_grade.items():
        ids = np.array(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(n * cfg.ratio[0])
        n_va = int(n * cfg.ratio[1])
        out["train"].extend(ids[:n_tr].tolist())
        out["val"].extend(ids[n_tr:n_tr + n_va].tolist())
        out["test"].extend(ids[n_tr + n_va:].tolist())
    return {s: np.array(sorted(v), dtype=np.int64) for s, v in out.items()}


def group_split(rows: list[dict], groups: np.ndarray, cfg: SplitCfg) -> dict[str, np.ndarray]:
    """
    按组分配 split：组是原子的，整组只进一个 split。

    贪心按体积装箱会把稀有级别（G1/G2 只占 11%/4%）几乎全塞进训练集，
    导致测试集里根本没有中度和轻微样本，分级指标失去意义。
    所以这里做**按级别分层的随机重启搜索**：多次随机分配组→split，
    用"各级别在三个 split 中的比例与目标比例的加权偏差"打分，取最优。
    """
    rng = np.random.default_rng(cfg.seed)
    gids = np.unique(groups)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[int(g)].append(i)
    n_groups = len(gids)
    total = len(rows)

    # 每个组在各等级上的计数矩阵 (G, K)
    grade_index = {g: k for k, g in enumerate(GRADES)}
    G = np.zeros((n_groups, len(GRADES)), dtype=np.int64)
    gpos = {int(g): j for j, g in enumerate(gids)}
    for i, g in enumerate(groups):
        G[gpos[int(g)], grade_index[rows[i]["grade"]]] += 1

    tot_k = G.sum(axis=0).astype(np.float64)          # 各等级总量
    ratio = np.asarray(cfg.ratio, dtype=np.float64)
    ratio = ratio / ratio.sum()
    target = np.outer(ratio, tot_k)                   # (3, K) 目标矩阵

    def score(mat: np.ndarray) -> float:
        """mat: (3, K) 实际分布。按等级样本量加权，稀有等级同样重要。"""
        dev = np.abs(mat - target) / np.maximum(target, 1.0)
        w = 1.0 / np.maximum(tot_k, 1.0)
        w = w / w.sum()
        return float((dev.mean(axis=0) * w).sum())

    best_mat, best_assign = None, None
    for _ in range(3000):
        assign = rng.integers(0, 3, size=n_groups)
        mat = np.zeros((3, len(GRADES)), dtype=np.float64)
        for s in range(3):
            mat[s] = G[assign == s].sum(axis=0)
        # 空 split 直接判负
        if (mat.sum(axis=1) == 0).any():
            continue
        sc = score(mat)
        if best_mat is None or sc < score(best_mat):
            best_mat, best_assign = mat.copy(), assign.copy()

    out = {}
    for k, s in enumerate(SPLITS):
        idx = [i for i, g in enumerate(groups) if best_assign[gpos[int(g)]] == k]
        out[s] = np.array(sorted(idx), dtype=np.int64)
    return out


# ------------------------------------------------------------------ 落盘
def build_yolo_dirs(rows: list[dict], split_idx: dict[str, np.ndarray], root: Path,
                    label_key: str = "grade", classes: list[str] | None = None,
                    copy: bool = True) -> dict[str, int]:
    """
    生成 YOLO 分类目录结构：
        root/train/<class>/xxx.png
        root/val/<class>/xxx.png
        root/test/<class>/xxx.png
    label_key: "grade" → 四级；"binary" → normal/defect 二分类（基线用）
    """
    classes = classes or GRADES
    if root.exists():
        shutil.rmtree(root)
    stat: dict[str, int] = {}
    for split in SPLITS:
        for c in classes:
            (root / split / c).mkdir(parents=True, exist_ok=True)
    for split, idxs in split_idx.items():
        for i in idxs:
            r = rows[int(i)]
            label = r[label_key]
            dst = root / split / label / Path(r["file"]).name
            if copy:
                shutil.copyfile(r["path"], dst)
            stat[f"{split}/{label}"] = stat.get(f"{split}/{label}", 0) + 1
    return stat


def oversample_train(root: Path, factors: dict[str, int]) -> dict[str, int]:
    """
    只对 train/<class>/ 做复制过采样（文件名加 _osN 后缀）。

    为什么在数据层做而不是用损失权重：
    ultralytics 的分类训练没有暴露 per-class 权重接口，改损失要侵入源码；
    而复制的效果等价（每个 epoch 这些样本被采到的概率按倍数提高），
    且**完全可见**——打开目录就能数出有多少张，答辩时能解释清楚。
    复制件只存在于 train，val/test 不受影响，测试指标仍然干净。
    """
    added: dict[str, int] = {}
    train = root / "train"
    for cls, k in factors.items():
        if k <= 1:
            continue
        d = train / cls
        if not d.exists():
            continue
        originals = sorted(p for p in d.glob("*.png") if "_os" not in p.stem)
        for n in range(1, k):
            for p in originals:
                shutil.copyfile(p, d / f"{p.stem}_os{n}.png")
        added[cls] = len(originals) * (k - 1)
    return added


def run(cfg: SplitCfg | None = None, verbose: bool = True) -> dict:
    cfg = cfg or SplitCfg()
    rows = read_labels()
    if verbose:
        print(f"读取 {len(rows)} 条标注")

    if verbose:
        print("计算 16×16 外观特征…")
    feats = _appearance_vectors([r["path"] for r in rows])

    if verbose:
        print(f"按外观聚类成 {cfg.target_groups} 组（近似还原组件归属）…")
    groups = kmeans_groups(feats, cfg.target_groups, seed=cfg.seed)

    if verbose:
        print(f"合并重复样本（余弦 ≥ {cfg.dup_similarity}）…")
    dups = duplicate_pairs(feats, cfg.dup_similarity)
    groups = merge_duplicates(groups, dups)
    if verbose:
        print(f"  重复对 {len(dups)} 个，合并后 {int(groups.max()) + 1} 组")

    n_groups = int(groups.max() + 1)
    sizes = np.bincount(groups)
    if verbose:
        print(f"  组数 {n_groups}  每组样本数 min={sizes.min()} "
              f"median={int(np.median(sizes))} max={sizes.max()}")

    split_idx = group_split(rows, groups, cfg)
    split_of = {}
    for s, idxs in split_idx.items():
        for i in idxs:
            split_of[int(i)] = s

    # 二分类标签（基线）：G0 → normal，其余 → defect
    for i, r in enumerate(rows):
        r["binary"] = "normal" if r["grade"] == GRADES[0] else "defect"
        r["three"] = GRADE3_OF[r["grade"]]
        r["group"] = int(groups[i])
        r["split"] = split_of[i]

    # 1) 四级分级数据集（分组切分）
    stat4 = build_yolo_dirs(rows, split_idx, WORK / "ds4", "grade", GRADES)
    # 1b) 训练集按级别过采样（只影响 train）
    from .config import TRAIN_OVERSAMPLE
    added = oversample_train(WORK / "ds4", TRAIN_OVERSAMPLE)
    # 2) 二分类数据集（基线对照，同样分组切分 + 过采样）
    stat2 = build_yolo_dirs(rows, split_idx, WORK / "ds2", "binary", ["normal", "defect"])
    oversample_train(WORK / "ds2", {"defect": 2})
    # 3) 四级分级数据集（随机切分，用于证明分组切分的必要性）
    rand_idx = random_split(rows, cfg)
    stat4r = build_yolo_dirs(rows, rand_idx, WORK / "ds4_rand", "grade", GRADES)
    oversample_train(WORK / "ds4_rand", TRAIN_OVERSAMPLE)
    # 4) 操作级三档（分组切分）：轻微+中度 合并为"待复核"
    stat3 = build_yolo_dirs(rows, split_idx, WORK / "ds3", "three", GRADES3)
    oversample_train(WORK / "ds3", {"P1_待复核": 2})

    # 3) 清单
    csv_path = WORK / "samples.csv"
    WORK.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "prob", "type", "grade", "binary", "group", "split"])
        for r in rows:
            w.writerow([r["file"], f"{r['prob']:.4f}", r["type"], r["grade"],
                        r["binary"], r["group"], r["split"]])

    # 4) 泄漏检查
    leak = check_leakage(rows, feats, dup_similarity=cfg.dup_similarity)

    summary = {
        "groups": n_groups,
        "group_sizes": {"min": int(sizes.min()), "median": int(np.median(sizes)),
                        "max": int(sizes.max())},
        "samples": len(rows),
        "split_sizes": {s: int(len(v)) for s, v in split_idx.items()},
        "grade_dist": {s: dict(Counter(rows[int(i)]["grade"] for i in idxs))
                       for s, idxs in split_idx.items()},
        "binary_dist": {s: dict(Counter(rows[int(i)]["binary"] for i in idxs))
                        for s, idxs in split_idx.items()},
        "leakage": leak,
        "ds4": stat4,
        "ds2": stat2,
        "ds3": stat3,
        "ds4_rand": stat4r,
        "oversample_added": added,
        "random_split_sizes": {s: int(len(v)) for s, v in rand_idx.items()},
    }
    (WORK / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print("\n切分结果：")
        for s in SPLITS:
            d = summary["grade_dist"][s]
            print(f"  {s:5s} {summary['split_sizes'][s]:5d} 张  "
                  + "  ".join(f"{k}:{d.get(k, 0)}" for k in GRADES))
        print(f"\n泄漏检查：{'通过' if leak['ok'] else '未通过'}")
        print(f"  同组跨 split：{leak['cross_groups']} 处")
        print(f"  重复样本跨 split（余弦 ≥ {leak['dup_similarity']}）："
              f"{leak['cross_dups']} 对 / 共 {leak['dup_pairs']} 对重复")
        print(f"样本清单：{csv_path}")
    return summary


def check_leakage(rows: list[dict], feats: np.ndarray, dup_similarity: float = 0.995) -> dict:
    """
    两道泄漏检查：
      1) 同一组是否跨 split（组是原子分配的，应恒为 0）；
      2) 重复样本对是否跨 split —— 即使分组不完美，这一条兜住真正的重复图。
    """
    g2s: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        g2s[r["group"]].add(r["split"])
    cross_groups = sum(1 for s in g2s.values() if len(s) > 1)

    split_arr = np.array([r["split"] for r in rows])
    pairs = duplicate_pairs(feats, dup_similarity)
    cross_dup = sum(1 for i, j in pairs if split_arr[i] != split_arr[j])

    return {
        "ok": cross_groups == 0 and cross_dup == 0,
        "groups": len(g2s),
        "cross_groups": cross_groups,
        "cross_dups": cross_dup,
        "dup_pairs": len(pairs),
        "dup_similarity": dup_similarity,
    }
