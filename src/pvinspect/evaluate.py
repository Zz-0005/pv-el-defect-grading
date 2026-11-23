# -*- coding: utf-8 -*-
"""
评测：混淆矩阵、分级指标、代价敏感分析、错例图。

关键口径（写进报告，避免"指标好看但没用"）：
  · 全部在**冻结的测试集**上算，测试集从未参与训练或调参；
  · 分级任务不只是看 accuracy —— 四级里 G2/G3 才是要拦截的，
    所以单独报"严重缺陷漏判率"和"加权代价"；
  · 代价矩阵只用于**评估与决策**，不参与训练，因此不存在"用测试集调参"的问题。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_COST, GRADE_ACTION, GRADE_CN, GRADES, MISCLASS_COST, WORK, EvalCfg,
)


def _prepare_env() -> None:
    from . import compat
    compat.configure_dirs()
    compat.patch_ultralytics()


# ------------------------------------------------------------------ 推理
def predict_dir(weights: Path, root: Path, imgsz: int = 224, batch: int = 64,
                device: str | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    对整个目录逐图预测，返回 (概率矩阵, 真值索引, 类别名)。
    概率矩阵行对应 root/<class>/<img>，按 class 目录顺序排。
    """
    _prepare_env()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    files: list[Path] = []
    truth: list[int] = []
    for ci, cls in enumerate(names):
        for p in sorted((root / cls).glob("*.png")):
            files.append(p)
            truth.append(ci)

    probs = np.zeros((len(files), len(names)), dtype=np.float32)
    for i, p in enumerate(files):
        r = model.predict(str(p), imgsz=imgsz, device=device, verbose=False)[0]
        probs[i] = r.probs.data.cpu().numpy()
    return probs, np.array(truth, dtype=np.int64), names


# ------------------------------------------------------------------ 指标
def confusion(pred: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(truth, pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> dict[int, dict[str, float]]:
    k = cm.shape[0]
    out = {}
    for i in range(k):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[i] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                  "support": int(cm[i].sum())}
    return out


def macro_f1(cm: np.ndarray) -> float:
    prf = per_class_prf(cm)
    return float(np.mean([v["f1"] for v in prf.values()]))


def weighted_cost(cm: np.ndarray, names: list[str]) -> float:
    """按代价矩阵计算每 1000 张的加权错误代价。"""
    k = cm.shape[0]
    total = 0.0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            total += int(cm[i, j]) * MISCLASS_COST.get((names[i], names[j]), DEFAULT_COST)
    n = max(int(cm.sum()), 1)
    return float(total / n * 1000)


def evaluate(weights: Path, split_dir: Path, names_ref: list[str],
             imgsz: int = 224, device: str | None = None) -> dict:
    probs, truth, names = predict_dir(weights, split_dir, imgsz=imgsz, device=device)
    pred = probs.argmax(axis=1)
    k = len(names)
    cm = confusion(pred, truth, k)
    prf = per_class_prf(cm)
    acc = float((pred == truth).mean())
    res = {
        "n": int(len(truth)),
        "classes": names,
        "accuracy": acc,
        "macro_f1": macro_f1(cm),
        "confusion": cm.tolist(),
        "per_class": {names[i]: prf[i] for i in range(k)},
        "cost_per_1000": weighted_cost(cm, names),
    }
    if "G3_严重" in names:
        gi = names.index("G3_严重")
        low = [names.index(c) for c in ("G0_良好", "G1_轻微") if c in names]
        n3 = int(cm[gi].sum())
        res["severe_miss_rate"] = float(cm[gi, low].sum() / max(n3, 1))
        res["severe_support"] = n3
    return res


def class_priors(train_dir: Path, names: list[str]) -> np.ndarray:
    """训练集各类样本占比（含过采样后的复制件，反映模型实际见到的先验）。"""
    counts = np.array([len(list((train_dir / c).glob("*.png"))) for c in names], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    return counts / counts.sum()


def adjust_probs(probs: np.ndarray, priors: np.ndarray, tau: float) -> np.ndarray:
    """
    先验调整（logit adjustment 的概率形式）： p'(c|x) ∝ p(c|x) / π_c^τ

    动机：训练集长尾时，模型输出的 argmax 会被多数类先验主导，
    表现为"只输出两极、中间级别永不命中"（本项目第一版实测 G1/G2 召回均为 0）。
    把训练先验除掉，等于把决策边界往少数类方向推。

    τ=0 表示不调整，退化为原始 argmax；τ=1 是完整的先验校正。
    τ 只在**验证集**上选，测试集只用来报告最终指标。
    """
    if tau <= 0:
        return probs
    s = probs / np.power(priors[None, :], tau)
    return s / np.maximum(s.sum(axis=1, keepdims=True), 1e-12)


def metrics_from_probs(probs: np.ndarray, truth: np.ndarray, names: list[str]) -> dict:
    pred = probs.argmax(axis=1)
    k = len(names)
    cm = confusion(pred, truth, k)
    prf = per_class_prf(cm)
    res = {
        "n": int(len(truth)),
        "classes": names,
        "accuracy": float((pred == truth).mean()),
        "macro_f1": macro_f1(cm),
        "confusion": cm.tolist(),
        "per_class": {names[i]: prf[i] for i in range(k)},
        "cost_per_1000": weighted_cost(cm, names),
    }
    if "G3_严重" in names:
        gi = names.index("G3_严重")
        low = [names.index(c) for c in ("G0_良好", "G1_轻微") if c in names]
        n3 = int(cm[gi].sum())
        res["severe_miss_rate"] = float(cm[gi, low].sum() / max(n3, 1))
        res["severe_support"] = n3
    return res


def pick_tau(val_probs: np.ndarray, val_truth: np.ndarray, priors: np.ndarray,
             names: list[str], grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
             objective: str = "macro_f1") -> tuple[float, dict[float, float]]:
    """在验证集上扫描 τ，返回 (最优τ, 每个τ的指标)。"""
    scores: dict[float, float] = {}
    for t in grid:
        m = metrics_from_probs(adjust_probs(val_probs, priors, t), val_truth, names)
        scores[t] = m["macro_f1"] if objective == "macro_f1" else -m["cost_per_1000"]
    best = max(scores, key=lambda k: scores[k])
    return best, scores


def evaluate_adjusted(weights: Path, dirs: dict[str, Path], imgsz: int = 224,
                      device: str | None = None) -> dict:
    """
    完整流程：验证集选 τ → 测试集报指标。
    返回 {"raw": 测试集 argmax 结果, "adjusted": 先验调整后结果, "tau": ..., "tau_scores": ...}
    """
    probs_val, truth_val, names = predict_dir(weights, dirs["val"], imgsz=imgsz, device=device)
    probs_test, truth_test, _ = predict_dir(weights, dirs["test"], imgsz=imgsz, device=device)
    priors = class_priors(dirs["train"], names)
    tau, scores = pick_tau(probs_val, truth_val, priors, names)
    raw = metrics_from_probs(probs_test, truth_test, names)
    adj = metrics_from_probs(adjust_probs(probs_test, priors, tau), truth_test, names)
    return {"raw": raw, "adjusted": adj, "tau": tau,
            "tau_scores": {float(k): float(v) for k, v in scores.items()},
            "val": metrics_from_probs(adjust_probs(probs_val, priors, tau), truth_val, names)}


def report(results: dict[str, dict], out_csv: Path, out_json: Path) -> None:
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["实验", "说明", "样本数", "准确率", "宏平均F1",
                    "严重漏判率", "加权代价/千张"])
        for name, r in results.items():
            w.writerow([
                name, r.get("desc", ""), r["n"],
                f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}",
                f"{r.get('severe_miss_rate', float('nan')):.4f}",
                f"{r['cost_per_1000']:.2f}",
            ])


# ------------------------------------------------------------------ 出图
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for cand in ("Microsoft YaHei", "SimHei", "SimSun"):
        if any(cand in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [cand]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_confusion(cm: np.ndarray, names: list[str], path: Path, title: str) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([GRADE_CN.get(n, n) for n in names], fontsize=9)
    ax.set_yticklabels([GRADE_CN.get(n, n) for n in names], fontsize=9)
    ax.set_xlabel("预测", fontsize=10)
    ax.set_ylabel("真实", fontsize=10)
    ax.set_title(title, fontsize=11)
    for i in range(len(names)):
        for j in range(len(names)):
            v = int(cm[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if v > cm.max() * 0.55 else "#1a1a1a")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(results: dict[str, dict], path: Path) -> None:
    plt = _mpl()
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    colors = ["#2E7D8F" if "grouped" in n else "#C9762A" for n in names]
    bars = ax.bar(range(len(names)), accs, color=colors, width=0.55)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("测试集准确率")
    ax.set_ylim(0, 1.05)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.3f}",
                ha="center", fontsize=9)
    ax.set_title("切分方式与任务粒度对指标的影响")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_error_gallery(weights: Path, test_dir: Path, names: list[str], path: Path,
                       imgsz: int = 224, device: str | None = None,
                       per_kind: int = 4) -> dict:
    """
    把错分样本按"错误类型"拼成一张图，尤其是 G3→G0 这种致命错误。

    报告里放一张真实错例图，比放十个指标都有说服力：
    评委能直接看到模型在什么情况下会出错。
    """
    _prepare_env()
    plt = _mpl()
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(str(weights))
    collected: dict[str, list[tuple[Path, float]]] = {}
    for ti, cls in enumerate(names):
        for p in sorted((test_dir / cls).glob("*.png")):
            r = model.predict(str(p), imgsz=imgsz, device=device, verbose=False)[0]
            probs = r.probs.data.cpu().numpy()
            pi = int(probs.argmax())
            if pi == ti:
                continue
            kind = f"{names[ti]}→{names[pi]}"
            bucket = collected.setdefault(kind, [])
            if len(bucket) < per_kind:
                bucket.append((p, float(probs[pi])))

    def severity(kind: str) -> int:
        t = kind.split("→")[0]
        return -int(t[1]) if len(t) > 1 and t[1].isdigit() else 0

    kinds = sorted(collected, key=severity)
    if not kinds:
        return {"kinds": 0}

    cols, rows = per_kind, len(kinds)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.0 * rows), dpi=140,
                             squeeze=False)
    for r_i, kind in enumerate(kinds):
        items = collected[kind]
        for c_i in range(cols):
            ax = axes[r_i][c_i]
            ax.axis("off")
            if c_i < len(items):
                p, conf = items[c_i]
                with Image.open(p) as im:
                    ax.imshow(np.asarray(im.convert("L")), cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{kind}  p={conf:.2f}", fontsize=7.5)
            if c_i == 0:
                ax.text(-0.08, 0.5, kind, transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=8)
    fig.suptitle("错分样本（标题为 真实→预测 与预测置信度）", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"kinds": len(kinds), "counts": {k: len(v) for k, v in collected.items()}}
