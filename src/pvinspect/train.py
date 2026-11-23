# -*- coding: utf-8 -*-
"""
训练：YOLOv8n-cls 的三个对照实验。

E1 泄漏对照   同一模型，ds4（分组切分） vs ds4_rand（随机切分）
E2 任务粒度   二分类 ds2（合格/不合格） vs 四级分级 ds4
E3 输入分辨率 224 vs 320（可选，看缺陷细节是否需要更高分辨率）

为什么用 -cls 而不是 -det：
ELPV 的标注是"这一个电池片整体的缺陷概率"，不是缺陷实例的包围盒。
硬套检测模型就要自己编框的真值，指标就成了自证的。分级本身就是分类任务，
用 YOLOv8 的分类头是任务与模型匹配的做法。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import EvalCfg, TrainCfg, WORK

EXPERIMENTS = {
    "E1_grouped": {"ds": "ds4", "classes": 4, "desc": "四级分级 · 分组切分（本方案）"},
    "E1_random": {"ds": "ds4_rand", "classes": 4, "desc": "四级分级 · 随机切分（泄漏对照）"},
    "E2_binary": {"ds": "ds2", "classes": 2, "desc": "二分类 · 合格/不合格（基线）"},
    "E3_three": {"ds": "ds3", "classes": 3, "desc": "三档操作级 · 分组切分（合并轻微/中度）"},
}


def train_one(ds: str, name: str, tcfg: TrainCfg, imgsz: int | None = None) -> Path:
    from . import compat
    compat.configure_dirs()
    compat.patch_ultralytics(verbose=True)
    compat.patch_torch_threads(4)
    from ultralytics import YOLO

    data_dir = WORK / ds
    if not data_dir.exists():
        raise FileNotFoundError(f"数据集不存在：{data_dir}，先运行 prepare")

    model = YOLO(tcfg.model)
    model.train(
        data=str(data_dir),
        epochs=tcfg.epochs,
        imgsz=imgsz or tcfg.imgsz,
        batch=tcfg.batch,
        patience=tcfg.patience,
        seed=tcfg.seed,
        device=tcfg.device or None,
        workers=tcfg.workers,
        cache=tcfg.cache,
        cos_lr=tcfg.cos_lr,
        label_smoothing=tcfg.label_smoothing,
        project=str(tcfg.runs),
        name=name,
        exist_ok=True,
        pretrained=True,
        verbose=False,
        plots=True,
        val=True,
    )
    best = tcfg.runs / name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"训练未产出权重：{best}")
    return best


def run(tcfg: TrainCfg | None = None, only: list[str] | None = None) -> dict:
    tcfg = tcfg or TrainCfg()
    tcfg.runs.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, spec in EXPERIMENTS.items():
        if only and name not in only:
            continue
        print(f"\n=== {name}：{spec['desc']} ===", flush=True)
        best = train_one(spec["ds"], name, tcfg)
        # 把权重收敛到一个稳定位置，避免依赖 run 目录结构
        dst = WORK / "models" / f"{name}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, dst)
        summary[name] = {"weights": str(dst), "desc": spec["desc"], "ds": spec["ds"]}
        print(f"  -> {dst}", flush=True)

    out = WORK / "models" / "train_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
    prev.update(summary)
    out.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
