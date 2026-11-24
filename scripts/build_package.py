# -*- coding: utf-8 -*-
"""
组装交付包：把训练成果整理成「双击即用」的形式。

产物结构（对齐参考交付包）：
    光伏电池片EL缺陷分级_交付包/
    ├── 使用说明.md / 启动.bat / *.pyw / 创建快捷方式.ps1 / app.ico
    └── pv-demo/
        ├── README.md run.py requirements.txt LICENSE
        ├── models/   onnx 权重（不需要 torch）
        ├── demo/ demo2/   精挑演示图 + 标签
        ├── train/ valid/ test/   全量数据 + 标签
        ├── out/      评估结果 json
        ├── tools/    推理 / 评估 / 渲染 / 建网页
        ├── docs/     PPT
        └── web/      离线技术报告
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PKG = ROOT.parent / "光伏电池片EL缺陷分级_交付包"
DEMO = PKG / "pv-demo"
ART = ROOT / "artifacts"

GRADE_CN = {"P0_合格": "合格", "P1_待复核": "待复核", "P2_判废": "判废"}
GRADE_ACTION = {"P0_合格": "正常流转", "P1_待复核": "返修 / 降级", "P2_判废": "判废"}


def fresh(p: Path) -> Path:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_split(split: str, src: Path, dst_root: Path) -> int:
    """test/val/train 结构：images/*.png + labels/*.txt（一行：级别 概率）。"""
    img_dir = dst_root / "images"
    lbl_dir = dst_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cls_dir in sorted((src / split).iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        for p in sorted(cls_dir.glob("*.png")):
            if "_os" in p.stem:      # 过采样复制件不进交付数据
                continue
            shutil.copyfile(p, img_dir / p.name)
            (lbl_dir / f"{p.stem}.txt").write_text(f"{cls}\n", encoding="utf-8")
            n += 1
    return n


def main() -> int:
    print(f"交付包: {PKG}")
    fresh(PKG)
    fresh(DEMO)

    # ---------- 1) 数据集
    print("[1/6] 复制数据集 ...")
    counts = {}
    for split, name in (("train", "train"), ("val", "valid"), ("test", "test")):
        counts[name] = copy_split(split, ART / "ds3", DEMO / name)
        print(f"   {name}: {counts[name]} 张")

    # demo 集：每档挑几张高置信度正确的（已有 demo 样本，复用其来源）
    print("[2/6] 复制演示集 ...")
    rows = list(csv.DictReader((ROOT / "demo" / "out" / "samples.csv").open(encoding="utf-8-sig")))
    demos = ROOT / "demo" / "样本"
    (DEMO / "demo" / "images").mkdir(parents=True, exist_ok=True)
    (DEMO / "demo" / "labels").mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows, 1):
        src = demos / r["sample"]
        if not src.exists():
            continue
        dst = DEMO / "demo" / "images" / f"demo_{i:02d}.png"
        shutil.copyfile(src, dst)
        true_cn = r["true"]
        cls = [k for k, v in GRADE_CN.items() if v == true_cn]
        lbl = cls[0] if cls else "P1_待复核"
        (DEMO / "demo" / "labels" / f"demo_{i:02d}.txt").write_text(f"{lbl}\n", encoding="utf-8")
    print(f"   demo: {len(list((DEMO / 'demo' / 'images').glob('*.png')))} 张")

    # demo2：从 test 里每档再抽多一些，覆盖更多组件
    print("[3/6] 构建扩展演示集 demo2 ...")
    (DEMO / "demo2" / "images").mkdir(parents=True, exist_ok=True)
    (DEMO / "demo2" / "labels").mkdir(parents=True, exist_ok=True)
    n2 = 0
    for cls, k in (("P0_合格", 20), ("P1_待复核", 15), ("P2_判废", 25)):
        for p in sorted((ART / "ds3" / "test" / cls).glob("*.png"))[k * 0:k * 2:k]:
            pass
        files = sorted((ART / "ds3" / "test" / cls).glob("*.png"))
        step = max(1, len(files) // k)
        for p in files[::step][:k]:
            n2 += 1
            shutil.copyfile(p, DEMO / "demo2" / "images" / f"x_{n2:03d}.png")
            (DEMO / "demo2" / "labels" / f"x_{n2:03d}.txt").write_text(f"{cls}\n", encoding="utf-8")
    print(f"   demo2: {n2} 张")

    # ---------- 2) 模型
    print("[4/6] 复制 ONNX 权重 ...")
    (DEMO / "models").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ART / "models" / "pv_el_yolov8s_cls.onnx",
                    DEMO / "models" / "pv_el_yolov8s_cls.onnx")
    (DEMO / "models" / "labels.txt").write_text(
        "\n".join(f"{k}\t{v}\t{GRADE_ACTION[k]}" for k, v in GRADE_CN.items()) + "\n",
        encoding="utf-8")

    # ---------- 3) 目录占位
    for d in ("out", "tools", "docs", "web"):
        (DEMO / d).mkdir(parents=True, exist_ok=True)

    meta = {
        "task": "光伏电池片 EL 图像缺陷分级（3 档）",
        "model": "YOLOv8s-cls 导出 ONNX，输入 1x3x224x224，输出 3 类 softmax 概率",
        "classes": list(GRADE_CN.keys()),
        "class_cn": GRADE_CN,
        "action": GRADE_ACTION,
        "splits": counts,
        "demo": len(list((DEMO / "demo" / "images").glob("*.png"))),
        "demo2": n2,
    }
    (DEMO / "models" / "model_card.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[5/6] 写出 model_card.json")

    size = sum(p.stat().st_size for p in PKG.rglob("*") if p.is_file())
    print(f"[6/6] 完成，当前 {size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
