# -*- coding: utf-8 -*-
"""配置：集中所有可调项，避免散落在脚本里的魔法数字。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "elpv" / "src" / "elpv_dataset" / "data"
LABELS = DATA_RAW / "labels.csv"
IMAGES = DATA_RAW / "images"
WORK = ROOT / "artifacts"

# ---------------------------------------------------------------- 分级定义
# ELPV 的标注不是连续概率，而是 4 档离散值（0.0 / 1/3 / 2/3 / 1.0）。
# 因此"分级"不是我们人为切出来的，而是数据集本身的标注粒度。
GRADES = ["G0_良好", "G1_轻微", "G2_中度", "G3_严重"]

# 操作级三档：把"轻微/中度"合并为"待复核"。
# 依据：这两档在真实产线上的处置动作同属"返修/降级"，且二者边界最依赖经验，
# 四级模型在跨组件泛化时恰好在这两档坍塌（见 docs/实验记录.md）。
GRADES3 = ["P0_合格", "P1_待复核", "P2_判废"]
GRADE3_OF = {
    "G0_良好": "P0_合格",
    "G1_轻微": "P1_待复核",
    "G2_中度": "P1_待复核",
    "G3_严重": "P2_判废",
}

GRADE_CN = {
    "G0_良好": "良好",
    "G1_轻微": "轻微缺陷",
    "G2_中度": "中度缺陷",
    "G3_严重": "严重缺陷",
    "P0_合格": "合格",
    "P1_待复核": "待复核",
    "P2_判废": "判废",
    "normal": "合格",
    "defect": "不合格",
}
# 业务含义：G0 直接出厂；G1/G2 降级或返修；G3 判废。
GRADE_ACTION = {
    "G0_良好": "正常流转",
    "G1_轻微": "降级使用",
    "G2_中度": "返修复检",
    "G3_严重": "判废",
    "P0_合格": "正常流转",
    "P1_待复核": "返修 / 降级",
    "P2_判废": "判废",
}
# 代价矩阵的第一版：把“严重缺陷判成良好”的代价设为 10，
# 其余错误代价按档位差递增。用于代价敏感阈值搜索。
MISCLASS_COST = {
    # (真实, 预测): 代价
    ("G3_严重", "G0_良好"): 20.0,   # 最严重：判废品流入产线
    ("G3_严重", "G1_轻微"): 12.0,
    ("G3_严重", "G2_中度"): 5.0,
    ("G2_中度", "G0_良好"): 10.0,
    ("G2_中度", "G1_轻微"): 5.0,
    ("G1_轻微", "G0_良好"): 4.0,
    ("G0_良好", "G3_严重"): 2.5,    # 误判废：损失良品
    ("G0_良好", "G2_中度"): 2.0,
    ("G0_良好", "G1_轻微"): 1.0,
}
DEFAULT_COST = 1.0


@dataclass
class SplitCfg:
    """分组切分配置。"""
    hash_size: int = 8          # 感知哈希边长，8 → 64 bit
    ratio: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 20261119
    # 分组阈值：Hamming 距离 ≤ 该值视为同一组件。None 表示自动搜索到目标组数。
    hamming: int | None = None
    target_groups: int = 44     # ELPV 论文说明数据来自 44 块组件，用它校准阈值
    dup_similarity: float = 0.995   # 16×16 外观余弦相似度达到该值视为重复样本


@dataclass
class TrainCfg:
    model: str = "yolov8s-cls.pt"
    imgsz: int = 224
    epochs: int = 60
    batch: int = 64
    patience: int = 20
    seed: int = 0
    device: str = ""            # "" = 自动（有 GPU 用 GPU）
    workers: int = 0            # 沙箱内多进程受限，固定 0
    # 数据量只有 1600 张，全部缓存进内存；不缓存时单进程解码 300×300 PNG
    # 会成为瓶颈（实测 2.8 s/batch，比 GPU 计算慢一个数量级）。
    cache: str = "ram"
    cos_lr: bool = True
    label_smoothing: float = 0.1
    runs: Path = field(default_factory=lambda: WORK / "runs")


# 训练集过采样倍数：四级样本量 911/215/73/408，中间两档严重不足，
# 不处理时模型会直接放弃这两类（实测召回 0.000）。
# 注意只作用于 train，绝不复制到 val/test。
TRAIN_OVERSAMPLE = {
    "G0_良好": 1,
    "G1_轻微": 3,
    "G2_中度": 7,
    "G3_严重": 2,
}


@dataclass
class EvalCfg:
    imgsz: int = 224
    batch: int = 64
