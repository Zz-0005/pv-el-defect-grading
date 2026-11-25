# 光伏电池片 EL 图像缺陷分级

用 YOLOv8 分类头把一片光伏电池片的**电致发光（EL）图像**判成三档——
**合格 / 待复核 / 判废**，并给出处置动作。数据是 ELPV 公开的 2624 张真实产线 EL 图像，
全部指标可在本机一条命令复现。

```
EL 图像 ──► ONNX 模型 ──► 三档概率 ──► 处置动作
(300×300)   YOLOv8s-cls              放行 / 返修 / 报废
```

| 准确率 | 宏平均 F1 | 单张推理 | 模型大小 |
|---|---|---|---|
| **0.761** | 0.621 | **9.5 ms**（CPU） | 20 MB（ONNX） |

> 不需要 GPU、不需要 torch 就能跑推理。交付包双击即用。

---

## 这个项目解决什么问题

EL 成像是给电池片通电、让它自己发光，再拍近红外图像。有隐裂、断栅的片子发光不均，在图上就是暗纹。

现场靠人工看图分档：一天上千张，人会疲劳、判级会漂移；而"轻微"和"中度"的界限本来就靠经验。

**关键不是判断"有没有缺陷"**——那只是把人工判断换个说法。系统要输出的是**处置动作**：
放行、返修、还是报废。因为"不合格之后怎么办"决定了成本差一个数量级。

## 结果（冻结测试集 582 张，来自训练时未见过的组件）

| 档位 | 精确率 | 召回率 | F1 | 样本 |
|---|---|---|---|---|
| 合格 | 0.725 | **0.936** | 0.817 | 313 |
| 待复核 | 0.444 | 0.211 | 0.286 | 57 |
| 判废 | **0.914** | 0.651 | 0.760 | 212 |

- 判废漏判率 **34.9%**（真实判废被判成合格/待复核）——最该改进的一项
- 加权代价 **2753 / 千张**（判废漏出去记 20 分，多看一遍记 1 分）

## 三个关键设计

### 1. 按组件分组切分，防止数据泄漏

2624 张图来自 44 块真实组件，同一组件的相邻电池片成像几乎一致。按图片随机切分会让孪生样本
同时进训练和测试。ELPV 不提供组件 ID，所以用 16×16 外观向量 k-means 还原 42 组，按组切分，
并做两道泄漏检查（同组跨 split = 0，重复样本跨 split = 0）。

实测：同一模型随机切分 **0.7330** vs 分组切分 **0.7045**，差 **+2.85 个百分点**——那就是泄漏。

### 2. 四级分级会失败，所以降到三档

第一版做四级（良好/轻微/中度/严重），中间两档召回只有 0.150 / 0.059，模型几乎放弃了它们。
把「轻微/中度」合并成「待复核」后宏 F1 从 0.437 升到 0.621。

合并的依据不是"指标不好看"，而是这两档在产线上的处置动作本就同属返修/降级。

### 3. 拒识：把最没把握的 5% 转人工

固定人工预算，把置信度最低的 5% 改判「待复核」：宏 F1 0.621→0.638，代价降 6%，
准确率只掉 0.5 个点。

> 这里有一次失败：先做无约束的代价优化，算法给出"置信度 < 90% 就判待复核"，
> 代价降 57% 但把 **73% 的合格品**标成待复核。那不是模型变好，是评价函数允许它作弊。
> 所以改成固定人工预算——预算只能由人定。

## 仓库结构

```
.
├── src/pvinspect/       训练与评估源码（数据准备、训练、评测、Grad-CAM）
├── scripts/             数据下载、ONNX 导出、探针、打包脚本
├── docs/                实验记录、局限与边界、拒识策略、交付说明
├── delivery/            ★ 交付包：ONNX 权重 + 演示 + 离线技术报告 + PPT
│   ├── 使用说明.md
│   ├── 演示脚本.md       现场 60 秒逐字台词
│   ├── 演讲稿.md         1.5 / 3.5 / 7 分钟三个长度
│   └── pv-demo/
│       ├── models/       ONNX 权重（20 MB）
│       ├── demo/ demo2/ fails/   演示图与错例
│       ├── tools/        推理、评估、渲染、建网页
│       ├── web/          离线技术报告（13 章，含三视图查看器）
│       └── docs/         答辩 PPT
└── artifacts/           训练产物（不进仓库，可用脚本重建）
```

**完整数据集不进仓库**（2624 张图约 50 MB，且 ELPV 为 CC-BY-NC-SA 协议）。
需要完整跑通时按下面的"从零训练"下载。

## 快速开始

### 方式一：直接看演示（不训练）

```bash
cd delivery/pv-demo
pip install -r requirements.txt      # numpy / onnxruntime / pillow，约 120 MB

python tools/onnx_predict.py --images demo/images --json out/pred_demo.json
python run.py --web                  # 重建离线技术报告
```

或直接双击 `delivery/pv-demo/web/index.html`（纯静态，不联网）。
Windows 下也可以双击 `delivery/` 里的启动脚本。

### 方式二：从零训练（需要 torch + GPU）

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python scripts/fetch_data.py                 # 下载 ELPV（93 MB）
python -m pvinspect prepare                  # 分级标注 + 分组切分 + 泄漏检查
python -m pvinspect train --epochs 50        # 训练四组对照实验
python -m pvinspect eval                     # 冻结测试集评测
python scripts/make_report.py                # 生成 docs/实验记录.md
python scripts/export_onnx.py                # 导出 ONNX 给交付包用
```

自检（不需要权重，10 秒）：

```bash
python -m pvinspect selftest     # 23 项
```

## 四组对照实验

| 实验 | 准确率 | 宏F1 | 加权代价/千张 |
|---|---|---|---|
| `E1_grouped` 四级 · 分组切分 | 0.7045 | 0.4371 | 2732.0 |
| `E1_random` 四级 · 随机切分（泄漏对照） | 0.7330 | 0.5538 | 2027.7 |
| `E2_binary` 二分类 | 0.7182 | 0.7166 | 3170.1 |
| **`E3_three` 三档（采用）** | **0.7612** | 0.6211 | 2753.4 |

宏平均 F1 不能跨类别数直接比较（两类瞎猜约 0.50，三类约 0.33），
横向比较只看同一套代价矩阵下的加权代价。

## 已知局限

1. **分级档位来自 ELPV 标注，不是行业判废标准**（IEC 61215 更细）；
2. **组件分组是 k-means 近似还原**，不能保证零泄漏；
3. **没有缺陷类型**（隐裂/断栅/黑斑），只有整体严重度；
4. **单晶/多晶混合训练**，未分型建模；
5. **跨组件泛化是根本瓶颈**：训练损失 0.18、验证损失 1.34，模型在记训练组件；
6. **判废漏判率 34.9%**，目前不能接受；
7. **置信度未做概率校准**：0.9 以上可信（实测准确率 0.99），0.5–0.9 之间不可信，
   所以演示页看的是 top1−top2 的差距，不是绝对值。

详见 [`docs/局限与边界.md`](docs/局限与边界.md)。

## 数据与许可

- **数据**：[ELPV Dataset](https://github.com/zae-bayern/elpv-dataset)
  （Buerhop-Lutz 等，EU PVSEC 2018），**CC-BY-NC-SA-4.0**，仅用于教学与研究。
  2624 张 300×300 真实电致发光图像，来自 44 块真实组件。
- **本仓库自有代码**（`src/`、`scripts/`、`delivery/pv-demo/tools/` 等）：MIT，见 [`LICENSE`](LICENSE)。
- **模型权重**（`delivery/pv-demo/models/*.onnx`）：由 **Ultralytics YOLOv8** 微调并导出。
  Ultralytics YOLOv8 采用 **AGPL-3.0**，因此该权重及其衍生文件按 AGPL-3.0 处理，
  不在 MIT 范围内。用于教学与个人作品展示；商用需另行取得 Ultralytics 的商业授权。
- 本仓库**不包含** ELPV 原始图像；交付包里的 `demo`/`demo2`/`fails` 仅保留少量样例用于演示。

> 如需纯 MIT 的版本，可只保留 `src/` 与 `scripts/`（不包含任何 YOLOv8 权重），
> 训练时由 ultralytics 自行下载预训练权重。

## 引用

```
Buerhop-Lutz, C.; Deitsch, S.; Maier, A.; Gallwitz, F.; Berger, S.; Doll, B.;
Haug, J.; Lüttgens, G.; Brabec, C. J.; Braband, S.; Rieger, F.; Gruber, T.; Reindl, L.
"A Benchmark for Visual Identification of Defective Solar Cells in
 Electroluminescence Imagery." EU PVSEC 2018.
```
