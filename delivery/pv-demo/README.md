# 光伏电池片 EL 缺陷分级 · 技术说明

用 YOLOv8 分类头把一片光伏电池片的电致发光（EL）图像判成 **三档：合格 / 待复核 / 判废**，
并给出处置动作与缺陷热力图。模型导出为 ONNX，**推理不需要 torch、不需要 CUDA**。

```
EL 图像 ──► ONNX 模型 ──► 三档概率 ──► 处置动作
 (300×300)   (YOLOv8s-cls)            合格 / 返修 / 判废
```

---

## 结果（冻结测试集 582 张，来自训练时未见过的组件）

| 档位 | 精确率 | 召回率 | F1 | 样本 |
|---|---|---|---|---|
| 合格 | 0.725 | **0.936** | 0.817 | 313 |
| 待复核 | 0.444 | 0.211 | 0.286 | 57 |
| 判废 | **0.914** | 0.651 | 0.760 | 212 |

- 准确率 **0.7612**，宏平均 F1 **0.6211**
- 判废漏判率 **34.9%**（真实判废被判成合格/待复核）——这是最该改进的一项
- 单张推理 **8.0 ms**（CPU，P95 8.8 ms），582 张约 6 秒

> 完整指标、混淆矩阵、A/B 对照、失败案例见 `web/index.html`（离线网页报告），
> 每个数字都来自 `out/*.json`，可用 `python run.py --all` 重建。

## 三个关键设计

**1. 按组件分组切分，防止数据泄漏**

2624 张图来自 44 块真实组件，同一组件的相邻电池片成像几乎一致。
按图片随机切分会让孪生样本同时进训练和测试。ELPV 不提供组件 ID，
所以用 16×16 外观向量 k-means 还原 42 组，按组切分，并做两道泄漏检查
（同组跨 split = 0，重复样本跨 split = 0）。

实测：同一模型随机切分 0.7330 vs 分组切分 0.7045，差 **+2.85 个百分点**——那部分就是虚高。

**2. 四级分级会失败，所以降到三档**

第一版做四级（良好/轻微/中度/严重），中间两档召回只有 0.150 / 0.059，模型几乎放弃了它们。
把「轻微/中度」合并成「待复核」后宏 F1 从 0.437 升到 0.621。
合并的依据不是"指标不好看"，而是这两档在产线上的处置动作本就同属返修/降级。

**3. 拒识：把最没把握的 5% 转人工**

固定人工预算，把置信度最低的 5% 改判「待复核」：宏 F1 0.621→0.638，代价降 6%，
准确率只掉 0.5 个点。

> 这里有一次失败：先做无约束的代价优化，算法给出"置信度 < 90% 就判待复核"，
> 代价降 57% 但把 73% 的合格品都标成待复核。那不是模型变好，是评价函数允许它作弊。
> 所以改成固定人工预算。详见网页第九章。

## 快速开始

```bash
pip install -r requirements.txt

# 单张 / 整目录推理
python tools/onnx_predict.py --images demo/images --json out/pred_demo.json

# 重跑网页上的全部数字（实测 21 秒）
python run.py --all
```

打开 `web/index.html` 即可看离线报告（不联网）。

## 目录

```
pv-demo/
├── models/   pv_el_yolov8s_cls.onnx（20.3 MB）+ labels.txt + model_card.json
├── demo/     7 张精挑演示图（含 1 张刻意保留的错例）+ 标签
├── demo2/    60 张扩展测试图 + 标签
├── fails/    9 张业务代价最高的错例
├── train/ valid/ test/   全量 2624 张（images + labels）
├── out/      pred_*.json / metrics_*.json / ab.json / failures.json
├── tools/    common / onnx_predict / evaluate / build_failures /
│             render_views / build_web / build_ppt / check_web
├── docs/     答辩 PPT（14 页）
└── web/      index.html + assets/（原图 / 真值 / 预测 三视图 + 热力图）
```

## 已知局限

1. 分级档位来自 ELPV 标注，**不是行业判废标准**（IEC 61215 更细）；
2. 组件分组是 k-means 近似还原，不能保证零泄漏；
3. 没有缺陷类型（隐裂/断栅/黑斑），只有整体严重度；
4. 单晶/多晶混合训练，未分型建模；
5. **跨组件泛化是根本瓶颈**：训练损失 0.18、验证损失 1.34，模型在记训练组件。

## 许可与引用

代码 MIT；数据集 ELPV 为 CC-BY-NC-SA-4.0。

```
Buerhop-Lutz, C.; Deitsch, S.; Maier, A.; Gallwitz, F.; Berger, S.; Doll, B.;
Haug, J.; Lüttgens, G.; Brabec, C. J.; Braband, S.; Rieger, F.; Gruber, T.; Reindl, L.
"A Benchmark for Visual Identification of Defective Solar Cells in
 Electroluminescence Imagery." EU PVSEC 2018.
```
