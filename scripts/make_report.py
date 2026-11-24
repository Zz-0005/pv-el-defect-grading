# -*- coding: utf-8 -*-
"""
把评测结果生成 docs/实验记录.md。

为什么用脚本生成而不是手写：
报告里的每个数字都必须能追溯到 artifacts/results/results.json，
手抄一次就多一次出错的机会。改完模型重跑一遍，文档自动同步。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "实验记录.md"

CN = {
    "G0_良好": "良好", "G1_轻微": "轻微", "G2_中度": "中度", "G3_严重": "严重",
    "P0_合格": "合格", "P1_待复核": "待复核", "P2_判废": "判废",
    "normal": "合格", "defect": "不合格",
}


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def main() -> int:
    res_path = ART / "results" / "results.json"
    if not res_path.exists():
        print(f"缺少 {res_path}，先运行 eval")
        return 1
    results = json.loads(res_path.read_text(encoding="utf-8"))
    summary_path = ART / "dataset_summary.json"
    ds = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    L: list[str] = []
    a = L.append
    a("# 实验记录")
    a("")
    a("> 本文件由 `scripts/make_report.py` 从 `artifacts/results/results.json` 自动生成，")
    a("> 数字与产物一一对应，不做手工誊写。")
    a("")

    # ---------------- 数据
    a("## 1. 数据集与切分")
    a("")
    if ds:
        a(f"- 样本总数：**{ds.get('samples')}** 张（ELPV，300×300 灰度，来自 44 块真实组件）")
        gs = ds.get("group_sizes", {})
        if gs:
            a(f"- 分组数：**{ds.get('groups')}** 组"
              f"（每组样本数 min={gs.get('min')} median={gs.get('median')} max={gs.get('max')}）")
        leak = ds.get("leakage", {})
        if leak:
            a(f"- 泄漏检查：同组跨 split **{leak.get('cross_groups')}** 处，"
              f"重复样本跨 split **{leak.get('cross_dups')}** 对"
              f"（重复判定：16×16 外观余弦 ≥ {leak.get('dup_similarity')}）")
        ov = ds.get("oversample_added", {})
        if ov:
            a(f"- 训练集过采样（仅 train）：" + "，".join(f"{CN.get(k, k)}+{v}" for k, v in ov.items()))
        a("")
        a("| 切分 | 样本数 | 良好 | 轻微 | 中度 | 严重 |")
        a("|---|---|---|---|---|---|")
        for s in ("train", "val", "test"):
            d = ds.get("grade_dist", {}).get(s, {})
            a(f"| {s} | {ds.get('split_sizes', {}).get(s, '-')} | "
              + " | ".join(str(d.get(k, 0)) for k in
                           ("G0_良好", "G1_轻微", "G2_中度", "G3_严重")) + " |")
        a("")

    # ---------------- 主表
    a("## 2. 评测结果（冻结测试集）")
    a("")
    a("| 实验 | 说明 | 样本 | 准确率 | 宏平均F1 | 严重漏判率 | 加权代价/千张 |")
    a("|---|---|---|---|---|---|---|")
    order = ["E1_grouped", "E1_grouped+adj", "E1_random", "E1_random+adj",
             "E2_binary", "E3_three", "E3_three+adj"]
    keys = [k for k in order if k in results] + [k for k in results if k not in order]
    for name in keys:
        r = results[name]
        sm = r.get("severe_miss_rate")
        a(f"| `{name}` | {r.get('desc', '')} | {r['n']} | "
          f"**{fmt(r['accuracy'])}** | {fmt(r['macro_f1'])} | "
          f"{fmt(sm) if sm is not None else '—'} | {r['cost_per_1000']:.1f} |")
    a("")

    # ---------------- 泄漏对照
    if "E1_grouped" in results and "E1_random" in results:
        a("### 2.1 泄漏对照：随机切分把指标抬高了多少")
        a("")
        g, rg = results["E1_grouped"], results["E1_random"]
        a("同一模型、同一批样本、同一超参，只换切分方式：")
        a("")
        a(f"- 分组切分（本方案）：准确率 {fmt(g['accuracy'])}，宏F1 {fmt(g['macro_f1'])}")
        a(f"- 随机切分（泄漏对照）：准确率 {fmt(rg['accuracy'])}，宏F1 {fmt(rg['macro_f1'])}")
        a(f"- 差值：准确率 **{(rg['accuracy'] - g['accuracy']) * 100:+.2f}** 个百分点，"
          f"宏F1 **{(rg['macro_f1'] - g['macro_f1']) * 100:+.2f}** 个百分点")
        a("")
        a("随机切分的指标更高，但这不是模型更强，而是同一块组件的孪生样本")
        a("同时出现在训练集和测试集里。**这个差值就是泄漏带来的虚高。**")
        a("")

    # ---------------- 先验调整
    adj_pairs = [(k, k + "+adj") for k in ("E1_grouped", "E1_random", "E3_three")
                 if k in results and k + "+adj" in results]
    if adj_pairs:
        a("### 2.2 先验调整：让模型敢输出中间档")
        a("")
        a("长尾分级里，模型的 argmax 会被多数类先验主导，典型表现是"
          "**中间级别几乎永不命中**（本项第一版实测 G1/G2 召回为 0）。")
        a("先验调整把训练先验从输出概率里除掉：")
        a("")
        a("```")
        a("p'(c|x) ∝ p(c|x) / π_c^τ")
        a("```")
        a("")
        a("τ 只在**验证集**上按宏平均 F1 选择，测试集只用来报告最终结果。")
        a("")
        a("| 实验 | τ | 准确率 | 宏平均F1 | 严重漏判率 | 代价/千张 |")
        a("|---|---|---|---|---|---|")
        for raw_k, adj_k in adj_pairs:
            r0, r1 = results[raw_k], results[adj_k]
            a(f"| `{raw_k}` | 0（argmax） | {fmt(r0['accuracy'])} | {fmt(r0['macro_f1'])} | "
              f"{fmt(r0['severe_miss_rate']) if 'severe_miss_rate' in r0 else '—'} | {r0['cost_per_1000']:.1f} |")
            a(f"| `{adj_k}` | **{r1.get('tau', 0):.2f}** | {fmt(r1['accuracy'])} | "
              f"{fmt(r1['macro_f1'])} | "
              f"{fmt(r1['severe_miss_rate']) if 'severe_miss_rate' in r1 else '—'} | {r1['cost_per_1000']:.1f} |")
        a("")

    # ---------------- 三档
    if "E3_three" in results:
        a("### 2.3 四级 vs 三档操作级")
        a("")
        a("四级里的「轻微」「中度」在跨组件泛化时最难区分（两者在真实产线上的")
        a("处置动作同属返修/降级）。把这两档合并为「待复核」后：")
        a("")
        e3 = results["E3_three"]
        e1 = results.get("E1_grouped")
        a(f"- 三档：准确率 {fmt(e3['accuracy'])}，宏F1 {fmt(e3['macro_f1'])}，"
          f"加权代价 {e3['cost_per_1000']:.1f}")
        if e1:
            a(f"- 四级：准确率 {fmt(e1['accuracy'])}，宏F1 {fmt(e1['macro_f1'])}，"
              f"加权代价 {e1['cost_per_1000']:.1f}")
            dp = (e3["accuracy"] - e1["accuracy"]) * 100
            dc = (e3["cost_per_1000"] - e1["cost_per_1000"]) / e1["cost_per_1000"] * 100
            a("")
            a(f"**准确率 {dp:+.2f} 个百分点，加权代价 {dc:+.1f}%。**")
            a("")
            a("这一点很重要，早期结论在这里出过错：")
            a("降粒度提高了准确率，但**没有降低加权代价**——因为「待复核」一旦被误判成")
            a("「合格」，代价同样是 20 分。真正能降代价的是拒识策略，不是分级粒度。")
        a("")
        a("另外：三档的宏F1 不能和四级直接比大小（类别数不同，随机基线不同）。")
        a("这一组的价值在于说明**分级粒度需要由数据支撑**，不是越细越好。")
        a("")
        a("拒识策略的完整实验见 [`拒识策略.md`](拒识策略.md)。")
        a("")

    # ---------------- 分类别
    a("## 3. 分类别指标")
    a("")
    for name in keys:
        r = results[name]
        if r["classes"][0] not in ("G0_良好", "P0_合格", "normal"):
            continue
        a(f"### `{name}`")
        a("")
        a("| 级别 | 精确率 | 召回率 | F1 | 支持数 |")
        a("|---|---|---|---|---|")
        for c in r["classes"]:
            m = r["per_class"][c]
            a(f"| {CN.get(c, c)} | {fmt(m['precision'], 3)} | {fmt(m['recall'], 3)} | "
              f"{fmt(m['f1'], 3)} | {m['support']} |")
        a("")

    # ---------------- 混淆矩阵
    a("## 4. 混淆矩阵（行=真实，列=预测）")
    a("")
    for name in keys:
        r = results[name]
        names = r["classes"]
        a(f"### `{name}`")
        a("")
        a("| 真实\\预测 | " + " | ".join(CN.get(n, n) for n in names) + " |")
        a("|---|" + "---|" * len(names))
        for i, n in enumerate(names):
            a(f"| **{CN.get(n, n)}** | " + " | ".join(str(v) for v in r["confusion"][i]) + " |")
        a("")

    # ---------------- 产物
    a("## 5. 产物")
    a("")
    a("```")
    a("artifacts/results/results.csv            汇总表")
    a("artifacts/results/results.json           完整指标（含混淆矩阵与 τ 扫描）")
    a("artifacts/results/confusion_*.png        混淆矩阵图")
    a("artifacts/results/accuracy_compare.png   准确率对比")
    a("artifacts/results/errors_E1_grouped.png  错分样本图")
    a("artifacts/models/*.pt                    各组权重")
    a("```")
    a("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"已生成 {OUT}（{len(L)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
