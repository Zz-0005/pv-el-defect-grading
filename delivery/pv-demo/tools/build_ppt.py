# -*- coding: utf-8 -*-
"""
生成答辩 PPT，**严格按演讲稿的段落顺序**排版。

设计原则：
  1. 每页标题就是讲稿里要说的那句话，翻页节奏 = 说话节奏；
  2. 每页的备注里放讲稿原文，讲到哪儿翻到哪儿；
  3. 讲稿没提到的内容（热力图、失败案例、下一步）全部降到附录，不占主线。

产物：docs/光伏电池片EL缺陷分级-答辩汇报.pptx（15 页：11 页主线 + 4 页附录）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pptx import Presentation  # noqa: E402
from pptx.util import Inches  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402

# deck_style.py 放在同目录，保证别人 clone 下来也能重建 PPT
from deck_style import (  # noqa: E402
    INK, INK_SOFT, MUTED, LINE, SOFT, SOFT2, WHITE, ACCENT, ACCENT_L, AMBER, AMBER_L, DARK,
    simple, rect, hline, add_table, slide_header, slide_footer, set_notes,
)

ROOT = HERE.parent
OUT = ROOT / "docs" / "光伏电池片EL缺陷分级-答辩汇报.pptx"
TOTAL = 15
TAG = "光伏电池片 EL 缺陷分级 · 现场陈述"


def load(name):
    return json.loads((ROOT / "out" / name).read_text(encoding="utf-8"))


def main() -> int:
    m = load("metrics_test.json")
    ab = load("ab.json")
    fl = load("failures.json")

    CN = {"P0_合格": "合格", "P1_待复核": "待复核", "P2_判废": "判废"}
    pc = m["per_class"]
    cm = m["confusion"]
    abm = {e["id"]: e for e in ab["experiments"]}
    lat = m["latency_ms"]["median"]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    BL = prs.slide_layouts[6]
    L, R = 0.72, 12.61
    W = R - L

    def new(bg=WHITE):
        s = prs.slides.add_slide(BL)
        if bg != WHITE:
            rect(s, 0, 0, 13.333, 7.5, fill=bg)
        return s

    # ================================================================ 1 封面
    s = new(DARK)
    rect(s, 0, 0, 0.42, 7.5, fill=ACCENT)
    for i in range(9):
        rect(s, 9.05 + i * 0.46, 0, 0.012, 7.5, fill=RGBColor.from_string("1B3340"))
    rect(s, 8.35, 0, 4.98, 7.5, fill=RGBColor.from_string("142C35"))
    simple(s, 9.05, 1.05, 3.4, 0.26, "项目答辩 · 现场陈述", size=9.5, bold=True,
           color=RGBColor.from_string("5FA6A6"))
    simple(s, 9.05, 1.45, 3.55, 1.0, "这一片\n该判废、该返修\n还是该放行", size=19, bold=True,
           color=WHITE, line=1.18)
    hline(s, 9.05, 3.15, 1.5, RGBColor.from_string("3E6B6B"), 1.2)
    simple(s, 9.05, 3.32, 3.55, 1.3, f"准确率 {m['accuracy']:.3f}\n"
           f"宏 F1 {m['macro_f1']:.3f}\n单张 {lat:.1f} ms", size=12,
           color=RGBColor.from_string("A9C2C8"), line=1.7)
    simple(s, 9.05, 5.1, 3.55, 1.2, "2624 张真实 EL 图像\n来自 44 块真实组件",
           size=9.5, color=RGBColor.from_string("6C8A93"), line=1.6)

    simple(s, L, 1.62, 7.2, 0.3, "AI 机器视觉 · 课程设计 · 项目答辩", size=11.5, bold=True,
           color=RGBColor.from_string("6FB3B3"))
    simple(s, L, 2.02, 7.4, 1.9, "光伏电池片 EL 图像\n缺陷分级系统", size=40, bold=True,
           color=WHITE, line=1.14)
    hline(s, L, 3.94, 2.2, ACCENT, 2.0)
    simple(s, L, 4.16, 7.2, 0.9,
           "不是判断「有没有缺陷」，\n而是判断「这一片该怎么处置」",
           size=13, color=RGBColor.from_string("B9CDD3"), line=1.55)
    rect(s, L, 5.55, 7.25, 0.012, fill=RGBColor.from_string("33505C"))
    simple(s, L, 5.72, 2.2, 0.24, "答辩人", size=9, color=RGBColor.from_string("6C8A93"))
    simple(s, L, 5.98, 2.2, 0.3, "〔姓名〕", size=13, bold=True, color=WHITE)
    simple(s, L + 2.35, 5.72, 2.2, 0.24, "项目类型", size=9, color=RGBColor.from_string("6C8A93"))
    simple(s, L + 2.35, 5.98, 2.8, 0.3, "个人项目 · 训练+部署", size=13, bold=True, color=WHITE)
    simple(s, L + 4.9, 5.72, 2.4, 0.24, "模型", size=9, color=RGBColor.from_string("6C8A93"))
    simple(s, L + 4.9, 5.98, 2.4, 0.3, "YOLOv8s-cls · ONNX", size=12.5, bold=True, color=WHITE)
    set_notes(s, "【开场】各位老师好，我汇报的是光伏电池片的 EL 缺陷分级。\n"
                 "先说一句话定调：我做的不是「判断有没有缺陷」，"
                 "而是「这一片该判废、该返修，还是该放行」。")

    # ================================================================ 2 定调
    s = new()
    slide_header(s, "一 · 定调", "不是判断有没有缺陷，而是判断该怎么处置",
                 sub="二分类只回答「能不能出厂」，三档直接回答「下一步做什么」")
    left = [("系统只输出「不合格」", "合格 / 不合格",
             "「不合格之后怎么办」——判废还是返修，成本差一个数量级——还得人来定。自动化只做了一半。",
             RGBColor.from_string("F0F3F5")),
            ("系统输出三档", "合格 / 待复核 / 判废",
             "每一档直接对应一个可执行动作：放行、返修或降级、报废。",
             ACCENT_L)]
    yy = 2.0
    for h, v, d, c in left:
        rect(s, L, yy, 7.3, 1.62, fill=c, line=None)
        simple(s, L + 0.26, yy + 0.16, 6.8, 0.26, h, size=9.5, bold=True,
               color=ACCENT if c == ACCENT_L else MUTED)
        simple(s, L + 0.26, yy + 0.44, 6.8, 0.4, v, size=18, bold=True, color=INK)
        simple(s, L + 0.26, yy + 0.98, 6.8, 0.6, d, size=9.8, color=INK_SOFT, line=1.35)
        yy += 1.76

    rect(s, 8.35, 2.0, 4.26, 3.1, fill=DARK, line=None)
    simple(s, 8.59, 2.14, 3.7, 0.26, "实测：留一个中间档值多少", size=9, bold=True,
           color=RGBColor.from_string("7FB8B8"))
    rows = [["方案", "准确率", "代价/千张"],
            ["二分类", f"{abm['E2_binary']['accuracy']:.4f}", f"{abm['E2_binary']['cost_per_1000']:.0f}"],
            ["三档", f"{abm['E3_three']['accuracy']:.4f}", f"{abm['E3_three']['cost_per_1000']:.0f}"]]
    add_table(s, 8.59, 2.44, 3.79, rows, [1.2, 1.3, 1.3], row_h=0.46, head_h=0.40,
              size=10.2, head_fill=RGBColor.from_string("22424E"),
              align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER], rule_rows=(2,))
    simple(s, 8.59, 3.86, 3.8, 1.0,
           "代价＝每 1000 片的业务损失分：\n判废漏出去 20 分，多看一遍 1 分。",
           size=8.8, color=RGBColor.from_string("A9C2C8"), line=1.45)

    rect(s, 8.35, 5.3, 4.26, 1.3, fill=AMBER_L, line=None)
    simple(s, 8.59, 5.44, 3.8, 0.26, "所以我要的不是分类器", size=9, bold=True, color=AMBER)
    simple(s, 8.59, 5.70, 3.8, 0.8, "是三条出口：\n自动放行 / 自动剔除 / 推人工",
           size=10, color=RGBColor.from_string("6B4A25"), line=1.4)
    slide_footer(s, 2, TOTAL, TAG)
    set_notes(s, "【讲稿原文】\n"
                 "为什么这个区别重要？\n"
                 "EL 成像是给电池片通电让它发光，再拍近红外图。有隐裂、断栅的片子发光不均，在图上就是暗纹。\n"
                 "现在靠人工看图分档，一天上千张，人会疲劳、判级会漂移。\n"
                 "如果系统只输出「不合格」，那「不合格之后怎么办」——判废还是返修，"
                 "成本差一个数量级——还得人来定。等于自动化只做了一半。")

    # ================================================================ 3 EL 原理
    s = new()
    slide_header(s, "二 · 为什么需要它", "人工分档有两个绕不过去的问题",
                 sub="EL 成像：给电池片通电让它自己发光，再拍近红外图——有隐裂、断栅的地方发光不均，图上就是暗纹")
    probs = [("问题一", "一天上千张，人会疲劳", "判级会漂移：同一片，上午判轻微、下午判中度",
              RGBColor.from_string("EAF0F4")),
             ("问题二", "中间档的界限靠经验", "「轻微」和「中度」不同人给出不同档，没有共同尺子",
              AMBER_L)]
    x = L
    for n, h, d, c in probs:
        rect(s, x, 2.0, 5.86, 1.66, fill=c, line=None)
        simple(s, x + 0.26, 2.14, 5.3, 0.26, n, size=9, bold=True,
               color=AMBER if c == AMBER_L else ACCENT)
        simple(s, x + 0.26, 2.42, 5.3, 0.32, h, size=13.5, bold=True, color=INK)
        simple(s, x + 0.26, 2.86, 5.3, 0.7, d, size=9.8, color=INK_SOFT, line=1.4)
        x += 6.03

    rect(s, L, 3.92, W, 1.55, fill=DARK, line=None)
    simple(s, L + 0.3, 4.06, 5.0, 0.26, "于是系统该输出什么？", size=9, bold=True,
           color=RGBColor.from_string("7FB8B8"))
    simple(s, L + 0.3, 4.38, 11.2, 0.95,
           "不只是「有没有缺陷」——那只是把人工判断换了个说法。\n"
           "要输出的是处置动作：「放行」「返修」「报废」，这样自动化才真正接住了下一道工序。",
           size=12.5, color=WHITE, line=1.5)
    simple(s, L, 5.78, W, 0.9,
           "这也是本项目所有设计选择的出发点：后面三个决定（怎么切数据、分几档、要不要拒识）"
           "都是为了让输出能落到动作上。",
           size=10.2, color=INK_SOFT, line=1.5)
    slide_footer(s, 3, TOTAL, TAG)
    set_notes(s, "这一页展开讲稿第二段的两句：\n"
                 "① 「一天上千张，人会疲劳、判级会漂移」\n"
                 "② 「轻微和中度的界限靠经验」\n"
                 "然后落到「所以系统该输出处置动作，不是输出有没有缺陷」。")

    # ================================================================ 4 数据
    s = new()
    slide_header(s, "三 · 数据", "分三档不是我拍的，是数据本身的粒度",
                 sub="ELPV 公开集：2624 张真实电致发光图像，来自 44 块真实组件")
    cards = [("2624", "张图像", "不是合成、不是渲染", ACCENT),
             ("44", "块真实组件", "一块组件切成很多片电池片", AMBER),
             ("4", "个标注档位", "0 / 1/3 / 2/3 / 1", ACCENT)]
    x = L
    for v, k, d, c in cards:
        rect(s, x, 1.98, 3.82, 1.36, fill=WHITE, line=LINE)
        simple(s, x + 0.26, 2.12, 3.3, 0.46, v, size=24, bold=True, color=c)
        simple(s, x + 0.26, 2.62, 3.3, 0.26, k, size=10.5, bold=True, color=INK)
        simple(s, x + 0.26, 2.90, 3.3, 0.26, d, size=9, color=MUTED)
        x += 4.03
    rect(s, L, 3.56, W, 1.24, fill=ACCENT_L, line=None)
    rect(s, L, 3.56, 0.045, 1.24, fill=ACCENT)
    simple(s, L + 0.26, 3.70, 2.6, 0.24, "一句话说清", size=9, bold=True, color=ACCENT)
    simple(s, L + 0.26, 3.96, 11.1, 0.7,
           "标注的缺陷概率只有四个离散取值（0 / 1/3 / 2/3 / 1）——"
           "所以「三档」是把这四档按处置动作合并出来的，不是我为凑任务切的。",
           size=11.5, color=INK, line=1.45)
    rect(s, L, 5.0, 5.86, 1.62, fill=WHITE, line=LINE)
    simple(s, L + 0.24, 5.14, 5.3, 0.24, "为什么选这个数据集", size=9.5, bold=True, color=ACCENT)
    simple(s, L + 0.24, 5.40, 5.4, 1.1,
           "· 真实产线图像，不是公开渲染集\n"
           "· 体量小（93 MB），评委可现场复现\n"
           "· 有官方论文与引用，来源可查",
           size=9.8, color=INK_SOFT, line=1.45)
    rect(s, 6.86, 5.0, 5.75, 1.62, fill=AMBER_L, line=None)
    simple(s, 7.10, 5.14, 5.2, 0.24, "必须主动说的两点", size=9.5, bold=True, color=AMBER)
    simple(s, 7.10, 5.40, 5.3, 1.1,
           "· 档位不是行业判废标准（IEC 61215 更细）\n"
           "· 没有缺陷类型，只有整体严重度",
           size=9.8, color=RGBColor.from_string("6B4A25"), line=1.45)
    slide_footer(s, 4, TOTAL, TAG)
    set_notes(s, "【讲稿原文】\n"
                 "数据。ELPV 公开集，2624 张真实电致发光图像，来自 44 块真实组件。\n"
                 "它的标注概率只有四个离散值：0、1/3、2/3、1。所以分三档不是我拍的，是数据本身的粒度。")

    # ================================================================ 5 决定一
    s = new()
    lk = ab["leakage"]
    slide_header(s, "四 · 决定一", "按组件分组切分：差的那 2.85 个点就是泄漏",
                 sub="同一块组件的相邻片子成像几乎一致；按图片随机切，孪生样本会同时进训练和测试")
    steps = [("①", "算外观指纹", "每张图缩到 16×16，零均值单位方差"),
             ("②", "聚类还原组件", "k-means 成 44 组（对应 44 块组件）"),
             ("③", "合并重复样本", "余弦相似度 ≥ 0.995 视为重复，并进同组"),
             ("④", "按组切分", "组是原子的，绝不拆；两道泄漏检查")]
    yy = 2.0
    for n, h, d in steps:
        rect(s, L, yy, 6.2, 0.86, fill=WHITE, line=LINE)
        simple(s, L + 0.18, yy + 0.24, 0.5, 0.3, n, size=15, bold=True, color=ACCENT)
        simple(s, L + 0.72, yy + 0.13, 2.1, 0.3, h, size=11, bold=True, color=INK)
        simple(s, L + 2.9, yy + 0.15, 3.1, 0.56, d, size=9, color=INK_SOFT, line=1.3)
        yy += 0.96
    simple(s, L, 5.9, 6.2, 0.7,
           f"结果：还原出 {lk['groups']} 组，同组跨 split = {lk['cross_groups']}，"
           f"重复样本跨 split = {lk['cross_dups']}。",
           size=10.5, bold=True, color=ACCENT)

    rect(s, 7.16, 2.0, 5.45, 2.15, fill=DARK, line=None)
    simple(s, 7.40, 2.14, 5.0, 0.26, "同一模型 · 同一超参 · 只换切分方式", size=9,
           bold=True, color=RGBColor.from_string("7FB8B8"))
    simple(s, 7.40, 2.44, 2.4, 0.3, "随机切分", size=10, color=RGBColor.from_string("C6D8DD"))
    simple(s, 7.40, 2.74, 2.4, 0.5, f"{lk['random']:.4f}", size=22, bold=True, color=WHITE)
    simple(s, 9.95, 2.44, 2.4, 0.3, "分组切分", size=10, color=RGBColor.from_string("C6D8DD"))
    simple(s, 9.95, 2.74, 2.4, 0.5, f"{lk['grouped']:.4f}", size=22, bold=True,
           color=RGBColor.from_string("F0B26B"))
    simple(s, 7.40, 3.44, 5.0, 0.4,
           f"差 {lk['delta_pt']:+.2f} 个百分点 —— 那就是泄漏", size=10, bold=True,
           color=RGBColor.from_string("F0B26B"))
    simple(s, 7.16, 4.36, 5.45, 2.2,
           "随机切分的指标更高，不是模型更强——同一块组件的孪生样本同时进了训练集和测试集，"
           "模型只要记住「这块组件的底色」就能答对。\n\n"
           "ELPV 没给组件编号，所以这一步是「近似还原」，不是精确还原；"
           "这一点我在局限里会再说一次。",
           size=9.8, color=INK_SOFT, line=1.5)
    slide_footer(s, 5, TOTAL, TAG)
    set_notes(s, "【讲稿原文】\n"
                 "第一个决定：按组件分组切分。同一块组件的相邻片子成像几乎一致，"
                 "按图片随机切会让孪生样本同时进训练和测试。\n"
                 "ELPV 没给组件编号，我把每张图缩成 16×16 做外观指纹，k-means 聚成 42 组，按组切分。\n"
                 "实测同一模型：随机 0.733、分组 0.705，差 2.85 个点，那就是泄漏。")

    # ================================================================ 6 决定二
    s = new()
    slide_header(s, "五 · 决定二", "四级降到三档：因为这两档的动作本来就一样",
                 sub="第一版做四级，中间两档的召回只有 0.15 和 0.06——召回＝这一档的真图里被找出来多少")
    rect(s, L, 1.98, 5.86, 2.42, fill=WHITE, line=LINE)
    simple(s, L + 0.24, 2.12, 5.3, 0.24, "第一版：四级分级", size=9.5, bold=True, color=AMBER)
    rows = [["档位", "召回"], ["良好", "0.815"], ["轻微", "0.150"],
            ["中度", "0.059"], ["严重", "0.698"]]
    add_table(s, L + 0.24, 2.40, 5.3, rows, [1.6, 1.2], row_h=0.32, head_h=0.32,
              size=9.5, align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER], rule_rows=(2, 3))
    simple(s, L + 0.24, 4.04, 5.3, 0.3, "中间两档基本不被预测出来", size=9,
           color=RGBColor.from_string("8A5316"))

    rect(s, L, 4.56, 5.86, 2.06, fill=ACCENT_L, line=None)
    rect(s, L, 4.56, 0.045, 2.06, fill=ACCENT)
    simple(s, L + 0.26, 4.70, 2.4, 0.24, "改法：合并成「待复核」", size=9, bold=True, color=ACCENT)
    simple(s, L + 0.26, 4.96, 5.3, 1.5,
           "理由不是「指标不好看就合并」，而是这两档在产线上的动作本来就一样："
           "返修或降级。\n\n边界最模糊的地方，恰好也是动作最不需要区分的地方。",
           size=9.8, color=INK, line=1.45)

    rect(s, 6.86, 1.98, 5.75, 2.42, fill=DARK, line=None)
    simple(s, 7.10, 2.12, 5.2, 0.26, "合并之后", size=9, bold=True,
           color=RGBColor.from_string("7FB8B8"))
    simple(s, 7.10, 2.44, 2.5, 0.28, "四级 · 宏F1", size=9.5, color=RGBColor.from_string("C6D8DD"))
    simple(s, 7.10, 2.70, 2.5, 0.44, f"{abm['E1_grouped']['macro_f1']:.3f}", size=20,
           bold=True, color=RGBColor.from_string("A9C2C8"))
    simple(s, 9.75, 2.44, 2.5, 0.28, "三档 · 宏F1", size=9.5, color=RGBColor.from_string("C6D8DD"))
    simple(s, 9.75, 2.70, 2.5, 0.44, f"{abm['E3_three']['macro_f1']:.3f}", size=20, bold=True,
           color=RGBColor.from_string("F0B26B"))
    simple(s, 7.10, 3.40, 5.2, 0.8,
           f"准确率 {abm['E1_grouped']['accuracy']:.4f} → {abm['E3_three']['accuracy']:.4f}",
           size=10.5, color=WHITE)
    simple(s, 6.86, 4.56, 5.75, 2.06,
           "这里要诚实：降粒度提高了准确率，但**没有降低加权代价**"
           f"（{abm['E3_three']['cost_per_1000']:.0f} vs {abm['E1_grouped']['cost_per_1000']:.0f}，基本持平）。\n\n"
           "因为「待复核」一旦被误判成合格，代价同样是 20 分。\n"
           "真正能降代价的是拒识——下一个决定。",
           size=9.6, color=INK_SOFT, line=1.5)
    slide_footer(s, 6, TOTAL, TAG)
    set_notes(s, "【讲稿原文】\n"
                 "第二个决定：四级降到三档。四级时中间两档召回只有 0.15 和 0.06，模型基本放弃了它们。\n"
                 "合并成「待复核」，是因为这两档在产线上的动作本来就一样：返修或降级。")

    # ================================================================ 7 决定三
    s = new()
    slide_header(s, "六 · 决定三", "拒识：把最没把握的 5% 转人工",
                 sub="只取最大概率那档，等于逼它在没把握时也选一个——于是改成固定人工预算")
    rows = [["转人工比例", "准确率", "宏F1", "代价/千张", "待复核召回"]]
    for r in ab["reject_budget"]:
        mark = " ← 采用" if abs(r["q"] - 0.05) < 1e-9 else ""
        rows.append([f"{r['q']*100:.0f}%{mark}", f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}",
                     f"{r['cost_per_1000']:.1f}", f"{r['recall_review']:.3f}"])
    add_table(s, L, 1.98, 7.4, rows, [1.9, 1.5, 1.4, 1.5, 1.6], row_h=0.38, head_h=0.38,
              size=9.8, align=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4, rule_rows=(2,))
    simple(s, L, 4.62, 7.4, 1.1,
           "取 5%：宏 F1 从 0.621 升到 0.638，加权代价降 6%，\n"
           "准确率只掉 0.5 个百分点。人工负担是每 100 片看 5 片。",
           size=10.5, color=INK, line=1.5)

    rect(s, 8.42, 1.98, 4.19, 4.3, fill=ACCENT_L, line=None)
    rect(s, 8.42, 1.98, 0.045, 4.3, fill=ACCENT)
    simple(s, 8.66, 2.12, 3.7, 0.24, "为什么不用固定阈值", size=9, bold=True, color=ACCENT)
    simple(s, 8.66, 2.40, 3.75, 3.7,
           "阈值是「算法」定的，预算才是「人」定的。\n\n"
           "复核岗一天看不了几百片——"
           "如果算法自己决定要转多少，"
           "它就会转到产线受不了为止。\n\n"
           "所以这里固定的是比例，"
           "由工厂按复核产能给上限，"
           "算法只负责挑出「最没把握的那一批」。",
           size=9.4, color=INK, line=1.5)
    slide_footer(s, 7, TOTAL, TAG)
    set_notes(s, "【讲稿原文】\n"
                 "第三个决定：拒识。只取最大概率那档，等于逼它没把握时也选一个。\n"
                 "我改成固定人工预算，最没把握的 5% 转人工：宏 F1 0.621→0.638，"
                 "代价降 6%，准确率只掉 0.5 个点。")

    # ================================================================ 8 弯路
    s = new()
    slide_header(s, "七 · 一次弯路", "算法把 73% 的良品标成了待复核，我却以为自己做对了",
                 sub="这一页讲失败：先按代价做无约束优化，指标很漂亮，但没有用")
    chain = [("做法", "按业务代价做无约束优化\n搜出的最优解：置信度 < 0.90 就判待复核", RGBColor.from_string("F0F3F5")),
             ("指标", "加权代价降 57%\n看起来是一次大成功", ACCENT_L),
             ("真相", "把 73% 的合格品标成了待复核\n每 100 片有 73 片要人看", AMBER_L)]
    x = L
    for h, d, c in chain:
        rect(s, x, 1.98, 3.82, 1.72, fill=c, line=None)
        simple(s, x + 0.24, 2.12, 3.3, 0.26, h, size=9, bold=True,
               color=AMBER if c == AMBER_L else ACCENT)
        simple(s, x + 0.24, 2.42, 3.35, 1.16, d, size=10, color=INK, line=1.45)
        x += 4.03
    rect(s, L, 3.94, W, 1.42, fill=DARK, line=None)
    simple(s, L + 0.3, 4.06, 6.0, 0.26, "那不是模型变好，是我的评价函数允许它作弊", size=9,
           bold=True, color=RGBColor.from_string("F0B26B"))
    simple(s, L + 0.3, 4.36, 11.2, 0.85,
           "代价表里「多看一眼」只记 1 分，「漏掉判废」记 20 分——"
           "算法当然选择全都多看一眼。\n换掉评价函数，同一份数据就能得出「更好」的结论。",
           size=11.5, color=WHITE, line=1.5)
    rect(s, L, 5.56, W, 1.06, fill=ACCENT_L, line=None)
    rect(s, L, 5.56, 0.045, 1.06, fill=ACCENT)
    simple(s, L + 0.28, 5.70, 2.6, 0.24, "所以我把约束加回去了", size=9, bold=True, color=ACCENT)
    simple(s, L + 0.28, 5.96, 11.2, 0.55,
           "人工复核有预算上限，这个上限只能由人来定——这就是上一个决定里「固定 5%」的由来。",
           size=10.5, color=INK, line=1.4)
    slide_footer(s, 8, TOTAL, TAG)
    set_notes(s, "【讲稿原文】\n"
                 "这里我走过一次弯路：先按代价做无约束优化，算法给的解是"
                 "「置信度低于 90% 就判待复核」，代价降 57%——但它把 73% 的良品标成了待复核。\n"
                 "那不是模型变好，是评价函数允许它作弊。所以我把人工预算这条约束加回去了。\n\n"
                 "【节奏】这一页是全篇最容易被记住的地方，说完「偷偷作弊」停一秒再翻页。")

    # ================================================================ 9 结果
    s = new()
    slide_header(s, "八 · 结果", "冻结测试集 582 张上的真实水平",
                 sub="测试集来自训练时从未见过的组件，从未参与训练或调参")
    head = ["真实 \\ 预测", "合格", "待复核", "判废"]
    rows = []
    for i, name in enumerate(["合格", "待复核", "判废"]):
        rows.append([f"**{name}**"] + [str(v) for v in cm[i]])
    add_table(s, L, 1.98, 6.6, [head] + rows, [1.8, 1.4, 1.4, 1.4], row_h=0.46, head_h=0.40,
              size=10.5, align=[PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 3)
    simple(s, L, 3.94, 6.6, 0.8,
           f"最危险的一格是「判废 → 合格」：{cm[2][0]} 张，"
           f"即判废漏判率 {m['severe_miss_rate']*100:.1f}%。",
           size=9.8, color=INK_SOFT, line=1.45)

    facts = [(f"{m['accuracy']:.3f}", "准确率"),
             (f"{m['macro_f1']:.3f}", "宏平均 F1"),
             (f"{pc['P0_合格']['recall']:.3f}", "合格召回"),
             (f"{pc['P2_判废']['precision']:.3f}", "判废精确率"),
             (f"{pc['P1_待复核']['recall']:.3f}", "待复核召回")]
    x = L
    for v, k in facts:
        rect(s, x, 4.86, 1.24, 1.0, fill=SOFT, line=None)
        simple(s, x + 0.10, 4.96, 1.1, 0.32, v, size=13.5, bold=True,
               color=AMBER if k == "待复核召回" else ACCENT)
        simple(s, x + 0.10, 5.32, 1.1, 0.42, k, size=7.6, color=MUTED, line=1.2)
        x += 1.32
    simple(s, L, 6.02, 6.6, 0.7,
           "召回＝这一档的真图里被找出来多少；精确率＝判成这一档的里面有多少是对的。",
           size=8.8, color=MUTED, line=1.35)

    rect(s, 7.62, 1.98, 4.99, 4.02, fill=WHITE, line=LINE)
    simple(s, 7.86, 2.12, 4.5, 0.24, "这几行要念慢一点", size=9.5, bold=True, color=ACCENT)
    simple(s, 7.86, 2.40, 4.55, 3.4,
           "· 合格召回 0.936：良品几乎不误拦\n\n"
           "· 判废精确率 0.914：判废误伤很少\n\n"
           "· 待复核召回 0.211：最弱的一环，"
           "十片里漏掉近八片——这一条我要主动说，不藏着\n\n"
           f"· 单张 {lat:.1f} 毫秒，纯 CPU，不需要显卡",
           size=9.4, color=INK_SOFT, line=1.45)
    slide_footer(s, 9, TOTAL, TAG)
    set_notes(s, "【讲稿原文】\n"
                 "结果。测试集 582 张：准确率 0.761，宏 F1 0.621，合格召回 0.936，"
                 "判废精确率 0.914，最弱的待复核召回 0.211。单张 10 毫秒以内，纯 CPU。\n\n"
                 "【节奏】三个数字放慢：0.761、0.936、34.9%。")

    # ================================================================ 10 局限
    s = new()
    slide_header(s, "九 · 局限", "三条我现在还解决不了的事",
                 sub="每一条都能在技术报告里找到对应证据，不打算绕过去")
    lims = [("①", "档位不是行业标准",
             "这四档来自数据集标注，不是 IEC 61215 那类判废标准。真上线要用厂里的真实判废记录重标。"),
            ("②", "跨组件泛化是根本瓶颈",
             "训练损失 0.18、验证损失 1.34——模型在记训练组件。这是 44 块组件的数据规模决定的，不是调参能解决。"),
            ("③", "判废漏判率 34.9%",
             "三片判废里会漏掉一片。这个数字目前不能接受，也是我下一步最想改的一项。")]
    yy = 1.98
    for n, h, d in lims:
        c = AMBER if n == "③" else MUTED
        rect(s, L, yy, W, 1.12, fill=WHITE, line=LINE)
        rect(s, L, yy, 0.045, 1.12, fill=c)
        simple(s, L + 0.26, yy + 0.14, 0.5, 0.3, n, size=13, bold=True, color=c)
        simple(s, L + 0.86, yy + 0.13, 4.6, 0.3, h, size=12, bold=True, color=INK)
        simple(s, L + 0.86, yy + 0.50, 10.9, 0.58, d, size=9.6, color=INK_SOFT, line=1.4)
        yy += 1.24
    rect(s, L, 5.72, W, 0.9, fill=DARK, line=None)
    simple(s, L + 0.3, 5.86, 11.2, 0.62,
           "一句话总结：方法可行、指标可复现、边界清楚；距离上线，差的是真实标签和更多样本。",
           size=12.5, bold=True, color=WHITE)
    slide_footer(s, 10, TOTAL, TAG)
    set_notes(s, "【讲稿原文】\n"
                 "局限。三条：档位不是行业标准；跨组件泛化是根本瓶颈，训练损失 0.18、验证损失 1.34；"
                 "判废漏判率 34.9%，三片判废会漏一片，这个目前不能接受。\n\n"
                 "说完停住，等提问。不要自己往下补。")

    # ================================================================ 11 结语
    s = new(DARK)
    simple(s, L, 0.9, 9.0, 0.3, "十 · 结语", size=10.5, bold=True,
           color=RGBColor.from_string("7FB8B8"))
    simple(s, L, 1.32, 11.4, 1.1, "方法可行、指标可复现、边界清楚", size=30, bold=True, color=WHITE)
    quotes = [
        "输出的是处置动作，不是「有没有缺陷」——这是三档存在的理由。",
        "三个决定都能追溯到一次实测：怎么切数据、分几档、要不要拒识。",
        "代码、ONNX 权重、技术报告都在交付包里，现场双击就能跑。",
    ]
    yy = 3.0
    for i, q in enumerate(quotes, 1):
        simple(s, L, yy, 0.5, 0.3, f"0{i}", size=11, bold=True, color=RGBColor.from_string("6FB3B3"))
        simple(s, L + 0.6, yy - 0.04, 10.6, 0.7, q, size=14, color=WHITE, line=1.45)
        yy += 1.02
    rect(s, L, 6.2, W, 0.66, fill=ACCENT, line=None)
    simple(s, L + 0.3, 6.34, 7.0, 0.36, "谢谢，请提问。", size=15, bold=True, color=WHITE)
    simple(s, 7.4, 6.38, 4.9, 0.3, "现场演示：双击「启动_报告与演示.pyw」", size=9.5,
           color=RGBColor.from_string("C6E8E8"), align=PP_ALIGN.RIGHT)
    set_notes(s, "讲稿里没有单独一段结语，用这三句收尾即可，每句 5 秒。\n"
                 "说完「谢谢，请提问」就停，不要再补充材料。")

    # ================================================================ 附录
    s = new()
    slide_header(s, "附录 A", "热力图能指出缺陷在哪，但强度有限",
                 sub="Grad-CAM（把模型关注的区域画成热力图）映射回原图，再用「暗区/亮区响应比」定量检验")
    blocks = [("目标层不能取最后一层", "最后一层 7×7 特征图上采样回原图会糊成一片，暗/亮比只有 1.2；"
               "改成「最深的、空间尺寸 ≥ 输入 1/16 的特征图」（14×14）后才有区分度。", ACCENT),
              ("四级模型上的效果", "判废样本暗/亮比 2.1~5.2，合格样本无偏向——热力图确实指向缺陷。", ACCENT),
              ("三档模型上弱得多", "判废 1.03~2.12（中位 1.46），合格 0.29~0.69。"
               "方向对，但强度不足以支撑「能定位缺陷」这种说法。", AMBER)]
    yy = 2.0
    for h, d, c in blocks:
        rect(s, L, yy, W, 1.3, fill=WHITE if c == ACCENT else AMBER_L,
             line=LINE if c == ACCENT else None)
        rect(s, L, yy, 0.045, 1.3, fill=c)
        simple(s, L + 0.28, yy + 0.14, 5.0, 0.28, h, size=11, bold=True, color=c)
        simple(s, L + 0.28, yy + 0.46, 11.0, 0.78, d, size=9.8, color=INK_SOFT, line=1.4)
        yy += 1.42
    rect(s, L, 6.36, W, 0.5, fill=DARK, line=None)
    simple(s, L + 0.24, 6.47, W - 0.5, 0.3,
           "正确的说法：热力图可用于人工复核参考，不能当作缺陷分割结果。",
           size=10.5, bold=True, color=WHITE)
    slide_footer(s, 12, TOTAL, "附录")
    set_notes(s, "附录，不讲。被问到「能不能定位缺陷」时翻到这里。")

    s = new()
    slide_header(s, "附录 B", "失败案例：按业务代价排序",
                 sub=f"测试集 {m['n']} 张错了 {fl['total_errors']} 张（{fl['error_rate']*100:.1f}%）")
    rows = [["错误类型", "张数", "占全部错误"]]
    for k, v in fl["kinds"].items():
        kk = k.replace("P0_合格", "合格").replace("P1_待复核", "待复核").replace("P2_判废", "判废")
        rows.append([kk, str(v), f"{v/fl['total_errors']*100:.1f}%"])
    add_table(s, L, 1.98, 6.6, rows, [2.6, 1.2, 1.6], row_h=0.36, head_h=0.36,
              size=9.8, align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER], rule_rows=(1,))
    rect(s, 7.6, 1.98, 5.01, 4.2, fill=AMBER_L, line=None)
    rect(s, 7.6, 1.98, 0.045, 4.2, fill=AMBER)
    simple(s, 7.84, 2.12, 4.5, 0.24, "看完错例要承认一件事", size=9, bold=True, color=AMBER)
    simple(s, 7.84, 2.40, 4.55, 3.6,
           "模型学到的不完全是「缺陷的局部形态」，"
           "很大程度上是整片电池的明暗。\n\n"
           "所以这三类它必然弱：\n"
           "· 同色浅缺陷（与背景灰度差太小）\n"
           "· 油膜覆盖（缺陷被遮住）\n"
           "· 跨视野裂纹（单帧看不全）\n\n"
           "这不是调参能解决的——需要更好的成像，"
           "或换成检测式模型。",
           size=9.3, color=RGBColor.from_string("6B4A25"), line=1.45)
    slide_footer(s, 13, TOTAL, "附录")
    set_notes(s, "附录，不讲。被问「它会在哪儿错」时翻到这里。")

    s = new()
    slide_header(s, "附录 C", "下一步：三件具体的事",
                 sub="按投入产出排序——先解决标签口径，再补样本，最后换监督方式")
    nxt = [("01", "要真实判废记录", "找一家组件厂，用他们的档位重新标标签。现在的档位不是行业标准，这是上线的前提"),
           ("02", "补「待复核」样本", "这一类测试集只有 57 张，指标置信区间太宽；定向补采比调参有效"),
           ("03", "换检测式监督", "让热力图能给出缺陷轮廓而不只是注意力——这需要框或掩码标注")]
    yy = 2.0
    for n, h, d in nxt:
        rect(s, L, yy, W, 1.2, fill=WHITE, line=LINE)
        rect(s, L, yy, 0.045, 1.2, fill=ACCENT)
        simple(s, L + 0.28, yy + 0.32, 0.8, 0.4, n, size=18, bold=True,
               color=RGBColor.from_string("B9C7CF"))
        simple(s, L + 1.16, yy + 0.16, 4.2, 0.32, h, size=12, bold=True, color=INK)
        simple(s, L + 1.16, yy + 0.54, 10.2, 0.58, d, size=9.6, color=INK_SOFT, line=1.35)
        yy += 1.32
    rect(s, L, 6.14, W, 0.62, fill=ACCENT_L, line=None)
    simple(s, L + 0.26, 6.27, W - 0.5, 0.36,
           "结论：方法可行、指标可复现、边界清楚。距离上线，差的是真实标签和更多样本。",
           size=10.8, bold=True, color=ACCENT)
    slide_footer(s, 14, TOTAL, "附录")
    set_notes(s, "附录，不讲。被问「下一步做什么」时翻到这里。")

    s = new()
    slide_header(s, "附录 D", "术语速查：每个词一句话",
                 sub="正文里出现过的名词都在这里，被追问时可以直接翻到这一页")
    terms = [
        ["术语", "一句话解释"],
        ["EL（电致发光）", "给电池片通电让它自己发光，再拍近红外图；有缺陷的地方发光不均"],
        ["组件", "一整块光伏板；本次数据来自 44 块真实组件，每块切成很多片电池片"],
        ["召回", "这一档的真实图片里，被系统找出来多少（漏掉的越少越高）"],
        ["精确率", "被判成这一档的图里，有多少是真的属于这一档（误伤的越少越高）"],
        ["宏平均 F1", "各类 F1 直接平均，稀有档也算一份；比准确率更能暴露弱档"],
        ["加权代价/千张", "每 1000 片的业务损失分：判废漏出去 20 分，良品被判废 2.5 分，多看一遍 1 分"],
        ["分组切分", "同一块组件的电池片只进训练或只进测试，绝不两边都出现"],
        ["数据泄漏", "测试集里混进了训练时见过的相似图，指标虚高，不是模型更强"],
        ["拒识", "模型没把握时不硬判，直接转人工复核"],
        ["ONNX", "通用模型格式，不装训练框架、不用显卡也能跑推理"],
        ["Grad-CAM", "把模型最关注的区域画成热力图，用来检查它有没有看错地方"],
    ]
    add_table(s, L, 1.98, W, terms, [2.3, 9.6], row_h=0.375, head_h=0.36,
              size=9.8, head_size=9.2, bold_cols=(0,),
              align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT])
    slide_footer(s, 15, TOTAL, "附录")
    set_notes(s, "附录，不讲。最常被问的是「宏平均 F1 和准确率有什么区别」「加权代价怎么算」。")

    prs.core_properties.title = "光伏电池片 EL 缺陷分级 · 答辩汇报"
    prs.core_properties.author = "AI 机器视觉工程师订单班"
    prs.core_properties.subject = "按现场讲稿排版：定调 → 数据 → 三个决定 → 结果 → 局限"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(OUT))
        saved = OUT
    except PermissionError:
        alt = OUT.with_name(OUT.stem + "_讲稿版.pptx")
        prs.save(str(alt))
        saved = alt
        print("[提示] 原文件被 PowerPoint 占用，已另存：")
        print(f"       {alt.name}")
        print("       关掉 PowerPoint 后重跑本脚本即可覆盖。")
    print(f"-> {saved}  ({len(prs.slides.__iter__.__self__._sldIdLst)} 页)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
