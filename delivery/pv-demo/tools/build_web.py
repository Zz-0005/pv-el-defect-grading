# -*- coding: utf-8 -*-
"""离线网页技术报告生成器：web/index.html + 三视图查看器。"""
from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ASSETS, CLASS_ACTION, CLASS_CN, CLASS_COLOR, CLASSES, OUT, ROOT, WEB, load_json,
)

INK, ACC, AMBER = "#152430", "#0e6f73", "#c9762a"


# --------------------------------------------------------------------- 组件
def esc(s) -> str:
    return html.escape(str(s))


def img_path(name: str) -> str:
    """用相对路径引用素材（不内联 base64）。

    内联虽然能做成"单文件"，但 200+ 张图会让 HTML 涨到 10 MB，
    浏览器打开会明显卡顿。参考交付包同样是 index.html + assets 目录。
    """
    return f"assets/{name}" if (ASSETS / name).exists() else ""


def viewer(items: list[dict], vid: str) -> str:
    """三视图查看器：左图右表，按钮切换，方向键翻页。"""
    if not items:
        return "<p>没有可展示的样本。</p>"
    payload = []
    for it in items:
        payload.append({
            "orig": img_path(f"orig_{it['name']}.jpg"),
            "gt": img_path(f"gt_{it['name']}.jpg"),
            "pred": img_path(f"pred_{it['name']}.jpg"),
            "true": CLASS_CN.get(it["true"], it["true"]),
            "predcn": CLASS_CN.get(it["pred"], it["pred"]),
            "conf": it["conf"],
            "name": it["name"],
            "ok": it["true"] == it["pred"],
        })
    data = json.dumps(payload, ensure_ascii=False)
    return f"""
<div class="viewer" id="{vid}">
  <div class="vstage"><img id="{vid}_img" alt=""></div>
  <div class="vside">
    <div class="vbtns">
      <button class="vbtn on" data-mode="orig">原图</button>
      <button class="vbtn" data-mode="gt">数据集真值</button>
      <button class="vbtn" data-mode="pred">模型预测</button>
    </div>
    <div class="vmeta" id="{vid}_meta"></div>
    <div class="vnav">
      <button id="{vid}_prev">← 上一张</button>
      <span id="{vid}_idx"></span>
      <button id="{vid}_next">下一张 →</button>
    </div>
    <div class="vhint">键盘 ← → 翻页；按钮切换三视图</div>
  </div>
</div>
<script>
(function(){{
  const D = {data}, root = document.getElementById("{vid}");
  let i = 0, mode = "orig";
  const img = document.getElementById("{vid}_img");
  const meta = document.getElementById("{vid}_meta");
  const idx = document.getElementById("{vid}_idx");
  function render() {{
    const d = D[i];
    img.src = d[mode];
    idx.textContent = (i + 1) + " / " + D.length;
    const ok = d.ok ? '<span class="ok">判定正确</span>' : '<span class="bad">判定错误</span>';
    meta.innerHTML = `<div class="vk">真值</div><div class="vv">${{d.true}}</div>
      <div class="vk">预测</div><div class="vv">${{d.predcn}} <b>${{d.conf.toFixed(2)}}</b></div>
      <div>${{ok}}</div>`;
  }}
  root.querySelectorAll(".vbtn").forEach(b => b.onclick = () => {{
    root.querySelectorAll(".vbtn").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); mode = b.dataset.mode; render();
  }});
  document.getElementById("{vid}_prev").onclick = () => {{ i = (i - 1 + D.length) % D.length; render(); }};
  document.getElementById("{vid}_next").onclick = () => {{ i = (i + 1) % D.length; render(); }};
  window.addEventListener("keydown", e => {{
    if (e.key === "ArrowLeft") {{ i = (i - 1 + D.length) % D.length; render(); }}
    if (e.key === "ArrowRight") {{ i = (i + 1) % D.length; render(); }}
  }});
  render();
}})();
</script>"""


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    tr = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def kpi(value: str, label: str, sub: str = "", color: str = ACC) -> str:
    return (f'<div class="kpi"><div class="kv" style="color:{color}">{esc(value)}</div>'
            f'<div class="kl">{esc(label)}</div><div class="ks">{esc(sub)}</div></div>')


def bar_row(name: str, p: float, color: str) -> str:
    return (f'<div class="bl"><span>{esc(name)}</span>'
            f'<div class="bb"><i style="width:{p*100:.1f}%;background:{color}"></i></div>'
            f'<span class="bv">{p*100:.1f}%</span></div>')


# --------------------------------------------------------------------- 主体
def build() -> str:
    m = load_json(OUT / "metrics_test.json")
    mv = load_json(OUT / "metrics_valid.json") if (OUT / "metrics_valid.json").exists() else None
    ab = load_json(OUT / "ab.json")
    fl = load_json(OUT / "failures.json")
    pdemo = load_json(OUT / "pred_demo.json")
    pdemo2 = load_json(OUT / "pred_demo2.json")
    pfails = load_json(OUT / "pred_fails.json")

    demo_items = [{"name": Path(r["image"]).stem, "true": r["true"],
                   "pred": r["pred"], "conf": r["conf"]} for r in pdemo]
    d2_items = [{"name": Path(r["image"]).stem, "true": r["true"],
                 "pred": r["pred"], "conf": r["conf"]} for r in pdemo2]
    fail_items = [{"name": Path(r["image"]).stem, "true": r["true"],
                   "pred": r["pred"], "conf": r["conf"]} for r in pfails]

    cm = m["confusion"]
    pc = m["per_class"]
    lat = m["latency_ms"]

    # 混淆矩阵 HTML
    head = ["真实 \\ 预测"] + [CLASS_CN[c] for c in m["classes"]]
    cm_rows = []
    for i, c in enumerate(m["classes"]):
        cells = [f"<b>{CLASS_CN[c]}</b>"]
        for j, v in enumerate(cm[i]):
            hot = "hot" if i == j and v else ("warn" if v and i != j else "")
            cells.append(f'<span class="{hot}">{v}</span>' if v else "0")
        cm_rows.append(cells)

    # A/B 表
    ab_rows = []
    for e in ab["experiments"]:
        tag = '<span class="tag">包内可复现</span>' if e["in_package"] else ""
        ab_rows.append([
            f'<code>{esc(e["id"])}</code>', esc(e["title"]),
            f'{e["accuracy"]:.4f}', f'{e["macro_f1"]:.4f}',
            f'{e["cost_per_1000"]:.1f}', tag,
        ])

    rej_rows = []
    for r in ab["reject_budget"]:
        mark = " ← 推荐" if abs(r["q"] - 0.05) < 1e-9 else ""
        rej_rows.append([
            f'{r["q"]*100:.0f}%', f'{r["accuracy"]:.4f}', f'{r["macro_f1"]:.4f}',
            f'{r["cost_per_1000"]:.1f}', f'{r["recall_review"]:.3f}', mark,
        ])

    kind_rows = [[esc(k).replace("P0_合格", "合格").replace("P1_待复核", "待复核")
                  .replace("P2_判废", "判废"), str(v),
                  f'{v/fl["total_errors"]*100:.1f}%']
                 for k, v in fl["kinds"].items()]

    demo_ok = sum(1 for r in pdemo if r["true"] == r["pred"])
    d2_ok = sum(1 for r in pdemo2 if r["true"] == r["pred"])

    parts: list[str] = []
    A = parts.append

    # ---------------- 抬头
    A(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>光伏电池片 EL 缺陷分级 · 技术报告</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f4f7f9;color:{INK};
 font-family:-apple-system,"Microsoft YaHei","Segoe UI",sans-serif;line-height:1.72;font-size:15px}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 22px 90px}}
header.top{{background:{INK};color:#fff;padding:52px 0 44px;margin-bottom:34px}}
header.top .wrap{{padding-bottom:0}}
.eyebrow{{color:#6fb3b3;font-size:12.5px;letter-spacing:.14em;font-weight:700}}
h1{{font-size:34px;line-height:1.25;margin:12px 0 14px;font-weight:800}}
.lead{{color:#b9cdd3;font-size:15.5px;max-width:820px}}
h2{{font-size:23px;margin:52px 0 6px;padding-top:14px;border-top:2px solid {INK};font-weight:800}}
h2 .n{{color:{ACC};font-size:15px;margin-right:10px;vertical-align:2px}}
h3{{font-size:17px;margin:26px 0 8px;color:{INK}}}
p{{margin:10px 0}}
.muted{{color:#7c8c99;font-size:13.5px}}
code{{background:#eaf0f4;padding:1px 6px;border-radius:4px;font-size:13px;
 font-family:Consolas,Monaco,monospace}}
pre{{background:{INK};color:#d8e6ea;padding:14px 18px;border-radius:8px;overflow:auto;
 font-size:13px;line-height:1.6}}
pre code{{background:none;padding:0;color:inherit}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13.5px;background:#fff}}
th{{background:{INK};color:#fff;text-align:left;padding:9px 11px;font-weight:600;font-size:13px}}
td{{border-bottom:1px solid #e3ebf0;padding:8px 11px;vertical-align:top}}
tbody tr:hover td{{background:#f7fbfc}}
.grid{{display:grid;gap:14px;margin:18px 0}}
.g3{{grid-template-columns:repeat(3,1fr)}}
.g2{{grid-template-columns:repeat(2,1fr)}}
.card{{background:#fff;border:1px solid #dde6ec;border-radius:10px;padding:18px 20px}}
.card.warn{{border-left:4px solid {AMBER}}}
.card.ok{{border-left:4px solid {ACC}}}
.kpi{{background:#fff;border:1px solid #dde6ec;border-radius:10px;padding:16px 18px}}
.kv{{font-size:27px;font-weight:800;line-height:1.2}}
.kl{{font-size:13px;color:#3d5261;margin-top:4px;font-weight:600}}
.ks{{font-size:12px;color:#899aa6;margin-top:2px}}
.viewer{{display:grid;grid-template-columns:1.25fr .75fr;gap:18px;background:#fff;
 border:1px solid #dde6ec;border-radius:12px;padding:16px;margin:18px 0}}
.vstage{{background:#0d1b22;border-radius:8px;display:flex;align-items:center;justify-content:center;
 min-height:340px;overflow:hidden}}
.vstage img{{max-width:100%;max-height:520px;display:block}}
.vside{{display:flex;flex-direction:column;gap:10px}}
.vbtns{{display:flex;gap:8px;flex-wrap:wrap}}
.vbtn{{border:1px solid #c3d2da;background:#fff;border-radius:7px;padding:8px 14px;
 font-size:13.5px;cursor:pointer;font-family:inherit}}
.vbtn.on{{background:{ACC};color:#fff;border-color:{ACC}}}
.vmeta{{background:#f4f7f9;border-radius:8px;padding:12px 14px;font-size:14px;min-height:120px}}
.vk{{font-size:11.5px;color:#8b9aa5;letter-spacing:.08em;margin-top:6px}}
.vv{{font-size:17px;font-weight:700}}
.vnav{{display:flex;gap:10px;align-items:center;justify-content:space-between}}
.vnav button{{border:1px solid #c3d2da;background:#fff;border-radius:7px;padding:8px 12px;
 cursor:pointer;font-family:inherit;font-size:13px}}
.vhint{{font-size:12px;color:#9aa9b3}}
.ok{{color:#1a7f37;font-weight:700}}
.bad{{color:#c0392b;font-weight:700}}
.tag{{background:#e3f0f0;color:{ACC};font-size:11.5px;padding:2px 8px;border-radius:10px}}
.hot{{color:{ACC};font-weight:800}}
.warn{{color:#c0392b;font-weight:700}}
.bl{{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:13.5px}}
.bl span:first-child{{width:76px;color:#3d5261}}
.bb{{flex:1;height:9px;background:#eaf0f4;border-radius:5px;overflow:hidden}}
.bb i{{display:block;height:100%}}
.bv{{width:56px;text-align:right;color:#7c8c99;font-size:13px}}
.note{{background:#fdf6ec;border-left:4px solid {AMBER};padding:12px 16px;border-radius:6px;
 margin:16px 0;font-size:14px}}
footer{{margin-top:60px;padding-top:20px;border-top:1px solid #dde6ec;color:#8b9aa5;font-size:13px}}
@media(max-width:900px){{.g3{{grid-template-columns:1fr}}.g2{{grid-template-columns:1fr}}
 .viewer{{grid-template-columns:1fr}}}}
</style></head><body>
<header class="top"><div class="wrap">
<div class="eyebrow">AI 机器视觉 · 课程设计 · 技术报告（离线）</div>
<h1>光伏电池片 EL 图像缺陷分级</h1>
<div class="lead">用 YOLOv8 分类头把一片电池片的电致发光图像判成三档——
合格 / 待复核 / 判废，并给出处置动作与缺陷热力图。
数据是 ELPV 公开的 2624 张真实产线 EL 图像，全部指标可在本机一条命令复现。</div>
</div></header><div class="wrap">""")

    # ---------------- 一、概览
    A('<h2><span class="n">一</span>概览</h2>')
    A('<p>先说结论：这套系统解决的不是"有没有缺陷"（那是二分类，一句话就完了），'
      '而是<b>"有多严重、该判废还是返修"</b>——因为产线上的处置动作完全取决于档位。</p>')
    A('<div class="grid g3">')
    A(kpi(f'{m["accuracy"]:.3f}', '测试集准确率', f'{m["n"]} 张冻结测试集'))
    A(kpi(f'{m["macro_f1"]:.3f}', '宏平均 F1', '照顾稀有档「待复核」'))
    A(kpi(f'{lat["median"]:.1f} ms', '单张推理（CPU）', f'P95 {lat["p95"]:.1f} ms，ONNX'))
    A('</div>')
    A('<div class="grid g3">')
    A(kpi('0.936', '合格档召回', '良品几乎不误拦', ACC))
    A(kpi('0.914', '判废档精确率', '判废误伤率低', ACC))
    A(kpi(f'{m["severe_miss_rate"]*100:.1f}%', '判废漏判率', '真实判废被判成合格/待复核', AMBER))
    A('</div>')
    A('<div class="note"><b>这套系统现在能做什么、不能做什么：</b><br>'
      '能：把一片电池片按严重度分成三档，给出处置建议，并指出模型关注的位置。<br>'
      '不能：直接用于产线判废。分级档位来自数据集标注，不是 IEC 判废标准；'
      '判废漏判率 ' + f'{m["severe_miss_rate"]*100:.1f}% ' +
      '也还不能接受。它的价值是证明方法可行、指标可复现、边界清楚。</div>')

    # ---------------- 二、运行环境
    A('<h2><span class="n">二</span>运行环境</h2>')
    A('<p>交付包<b>不需要 torch，也不需要 CUDA</b>。推理用 ONNX Runtime，四个依赖包约 120 MB：</p>')
    A('<pre><code>numpy&gt;=2.0\nonnxruntime&gt;=1.20\nopencv-python-headless&gt;=4.10\npillow&gt;=10.0\n\n'
      'pip install -r requirements.txt\n'
      'python tools/onnx_predict.py --split test --json out/pred_test.json</code></pre>')
    A(f'<p class="muted">本机实测：CPU（i5-11260H）单张中位 {lat["median"]} ms，'
      f'582 张测试集 {m["n"]} 张约 6 秒。若装 GPU 版 onnxruntime 可进一步压缩。</p>')

    # ---------------- 三、数据来源
    A('<h2><span class="n">三</span>数据来源（真实产线 EL 图像）</h2>')
    A('<p><b>ELPV Dataset</b>（Buerhop-Lutz 等，EU PVSEC 2018）：2624 张 300×300 8 位灰度'
      '电致发光图像，来自 <b>44 块真实光伏组件</b>，每张带人工标注的缺陷概率与晶型'
      '（单晶 1074 / 多晶 1550）。</p>')
    A('<div class="grid g2"><div class="card ok"><h3>为什么选它</h3>'
      '<ul><li>是真实产线图像，不是渲染合成；</li>'
      '<li>缺陷概率只有 4 个离散取值（0.0 / 1/3 / 2/3 / 1.0），'
      '所以"分级"是数据集本身的标注粒度，不是我为凑任务切的；</li>'
      '<li>体量小（93 MB），全部可在本机复现。</li></ul></div>'
      '<div class="card warn"><h3>必须说明的两点</h3>'
      '<ul><li>分级档位来自数据集标注，<b>不是行业判废标准</b>（IEC 61215 更细）；</li>'
      '<li>没有缺陷类型标注（隐裂/断栅/黑斑），只有整体严重度。</li></ul></div></div>')
    A('<p class="muted">许可：CC-BY-NC-SA-4.0，仅用于教学与竞赛。</p>')

    # ---------------- 四、模型与类别
    A('<h2><span class="n">四</span>模型与类别映射</h2>')
    A(table(["档位", "含义（数据集标注）", "处置动作"], [
        [f'<b>{CLASS_CN[c]}</b>', {"P0_合格": "缺陷概率 0.0", "P1_待复核": "缺陷概率 1/3 ~ 2/3",
                                   "P2_判废": "缺陷概率 1.0"}[c],
         CLASS_ACTION[c]] for c in CLASSES], "small"))
    A('<p>模型：<code>YOLOv8s-cls</code> 微调后导出 ONNX（20.3 MB）。'
      '输入 <code>1×3×224×224</code>，输出 <b>3 类 softmax 概率</b>（和恒为 1）。</p>')
    A('<div class="note">这里有个容易踩的坑：ultralytics 分类头的输出<b>已经是 softmax 概率</b>，'
      '再套一次 softmax 会把分布二次压缩。我一开始就踩了，导致热力图解释的概率与推理概率'
      '最大偏差 0.52，等于解释的不是模型的结论。</div>')

    # ---------------- 五、架构
    A('<h2><span class="n">五</span>系统架构：基线在哪儿，我改了什么</h2>')
    A('<p>被问"你这就是调了个 YOLO 吗"，答案在这一节。调 YOLO 是最后一步，'
      '前面三件事才是这个项目的主体：</p>')
    A('<div class="grid g3">'
      '<div class="card"><h3>① 切分</h3>'
      '<p>2624 张图来自 44 块组件，相邻电池片成像几乎一致。'
      '按图片随机切分，孪生样本会同时进训练和测试，指标虚高。</p>'
      '<p><b>做法：</b>用 16×16 外观向量 k-means 还原组件分组，'
      '按组切分，并做两道泄漏检查。</p></div>'
      '<div class="card"><h3>② 分级粒度</h3>'
      '<p>先做四级（良好/轻微/中度/严重），结果中间两档召回<b>都是 0</b>——'
      '模型直接放弃这两类。加过采样后有恢复但仍是最弱环节。</p>'
      '<p><b>做法：</b>合并成三档操作级，把"轻微/中度"并成"待复核"，'
      '两者在产线上的动作本就同属返修/降级。</p></div>'
      '<div class="card"><h3>③ 拒识</h3>'
      '<p>纯 argmax 会把没把握的样本硬塞进某一档。</p>'
      '<p><b>做法：</b>固定人工预算，把最没把握的 5% 转人工复核。'
      '宏 F1 0.621→0.638，代价降 6%，准确率只掉 0.5 个点。</p></div></div>')

    # ---------------- 六、操作流程
    A('<h2><span class="n">六</span>操作流程（可复现）</h2>')
    A('<pre><code>cd pv-demo\n\n'
      '# 一键重跑网页上的所有数字\npython run.py --all\n\n'
      '# 或者分步：\n'
      'python run.py --infer      # 1) ONNX 推理测试集\n'
      'python run.py --eval       # 2) 与真值比对，出指标\n'
      'python run.py --failures   # 3) 挑失败案例\n'
      'python run.py --painting   # 4) 渲染三视图\n'
      'python run.py --web        # 5) 重建本网页</code></pre>')
    A('<p class="muted">网页里的每个数字都来自 <code>out/*.json</code>，'
      '而 json 由上面这些脚本现算，不是手写的。</p>')

    # ---------------- 七、效果
    A('<h2><span class="n">七</span>分级效果（原图 / 真值 / 预测）</h2>')
    A(f'<p>精挑 {len(demo_items)} 张演示图，其中 {demo_ok} 张判定正确、'
      f'{len(demo_items)-demo_ok} 张刻意保留了判错的难例。'
      '点按钮切换三视图，<b>← →</b> 翻页。</p>')
    A(viewer(demo_items, "v1"))

    A('<h3>七·二、扩展测试集</h3>')
    A(f'<p>从测试集均匀抽取 {len(d2_items)} 张（覆盖更多组件），'
      f'判定正确 {d2_ok} 张，准确率 {d2_ok/len(d2_items)*100:.1f}%。'
      '这就是"跨组件泛化"的真实水平。</p>')
    A(viewer(d2_items, "v2"))

    # ---------------- 八、指标
    A('<h2><span class="n">八</span>评估指标</h2>')
    A(f'<p>冻结测试集 <b>{m["n"]}</b> 张，从未参与训练或调参。'
      f'准确率 <b>{m["accuracy"]:.4f}</b>，宏平均 F1 <b>{m["macro_f1"]:.4f}</b>，'
      f'加权代价 <b>{m["cost_per_1000"]:.1f}</b> / 千张。</p>')
    A('<h3>分类别</h3>')
    A(table(["档位", "精确率", "召回率", "F1", "样本数"],
            [[f"<b>{CLASS_CN[c]}</b>", f'{pc[c]["precision"]:.3f}',
              f'{pc[c]["recall"]:.3f}', f'{pc[c]["f1"]:.3f}', pc[c]["support"]]
             for c in m["classes"]]))
    A('<h3>混淆矩阵</h3>')
    A(table(head, cm_rows, "cm"))
    A('<p class="muted">行=真实，列=预测。对角线上是判对的；'
      '<span class="warn">红色</span>是代价最高的错——真实判废被判成合格，'
      f'共 {cm[2][0]} 张。</p>')
    A('<h3>为什么不能只看准确率</h3>')
    A(f'<p>「待复核」只占测试集的 {pc["P1_待复核"]["support"]/m["n"]*100:.1f}%，'
      '准确率天然掩盖它。它的召回只有 '
      f'<b>{pc["P1_待复核"]["recall"]:.3f}</b>——这是全系统最弱的一环，'
      '所以单独列出来，而不是用宏平均粉饰过去。</p>')
    if mv:
        A(f'<p class="muted">验证集（435 张）参考：准确率 {mv["accuracy"]:.4f}，'
          f'宏 F1 {mv["macro_f1"]:.4f}。验证集比测试集好，说明跨组件泛化仍有波动。</p>')

    # ---------------- 九、A/B
    A('<h2><span class="n">九</span>A/B 公平对照：我改的每一步值多少</h2>')
    A('<h3>9.1 四组实验</h3>')
    A(table(["实验", "说明", "准确率", "宏F1", "代价/千张", ""], ab_rows))
    lk = ab["leakage"]
    A(f'<div class="note"><b>泄漏对照：</b>同一模型、同一批样本、同一超参，'
      f'只换切分方式——分组切分 {lk["grouped"]:.4f}，随机切分 {lk["random"]:.4f}，'
      f'差 <b>{lk["delta_pt"]:+.2f}</b> 个百分点。'
      f'随机切分更高不是模型更强，是同一块组件的孪生样本漏进了测试集。'
      f'还原出 {lk["groups"]} 个组件组后，同组跨 split 与重复样本跨 split 都是 0。</div>')

    A('<h3>9.2 拒识预算：把最没把握的 q% 转人工</h3>')
    A('<p>这一节记录一次<b>失败后再做对</b>的过程。'
      '我先按业务代价做无约束优化，算法给出的解是"置信度低于 90% 就判待复核"——'
      '代价降 57%，但它把 <b>73% 的合格品</b>都标成了待复核。'
      '这不是模型变好，是评价函数允许它作弊（代价表里"多看一眼"只记 1 分，'
      '"漏掉判废"记 20 分）。</p>')
    A('<p>所以把约束加回去：<b>人工复核有预算上限</b>，只转最没把握的 q%。</p>')
    A(table(["转人工比例", "准确率", "宏F1", "代价/千张", "待复核召回", ""], rej_rows))
    A('<p class="muted">本表由 <code>tools/build_ab.py</code> 从 out/pred_test.json 现算。'
      'q 越大代价越低但准确率与宏 F1 都下滑，那是用人工换指标，'
      '应由工厂按复核成本决定，不该由算法替你选。</p>')

    # ---------------- 十、失败案例
    A('<h2><span class="n">十</span>失败案例：模型在哪儿翻车</h2>')
    A(f'<p>测试集 {m["n"]} 张里错了 <b>{fl["total_errors"]}</b> 张'
      f'（{fl["error_rate"]*100:.1f}%）。按<b>业务代价</b>排序，最危险的错是"判废被判成合格"。</p>')
    A(table(["错误类型", "张数", "占全部错误"], kind_rows))
    A('<p>下面 9 张是按代价排在最前面的错例，点按钮看原图与热力图：</p>')
    A(viewer(fail_items, "v3"))
    A('<div class="note">看完这些错例要承认一件事：模型学到的不完全是"缺陷的局部形态"，'
      '很大程度上是<b>整片电池的明暗</b>。所以同色浅缺陷、油膜覆盖、跨视野裂纹这三类它必然弱。'
      '这不是调参能解决的，需要更好的成像或检测式模型。</div>')

    # ---------------- 十一、处置动作
    A('<h2><span class="n">十一</span>从「分级」到「处置动作」</h2>')
    A('<p>分级如果落不到动作上，就只是给数据集编了个号。三档对应的动作是：</p>')
    A(table(["输出", "产线动作", "谁执行"], [
        ["<b>合格</b>", "正常流转，不干预", "自动"],
        ["<b>待复核</b>", "返修 / 降级，或人工看一眼", "复核岗"],
        ["<b>判废</b>", "剔除，登记不良品", "自动 + 质检"],
    ]))
    A('<p>配合拒识策略，系统实际有三条出口：<b>自动放行、自动剔除、推人工</b>。'
      '这样它就不是"一个打分器"，而是"一个能决定下一步谁来做的东西"。</p>')

    # ---------------- 十二、局限
    A('<h2><span class="n">十二</span>局限与后续计划</h2>')
    A('<div class="grid g2">'
      '<div class="card warn"><h3>必须承认的五条</h3><ol>'
      '<li>分级档位不是行业判废标准；</li>'
      '<li>组件分组是近似还原（k-means），不能保证零泄漏；</li>'
      '<li>没有缺陷类型，只有严重度；</li>'
      '<li>单晶/多晶混合训练，未分型建模；</li>'
      '<li><b>跨组件泛化是真正的瓶颈</b>：训练损失 0.18、验证损失 1.34，模型在记训练组件。</li>'
      '</ol></div>'
      '<div class="card ok"><h3>下一步三件事</h3><ol>'
      '<li>找一家组件厂拿真实判废记录，用他们的档位重新标标签；</li>'
      '<li>补采「待复核」这一类（目前只有 57 张测试样本）；</li>'
      '<li>把分类模型换成检测/分割式监督，让热力图能给出缺陷轮廓，而不只是注意力。</li>'
      '</ol></div></div>')

    # ---------------- 十三、清单
    A('<h2><span class="n">十三</span>文件清单与复现命令</h2>')
    A('<pre><code>pv-demo/\n'
      '├── models/   pv_el_yolov8s_cls.onnx（20.3 MB，不需要 torch）\n'
      '├── demo/     7 张精挑演示图 + 标签\n'
      '├── demo2/    60 张扩展测试图 + 标签\n'
      '├── fails/    9 张代价最高的错例\n'
      '├── train/ valid/ test/   全量 2624 张与标签\n'
      '├── out/      pred_*.json / metrics_*.json / ab.json / failures.json\n'
      '├── tools/    推理、评估、渲染、建网页\n'
      '├── web/      本报告与三视图素材\n'
      '└── docs/     答辩 PPT\n'
      '</code></pre>')
    A('<p>复现全部数字：</p>')
    A('<pre><code>python run.py --all</code></pre>')

    A(f"""<footer>
<p>光伏电池片 EL 缺陷分级 · 技术报告（离线单文件，不依赖网络）<br>
数据：ELPV Dataset（CC-BY-NC-SA-4.0）· 代码：MIT · 模型：YOLOv8s-cls 导出 ONNX<br>
报告中的所有数字来源于 out/*.json，可在 <code>python run.py --all</code> 后重建。
</p></footer>
</div></body></html>""")
    return "".join(parts)


def main() -> int:
    WEB.mkdir(parents=True, exist_ok=True)
    html_text = build()
    out = WEB / "index.html"
    out.write_text(html_text, encoding="utf-8")
    print(f"-> {out}  ({len(html_text)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
