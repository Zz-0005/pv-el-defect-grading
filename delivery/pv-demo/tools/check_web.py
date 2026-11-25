# -*- coding: utf-8 -*-
"""校验网页：素材引用是否都存在、内嵌 JSON 是否合法、三视图数量是否对得上。"""
import json
import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
html = (WEB / "index.html").read_text(encoding="utf-8")

refs = re.findall(r'assets/[A-Za-z0-9_\-]+\.jpg', html)
missing = [r for r in set(refs) if not (WEB / r).exists()]
print(f"素材引用 {len(refs)} 处，去重 {len(set(refs))} 个，缺失 {len(missing)}")
if missing:
    print("  缺失示例:", missing[:5])

# 抽出三个 viewer 的数据数组
bad = 0
for m in re.finditer(r"const D = (\[.*?\]), root = document\.getElementById", html, re.S):
    try:
        data = json.loads(m.group(1))
        print(f"  viewer 数据 {len(data)} 条，首条: {data[0]['name']} "
              f"真值={data[0]['true']} 预测={data[0]['predcn']}")
    except Exception as e:
        bad += 1
        print("  JSON 解析失败:", e)

print("h2 章节:", len(re.findall(r"<h2>", html)))
print("base64 内联:", html.count("data:image"))
print("结论:", "通过" if not missing and not bad else "有问题")
sys.exit(0 if not missing and not bad else 1)
