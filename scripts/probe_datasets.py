# -*- coding: utf-8 -*-
"""
数据集可达性探测：从本机实测哪些公开数据集真能下、多大、要不要授权。

只列链接而不验证，等于没查。这里对候选做三件事：
  1. HuggingFace datasets API 搜索（无需 token）
  2. 固定候选 URL 的 HEAD 探测（看状态码与体积）
  3. 记录失败原因（DNS / TLS / 403 / 404），区分"没有"和"��访问不到"
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def head(url: str, timeout: int = 12) -> tuple[str, str]:
    try:
        req = urllib.request.Request(url, headers=H, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            size = f"{int(n)/1e6:.1f} MB" if n and n.isdigit() else "体积未知"
            return f"OK {r.status}", size
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}", ""
    except Exception as e:
        return f"FAIL {type(e).__name__}", ""


def get_json(url: str, timeout: int = 15):
    req = urllib.request.Request(url, headers=H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def hf_search(term: str, limit: int = 12):
    url = f"https://huggingface.co/api/datasets?search={urllib.parse.quote(term)}&limit={limit}"
    try:
        rows = get_json(url)
        out = []
        for d in rows:
            out.append({
                "id": d.get("id"),
                "downloads": d.get("downloads", 0),
                "likes": d.get("likes", 0),
                "tags": [t for t in d.get("tags", []) if t in
                         ("vision", "image-classification", "object-detection",
                          "image-segmentation", "anomaly-detection")],
            })
        return out
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]


CANDIDATES = [
    ("ELPV（本项目已用）", "https://github.com/zae-bayern/elpv-dataset"),
    ("PVEL-AD（36,543 张 EL，12 类带框）", "https://github.com/binyisu/PVEL-AD"),
    ("PVEL-AD README", "https://raw.githubusercontent.com/binyisu/PVEL-AD/main/README.md"),
    ("MVTec AD 数据集页", "https://www.mvtec.com/company/research/datasets/mvtec-ad"),
    ("Kolektor SDD2", "https://www.vicos.si/resources/kolektorsdd2/"),
    ("DAGM 2007 弱监督缺陷", "https://hci.iwr.uni-heidelberg.de/node/3616"),
    ("NEU 表面缺陷（钢）", "http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/"),
    ("HuggingFace 主页", "https://huggingface.co"),
    ("Kaggle API 数据集搜索", "https://www.kaggle.com/api/v1/datasets/list?search=solar"),
    ("Zenodo", "https://zenodo.org"),
    ("Roboflow Universe", "https://universe.roboflow.com"),
    ("Papers with Code 数据集", "https://paperswithcode.com/datasets"),
]


def main() -> int:
    import platform
    print(f"本机: {platform.system()} {platform.release()}")
    print("\n=== 1. 固定候选 URL（HEAD 探测）===")
    for name, url in CANDIDATES:
        status, size = head(url)
        print(f"  [{status:14s}] {size:>10s}  {name}")

    print("\n=== 2. HuggingFace 数据集搜索 ===")
    for term in ("photovoltaic", "electroluminescence", "solar cell defect",
                 "wafer defect", "pcb defect", "solder joint"):
        print(f"\n  关键词: {term}")
        rows = hf_search(term)
        if rows and "error" in rows[0]:
            print("    搜索失败:", rows[0]["error"])
            continue
        if not rows:
            print("    无结果")
        for d in rows[:8]:
            print(f"    {d['id']:<55s} ↓{d['downloads']:<7d} ♥{d['likes']:<4d} "
                  f"{','.join(d['tags'])}")
    return 0


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  （hf_search 里用到）
    raise SystemExit(main())
