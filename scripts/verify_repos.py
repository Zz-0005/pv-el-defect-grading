# -*- coding: utf-8 -*-
"""核实候选仓库里到底有没有数据：列目录、算体积、数图片文件。"""
from __future__ import annotations

import json
import urllib.request

H = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
API = "https://api.github.com"

REPOS = [
    ("QQ767172261", "How-to-use-Yolov5-training-Solar-cell-monocrystalline-silicon-defect-detection-dataset-owns-1108-ima"),
    ("wch313", "PSCDE-Dataset"),
    ("abj247", "PV-S3"),
    ("binyisu", "PVEL-AD"),
    ("Lanpangzihzn", "DM-PV-dataset"),
    ("lyc686", "MNFI-Net"),
    ("umair-naeem1", "EL-Image-curation"),
    ("tangsanli5201", "DeepPCB"),
    ("ismh16", "PCBA-Dataset"),
]


def get(url: str, timeout: int = 20):
    with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
        return json.loads(r.read())


def human(n: int) -> str:
    return f"{n/1024/1024:.1f} MB" if n >= 1 << 20 else f"{n/1024:.0f} KB"


def probe(owner: str, repo: str) -> None:
    full = f"{owner}/{repo}"
    print(f"\n{'='*70}\n{full}")
    try:
        meta = get(f"{API}/repos/{full}")
    except Exception as e:
        print(f"  仓库不可访问: {type(e).__name__}")
        return
    print(f"  ★{meta['stargazers_count']}  fork {meta['forks_count']}  "
          f"仓库体积 {meta['size']/1024:.1f} MB  默认分支 {meta['default_branch']}")
    lic = (meta.get("license") or {}).get("spdx_id", "无")
    print(f"  许可证: {lic}")

    try:
        items = get(f"{API}/repos/{full}/contents/")
    except Exception as e:
        print(f"  目录读取失败: {type(e).__name__}")
        return
    for it in items:
        kind = "DIR " if it["type"] == "dir" else "file"
        size = human(it["size"]) if it["type"] == "file" else ""
        print(f"    {kind} {it['name']:<48s} {size}")

    # 递归统计图片与标注文件（只看前 3 层，避免请求过多）
    n_img = n_xml = n_txt = n_json = 0
    def walk(path: str, depth: int):
        nonlocal n_img, n_xml, n_txt, n_json
        if depth > 3:
            return
        try:
            rows = get(f"{API}/repos/{full}/contents/{path}")
        except Exception:
            return
        if isinstance(rows, dict):
            return
        for r in rows:
            if r["type"] == "dir":
                walk(r["path"], depth + 1)
            else:
                ext = r["name"].rsplit(".", 1)[-1].lower()
                if ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
                    n_img += 1
                elif ext == "xml":
                    n_xml += 1
                elif ext == "txt":
                    n_txt += 1
                elif ext == "json":
                    n_json += 1

    for it in items:
        if it["type"] == "dir":
            walk(it["path"], 1)
    print(f"  → 图片 {n_img} 张，XML 标注 {n_xml}，TXT 标注 {n_txt}，JSON {n_json}")


def main() -> int:
    for owner, repo in REPOS:
        probe(owner, repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
