# -*- coding: utf-8 -*-
"""在可达的渠道（GitHub / Zenodo）里搜索公开数据集，并核实每个来源确实存在。"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

H = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}


def get_json(url: str, timeout: int = 20):
    with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=timeout) as r:
        return json.loads(r.read())


def gh_repos(term: str, n: int = 8):
    url = ("https://api.github.com/search/repositories?q="
           + urllib.parse.quote(term) + f"&sort=stars&order=desc&per_page={n}")
    try:
        d = get_json(url)
    except Exception as e:
        print(f"    查询失败: {type(e).__name__}")
        return []
    return d.get("items", [])


def zenodo(term: str, n: int = 8):
    url = ("https://zenodo.org/api/records?q=" + urllib.parse.quote(term)
           + f"&size={n}&sort=mostrecent")
    try:
        d = get_json(url)
    except Exception as e:
        print(f"    查询失败: {type(e).__name__}")
        return []
    return d.get("hits", {}).get("hits", [])


def main() -> int:
    print("=== GitHub 仓库搜索（按 star 排序）===")
    for term in ("electroluminescence dataset",
                 "solar cell defect dataset",
                 "photovoltaic defect detection",
                 "wafer defect dataset",
                 "PCB defect dataset"):
        print(f"\n[{term}]")
        for it in gh_repos(term):
            print(f"  ★{it['stargazers_count']:<6d} {it['full_name']}")
            desc = (it.get("description") or "")[:90]
            if desc:
                print(f"          {desc}")

    print("\n\n=== Zenodo 记录搜索 ===")
    for term in ("photovoltaic electroluminescence", "solar cell defect",
                 "PV module defect"):
        print(f"\n[{term}]")
        for rec in zenodo(term):
            md = rec.get("metadata", {})
            title = (md.get("title") or "")[:80]
            size = rec.get("files", [])
            total = sum(f.get("size", 0) for f in size) / 1e6
            print(f"  {title}")
            print(f"    文件 {len(size)} 个，合计 {total:.1f} MB  doi={rec.get('doi')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
