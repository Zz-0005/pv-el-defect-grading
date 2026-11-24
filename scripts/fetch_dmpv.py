# -*- coding: utf-8 -*-
"""
下载 DM-PV 数据集（红外 + 可见光双模态光伏缺陷，6 类，YOLO 标注）。

为什么需要这个脚本而不用 git clone：
该仓库用 **Git LFS** 存图像，普通 zip 下载只能拿到 131 字节的指针文件
（本机也没装 git-lfs）。但 LFS 内容可以从 media.githubusercontent.com 直接取，
所以这里按需逐张下载。

用法：
    python scripts/fetch_dmpv.py --limit 40 --modality ir
    python scripts/fetch_dmpv.py --limit 0 --modality both   # 0 = 全部

注意：全部下载 about 3.1 GB（IR 2376×0.38MB + VIS 2376×0.94MB），
本机实测速度约 0.5~1 MB/s，全量需要 1 小时以上。答辩前只建议取小样本。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = "https://media.githubusercontent.com/media/Lanpangzihzn/DM-PV-dataset/main/"
RAW = "https://raw.githubusercontent.com/Lanpangzihzn/DM-PV-dataset/main/"
H = {"User-Agent": "Mozilla/5.0"}

CLASSES = ["Single Hotspot", "Large-area Hotspot", "Diode Short Circuit",
           "Bird Droppings", "Weed Occlusion", "Sunlight Reflection"]


def fetch(url: str, dst: Path, retries: int = 4, min_size: int = 1024) -> bool:
    """min_size 只对图像有意义：YOLO 标注文件本来就只有几百字节。"""
    if dst.exists() and dst.stat().st_size >= min_size:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=60) as r:
                data = r.read()
            if len(data) < min_size:
                raise IOError(f"只取到 {len(data)} 字节，可能是 LFS 指针")
            dst.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:         # 不存在就别重试，省时间
                return False
            if i == retries - 1:
                print(f"    FAIL {dst.name}: HTTP {e.code}")
                return False
            time.sleep(1.5 * (i + 1))
        except Exception as e:
            if i == retries - 1:
                print(f"    FAIL {dst.name}: {type(e).__name__} {e}")
                return False
            time.sleep(1.5 * (i + 1))
    return False


def discover_names(zip_path: Path | None) -> list[str]:
    """
    取标注文件名。优先读本地 zip（标注文件不是 LFS 对象，能直接读），
    这样不依赖 GitHub API（免限流），也不用猜编号。
    """
    if zip_path and zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path) as z:
            names = [Path(n).name for n in z.namelist()
                     if "/labels/" in n.replace("\\", "/") and n.endswith(".txt")]
        if names:
            print(f"[源] 本地 zip：{zip_path.name}，{len(names)} 个标注")
            return sorted(names)
    try:
        import json
        with urllib.request.urlopen(urllib.request.Request(
                "https://api.github.com/repos/Lanpangzihzn/DM-PV-dataset/contents/labels",
                headers=H), timeout=30) as r:
            names = sorted(x["name"] for x in json.loads(r.read()) if x["name"].endswith(".txt"))
        print(f"[源] GitHub API：{len(names)} 个标注")
        return names
    except Exception as e:
        print(f"[错误] 拿不到文件清单（{type(e).__name__}）。")
        print("       请先下载仓库 zip 并用 --zip 指定，例如：")
        print("       curl -L -o dm-pv.zip https://codeload.github.com/Lanpangzihzn/DM-PV-dataset/zip/refs/heads/main")
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40, help="取多少个编号对（0=全部）")
    ap.add_argument("--modality", choices=("ir", "vis", "both"), default="ir")
    ap.add_argument("--out", default=str(ROOT / "data" / "dmpv"))
    ap.add_argument("--zip", default="", help="仓库 zip 路径，用于读取真实文件名")
    a = ap.parse_args()

    out = Path(a.out)
    for sub in ("images_ir", "images_vis", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    zip_path = Path(a.zip) if a.zip else None
    if zip_path is None:
        for cand in (ROOT.parent / ".dsprobe" / "DM-PV.zip", ROOT / "dm-pv.zip"):
            if Path(cand).exists():
                zip_path = Path(cand)
                break
    files = discover_names(zip_path)
    if not files:
        return 2

    import json
    (out / "classes.json").write_text(
        json.dumps({"classes": CLASSES}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    picked = files[: a.limit] if a.limit else files
    print(f"目标 {len(picked)} 组，模态 {a.modality}")
    ok = 0
    t0 = time.time()
    for i, fn in enumerate(picked, 1):
        stem = fn[:-4]
        got = True
        if a.modality in ("ir", "both"):
            got &= fetch(MEDIA + f"images_ir/{stem}.jpg", out / "images_ir" / f"{stem}.jpg")
        if a.modality in ("vis", "both"):
            got &= fetch(MEDIA + f"images_vis/{stem}.jpg", out / "images_vis" / f"{stem}.jpg")
        got &= fetch(RAW + f"labels/{fn}", out / "labels" / fn, min_size=1)
        ok += int(got)
        if i % 10 == 0 or i == len(picked):
            dt = time.time() - t0
            print(f"  {i}/{len(picked)}  成功 {ok}  用时 {dt:.0f}s")

    n_img = len(list((out / "images_ir").glob("*.jpg"))) + len(list((out / "images_vis").glob("*.jpg")))
    n_lab = len(list((out / "labels").glob("*.txt")))
    print(f"\n完成：图像 {n_img} 张，标注 {n_lab} 个 -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
