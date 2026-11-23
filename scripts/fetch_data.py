# -*- coding: utf-8 -*-
"""下载 ELPV 数据集仓库并解包到 data/elpv。"""
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
ZIP_URL = "https://codeload.github.com/zae-bayern/elpv-dataset/zip/refs/heads/master"
H = {"User-Agent": "Mozilla/5.0"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "elpv.zip"
    if not dst.exists() or dst.stat().st_size < 1_000_000:
        print(f"下载 {ZIP_URL}", flush=True)
        t0 = time.time()
        buf = io.BytesIO()
        with urllib.request.urlopen(urllib.request.Request(ZIP_URL, headers=H), timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                chunk = r.read(1 << 18)
                if not chunk:
                    break
                buf.write(chunk)
                got += len(chunk)
                if total and got % (8 << 20) < (1 << 18):
                    pct = got * 100 / total
                    print(f"  {got/1e6:6.1f}/{total/1e6:.1f} MB  {pct:5.1f}%  {got/1e6/(time.time()-t0):.1f} MB/s", flush=True)
        dst.write_bytes(buf.getvalue())
        print(f"  完成 {dst.stat().st_size/1e6:.1f} MB  用时 {time.time()-t0:.1f}s", flush=True)

    tgt = OUT / "elpv"
    if (tgt / "src" / "elpv_dataset" / "data" / "labels.csv").exists():
        print("已解包，跳过")
        return
    print("解包...", flush=True)
    with zipfile.ZipFile(dst) as z:
        names = z.namelist()
        root = names[0].split("/")[0]
        for n in names:
            if n.endswith("/"):
                continue
            rel = n[len(root) + 1:]
            if not rel:
                continue
            p = tgt / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, p.open("wb") as f:
                f.write(src.read())
    n_img = len(list((tgt / "src" / "elpv_dataset" / "data" / "images").glob("*.png")))
    print(f"解包完成，图片 {n_img} 张")


if __name__ == "__main__":
    sys.exit(main())
