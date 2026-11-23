# -*- coding: utf-8 -*-
"""
分块续传下载器。

pip 在这种网络下会卡死：单条长连接传 2.4 GB 的 wheel，中途断流后
既不报错也不重试，进程挂着不动（实测 45 秒内文件增长 0 字节）。
这里改成用 HTTP Range 分块拉取，每块独立重试，断点可续。
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def head_size(url: str, timeout: int = 30) -> int:
    req = urllib.request.Request(url, headers=H, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return int(r.headers.get("Content-Length") or 0)


def download(url: str, dst: Path, chunk: int = 8 << 20, retries: int = 6,
             verbose: bool = True) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = head_size(url)
    if total == 0:
        raise RuntimeError("服务器未返回 Content-Length")
    if dst.exists() and dst.stat().st_size == total:
        if verbose:
            print(f"已存在完整文件：{dst.name} ({total/1e6:.1f} MB)")
        return dst

    pos = dst.stat().st_size if dst.exists() else 0
    if pos > total:
        pos = 0
        dst.unlink()

    t0 = time.time()
    base = pos
    with dst.open("ab") as f:
        while pos < total:
            end = min(pos + chunk - 1, total - 1)
            req = urllib.request.Request(url, headers={**H, "Range": f"bytes={pos}-{end}"})
            for attempt in range(retries):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = r.read()
                    if len(data) != end - pos + 1:
                        raise IOError(f"块长度不符 {len(data)} != {end - pos + 1}")
                    f.write(data)
                    f.flush()
                    pos += len(data)
                    break
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    if verbose:
                        print(f"    块重试 {attempt + 1}/{retries}: {type(e).__name__}", flush=True)
                    time.sleep(1.5 * (attempt + 1))
            if verbose:
                done = pos - base
                dt = max(time.time() - t0, 1e-6)
                pct = pos * 100 / total
                eta = (total - pos) / max(done / dt, 1e-6)
                print(f"  {pos/1e6:7.1f}/{total/1e6:.1f} MB  {pct:5.1f}%  "
                      f"{done/1e6/dt:5.2f} MB/s  剩余 {eta/60:4.1f} min", flush=True)
    if dst.stat().st_size != total:
        raise RuntimeError(f"下载不完整 {dst.stat().st_size} != {total}")
    return dst


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("用法: resumable_download.py <url> <输出文件>")
        return 2
    download(argv[1], Path(argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
