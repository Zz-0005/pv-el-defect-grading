# -*- coding: utf-8 -*-
"""
受限执行环境下的兼容层。

问题：ultralytics 在扫描数据集时用 `multiprocessing.pool.ThreadPool` 并行校验图像，
而 ThreadPool 内部会创建 `SimpleQueue` → `Pipe`，在 Windows 上落到
`_winapi.CreateFile` 创建命名管道。某些沙箱/受限令牌环境会直接拒绝
（表现为 `PermissionError: [WinError 5] 拒绝访问`，堆栈停在 pool.py 的 SimpleQueue）。

做法：把该模块里的 ThreadPool 换成一个串行实现，接口保持
`with ... as pool: pool.imap(func=..., iterable=...)`。
代价是图像校验从并行变串行（2624 张小图约 3~5 秒），换来的是在任何环境下都能跑。

这是**环境适配**，不是算法修改：不影响训练结果，只影响扫描速度。
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path


def configure_dirs() -> None:
    """
    把 ultralytics / matplotlib 的缓存目录指到工作区。

    默认它们会写 C:\\Users\\<user>\\AppData\\Roaming\\Ultralytics，
    在受限环境里这是只读的，会导致：字体下载 PermissionError、
    后续绘图与 settings.json 落盘失败。指到 artifacts/cache 下即可。
    """
    from .config import WORK
    pairs = {
        "YOLO_CONFIG_DIR": WORK / "cache" / "ultralytics",
        "MPLCONFIGDIR": WORK / "cache" / "matplotlib",
    }
    for var, path in pairs.items():
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(var, str(p))



class SerialPool:
    """ThreadPool 的最小串行替代品，只实现 ultralytics 用到的方法。"""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, func, iterable, chunksize=1):
        for item in iterable:
            yield func(item)

    def map(self, func, iterable, chunksize=1):
        return [func(x) for x in iterable]

    def close(self):
        pass

    def join(self):
        pass


def patch_ultralytics(verbose: bool = False) -> bool:
    """把 ultralytics 的并行校验改成串行。返回是否做了替换。"""
    try:
        import ultralytics.data.dataset as ds
    except Exception:
        return False
    if getattr(ds, "_pv_serial", False):
        return False
    ds.ThreadPool = SerialPool          # type: ignore[attr-defined]
    ds._pv_serial = True                # type: ignore[attr-defined]
    if verbose:
        print("[compat] 已将 ultralytics 数据集校验改为串行（沙箱不允许多进程管道）")
    return True


def patch_torch_threads(n: int = 4) -> None:
    """限制 torch 线程数，避免和 DataLoader 抢核导致的小图训练抖动。"""
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
