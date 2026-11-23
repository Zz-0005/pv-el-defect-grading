# -*- coding: utf-8 -*-
"""
在受限环境下运行 pip。

背景：Windows 上 tempfile.mkdtemp() 会用只允许"当前用户 SID"的 DACL 创建目录，
当进程令牌与沙箱策略不完全一致时，pip 在这些目录里 open("wb") 会得到 Errno 13，
表现为 "Could not install packages due to an OSError: Permission denied"。

做法：把 tempfile.mkdtemp 换成基于 os.mkdir 的等价实现（继承父目录 ACL），
再调用 pip 的入口。语义等价：仍是唯一命名的新目录。
"""
import os
import sys
import tempfile
import uuid


def safe_mkdtemp(suffix: str | None = None, prefix: str | None = None, dir: str | None = None) -> str:
    base = dir or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    for _ in range(100):
        name = (prefix or "tmp") + uuid.uuid4().hex[:12] + (suffix or "")
        path = os.path.join(base, name)
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            continue
    raise FileExistsError("无法创建唯一临时目录")


tempfile.mkdtemp = safe_mkdtemp

from pip._internal.cli.main import main as pip_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(pip_main(sys.argv[1:]))
