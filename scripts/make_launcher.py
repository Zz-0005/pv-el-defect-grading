# -*- coding: utf-8 -*-
"""
生成 Windows 一键启动脚本。

两条硬性要求，都是踩过的坑：
  1. **必须 CRLF**：cmd 不接受纯 LF，会把多行粘成一行然后报语法错。
  2. **内容只用 ASCII**：中文在 GBK 控制台下会被按字节切断，
     导致 "…已停止，可以关闭窗口" 这种半截指令被当成命令执行。
     （`chcp 65001` 也救不了——cmd 逐行读取时的解码时机早于它生效。）
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BAT = r"""@echo off
setlocal
cd /d "%~dp0"

set "PYTHONPATH=%~dp0src"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" goto noenv

echo ============================================
echo   PV EL defect grading - demo (batch mode)
echo ============================================
echo.
echo   A browser will open the demo page.
echo   Drag several EL images in, or use the
echo   sample buttons. Results export to CSV.
echo.
echo   Close this window to stop the server.
echo ============================================
echo.

rem Port is resolved by the program: if 8000 is busy it tries 8001, 8002, ...
"%PY%" -m pvinspect serve --port 8000 --open

echo.
echo Server stopped.
pause
exit /b 0

:noenv
echo [ERROR] venv not found: %PY%
echo Run this first:  python -m venv .venv
pause
exit /b 1
"""


def main() -> int:
    data = BAT.replace("\r\n", "\n").replace("\n", "\r\n")
    for name in ("启动演示.bat", "start_demo.bat"):
        p = ROOT / name
        p.write_bytes(data.encode("ascii"))
        print(f"written {p} ({len(data)} chars, CRLF, ASCII)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
