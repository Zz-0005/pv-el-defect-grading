# -*- coding: utf-8 -*-
"""生成 Windows 一键启动脚本（必须 CRLF，cmd 不接受纯 LF）。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BAT = r"""@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHONPATH=%~dp0src"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" goto noenv

echo ============================================
echo   光伏电池片 EL 缺陷分级 - 演示服务
echo ============================================
echo.
echo   浏览器会自动打开演示页面
echo   把 demo\样本 文件夹里的图片拖进去即可
echo.
echo   关闭本窗口 = 停止服务
echo ============================================
echo.

set "PORT=8000"
netstat -ano | findstr "LISTENING" | findstr ":8000 " >nul 2>&1
if not errorlevel 1 set "PORT=8010"

if "%PORT%"=="8010" echo [提示] 8000 端口被占用，改用 8010

start "" "http://127.0.0.1:%PORT%"
"%PY%" -m pvinspect serve --port %PORT%

echo.
echo 服务已停止，可以关闭窗口
pause
exit /b 0

:noenv
echo [错误] 找不到虚拟环境：%PY%
echo 请先在项目目录执行： python -m venv .venv
pause
exit /b 1
"""


def main() -> int:
    p = ROOT / "启动演示.bat"
    data = BAT.replace("\r\n", "\n").replace("\n", "\r\n")
    p.write_bytes(data.encode("utf-8"))
    print(f"written {p} ({len(data)} chars, CRLF)")

    # 另存一份纯 ASCII 的备用脚本：中文注释编码出问题时可退回到这个
    ascii_bat = BAT.replace("光伏电池片 EL 缺陷分级 - 演示服务", "PV EL defect grading - demo")
    ascii_bat = ascii_bat.replace("浏览器会自动打开演示页面", "Browser will open the demo page")
    ascii_bat = ascii_bat.replace("把 demo\\样本 文件夹里的图片拖进去即可", "Drag images from demo\\samples into the page")
    ascii_bat = ascii_bat.replace("关闭本窗口 = 停止服务", "Close this window to stop")
    ascii_bat = ascii_bat.replace("[提示] 8000 端口被占用，改用 8010", "[note] port 8000 busy, using 8010")
    ascii_bat = ascii_bat.replace("服务已停止，可以关闭窗口", "Server stopped")
    ascii_bat = ascii_bat.replace("[错误] 找不到虚拟环境：%PY%", "[ERROR] venv not found: %PY%")
    ascii_bat = ascii_bat.replace("请先在项目目录执行： python -m venv .venv", "Run first: python -m venv .venv")
    ascii_bat = ascii_bat.replace("chcp 65001 >nul\r\n", "")
    data2 = ascii_bat.replace("\r\n", "\n").replace("\n", "\r\n")
    p2 = ROOT / "start_demo.bat"
    p2.write_bytes(data2.encode("ascii", "ignore"))
    print(f"written {p2} (ascii fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
