# -*- coding: utf-8 -*-
"""生成交付包根目录的启动器、图标与快捷方式脚本（CRLF + UTF-8）。"""
from __future__ import annotations

import io
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT.parent / "光伏电池片EL缺陷分级_交付包"


def w(path: Path, text: str, crlf: bool = True, enc: str = "utf-8") -> None:
    data = text.replace("\r\n", "\n")
    if crlf:
        data = data.replace("\n", "\r\n")
    path.write_bytes(data.encode(enc))
    print(f"  {path.name}  ({len(data)} chars)")


# ------------------------------------------------------------------ 图标
def make_icon() -> None:
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([8, 8, size - 8, size - 8], radius=44, fill=(14, 111, 115, 255))
    # 电池片栅线
    for i in range(1, 5):
        x = 8 + (size - 16) * i / 5
        d.line([(x, 40), (x, size - 40)], fill=(255, 255, 255, 70), width=3)
    for i in range(1, 3):
        y = 40 + (size - 80) * i / 3
        d.line([(40, y), (size - 40, y)], fill=(255, 255, 255, 55), width=3)
    # 一道裂纹
    d.line([(70, 190), (110, 120), (150, 170), (196, 84)], fill=(240, 178, 107, 255), width=9)
    try:
        f = ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc", 54)
    except Exception:
        f = ImageFont.load_default()
    d.text((size / 2, 208), "EL", font=f, fill=(255, 255, 255, 230), anchor="mm")

    buf = io.BytesIO()
    img.save(buf, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    (PKG / "app.ico").write_bytes(buf.getvalue())
    img.resize((128, 128), Image.LANCZOS).save(PKG / "app.png")
    print(f"  app.ico / app.png  ({len(buf.getvalue())/1024:.1f} KB)")


# ------------------------------------------------------------------ 启动器
REPORT_PYW = '''# -*- coding: utf-8 -*-
"""只打开离线技术报告（不启动服务、不联网）。"""
import os
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
page = os.path.join(ROOT, "pv-demo", "web", "index.html")

if not os.path.exists(page):
    import tkinter.messagebox as mb
    mb.showerror("文件缺失", "找不到技术报告：\\n" + page)
    sys.exit(1)

webbrowser.open("file:///" + page.replace("\\\\", "/"))
'''

MAIN_PYW = '''# -*- coding: utf-8 -*-
"""主启动器：打开技术报告 + 启动本地演示服务（可在浏览器里传自己的图）。"""
import os
import socket
import subprocess
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(ROOT, "pv-demo")
PY = os.path.join(DEMO, "_python.txt")   # 由创建快捷方式时写入，可选
LOG = os.path.join(ROOT, "启动日志.txt")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("[%H:%M:%S] ") + str(msg) + "\\n")


def free_port(start=8000):
    for p in range(start, start + 40, 10):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
    return start


def main():
    page = os.path.join(DEMO, "web", "index.html")
    if not os.path.exists(page):
        log("缺少 web/index.html")
        return

    # 1) 技术报告（离线静态页）
    webbrowser.open("file:///" + page.replace("\\\\", "/"))
    log("打开技术报告")

    # 2) 本地演示服务：新开一个控制台窗口，关掉窗口即停止服务
    port = free_port(8000)
    try:
        subprocess.Popen(
            [sys.executable, os.path.join(DEMO, "tools", "serve_onnx.py"), "--port", str(port)],
            cwd=DEMO, creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        log("启动演示服务 :%d" % port)
        time.sleep(3.5)
        webbrowser.open("http://127.0.0.1:%d" % port)
    except Exception as e:
        log("启动服务失败: %r" % (e,))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("未捕获异常: %r" % (e,))
'''

BAT = '''@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   光伏电池片 EL 缺陷分级 - 启动
echo ============================================
echo.
echo  [1] 打开技术报告（离线网页）
start "" "pv-demo\\web\\index.html"
echo  [2] 启动本地演示服务 :8000
echo      关闭本窗口即停止服务
echo.
cd pv-demo
python tools\\serve_onnx.py --port 8000
if errorlevel 1 (
  echo.
  echo  [错误] 找不到 python 或依赖未安装
  echo  先执行： pip install -r pv-demo\\requirements.txt
)
pause
'''

SHORTCUT_PS1 = '''# 重建桌面快捷方式：右键本文件 ->「使用 PowerShell 运行」
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$ico = Join-Path $here "app.ico"

# 找一个 pythonw.exe
$pyw = $null
foreach ($c in @("pythonw.exe", "python.exe")) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { $pyw = $cmd.Source; if ($c -eq "pythonw.exe") { break } }
}
if (-not $pyw) {
    Write-Host "未找到 python，请先安装 Python 3.10+ 并加入 PATH" -ForegroundColor Red
    pause; exit 1
}
Write-Host "使用： $pyw"

$items = @(
    @{ name = "光伏电池片EL缺陷分级"; target = "启动_报告与演示.pyw"; desc = "技术报告 + 本地演示服务" },
    @{ name = "1_技术报告";           target = "只看技术报告.pyw";   desc = "离线网页技术报告" },
    @{ name = "2_本地演示";           target = "启动_报告与演示.pyw"; desc = "上传图片现场分级" }
)

$ws = New-Object -ComObject WScript.Shell
foreach ($it in $items) {
    $lnkPath = Join-Path $desktop ($it.name + ".lnk")
    $lnk = $ws.CreateShortcut($lnkPath)
    $lnk.TargetPath = $pyw
    $lnk.Arguments = '"' + (Join-Path $here $it.target) + '"'
    $lnk.WorkingDirectory = $here
    $lnk.IconLocation = $ico
    $lnk.Description = $it.desc
    $lnk.Save()
    Write-Host ("已创建：" + $lnkPath) -ForegroundColor Green
}
Write-Host "`n完成。若 pythonw 路径变化，重新运行本脚本即可。" -ForegroundColor Cyan
pause
'''


def main() -> int:
    print("生成启动器与图标：")
    make_icon()
    w(PKG / "只看技术报告.pyw", REPORT_PYW)
    w(PKG / "启动_报告与演示.pyw", MAIN_PYW)
    w(PKG / "启动.bat", BAT)
    w(PKG / "创建快捷方式.ps1", SHORTCUT_PS1, enc="utf-8-sig")   # PS1 带 BOM 才认中文
    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
