@echo off
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
