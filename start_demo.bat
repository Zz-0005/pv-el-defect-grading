@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PYTHONPATH=%~dp0src"
set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" goto noenv

echo ============================================
echo   PV EL defect grading - demo
echo ============================================
echo.
echo   Browser will open the demo page
echo   Drag images from demo\samples into the page
echo.
echo   Close this window to stop
echo ============================================
echo.

set "PORT=8000"
netstat -ano | findstr "LISTENING" | findstr ":8000 " >nul 2>&1
if not errorlevel 1 set "PORT=8010"

if "%PORT%"=="8010" echo [note] port 8000 busy, using 8010

start "" "http://127.0.0.1:%PORT%"
"%PY%" -m pvinspect serve --port %PORT%

echo.
echo Server stopped
pause
exit /b 0

:noenv
echo [ERROR] venv not found: %PY%
echo Run first: python -m venv .venv
pause
exit /b 1
