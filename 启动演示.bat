@echo off
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
