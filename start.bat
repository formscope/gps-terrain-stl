@echo off
cd /d "%~dp0"
echo ============================================
echo   GPS Terrain STL Generator
echo ============================================
echo.

:: Check if Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH!
    echo Please install Python or add it to your PATH.
    pause
    exit /b 1
)

:: Check if Flask is installed
python -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Flask...
    python -m pip install flask
)

echo Starting server on http://localhost:5000
echo Press Ctrl+C to stop the server.
echo.

:: Open browser after short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:5000"

:: Start Flask (stays in foreground)
python app.py

echo.
echo Server stopped.
pause
