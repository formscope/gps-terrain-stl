@echo off
cd /d "%~dp0"
echo ============================================
echo   GPS Terrain STL Generator (PUBLIC URL)
echo ============================================
echo.

:: --- Python check ---
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found in PATH.
    pause
    exit /b 1
)

:: --- Flask check ---
python -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Flask...
    python -m pip install flask
)

:: --- cloudflared check ---
set "CFD=%LocalAppData%\Microsoft\WinGet\Links\cloudflared.exe"
if not exist "%CFD%" (
    where cloudflared >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: cloudflared not found.
        echo Install with: winget install Cloudflare.cloudflared
        pause
        exit /b 1
    )
    set "CFD=cloudflared"
)

echo Starting Flask server in the background on http://localhost:5000 ...
start "GPS-Terrain-STL Flask" /min cmd /c "python app.py"

:: Give Flask a moment to come up before opening the tunnel
timeout /t 3 /nobreak >nul

echo.
echo Opening Cloudflare tunnel - your public URL will appear below.
echo Share the https://*.trycloudflare.com link with anyone you like.
echo Press Ctrl+C to stop the tunnel (Flask keeps running in its own window).
echo.

"%CFD%" tunnel --url http://localhost:5000

echo.
echo Tunnel stopped.
pause
