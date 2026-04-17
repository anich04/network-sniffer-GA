@echo off
echo ============================================================
echo   GA Network Capture -- Backend Setup
echo ============================================================

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ and retry.
    pause & exit /b 1
)

:: Install deps if not present
echo [*] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo [*] Starting Flask API on http://localhost:5050
echo     Press Ctrl+C to stop.
echo.
python app.py
pause
