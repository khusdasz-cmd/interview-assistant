@echo off
cd /d "%~dp0"
echo === Interview Assistant v2 ===

if not exist config.toml (
    echo.
    echo [First run] Please configure your API Key:
    copy config.example.toml config.toml >nul
    echo Generated config.toml. Open it and fill in your API Key.
    echo.
    pause
    start notepad config.toml
    exit /b
)

if not exist venv\ (
    echo [First run] Creating virtual environment...
    python -m venv venv
    echo Installing dependencies...
    call venv\Scripts\pip.exe install -r requirements.txt -q
    echo Dependencies installed.
)

echo Starting Interview Assistant...
call venv\Scripts\python.exe interview-assistant.py
pause
