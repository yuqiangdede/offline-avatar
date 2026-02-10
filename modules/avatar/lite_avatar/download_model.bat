@echo off
REM Download LiteAvatar model files using modelscope Python API

echo Downloading LiteAvatar model files...

set PYTHON_BIN=python
if exist "..\..\..\.venv\Scripts\python.exe" (
    set PYTHON_BIN=..\..\..\.venv\Scripts\python.exe
)

%PYTHON_BIN% download_model.py
if %errorlevel% neq 0 (
    echo Error downloading LiteAvatar model files
    pause
    exit /b 1
)

echo All model files downloaded successfully!
