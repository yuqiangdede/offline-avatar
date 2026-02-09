@echo off
REM Download LiteAvatar model files using modelscope Python API

echo Downloading LiteAvatar model files...

python download_model.py
if %errorlevel% neq 0 (
    echo Error downloading lite_avatar_weights
    pause
    exit /b 1
)

echo All model files downloaded successfully!
