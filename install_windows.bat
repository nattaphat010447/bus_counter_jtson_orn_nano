@echo off
REM ==========================================
REM Windows Installation Script
REM Focus: Virtual Environment & ONNX Runtime
REM ==========================================

echo [Info] Setting up Windows Environment...

REM Create and activate Virtual Environment
python -m venv venv
call venv\Scripts\activate

REM Install dependencies
echo [Info] Installing requirements...
pip install -U pip
pip install -r requirements.txt

REM Check and download missing models
echo [Info] Checking and downloading missing models...
python tools/download_models.py

REM Export YOLO to ONNX
echo [Info] Exporting YOLO models...
python tools/yolo_export.py

echo [Info] Windows setup completed.
pause