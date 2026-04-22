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
python.exe -m pip install --upgrade pip
pip install -U pip
pip install -r requirements.txt
pip install transformers

REM Check and download missing models
echo [Info] Checking and downloading missing models...
python tools/download_models.py

REM Update ultralytics
echo Updating ultralytics...
pip install -U ultralytics

REM Export YOLO to ONNX
echo [Info] Exporting YOLO models...
python tools/yolo_export.py

REM Install MiVOLO explicitly without build isolation
echo [Info] Installing MiVOLO...
pip install --upgrade wrapt 
pip install --no-build-isolation git+https://github.com/WildChlamydia/MiVOLO.git

REM miVOLO Conversion
echo [Info] Processing miVOLO model...
python tools/mivolo_export.py
pip install -U ultralytics

echo [Info] Windows setup completed.
pause